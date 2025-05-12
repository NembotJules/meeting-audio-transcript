import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import warnings
import torch
import torchaudio
import json
import re
import base64
from mistralai import Mistral

# Suppression des avertissements pour un affichage plus propre
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Outil de Transcription de Réunion", page_icon=":microphone:", layout="wide")

def transcribe_audio(audio_file, file_extension, model_size="base"):
    """Transcribe the uploaded audio file to text using the Whisper model"""
    try:
        model_id_mapping = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
        }
        model_id = model_id_mapping.get(model_size, "openai/whisper-base")
        transcriber = pipeline("automatic-speech-recognition", model=model_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
            temp_audio.write(audio_file.getvalue())
            temp_audio_path = temp_audio.name
        
        try:
            waveform, sample_rate = torchaudio.load(temp_audio_path, backend="ffmpeg")
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            result = transcriber({"raw": waveform[0].numpy(), "sampling_rate": 16000}, chunk_length_s=30, stride_length_s=5)
            return result["text"]
        finally:
            os.unlink(temp_audio_path)
    except Exception as e:
        st.error(f"Erreur lors de la transcription audio: {e}")
        return f"Erreur lors de la transcription audio: {e}"

def extract_context_from_report(file, mistral_api_key):
    """Extract text from the uploaded file using Mistral OCR for PDFs/images or python-docx for .docx."""
    if not file:
        return ""
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'docx':
        try:
            doc = Document(file)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error processing .docx file: {e}")
            return ""
    elif file_extension in ['pdf', 'png', 'jpg', 'jpeg']:
        if not mistral_api_key:
            st.error("Mistral API Key is required for processing PDF and image files.")
            return ""
        try:
            client = Mistral(api_key=mistral_api_key)
            file_content = file.read()
            base64_content = base64.b64encode(file_content).decode('utf-8')
            document = {
                "type": "document_base64",
                "document_base64": base64_content
            }
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document=document
            )
            context = ""
            for block in ocr_response.keys:
                if block.type == "text":
                    context += block.text + "\n"
            return context.strip()
        except Exception as e:
            st.error(f"Error processing file with Mistral OCR: {e}")
            return ""
    else:
        st.error("Unsupported file type.")
        return ""

def answer_question_with_context(question, context, deepseek_api_key):
    """Answer a question based on the extracted context using Deepseek API."""
    if not context:
        return "No context available."
    prompt = f"""
    As an assistant, answer the following question based on the provided context.

    **Context**:
    {context}

    **Question**:
    {question}

    **Answer**:
    """
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {deepseek_api_key}"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def extract_info(transcription, meeting_title, date, deepseek_api_key, previous_context=""):
    """Extract key information from the transcription using Deepseek API with previous context."""
    prompt = f"""
    Vous êtes un assistant IA spécialisé dans la rédaction de comptes rendus de réunion. 
    À partir de la transcription suivante et du contexte de la réunion précédente, extrayez les informations clés et retournez-les sous forme de JSON structuré en français.

    **Contexte de la réunion précédente** :
    {previous_context if previous_context else "Aucun contexte disponible."}

    **Transcription de la réunion actuelle** :
    {transcription}

    **Sections à extraire** :
    - **presence_list** : Liste des participants présents et absents sous forme de chaîne (ex. "Présents : Alice, Bob\nAbsents : Charlie"). Identifiez les noms mentionnés comme présents ou absents dans la transcription. Si non trouvé, utilisez "Présents : Non spécifié\nAbsents : Non spécifié".
    - **agenda_items** : Liste des points de l'ordre du jour sous forme de chaîne (ex. "1. Relecture du compte rendu\n2. Résolutions"). Déduisez les points discutés ou explicitement mentionnés comme ordre du jour. Si non trouvé, utilisez "Non spécifié".
    - **resolutions_summary** : Liste de résolutions sous forme de tableau (liste de dictionnaires avec les clés "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count"). "date", "deadline" et "execution_date" doivent être au format DD/MM/YYYY. "report_count" est une chaîne (ex. "0").
    - **sanctions_summary** : Liste de sanctions sous forme de tableau (liste de dictionnaires avec les clés "name", "reason", "amount", "date", "status"). "date" doit être au format DD/MM/YYYY, "amount" est une chaîne.
    - **start_time** : Heure de début de la réunion (format HHhMMmin, ex. 07h00min). Déduisez-la si mentionnée, sinon utilisez "Non spécifié".
    - **end_time** : Heure de fin de la réunion (format HHhMMmin, ex. 10h34min). Déduisez-la si mentionnée, sinon utilisez "Non spécifié".
    - **rapporteur** : Nom du rapporteur de la réunion. Déduisez-le si mentionné, sinon utilisez "Non spécifié".
    - **president** : Nom du président de la réunion. Déduisez-le si mentionné, sinon utilisez "Non spécifié".
    - **balance_amount** : Solde du compte DRI Solidarité (ex. "827540"). Déduisez-le si mentionné, sinon utilisez "Non spécifié".
    - **balance_date** : Date du solde (format DD/MM/YYYY). Déduisez-la si mentionnée, sinon utilisez la date fournie : {date}.

    **Instructions** :
    1. Pour chaque intervenant, identifiez ses contributions, résolutions et actions assignées.
    2. Si une information n’est pas trouvée, utilisez des valeurs par défaut raisonnables (ex. "Non spécifié" ou la date fournie : {date}).
    3. Assurez-vous que le JSON est bien formé et que toutes les dates respectent le format DD/MM/YYYY.
    4. Priorisez les informations claires et exploitables, en évitant les détails non pertinents.

    Retournez le résultat sous forme de JSON structuré, en français.
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            raw_response = response.json()["choices"][0]["message"]["content"].strip()
            st.write(f"Réponse brute de Deepseek : {raw_response}")
            try:
                extracted_data = json.loads(raw_response)
                extracted_data["date"] = date
                return extracted_data
            except json.JSONDecodeError as e:
                st.error(f"Erreur lors du parsing JSON : {e}")
                return None
        else:
            st.error(f"Erreur API Deepseek : Statut {response.status_code}, Message : {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des informations : {e}")
        return None

def extract_info_fallback(transcription, meeting_title, date):
    """Fallback mode for structuring information if Deepseek API fails"""
    extracted_data = {
        "date": date,
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "presence_list": "Présents : Non spécifié\nAbsents : Non spécifié",
        "agenda_items": "Non spécifié",
        "resolutions_summary": [{
            "date": date,
            "dossier": "Non spécifié",
            "resolution": "Non spécifié",
            "responsible": "Non spécifié",
            "deadline": "Non spécifié",
            "execution_date": "",
            "status": "En cours",
            "report_count": "0"
        }],
        "sanctions_summary": [{
            "name": "Aucun",
            "reason": "Aucune sanction mentionnée",
            "amount": "0",
            "date": date,
            "status": "Non appliqué"
        }],
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "rapporteur": "Non spécifié",
        "president": "Non spécifié",
        "transcription": transcription
    }

    start_time_match = re.search(r"(?:a commencé à|début.*?\b)(\d{1,2}h\d{2})\b", transcription, re.IGNORECASE)
    end_time_match = re.search(r"(?:s'est terminée à|fin.*?\b)(\d{1,2}h\d{2})\b", transcription, re.IGNORECASE)
    if start_time_match:
        extracted_data["start_time"] = start_time_match.group(1) + "min"
    if end_time_match:
        extracted_data["end_time"] = end_time_match.group(1) + "min"

    presence_match = re.search(r"(?:présents|présences)\s*:\s*([^.\n]+)", transcription, re.IGNORECASE)
    absence_match = re.search(r"(?:absents|absences)\s*:\s*([^.\n]+)", transcription, re.IGNORECASE)
    presents = presence_match.group(1).strip() if presence_match else "Non spécifié"
    absents = absence_match.group(1).strip() if absence_match else "Non spécifié"
    extracted_data["presence_list"] = f"Présents : {presents}\nAbsents : {absents}"

    agenda_match = re.search(r"(?:ordre du jour|agenda)\s*:\s*([\s\S]*?)(?=\n\n|\Z)", transcription, re.IGNORECASE)
    if agenda_match:
        agenda_lines = [line.strip("- \t") for line in agenda_match.group(1).split("\n") if line.strip()]
        extracted_data["agenda_items"] = "\n".join(agenda_lines) if agenda_lines else "Non spécifié"

    rapporteur_match = re.search(r"rapporteur\s*:\s*([^.\n]+)", transcription, re.IGNORECASE)
    president_match = re.search(r"(?:président|présidente)\s*(?:de\s*la\s*réunion)?\s*:\s*([^.\n]+)", transcription, re.IGNORECASE)
    if rapporteur_match:
        extracted_data["rapporteur"] = rapporteur_match.group(1).strip()
    if president_match:
        extracted_data["president"] = president_match.group(1).strip()

    balance_match = re.search(r"solde\s*(?:du compte)?\s*.*?(\d+(?:,\d{3})*)\s*(?:XAF|FCFA)?\s*(?:au|le)\s*(\d{1,2}/\d{1,2}/\d{4})", transcription, re.IGNORECASE)
    if balance_match:
        extracted_data["balance_amount"] = balance_match.group(1).replace(",", "")
        extracted_data["balance_date"] = balance_match.group(2)

    resolutions_section = re.search(r"(?:résolutions prises|résolutions)\s*:(.*?)(?=\n\n(?:sanctions|le solde|$))", transcription, re.IGNORECASE | re.DOTALL)
    if resolutions_section:
        resolution_lines = resolutions_section.group(1).strip().split("\n")
        resolutions = []
        for line in resolution_lines:
            match = re.match(r"-?\s*(?:sur\s*)?(.*?):\s*(.*?),\s*responsable\s*([^,]+),\s*statut\s*([^,]+),\s*(?:aucun report|nombre de reports\s*(\d+))\s*(?:d'ici le|pour le|avant le)?\s*(\d{1,2}/\d{1,2}/\d{4})?", line.strip(), re.IGNORECASE)
            if match:
                dossier, resolution, responsible, status = match.groups()[:4]
                report_count = "0" if "aucun report" in line.lower() else match.group(5) if match.group(5) else "0"
                deadline = match.group(6) if match.group(6) else "Non spécifié"
                resolutions.append({
                    "date": date,
                    "dossier": dossier.strip() if dossier else "Non spécifié",
                    "resolution": resolution.strip() if resolution else "Non spécifié",
                    "responsible": responsible.strip() if responsible else "Non spécifié",
                    "deadline": deadline,
                    "execution_date": "",
                    "status": status.strip() if status else "En cours",
                    "report_count": report_count
                })
        if resolutions:
            extracted_data["resolutions_summary"] = resolutions

    sanctions_section = re.search(r"sanctions\s*:(.*?)(?=\n\nle solde|$)", transcription, re.IGNORECASE | re.DOTALL)
    if sanctions_section:
        sanction_lines = sanctions_section.group(1).strip().split("\n")
        sanctions = []
        for line in sanction_lines:
            match = re.match(r"-?\s*([^,]+),\s*([^,]+),\s*(\d+)\s*(?:FCFA|XAF),\s*le\s*(\d{1,2}/\d{1,2}/\d{4}),\s*([^.]+)", line.strip(), re.IGNORECASE)
            if match:
                name, reason, amount, sanction_date, status = match.groups()
                sanctions.append({
                    "name": name.strip() if name else "Aucun",
                    "reason": reason.strip() if reason else "Aucune sanction mentionnée",
                    "amount": amount.strip() if amount else "0",
                    "date": sanction_date.strip() if sanction_date else date,
                    "status": status.strip() if status else "Non appliqué"
                })
        if sanctions:
            extracted_data["sanctions_summary"] = sanctions

    return extracted_data

def to_roman(num):
    """Convert an integer to a Roman numeral."""
    roman_numerals = {
        1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
        6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
    }
    return roman_numerals.get(num, str(num))

def set_cell_background(cell, rgb_color):
    """Set the background color of a table cell using RGB values."""
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), f"{rgb_color[0]:02X}{rgb_color[1]:02X}{rgb_color[2]:02X}")
    cell._element.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=0.1, bottom=0.1, left=0.1, right=0.1):
    """Set the margins of a table cell to adjust padding."""
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin, value in zip(['top', 'bottom', 'left', 'right'], [top, bottom, left, right]):
        margin_elm = OxmlElement(f'w:{margin}')
        margin_elm.set(qn('w:w'), str(int(value * 1440)))
        margin_elm.set(qn('w:type'), 'dxa')
        tcMar.append(margin_elm)
    tcPr.append(tcMar)

def set_table_width(table, width_in_inches):
    """Set the width of the table and allow columns to adjust proportionally."""
    table.autofit = False
    table.allow_autofit = False
    table_width = Inches(width_in_inches)
    table.width = table_width
    for row in table.rows:
        for cell in row.cells:
            cell.width = table_width
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def set_column_widths(table, widths_in_inches):
    """Set preferred widths for each column in the table."""
    for i, width in enumerate(widths_in_inches):
        for row in table.rows:
            cell = row.cells[i]
            cell.width = Inches(width)

def add_styled_paragraph(doc, text, font_name="Century", font_size=12, bold=False, color=None, alignment=WD_ALIGN_PARAGRAPH.LEFT):
    """Add a styled paragraph to the document."""
    p = doc.add_paragraph(text)
    p.alignment = alignment
    run = p.runs[0]
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color
    return p

def add_styled_table(doc, rows, cols, headers, data, header_bg_color=(0, 0, 0), header_text_color=(255, 255, 255), alt_row_bg_color=(192, 192, 192), column_widths=None, table_width=6.5):
    """Add a styled table to the document with background colors and custom widths."""
    table = doc.add_table(rows=rows, cols=cols)
    try:
        table.style = "Table Grid"
    except KeyError:
        st.warning("Le style 'Table Grid' n'est pas disponible. Utilisation du style par défaut.")
    
    set_table_width(table, table_width)
    if column_widths:
        set_column_widths(table, column_widths)
    
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        run = cell.paragraphs[0].runs[0]
        run.font.name = "Century"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(*header_text_color)
        set_cell_background(cell, header_bg_color)
    
    for i, row_data in enumerate(data):
        row = table.rows[i + 1]
        if (i + 1) % 2 == 0:
            for cell in row.cells:
                set_cell_background(cell, alt_row_bg_color)
        for j, cell_text in enumerate(row_data):
            cell = row.cells[j]
            cell.text = cell_text
            run = cell.paragraphs[0].runs[0]
            run.font.name = "Century"
            run.font.size = Pt(12)
    
    return table

def add_text_in_box(doc, text, bg_color=(192, 192, 192), font_size=14, box_width_in_inches=5.0):
    """Add text inside a single-cell table with a background color."""
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_width(table, box_width_in_inches)
    cell = table.cell(0, 0)
    cell.text = text
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.runs[0]
    run.font.name = "Century"
    run.font.size = Pt(font_size)
    run.font.bold = True
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=0.2, bottom=0.2, left=0.3, right=0.3)
    return table

def fill_template_and_generate_docx(extracted_info):
    """Build the Word document from scratch using python-docx"""
    try:
        doc = Document()

        # Extract presence list and split into present and absent attendees
        presence_list = extracted_info.get("presence_list", "Présents : Non spécifié\nAbsents : Non spécifié")
        present_attendees = []
        absent_attendees = []
        if "Présents :" in presence_list and "Absents :" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents :"):
                    presents = part.replace("Présents :", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                elif part.startswith("Absents :"):
                    absents = part.replace("Absents :", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Non spécifié" else []

        # Process agenda items
        agenda_list = extracted_info.get("agenda_items", "Non spécifié").split("\n")
        agenda_list = [f"{to_roman(idx)}. {item.strip()}" for idx, item in enumerate(agenda_list, 1) if item.strip() and item != "Non spécifié"]
        if not agenda_list:
            agenda_list = ["I. Non spécifié"]

        # Add header box
        add_text_in_box(
            doc,
            "Direction Recherches et Investissements",
            bg_color=(192, 192, 192),
            font_size=16,
            box_width_in_inches=5.0
        )

        # Add meeting title
        add_styled_paragraph(
            doc,
            "COMPTE RENDU DE REUNION HEBDOMADAIRE",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0),
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )

        # Add date
        add_styled_paragraph(
            doc,
            extracted_info.get("date", ""),
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0),
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )

        # Add start and end times
        add_styled_paragraph(
            doc,
            f"Heure de début : {extracted_info.get('start_time', 'Non spécifié')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        add_styled_paragraph(
            doc,
            f"Heure de fin : {extracted_info.get('end_time', 'Non spécifié')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )

        # Add rapporteur and president
        rapporteur = extracted_info.get("rapporteur", "Non spécifié")
        president = extracted_info.get("president", "Non spécifié")
        if rapporteur != "Non spécifié":
            add_styled_paragraph(
                doc,
                f"Rapporteur : {rapporteur}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )
        if president != "Non spécifié":
            add_styled_paragraph(
                doc,
                f"Président de Réunion : {president}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )

        # Add attendance table
        add_styled_paragraph(
            doc,
            "◆ LISTE DE PRÉSENCE/ABSENCE",
            font_name="Century",
            font_size=12,
            bold=True
        )

        if present_attendees or absent_attendees:
            max_rows = max(len(present_attendees), len(absent_attendees))
            if max_rows == 0:
                max_rows = 1
            attendance_data = []
            for i in range(max_rows):
                present_text = present_attendees[i] if i < len(present_attendees) else ""
                absent_text = absent_attendees[i] if i < len(absent_attendees) else ""
                attendance_data.append([present_text, absent_text])
            
            attendance_column_widths = [3.25, 3.25]
            add_styled_table(
                doc,
                rows=max_rows + 1,
                cols=2,
                headers=["PRÉSENCES", "ABSENCES"],
                data=attendance_data,
                header_bg_color=(0, 0, 0),
                header_text_color=(255, 255, 255),
                alt_row_bg_color=(192, 192, 192),
                column_widths=attendance_column_widths,
                table_width=6.5
            )
        else:
            add_styled_paragraph(
                doc,
                "Aucune présence ou absence spécifiée.",
                font_name="Century",
                font_size=12
            )

        # Add agenda items
        add_styled_paragraph(
            doc,
            "◆ Ordre du jour",
            font_name="Century",
            font_size=12,
            bold=True
        )
        for item in agenda_list:
            add_styled_paragraph(
                doc,
                item,
                font_name="Century",
                font_size=12
            )

        # Add resolutions summary
        resolutions = extracted_info.get("resolutions_summary", [])
        if not resolutions:
            resolutions = [{
                "date": extracted_info.get("date", ""),
                "dossier": "Non spécifié",
                "resolution": "Non spécifié",
                "responsible": "Non spécifié",
                "deadline": "Non spécifié",
                "execution_date": "",
                "status": "En cours",
                "report_count": "0"
            }]
        add_styled_paragraph(
            doc,
            "RÉCAPITULATIF DES RÉSOLUTIONS",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        resolutions_headers = ["DATE", "DOSSIERS", "RÉSOLUTIONS", "RESP.", "DÉLAI D'EXÉCUTION", "DATE D'EXÉCUTION", "STATUT", "NBR DE REPORT"]
        resolutions_data = []
        for resolution in resolutions:
            row_data = [
                resolution.get("date", ""),
                resolution.get("dossier", ""),
                resolution.get("resolution", ""),
                resolution.get("responsible", ""),
                resolution.get("deadline", ""),
                resolution.get("execution_date", ""),
                resolution.get("status", ""),
                str(resolution.get("report_count", ""))
            ]
            resolutions_data.append(row_data)
        resolutions_column_widths = [0.9, 1.2, 1.8, 0.8, 1.2, 0.9, 0.8, 0.9]
        add_styled_table(
            doc,
            rows=len(resolutions) + 1,
            cols=8,
            headers=resolutions_headers,
            data=resolutions_data,
            header_bg_color=(0, 0, 0),
            header_text_color=(255, 255, 255),
            alt_row_bg_color=(192, 192, 192),
            column_widths=resolutions_column_widths,
            table_width=7.5
        )

        # Add sanctions summary
        sanctions = extracted_info.get("sanctions_summary", [])
        if not sanctions:
            sanctions = [{
                "name": "Aucun",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info.get("date", ""),
                "status": "Non appliqué"
            }]
        add_styled_paragraph(
            doc,
            "RÉCAPITULATIF DES SANCTIONS",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        sanctions_headers = ["NOM", "MOTIF", "MONTANT (FCFA)", "DATE", "STATUT"]
        sanctions_data = []
        for sanction in sanctions:
            row_data = [
                sanction.get("name", ""),
                sanction.get("reason", ""),
                sanction.get("amount", ""),
                sanction.get("date", ""),
                sanction.get("status", "")
            ]
            sanctions_data.append(row_data)
        sanctions_column_widths = [1.5, 1.8, 1.4, 1.2, 1.6]
        add_styled_table(
            doc,
            rows=len(sanctions) + 1,
            cols=5,
            headers=sanctions_headers,
            data=sanctions_data,
            header_bg_color=(0, 0, 0),
            header_text_color=(255, 255, 255),
            alt_row_bg_color=(192, 192, 192),
            column_widths=sanctions_column_widths,
            table_width=7.5
        )

        # Add balance information
        add_styled_paragraph(
            doc,
            f"Le solde du compte DRI Solidarité (00001-00921711101-10) est de XAF {extracted_info.get('balance_amount', 'Non spécifié')} au {extracted_info.get('balance_date', '')}.",
            font_name="Century",
            font_size=12
        )

        # Save the document to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            with open(tmp.name, "rb") as f:
                docx_data = f.read()
            os.unlink(tmp.name)
        return docx_data

    except Exception as e:
        st.error(f"Erreur lors de la génération du document Word : {e}")
        return None

def main():
    st.title("Outil de Transcription de Réunion")
    
    # Sidebar for API keys and previous report
    st.sidebar.header("Configuration")
    st.session_state.mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    st.session_state.deepseek_api_key = st.sidebar.text_input("Deepseek API Key", type="password")
    
    st.sidebar.header("Previous Context")
    previous_report = st.sidebar.file_uploader("Upload Previous Report (optional)", type=["docx", "pdf", "png", "jpg", "jpeg"])
    if previous_report:
        status_text = st.sidebar.empty()
        status_text.text("Extracting context...")
        context = extract_context_from_report(previous_report, st.session_state.mistral_api_key)
        status_text.text("Context extracted successfully!")
        st.session_state.previous_context = context
    else:
        st.session_state.previous_context = ""
    
    # Context testing section
    st.sidebar.subheader("Test the Context")
    question = st.sidebar.text_input("Ask a question about the previous context:")
    if st.sidebar.button("Get Answer") and question and 'previous_context' in st.session_state:
        with st.spinner("Generating answer..."):
            answer = answer_question_with_context(question, st.session_state.previous_context, st.session_state.deepseek_api_key)
        st.sidebar.write("**Answer:**", answer)
    
    # Main app content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
    
    with col2:
        st.header("Transcription & Sortie")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])
        whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=1)
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if st.button("Transcribe Audio"):
                with st.spinner(f"Transcribing with Whisper {whisper_model}..."):
                    transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                    if transcription and not transcription.startswith("Erreur"):
                        st.session_state.transcription = transcription
                        st.text_area("Transcription", transcription, height=200)
        
        if 'transcription' in st.session_state:
            if st.button("Extract Information"):
                with st.spinner("Extracting information..."):
                    extracted_info = extract_info(
                        st.session_state.transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        st.session_state.deepseek_api_key,
                        st.session_state.get("previous_context", "")
                    )
                    if not extracted_info:
                        extracted_info = extract_info_fallback(
                            st.session_state.transcription,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y")
                        )
                    st.session_state.extracted_info = extracted_info
                    st.text_area("Extracted Information", json.dumps(extracted_info, indent=2), height=300)
            
            if 'extracted_info' in st.session_state:
                if st.button("Generate Document"):
                    with st.spinner("Generating document..."):
                        docx_data = fill_template_and_generate_docx(st.session_state.extracted_info)
                        if docx_data:
                            st.download_button(
                                label="Download Meeting Notes",
                                data=docx_data,
                                file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
if __name__ == "__main__":
    main()