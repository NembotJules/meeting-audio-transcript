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

def extract_context_from_report(docx_file):
    """Extract relevant context from the previous meeting's report"""
    try:
        doc = Document(docx_file)
        context = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if "RÉCAPITULATIF DES RÉSOLUTIONS" in text or "Ordre du Jour" in text:
                context += text + "\n"
        return context.strip() if context else "Aucun contexte pertinent trouvé."
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contexte: {e}")
        return "Erreur lors de l'extraction du contexte."

def extract_info(transcription, meeting_title, date, api_key, previous_context=""):
    """Extract key information from the transcription using Deepseek API with previous context"""
    prompt = f"""
    Vous êtes un assistant IA spécialisé dans la rédaction de comptes rendus de réunion. 
    À partir de la transcription suivante et du contexte de la réunion précédente, extrayez les informations clés et retournez-les sous forme de JSON structuré en français.

    **Contexte de la réunion précédente** :
    {previous_context if previous_context else "Aucun contexte disponible."}

    **Transcription de la réunion actuelle** :
    {transcription}

    **Sections à extraire** :
    - **presence_list** : Liste des participants présents et absents sous forme de chaîne (ex. "Présents : Alice, Bob\nAbsents : Charlie"). Si non trouvé, utilisez "Présents : Non spécifié\nAbsents : Non spécifié".
    - **agenda_items** : Liste des points de l'ordre du jour sous forme de chaîne (ex. "1. Relecture du compte rendu\n2. Résolutions"). Si non trouvé, utilisez "Non spécifié".
    - **resolutions_summary** : Liste de résolutions sous forme de tableau (liste de dictionnaires avec les clés "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count"). "date", "deadline" et "execution_date" au format DD/MM/YYYY.
    - **sanctions_summary** : Liste de sanctions sous forme de tableau (liste de dictionnaires avec les clés "name", "reason", "amount", "date", "status"). "date" au format DD/MM/YYYY.
    - **start_time** : Heure de début (format HHhMMmin). Si non trouvé, "Non spécifié".
    - **end_time** : Heure de fin (format HHhMMmin). Si non trouvé, "Non spécifié".
    - **rapporteur** : Nom du rapporteur. Si non trouvé, "Non spécifié".
    - **president** : Nom du président. Si non trouvé, "Non spécifié".
    - **balance_amount** : Solde du compte (ex. "827540"). Si non trouvé, "Non spécifié".
    - **balance_date** : Date du solde (format DD/MM/YYYY). Si non trouvé, utilisez {date}.

    **Instructions** :
    1. Utilisez le contexte de la réunion précédente pour comprendre les sujets récurrents et les décisions antérieures.
    2. Concentrez-vous sur la transcription actuelle pour extraire les nouvelles informations, mais utilisez le contexte pour clarifier les références ou les suivis.
    3. Si une information n’est pas trouvée, utilisez des valeurs par défaut raisonnables.
    4. Assurez-vous que le JSON est bien formé et que toutes les dates respectent le format DD/MM/YYYY.
    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
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
        
        agenda_list = extracted_info.get("agenda_items", "Non spécifié").split("\n")
        agenda_list = [f"{to_roman(idx)}. {item.strip()}" for idx, item in enumerate(agenda_list, 1) if item.strip() and item != "Non spécifié"]
        if not agenda_list:
            agenda_list = ["I. Non spécifié"]
        
        add_text_in_box(
            doc,
            "Direction Recherches et Investissements",
            bg_color=(192, 192, 192),
            font_size=16,
            box_width_in_inches=5.0
        )
        
        doc.add_paragraph()
        
        add_styled_paragraph(
            doc,
            "COMPTE RENDU DE REUNION HEBDOMADAIRE",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0),
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        add_styled_paragraph(
            doc,
            extracted_info.get("date", ""),
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0),
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
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
        
        doc.add_paragraph()
        
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
        
        doc.add_paragraph()
        
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
        
        doc.add_page_break()
        
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
        
        doc.add_paragraph()
        
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
        
        doc.add_paragraph()
        
        add_styled_paragraph(
            doc,
            f"Le solde du compte DRI Solidarité (00001-00921711101-10) est de XAF {extracted_info.get('balance_amount', 'Non spécifié')} au {extracted_info.get('balance_date', '')}.",
            font_name="Century",
            font_size=12
        )
        
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
    st.title("Outil de Transcription Audio de Réunion")
    
    try:
        from transformers import pipeline
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        st.warning("""
        ⚠️ Les dépendances nécessaires (transformers, torch, torchaudio) ne sont pas installées.
        Exécutez : `pip install transformers torch torchaudio`
        Assurez-vous que ffmpeg est installé : https://ffmpeg.org/download.html
        """)
    
    try:
        from docx import Document
        DOCX_AVAILABLE = True
    except ImportError:
        DOCX_AVAILABLE = False
        st.warning("""
        ⚠️ La bibliothèque python-docx n'est pas installée.
        Exécutez : `pip install python-docx`
        """)
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    with st.sidebar:
        st.header("Source de la Transcription")
        input_method = st.radio("Choisissez une méthode :", ("Télécharger un fichier audio", "Entrer une transcription manuelle"))
        
        if input_method == "Télécharger un fichier audio":
            uploaded_file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "m4a", "flac"])
            manual_transcript = None
        else:
            uploaded_file = None
            manual_transcript = st.text_area("Collez votre transcription ici :", height=200, key="manual_transcript_input")
        
        st.header("Options de Transcription (si fichier audio)")
        whisper_model = st.selectbox(
            "Taille du Modèle Whisper",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Les modèles plus grands sont plus précis mais plus lents",
            disabled=(input_method == "Entrer une transcription manuelle")
        )
        
        st.header("Paramètres API Deepseek")
        st.session_state.api_key = st.text_input("Clé API Deepseek", value=st.session_state.api_key, type="password")
        
        st.header("Contexte Précédent")
        previous_report = st.file_uploader("Uploader le compte rendu précédent (optionnel)", type=["docx"])
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"Fichier téléchargé: {uploaded_file.name}")
            transcribe_button = st.button("Transcrire l'Audio")
        elif manual_transcript:
            transcribe_button = False
            st.info("Transcription manuelle détectée. Passez directement au formatage.")
        else:
            transcribe_button = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
        
        if 'extracted_info' in st.session_state:
            extracted_info = st.session_state.extracted_info
            default_presence = extracted_info.get("presence_list", "Présents : Non spécifié\nAbsents : Non spécifié")
            default_agenda = extracted_info.get("agenda_items", "Non spécifié")
            default_start_time = extracted_info.get("start_time", "Non spécifié")
            default_end_time = extracted_info.get("end_time", "Non spécifié")
            default_rapporteur = extracted_info.get("rapporteur", "Non spécifié")
            default_president = extracted_info.get("president", "Non spécifié")
            default_balance_amount = extracted_info.get("balance_amount", "Non spécifié")
            default_balance_date = extracted_info.get("balance_date", meeting_date.strftime("%d/%m/%Y"))
        else:
            default_presence = "Présents : Non spécifié\nAbsents : Non spécifié"
            default_agenda = "Non spécifié"
            default_start_time = "Non spécifié"
            default_end_time = "Non spécifié"
            default_rapporteur = "Non spécifié"
            default_president = "Non spécifié"
            default_balance_amount = "Non spécifié"
            default_balance_date = meeting_date.strftime("%d/%m/%Y")
        
        st.subheader("Liste de Présence/Absence")
        presence_list = st.text_area("Présents et Absents (format: Présents : Nom1, Nom2\nAbsents : Nom3)", value=default_presence, height=100)
        
        st.subheader("Ordre du Jour")
        agenda_items = st.text_area("Points de l'ordre du jour (un par ligne)", value=default_agenda, height=150)
        
        st.subheader("Horaires")
        start_time = st.text_input("Heure de début (format HHhMMmin)", value=default_start_time)
        end_time = st.text_input("Heure de fin (format HHhMMmin)", value=default_end_time)
        
        st.subheader("Rôles")
        rapporteur = st.text_input("Rapporteur", value=default_rapporteur)
        president = st.text_input("Président de Réunion", value=default_president)
        
        st.subheader("Solde du Compte DRI Solidarité")
        balance_amount = st.text_input("Solde (en XAF)", value=default_balance_amount)
        balance_date = st.date_input("Date du solde", value=datetime.strptime(default_balance_date, "%d/%m/%Y") if default_balance_date != "Non spécifié" else meeting_date)
    
    with col2:
        st.header("Transcription & Sortie")
        
        if transcribe_button and WHISPER_AVAILABLE and uploaded_file is not None:
            with st.spinner(f"Transcription audio avec le modèle Whisper {whisper_model}..."):
                transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
            
            if transcription and not transcription.startswith("Erreur"):
                st.success("Transcription terminée!")
                st.session_state.transcription = transcription
        elif manual_transcript:
            transcription = manual_transcript.strip()
            if transcription:
                st.success("Transcription manuelle chargée!")
                st.session_state.transcription = transcription
            else:
                st.error("Veuillez entrer une transcription valide.")
                transcription = None
        else:
            transcription = getattr(st.session_state, 'transcription', None)
        
        if transcription:
            st.subheader("Transcription")
            st.text_area("Modifier si nécessaire:", transcription, height=200, key="edited_transcription")
            
            if st.button("Formater les Notes de Réunion") and DOCX_AVAILABLE:
                edited_transcription = st.session_state.get("edited_transcription", transcription)
                
                previous_context = ""
                if previous_report:
                    previous_context = extract_context_from_report(previous_report)
                
                if st.session_state.api_key:
                    with st.spinner("Extraction des informations avec Deepseek..."):
                        extracted_info = extract_info(
                            edited_transcription,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y"),
                            st.session_state.api_key,
                            previous_context
                        )
                        if not extracted_info:
                            extracted_info = extract_info_fallback(
                                edited_transcription,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y")
                            )
                else:
                    st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                    extracted_info = extract_info_fallback(
                        edited_transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y")
                    )
                
                if extracted_info:
                    if presence_list != default_presence:
                        extracted_info["presence_list"] = presence_list
                    if agenda_items != default_agenda:
                        extracted_info["agenda_items"] = agenda_items
                    if start_time != default_start_time:
                        extracted_info["start_time"] = start_time
                    if end_time != default_end_time:
                        extracted_info["end_time"] = end_time
                    if rapporteur != default_rapporteur:
                        extracted_info["rapporteur"] = rapporteur
                    if president != default_president:
                        extracted_info["president"] = president
                    if balance_amount != default_balance_amount:
                        extracted_info["balance_amount"] = balance_amount
                    if balance_date.strftime("%d/%m/%Y") != default_balance_date:
                        extracted_info["balance_date"] = balance_date.strftime("%d/%m/%Y")
                
                if extracted_info:
                    st.session_state.extracted_info = extracted_info
                    st.subheader("Informations Extraites")
                    st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
                    
                    with st.spinner("Génération du document Word..."):
                        docx_data = fill_template_and_generate_docx(extracted_info)
                    
                    if docx_data:
                        st.download_button(
                            label="Télécharger les Notes de Réunion",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
        
        elif hasattr(st.session_state, 'extracted_info') and DOCX_AVAILABLE:
            extracted_info = st.session_state.extracted_info
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
            
            with st.spinner("Génération du document Word..."):
                docx_data = fill_template_and_generate_docx(extracted_info)
            
            if docx_data:
                st.download_button(
                    label="Télécharger les Notes de Réunion",
                    data=docx_data,
                    file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du démarrage de l'application: {e}")
        st.write("""
        ### Dépannage
        Si vous rencontrez des erreurs, essayez les solutions suivantes :
        1. Installez toutes les dépendances :
           ```
           pip install streamlit transformers torch torchaudio python-docx requests
           ```
        2. Pour Streamlit Cloud, assurez-vous d'avoir un fichier `requirements.txt` :
           ```
           streamlit>=1.24.0
           transformers>=4.30.0
           torch>=2.0.1
           torchaudio>=2.0.2
           python-docx>=0.8.11
           requests>=2.28.0
           ```
        3. Installez ffmpeg pour les fichiers .m4a :
           - Sur Ubuntu : `sudo apt-get install ffmpeg`
           - Sur macOS : `brew install ffmpeg`
           - Sur Windows : Téléchargez depuis https://ffmpeg.org/download.html
        """)
        
        st.title("Mode Secours")
        st.warning("Application en mode limité. La transcription audio n'est pas disponible.")
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
        presence_list = st.text_area("Présents et Absents (format: Présents : Nom1, Nom2\nAbsents : Nom3)", value="Présents : Non spécifié\nAbsents : Non spécifié")
        agenda_items = st.text_area("Points de l'ordre du jour (un par ligne)", value="Non spécifié")
        start_time = st.text_input("Heure de début (format HHhMMmin)", value="Non spécifié")
        end_time = st.text_input("Heure de fin (format HHhMMmin)", value="Non spécifié")
        rapporteur = st.text_input("Rapporteur", value="Non spécifié")
        president = st.text_input("Président de Réunion", value="Non spécifié")
        balance_amount = st.text_input("Solde du compte DRI Solidarité (en XAF)", value="Non spécifié")
        balance_date = st.date_input("Date du solde", value=meeting_date)
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = extract_info_fallback(
                transcription,
                meeting_title,
                meeting_date.strftime("%d/%m/%Y")
            )
            extracted_info["presence_list"] = presence_list
            extracted_info["agenda_items"] = agenda_items
            extracted_info["start_time"] = start_time
            extracted_info["end_time"] = end_time
            extracted_info["rapporteur"] = rapporteur
            extracted_info["president"] = president
            extracted_info["balance_amount"] = balance_amount
            extracted_info["balance_date"] = balance_date.strftime("%d/%m/%Y")
            
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
            
            try:
                docx_data = fill_template_and_generate_docx(extracted_info)
                if docx_data:
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.warning(f"Erreur lors de la génération du document: {e}")