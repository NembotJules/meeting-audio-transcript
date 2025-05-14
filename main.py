import streamlit as st
from mistralai import Mistral
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Meeting Transcription Tool", page_icon=":microphone:", layout="wide")

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
        st.error(f"Error during audio transcription: {e}")
        return f"Error during audio transcription: {e}"

def extract_context_from_report(file, mistral_api_key):
    """Extract text from the uploaded file using Mistral OCR."""
    if not file or not mistral_api_key:
        return ""
    
    file_extension = file.name.split('.')[-1].lower()
    valid_extensions = ['pdf', 'png', 'jpg', 'jpeg']
    if file_extension not in valid_extensions:
        st.error("Unsupported file type. Please upload a PDF, PNG, JPG, or JPEG.")
        return ""
    
    try:
        client = Mistral(api_key=mistral_api_key)
        
        # Upload the file to Mistral's servers
        uploaded_file = client.files.upload(
            file={
                "file_name": file.name,
                "content": file.getvalue(),
            },
            purpose="ocr"
        )
        
        # Retrieve the signed URL
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        
        # Set document type based on file extension
        document_type = "document_url" if file_extension == 'pdf' else "image_url"
        
        # Construct the message with the correct structure
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from the document."
                    },
                    {
                        "type": document_type,
                        document_type: signed_url.url
                    }
                ]
            }
        ]
        
        # Call the Mistral chat API
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        
        # Return the extracted text
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error processing file with Mistral OCR: {e}")
        return ""

def answer_question_with_context(question, context, deepseek_api_key):
    """Answer a question based on the extracted context using Deepseek API."""
    if not context or not question or not deepseek_api_key:
        return "Please provide a question, context, and Deepseek API key."
    
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

def extract_info_fallback(transcription, meeting_title, date):
    """Fallback function to extract information using basic string parsing and regex."""
    extracted_data = {
        "presence_list": "Présents: Non spécifié\nAbsents: Non spécifié",
        "agenda_items": "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers",
        "resolutions_summary": [],
        "sanctions_summary": [],
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "rapporteur": "Non spécifié",
        "president": "Non spécifié",
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "date": date,
        "meeting_title": meeting_title
    }

    # Extract presence list (French keywords)
    present_match = re.search(r"(Présents|Présent|Présentes|Présente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    absent_match = re.search(r"(Absents|Absent|Absentes|Absente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    if present_match or absent_match:
        present = present_match.group(2).strip() if present_match else "Non spécifié"
        absent = absent_match.group(2).strip() if absent_match else "Non spécifié"
        extracted_data["presence_list"] = f"Présents: {present}\nAbsents: {absent}"
    else:
        # Fallback to infer presence from names mentioned in the transcript
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", transcription)
        if names:
            extracted_data["presence_list"] = f"Présents: {', '.join(set(names))}\nAbsents: Non spécifié"

    # Extract agenda items
    agenda_match = re.search(r"(Ordre du jour|Agenda)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if agenda_match:
        agenda_items = agenda_match.group(2).strip()
        items = [item.strip() for item in agenda_items.split("\n") if item.strip()]
        if items:
            extracted_data["agenda_items"] = "\n".join(items)

    # Extract start and end times
    time_pattern = r"\b(\d{1,2}(?:h\d{2}min|h:\d{2}|\d{2}min))\b"
    times = re.findall(time_pattern, transcription, re.IGNORECASE)
    if times:
        extracted_data["start_time"] = times[0]
        if len(times) > 1:
            extracted_data["end_time"] = times[-1]

    # Extract rapporteur and president
    rapporteur_match = re.search(r"(Rapporteur|Rapporteuse)[:\s]*(\w+)", transcription, re.IGNORECASE)
    president_match = re.search(r"(Président|Présidente|Prési)[:\s]*(\w+)", transcription, re.IGNORECASE)
    if rapporteur_match:
        extracted_data["rapporteur"] = rapporteur_match.group(2)
    if president_match:
        extracted_data["president"] = president_match.group(2)

    # Extract balance amount
    balance_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d+)", transcription, re.IGNORECASE)
    if balance_match:
        extracted_data["balance_amount"] = balance_match.group(2)

    # Extract balance date
    balance_date_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d{2}/\d{2}/\d{4})", transcription, re.IGNORECASE)
    if balance_date_match:
        extracted_data["balance_date"] = balance_date_match.group(2)

    # Extract resolutions (basic)
    resolution_match = re.search(r"(Résolution|Resolution)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if resolution_match:
        resolution_text = resolution_match.group(2).strip()
        extracted_data["resolutions_summary"] = [{
            "date": date,
            "dossier": "Non spécifié",
            "resolution": resolution_text,
            "responsible": "Non spécifié",
            "deadline": "Non spécifié",
            "execution_date": "",
            "status": "En cours",
            "report_count": "0"
        }]

    # Extract sanctions (basic)
    sanction_match = re.search(r"(Sanction|Amende)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if sanction_match:
        sanction_text = sanction_match.group(2).strip()
        extracted_data["sanctions_summary"] = [{
            "name": "Non spécifié",
            "reason": sanction_text,
            "amount": "0",
            "date": date,
            "status": "Non appliquée"
        }]

    return extracted_data

def extract_info(transcription, meeting_title, date, deepseek_api_key, previous_context=""):
    """Extract key information from the transcription using Deepseek API with previous context."""
    prompt = f"""
    Vous êtes un assistant IA expert en rédaction automatique de comptes rendus de réunion pour une institution bancaire. Votre mission est d’analyser un transcript de réunion en vous appuyant sur le rapport de la réunion précédente, notamment les tableaux de la Revue d’Activités, du Récapitulatif des Résolutions, et du Récapitulatif des Sanctions, afin de générer un rapport structuré et pertinent sous forme de JSON avec des clés en anglais.

    🧠 **Contexte** :
    Le rapport de la réunion précédente contient des informations-clés sur :
    - Les membres impliqués (noms, départements si disponibles)
    - Les dossiers traités par chacun
    - L’état d’avancement des travaux (Résultats / Perspectives)
    - Le tableau des résolutions (Date, Sujet, Responsable, Délai, Date d’exécution, Statut, Nombre de reports)
    - Le tableau des sanctions (Nom, Motif, Montant, Date, Statut)

    Voici le contenu du rapport de la réunion précédente :
    {previous_context if previous_context else "Aucun contexte disponible."}

    Utilisez ces éléments comme contexte de travail pour mieux comprendre les échanges dans le transcript à analyser.

    **Transcript de la réunion actuelle** :
    {transcription}

    ✅ **Informations à extraire du transcript** :
    À partir du transcript, extrayez et structurez les informations suivantes dans un objet JSON avec les clés en anglais spécifiées ci-dessous :

    1. **presence_list** : Liste des présents et absents sous forme de chaîne (par exemple, "Présents: Alice, Bob\nAbsents: Charlie").
       - **Liste des présents** : Identifiez les participants ayant pris la parole ou mentionnés comme présents (mots-clés : "Présents", "Présent", "Présente"). Si implicite, déduisez à partir des interventions (ex. : "Alice a dit…" implique qu'Alice est présente). Seule la transcription actuelle doit être utilisée pour déterminer les présents.
       - **Liste des absents** : Recherchez uniquement les mentions explicites dans le transcript (mots-clés : "Absents", "Absent", "Absente"). Ne déduisez pas les absents à partir du contexte précédent. Si aucune mention explicite, indiquez "Absents: Non spécifié".
       - Si aucune information sur les présents n’est trouvée, indiquez : "Présents: Non spécifié\nAbsents: Non spécifié".

    2. **agenda_items** : Liste des points de l'ordre du jour sous forme de chaîne (par exemple, "I- Revue des minutes\nII- Résolutions").
       - Recherchez des mots-clés comme "Ordre du jour" ou "Agenda" pour identifier une liste explicite.
       - Si aucun "Ordre du jour" n’est mentionné, déduisez les points discutés à partir des sujets abordés dans le transcript (ex. : "On a discuté des résolutions" peut indiquer un point sur les résolutions).
       - Si rien ne peut être déduit, utilisez cette liste par défaut :
         "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers".

    3. **president** : Président de séance.
       - Recherchez la personne associée aux mots-clés comme "Président", "Présidente", "Prési", ou des mentions comme "présidé par".
       - Si aucune information n’est trouvée, indiquez : "Non spécifié".

    4. **rapporteur** : Rapporteur de la réunion.
       - Recherchez des indices comme "Rapporteur", "Rapporteuse", ou toute mention indiquant qu’une personne est responsable de la rédaction du rapport (ex. : "Alice a rédigé…").
       - Si aucune information n’est trouvée, indiquez : "Non spécifié".

    5. **start_time** et **end_time** : Heure de début et de fin de la réunion (format HHhMMmin, par exemple, "07h00min").
       - Identifiez les horaires directement mentionnés dans le transcript (ex. : "La réunion a commencé à 10h00", "finie à 11h30").
       - Si non disponibles, utilisez "Non spécifié".

    6. **balance_amount** : Solde du compte solidarité DRI.
       - Recherchez les mots-clés "solde", "compte", "balance".
       - Si aucune information n’est trouvée, indiquez : "Non spécifié".

    7. **balance_date** : Date du solde (format JJ/MM/AAAA, par exemple, "14/05/2025").
       - Recherchez une date associée au solde.
       - Si non mentionnée explicitement, utilisez la date fournie : {date}.

    8. **resolutions_summary** : Tableau récapitulatif des résolutions.
       - Présentez sous forme de tableau (liste de dictionnaires) les résolutions abordées durant la réunion actuelle, avec les clés suivantes :
         - "date" : Date de la résolution (format JJ/MM/AAAA, utilisez {date} si non spécifiée).
         - "dossier" : Sujet ou dossier concerné (ex. : "Projet X", utilisez "Non spécifié" si non clair).
         - "resolution" : Description de la résolution (ex. : "Finaliser le rapport").
         - "responsible" : Personne responsable (ex. : "Alice", utilisez "Non spécifié" si non clair).
         - "deadline" : Délai d’exécution (format JJ/MM/AAAA, utilisez "Non spécifié" si non clair).
         - "execution_date" : Date d’exécution (format JJ/MM/AAAA, laissez vide "" si non exécutée).
         - "status" : Statut (ex. : "En cours", "Terminé", "Reporté").
         - "report_count" : Nombre de reports (chaîne, ex. : "0", "1").
       - Utilisez le contexte pour identifier les résolutions non résolues des réunions précédentes qui pourraient être mentionnées.

    9. **sanctions_summary** : Tableau récapitulatif des sanctions.
       - Présentez toutes les sanctions évoquées dans la réunion sous forme de tableau (liste de dictionnaires) avec les clés suivantes :
         - "name" : Nom de la personne sanctionnée.
         - "reason" : Motif de la sanction.
         - "amount" : Montant en FCFA (chaîne, ex. : "5000").
         - "date" : Date de la sanction (format JJ/MM/AAAA, utilisez {date} si non spécifiée).
         - "status" : Statut (ex. : "Appliquée", "Non appliquée").
       - Recherchez des mots-clés comme "Sanction", "Amende".

    **Instructions supplémentaires** :
    - Assurez-vous que la sortie est un objet JSON unique avec une syntaxe correcte (utilisez des guillemets doubles pour les clés et les valeurs de chaîne, pas de virgules finales).
    - Si vous ne pouvez pas extraire les informations ou rencontrez des problèmes, retournez un objet JSON avec une seule clé "error" expliquant le problème (ex. : {{"error": "Impossible de parser la transcription"}}).
    - Ne incluez aucun texte, explication, ou commentaire en dehors de l'objet JSON. La réponse doit être analysable par un parseur JSON.
    - Assurez-vous que toutes les dates dans la sortie sont au format JJ/MM/AAAA (par exemple, "14/05/2025").
    - Les clés du JSON doivent être en anglais (comme spécifié ci-dessus), mais les valeurs doivent refléter le texte original (donc en français).

    **Exemple de sortie** :
    {{"presence_list": "Présents: Alice, Bob\nAbsents: Charlie", "agenda_items": "I- Revue des minutes\nII- Résolutions", "president": "Alice", "rapporteur": "Bob", "start_time": "10h00min", "end_time": "11h00min", "balance_amount": "827540", "balance_date": "14/05/2025", "resolutions_summary": [{{"date": "14/05/2025", "dossier": "Projet X", "resolution": "Finaliser le rapport", "responsible": "Alice", "deadline": "20/05/2025", "execution_date": "", "status": "En cours", "report_count": "0"}}], "sanctions_summary": [{{"name": "Charlie", "reason": "Retard", "amount": "5000", "date": "14/05/2025", "status": "Appliquée"}}]}}

    **Exemple d'erreur** :
    {{"error": "Impossible de parser la transcription en raison d'un contenu peu clair"}}

    Retournez le résultat sous forme de JSON structuré.
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
            # Log the full response for debugging
            full_response = response.json()
            st.write(f"Full Deepseek response: {json.dumps(full_response, indent=2)}")
            
            # Extract the content
            raw_response = full_response["choices"][0]["message"]["content"].strip()
            st.write(f"Raw Deepseek response content: {raw_response}")
            
            # Validate the response before parsing
            if not raw_response:
                st.error("Deepseek API returned an empty response. Falling back to basic extraction.")
                return extract_info_fallback(transcription, meeting_title, date)
            
            # Attempt to parse the response as JSON
            extracted_data = json.loads(raw_response)
            
            # Check if the response contains an error key
            if "error" in extracted_data:
                st.error(f"Deepseek API error: {extracted_data['error']}. Falling back to basic extraction.")
                return extract_info_fallback(transcription, meeting_title, date)
            
            # Add meeting metadata
            extracted_data["date"] = date
            extracted_data["meeting_title"] = meeting_title
            return extracted_data
        else:
            st.error(f"Deepseek API error: Status {response.status_code}, Message: {response.text}. Falling back to basic extraction.")
            return extract_info_fallback(transcription, meeting_title, date)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}. Falling back to basic extraction.")
        return extract_info_fallback(transcription, meeting_title, date)
    except Exception as e:
        st.error(f"Error extracting information: {e}. Falling back to basic extraction.")
        return extract_info_fallback(transcription, meeting_title, date)

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
        st.warning("The 'Table Grid' style is not available. Using default style.")
    
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

def fill_template_and_generate_docx(extracted_info, meeting_title, meeting_date):
    """Build the Word document from scratch using python-docx and return the file data for download."""
    try:
        doc = Document()

        # Extract presence list and split into present and absent attendees
        presence_list = extracted_info.get("presence_list", "Présents: Non spécifié\nAbsents: Non spécifié")
        present_attendees = []
        absent_attendees = []
        if "Présents:" in presence_list and "Absents:" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents:"):
                    presents = part.replace("Présents:", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                elif part.startswith("Absents:"):
                    absents = part.replace("Absents:", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Non spécifié" else []

        # Process agenda items
        agenda_list = extracted_info.get("agenda_items", "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers").split("\n")
        agenda_list = [item.strip() for item in agenda_list if item.strip()]

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
            "COMPTE RENDU DE RÉUNION",
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
            f"Heure de début: {extracted_info.get('start_time', 'Non spécifié')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        add_styled_paragraph(
            doc,
            f"Heure de fin: {extracted_info.get('end_time', 'Non spécifié')}",
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
                f"Rapporteur: {rapporteur}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )
        if president != "Non spécifié":
            add_styled_paragraph(
                doc,
                f"Président de la réunion: {president}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )

        # Add attendance table
        add_styled_paragraph(
            doc,
            "◆ LISTE DE PRÉSENCE",
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
                headers=["PRÉSENTS", "ABSENTS"],
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
                "Aucune présence spécifiée.",
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
            "RÉSUMÉ DES RÉSOLUTIONS",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        resolutions_headers = ["DATE", "DOSSIER", "RÉSOLUTION", "RESP.", "ÉCHÉANCE", "DATE D'EXÉCUTION", "STATUT", "COMPTE RENDU"]
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
                "name": "Aucune",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info.get("date", ""),
                "status": "Non appliquée"
            }]
        add_styled_paragraph(
            doc,
            "RÉSUMÉ DES SANCTIONS",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        sanctions_headers = ["NOM", "RAISON", "MONTANT (FCFA)", "DATE", "STATUT"]
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
            f"Solde du compte de solidarité DRI (00001-00921711101-10) est de XAF {extracted_info.get('balance_amount', 'Non spécifié')} au {extracted_info.get('balance_date', '')}.",
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
        st.error(f"Error generating Word document: {e}")
        return None

def main():
    st.title("Meeting Transcription Tool")
    
    # Sidebar for API keys and previous report
    st.sidebar.header("Configuration")
    st.session_state.mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    st.session_state.deepseek_api_key = st.sidebar.text_input("Deepseek API Key", type="password")
    
    st.sidebar.header("Contexte Précédent")
    previous_report = st.sidebar.file_uploader("Télécharger le rapport précédent (optionnel)", type=["pdf", "png", "jpg", "jpeg"])
    if previous_report:
        st.session_state.previous_report = previous_report
        st.session_state.previous_context = ""  # Reset context until a question is asked
        st.sidebar.write("Rapport précédent téléchargé. Posez une question pour extraire le contexte.")
    else:
        st.session_state.previous_report = None
        st.session_state.previous_context = ""
    
    # Section to ask questions about the context
    st.sidebar.header("Tester le Contexte")
    question = st.sidebar.text_input("Posez une question sur le rapport précédent :")
    if st.sidebar.button("Poser la Question") and question:
        if not st.session_state.mistral_api_key:
            st.sidebar.error("Veuillez fournir une clé API Mistral pour extraire le contexte.")
        elif not st.session_state.previous_report:
            st.sidebar.error("Veuillez télécharger un rapport précédent pour extraire le contexte.")
        else:
            with st.spinner("Extraction du contexte..."):
                context = extract_context_from_report(
                    st.session_state.previous_report, 
                    st.session_state.mistral_api_key
                )
                if context:
                    st.session_state.previous_context = context
                    st.sidebar.text_area("Contexte Extrait", context, height=200)
                    st.sidebar.success("Contexte extrait avec succès !")
                else:
                    st.session_state.previous_context = ""
                    st.sidebar.error("Échec de l'extraction du contexte. Vérifiez la clé API ou le fichier.")
            
            # Now answer the question
            with st.spinner("Obtention de la réponse..."):
                answer = answer_question_with_context(
                    question, 
                    st.session_state.previous_context, 
                    st.session_state.deepseek_api_key
                )
            st.sidebar.write("**Réponse :**")
            st.sidebar.write(answer)
    
    # Main app content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
    
    with col2:
        st.header("Transcription & Résultat")
        input_method = st.radio("Choisissez la méthode d'entrée :", ("Télécharger Audio", "Entrer la Transcription"))
        
        if input_method == "Télécharger Audio":
            uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["mp3", "wav", "m4a", "flac"])
            whisper_model = st.selectbox("Modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
            
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if st.button("Transcrire l'Audio"):
                    with st.spinner(f"Transcription avec Whisper {whisper_model}..."):
                        transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                        if transcription and not transcription.startswith("Error"):
                            st.session_state.transcription = transcription
                            # Automatically extract information after transcription
                            with st.spinner("Extraction des informations..."):
                                extracted_info = extract_info(
                                    st.session_state.transcription,
                                    meeting_title,
                                    meeting_date.strftime("%d/%m/%Y"),
                                    st.session_state.deepseek_api_key,
                                    st.session_state.get("previous_context", "")
                                )
                                if extracted_info:
                                    st.session_state.extracted_info = extracted_info
                                    st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        else:
            transcription_input = st.text_area("Entrez la transcription de la réunion :", height=200)
            if st.button("Soumettre la Transcription") and transcription_input:
                st.session_state.transcription = transcription_input
                # Automatically extract information after submission
                with st.spinner("Extraction des informations..."):
                    extracted_info = extract_info(
                        st.session_state.transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        st.session_state.deepseek_api_key,
                        st.session_state.get("previous_context", "")
                    )
                    if extracted_info:
                        st.session_state.extracted_info = extracted_info
                        st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        
        if 'extracted_info' in st.session_state:
            if st.button("Générer et Télécharger le Document"):
                with st.spinner("Génération du document..."):
                    docx_data = fill_template_and_generate_docx(
                        st.session_state.extracted_info,
                        meeting_title,
                        meeting_date
                    )
                    if docx_data:
                        st.download_button(
                            label="Téléchargement du Document en Cours...",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download-button",
                            on_click=lambda: None
                        )

if __name__ == "__main__":
    main()