import streamlit as st
import os
import tempfile
from datetime import datetime, timedelta
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
from uuid import uuid4

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
            waveform, sample_rate = torchaudio.load(temp_audio_path)
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
        return None

def extract_info(transcription, meeting_title, date, attendees, absentees, api_key):
    """Extract key information from the transcription using Deepseek API with an improved prompt"""
    prompt = f"""
    Vous êtes un expert en rédaction de comptes rendus de réunion. À partir de la transcription suivante, extrayez et structurez les informations suivantes pour remplir un modèle de compte rendu de réunion. Retournez les informations sous forme de JSON avec les clés suivantes :

    - "date" : La date de la réunion (format DD/MM/YYYY). Si non trouvée, utilisez la date fournie ({date}).
    - "start_time" : L'heure de début de la réunion (format HHhMMmin, ex: 07h00min). Recherchez des mentions explicites (ex: "6:34AM") ou des indices dans l'en-tête.
    - "end_time" : L'heure de fin de la réunion (format HHhMMmin). Si une durée est mentionnée (ex: "2h 21m 21s"), calculez l'heure de fin à partir de l'heure de début. Sinon, utilisez "Non spécifié".
    - "president" : Le nom du président de la réunion. Identifiez-le via des mentions comme "président" ou "Monsieur le Président" (ex: "Cedric DONFACK" si appelé ainsi).
    - "rapporteur" : Le nom du rapporteur. Identifiez-le via des mentions de rédaction du rapport (ex: "Nous allons tester ces spécifications avec le rapport" peut indiquer Emmanuel TEINGA).
    - "presence_list" : Liste des participants présents (liste de noms). Identifiez les noms des personnes intervenant ou mentionnées comme présentes. Excluez les absents explicites (ex: "Brian n'est pas là").
    - "absence_list" : Liste des participants absents (liste de noms). Identifiez les noms mentionnés comme absents (ex: "Brian n'est pas là").
    - "agenda_items" : Liste des points discutés, déduits des sections ou rapports par département/sujet (ex: "Rapport sur la digitalisation", "Projet de migration"). Structurez comme une liste de chaînes.
    - "resolutions_summary" : Liste de résolutions sous forme de tableau (liste de dictionnaires avec les clés "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count"). 
      - "date" : Date de la résolution (généralement la date de la réunion).
      - "dossier" : Sujet spécifique (ex: "Campagne de communication", "Automatisation des dashboards").
      - "resolution" : Description claire de l'action à prendre (ex: "Préparer les templates pour la campagne").
      - "responsible" : Nom de la personne responsable (ex: "KAFO DJIMELI Christian").
      - "deadline" : Délai explicite (ex: "08/05/2025" ou "Lundi 05/05/2025"). Si non précisé, utilisez "Non spécifié".
      - "execution_date" : Date d'exécution, si mentionnée (sinon vide).
      - "status" : Statut (ex: "En cours", "Terminé"). Déduisez "En cours" si non terminé.
      - "report_count" : Nombre de reports (ex: "0" si non précisé).
    - "sanctions_summary" : Liste de sanctions sous forme de tableau (liste de dictionnaires avec les clés "name", "reason", "amount", "date", "status"). Si aucune sanction, retournez une liste vide.
    - "balance_amount" : Le solde du compte DRI Solidarité (ex: "682040"). Si non trouvé, utilisez "Non spécifié".
    - "balance_date" : La date du solde (format DD/MM/YYYY). Si non trouvée, utilisez la date de la réunion.

    Détails de la Réunion :
    - Titre : {meeting_title}
    - Date par défaut : {date}
    - Participants fournis : {attendees}
    - Absents fournis : {absentees}
    
    Transcription :
    {transcription}
    
    Instructions supplémentaires :
    - Priorisez les informations explicites dans la transcription (ex: "May 2, 2025, 6:34AM" pour l'heure de début).
    - Pour l'heure de fin, calculez à partir de la durée si fournie (ex: "2h 21m 21s" ajouté à 6:34AM donne 8:55:21).
    - Identifiez le président et le rapporteur en fonction du contexte (ex: qui dirige la réunion, qui rédige le rapport).
    - Pour les résolutions, extrayez chaque action assignée avec son responsable et son délai (ex: "Préparer les templates d'ici lundi" -> deadline "Lundi 05/05/2025").
    - Pour les points d'ordre du jour, regroupez les discussions par sujet ou département (ex: "Rapport de Christian sur la digitalisation").
    - Si une information est absente, utilisez "Non spécifié" ou une valeur déduite raisonnable.
    - Retournez un JSON bien formé, en français, avec des données précises et structurées.
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
            "max_tokens": 6000  # Augmenté pour gérer des transcriptions longues
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
                # Validation des données extraites
                if not extracted_data.get("presence_list"):
                    extracted_data["presence_list"] = attendees.split(",") if attendees else []
                if not extracted_data.get("absence_list"):
                    extracted_data["absence_list"] = absentees.split(",") if absentees else []
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

def extract_info_fallback(transcription, meeting_title, date, attendees, absentees, start_time="Non spécifié", end_time="Non spécifié", agenda_items=None, balance_amount="Non spécifié", balance_date=None, rapporteur="Non spécifié", president="Non spécifié"):
    """Fallback mode for structuring information if Deepseek API fails"""
    if agenda_items is None:
        agenda_items = ["Non spécifié dans la transcription."]
    if balance_date is None:
        balance_date = date
    
    # Extraction heuristique minimale
    presence_list = attendees.split(",") if attendees else []
    absence_list = absentees.split(",") if absentees else []
    
    # Recherche d'heure de début et de fin via regex
    start_time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', transcription, re.IGNORECASE)
    duration_match = re.search(r'(\d+h\s*\d+m\s*\d+s)', transcription)
    
    if start_time_match:
        start_time = start_time_match.group(1).replace("AM", "").replace("PM", "").strip()
        start_time = f"{start_time[:2]}h{start_time[3:5]}min"
    
    if duration_match and start_time_match:
        duration = duration_match.group(1)
        hours = int(re.search(r'(\d+)h', duration).group(1)) if 'h' in duration else 0
        minutes = int(re.search(r'(\d+)m', duration).group(1)) if 'm' in duration else 0
        seconds = int(re.search(r'(\d+)s', duration).group(1)) if 's' in duration else 0
        start_dt = datetime.strptime(start_time, "%Hh%Mmin")
        end_dt = start_dt + timedelta(hours=hours, minutes=minutes, seconds=seconds)
        end_time = end_dt.strftime("%Hh%Mmin")
    
    # Recherche du président (par exemple, mention de "président")
    president_match = re.search(r'(Monsieur le Président.*?)\s*(\w+\s+\w+)', transcription, re.IGNORECASE)
    if president_match:
        president = president_match.group(2)
    
    # Recherche du rapporteur (par exemple, mention de rédaction de rapport)
    rapporteur_match = re.search(r'(rapport\s*de\s*la\s*science|redaction\s*du\s*rapport).*?(\w+\s+\w+)', transcription, re.IGNORECASE)
    if rapporteur_match:
        rapporteur = rapporteur_match.group(2)
    
    # Extraction des points d'ordre du jour via les sections de rapport
    agenda_items = []
    rapport_sections = re.findall(r'(Rapport\s*de\s*(\w+\s+\w+)|Département\s*(\w+))', transcription, re.IGNORECASE)
    for section in rapport_sections:
        if section[1]:
            agenda_items.append(f"Rapport de {section[1]}")
        elif section[2]:
            agenda_items.append(f"Rapport du département {section[2]}")
    if not agenda_items:
        agenda_items = ["Non spécifié"]
    
    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "president": president,
        "rapporteur": rapporteur,
        "presence_list": presence_list,
        "absence_list": absence_list,
        "agenda_items": agenda_items,
        "resolutions_summary": [
            {
                "date": date,
                "dossier": "Non spécifié",
                "resolution": "Non spécifié",
                "responsible": "Non spécifié",
                "deadline": "Non spécifié",
                "execution_date": "",
                "status": "En cours",
                "report_count": "0"
            }
        ],
        "sanctions_summary": [],
        "balance_amount": balance_amount,
        "balance_date": balance_date,
        "transcription": transcription
    }

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
        margin_elm.set(qn('w:w'), str(int(value * 1440)))  # Convert inches to twips
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
    
    # Header row
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        run = cell.paragraphs[0].runs[0]
        run.font.name = "Century"
        run.font.size = Pt(10)  # Réduit pour les tableaux denses
        run.font.bold = True
        run.font.color.rgb = RGBColor(*header_text_color)
        set_cell_background(cell, header_bg_color)
        set_cell_margins(cell, top=0.05, bottom=0.05, left=0.1, right=0.1)
    
    # Data rows
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
            run.font.size = Pt(10)
            set_cell_margins(cell, top=0.05, bottom=0.05, left=0.1, right=0.1)
    
    return table

def add_text_in_box(doc, text, bg_color=(192, 192, 192), font_size=16, box_width_in_inches=5.0):
    """Add text inside a single-cell table to simulate a centered box."""
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

def fill_template_and_generate_docx(extracted_info, rapporteur, president):
    """Build the Word document from scratch using python-docx"""
    try:
        doc = Document()
        
        # Utiliser les valeurs extraites si disponibles, sinon les entrées manuelles
        rapporteur = extracted_info.get("rapporteur", rapporteur) or "Non spécifié"
        president = extracted_info.get("president", president) or "Non spécifié"
        presence_list = extracted_info.get("presence_list", [])
        absence_list = extracted_info.get("absence_list", [])
        
        # Préparer les listes de présence/absence
        present_attendees = [name.strip() for name in presence_list if name.strip()] if presence_list else ["Non spécifié"]
        absent_attendees = [name.strip() for name in absence_list if name.strip()] if absence_list else ["Non spécifié"]
        
        # Préparer les points d'ordre du jour
        agenda_list = extracted_info.get("agenda_items", [])
        if isinstance(agenda_list, str):
            agenda_list = agenda_list.split("\n") if agenda_list else ["Non spécifié"]
        agenda_list = [f"{to_roman(idx)}. {item.strip()}" for idx, item in enumerate(agenda_list, 1) if item.strip()]
        
        # --- Header Section ---
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
        
        # --- Date ---
        add_styled_paragraph(
            doc,
            extracted_info["date"],
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0),
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        # --- Start and End Time ---
        add_styled_paragraph(
            doc,
            f"Heure de début : {extracted_info['start_time']}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        add_styled_paragraph(
            doc,
            f"Heure de fin : {extracted_info['end_time']}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        # --- Rapporteur and President ---
        add_styled_paragraph(
            doc,
            f"Rapporteur : {rapporteur}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        add_styled_paragraph(
            doc,
            f"Président de Réunion : {president}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        
        doc.add_paragraph()
        
        # --- Attendance Table ---
        add_styled_paragraph(
            doc,
            "◆ LISTE DE PRÉSENCE/ABSENCE",
            font_name="Century",
            font_size=12,
            bold=True
        )
        
        max_rows = max(len(present_attendees), len(absent_attendees))
        if max_rows == 0:
            max_rows = 1
        attendance_data = []
        for i in range(max_rows):
            present_text = present_attendees[i] if i < len(present_attendees) and present_attendees[i] != "Non spécifié" else ""
            absent_text = absent_attendees[i] if i < len(absent_attendees) and absent_attendees[i] != "Non spécifié" else ""
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
        
        doc.add_paragraph()
        
        # --- Agenda Items ---
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
        
        # --- Resolutions Table ---
        resolutions = extracted_info.get("resolutions_summary", [])
        if not resolutions:
            resolutions = [{
                "date": extracted_info["date"],
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
                str(resolution.get("report_count", "0"))
            ]
            resolutions_data.append(row_data)
        
        # Largeurs optimisées pour 8 colonnes
        resolutions_column_widths = [0.8, 1.5, 2.0, 0.8, 1.2, 0.8, 0.8, 0.6]
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
        
        # --- Sanctions Table ---
        sanctions = extracted_info.get("sanctions_summary", [])
        if not sanctions:
            sanctions = [{
                "name": "Aucun",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info["date"],
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
        
        sanctions_column_widths = [1.4, 2.0, 1.2, 1.2, 1.7]
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
        
        # --- Balance Info ---
        add_styled_paragraph(
            doc,
            f"Le solde du compte DRI Solidarité (00001-00921711101-10) est de XAF {extracted_info['balance_amount']} au {extracted_info['balance_date']}.",
            font_name="Century",
            font_size=12
        )
        
        # Save the document
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
            help="Les modèles plus grands sont plus précis mais plus lent"
        )
        
        st.header("Paramètres API Deepseek")
        api_key = st.text_input("Clé API Deepseek", value=st.session_state.api_key, type="password")
        st.session_state.api_key = api_key
        
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
        start_time = st.text_input("Heure de début (format HHhMMmin, ex: 07h00min)", value="Non spécifié")
        end_time = st.text_input("Heure de fin (format HHhMMmin, ex: 10h34min)", value="Non spécifié")
        attendees = st.text_area("Participants Présents (séparés par des virgules)")
        absentees = st.text_area("Participants Absents (séparés par des virgules)")
        rapporteur = st.text_input("Rapporteur", value="Non spécifié")
        president = st.text_input("Président de Réunion", value="Non spécifié")
        
        st.subheader("Ordre du Jour")
        agenda_items_container = st.container()
        if 'agenda_items' not in st.session_state:
            st.session_state.agenda_items = [""]
        
        with agenda_items_container:
            new_agenda_items = []
            for i, item in enumerate(st.session_state.agenda_items):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    new_item = st.text_input(f"Point {i+1}", item, key=f"agenda_item_{i}")
                with cols[1]:
                    if st.button("𝗫", key=f"del_agenda_{i}"):
                        pass
                    else:
                        new_agenda_items.append(new_item)
            st.session_state.agenda_items = new_agenda_items if new_agenda_items else [""]
        
        if st.button("Ajouter un Point à l'Ordre du Jour"):
            st.session_state.agenda_items.append("")
            st.rerun()
        
        st.subheader("Solde du Compte DRI Solidarité")
        balance_amount = st.text_input("Solde (en XAF, ex: 682040)", value="Non spécifié")
        balance_date = st.date_input("Date du solde", value=meeting_date)
    
    with col2:
        st.header("Transcription & Sortie")
        
        # Handle the transcription source
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
        
        # Display and process the transcription
        if transcription:
            st.subheader("Transcription")
            st.text_area("Modifier si nécessaire:", transcription, height=200, key="edited_transcription")
            
            if st.button("Formater les Notes de Réunion") and DOCX_AVAILABLE:
                edited_transcription = st.session_state.get("edited_transcription", transcription)
                agenda_items = [item for item in st.session_state.agenda_items if item.strip()]
                
                if st.session_state.api_key:
                    with st.spinner("Extraction des informations avec Deepseek..."):
                        extracted_info = extract_info(
                            edited_transcription,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y"),
                            attendees,
                            absentees,
                            st.session_state.api_key
                        )
                        if not extracted_info:
                            extracted_info = extract_info_fallback(
                                edited_transcription,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                absentees,
                                start_time,
                                end_time,
                                agenda_items,
                                balance_amount,
                                balance_date.strftime("%d/%m/%Y"),
                                rapporteur,
                                president
                            )
                        else:
                            # Fusionner avec les entrées manuelles si non extraites
                            extracted_info["start_time"] = extracted_info.get("start_time", start_time)
                            extracted_info["end_time"] = extracted_info.get("end_time", end_time)
                            extracted_info["balance_amount"] = extracted_info.get("balance_amount", balance_amount)
                            extracted_info["balance_date"] = extracted_info.get("balance_date", balance_date.strftime("%d/%m/%Y"))
                            if agenda_items:
                                extracted_info["agenda_items"] = agenda_items
                else:
                    st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                    extracted_info = extract_info_fallback(
                        edited_transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        attendees,
                        absentees,
                        start_time,
                        end_time,
                        agenda_items,
                        balance_amount,
                        balance_date.strftime("%d/%m/%Y"),
                        rapporteur,
                        president
                    )
                
                if extracted_info:
                    st.session_state.extracted_info = extracted_info
                    st.subheader("Informations Extraites")
                    st.json(extracted_info)  # Affichage plus structuré
                    
                    with st.spinner("Génération du document Word..."):
                        docx_data = fill_template_and_generate_docx(extracted_info, rapporteur, president)
                    
                    if docx_data:
                        st.download_button(
                            label="Télécharger les Notes de Réunion",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
        
        elif hasattr(st.session_state, 'extracted_info') and DOCX_AVAILABLE:
            st.subheader("Informations Extraites")
            st.json(st.session_state.extracted_info)
            
            with st.spinner("Génération du document Word..."):
                docx_data = fill_template_and_generate_docx(st.session_state.extracted_info, rapporteur, president)
            
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
        start_time = st.text_input("Heure de début (format HHhMMmin, ex: 07h00min)", value="Non spécifié")
        end_time = st.text_input("Heure de fin (format HHhMMmin, ex: 10h34min)", value="Non spécifié")
        attendees = st.text_area("Participants Présents (séparés par des virgules)")
        absentees = st.text_area("Participants Absents (séparés par des virgules)")
        rapporteur = st.text_input("Rapporteur", value="Non spécifié")
        president = st.text_input("Président de Réunion", value="Non spécifié")
        balance_amount = st.text_input("Solde du compte DRI Solidarité (en XAF, ex: 682040)", value="Non spécifié")
        balance_date = st.date_input("Date du solde", value=meeting_date)
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = extract_info_fallback(
                transcription,
                meeting_title,
                meeting_date.strftime("%d/%m/%Y"),
                attendees,
                absentees,
                start_time=start_time,
                end_time=end_time,
                balance_amount=balance_amount,
                balance_date=balance_date.strftime("%d/%m/%Y"),
                rapporteur=rapporteur,
                president=president
            )
            st.subheader("Informations Extraites")
            st.json(extracted_info)
            
            try:
                docx_data = fill_template_and_generate_docx(extracted_info, rapporteur, president)
                if docx_data:
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.warning(f"Erreur lors de la génération du document: {e}")