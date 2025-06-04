import streamlit as st
from mistralai import Mistral
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
import base64
from vector_store import MeetingVectorStore
from prompt_manager import PromptManager

# Try to import tiktoken for accurate token counting; fall back to simple method if unavailable
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Meeting Transcription Tool", page_icon=":microphone:", layout="wide")

# Initialize vector store and prompt manager as global variables
vector_store = MeetingVectorStore()
prompt_manager = PromptManager()

# Add vector store persistence directory to session state
if 'vector_store_dir' not in st.session_state:
    st.session_state.vector_store_dir = "meeting_store"
    if os.path.exists(st.session_state.vector_store_dir):
        vector_store.load(st.session_state.vector_store_dir)

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
            answer = response.json()["choices"][0]["message"]["content"].strip()
            if "tableau récapitulatif des sanctions" in question.lower():
                sanctions = parse_sanctions_from_text(answer)
                if sanctions:
                    st.session_state.context_sanctions = sanctions
                    st.sidebar.success(f"Sanctions stored in session state!")
                else:
                    st.sidebar.warning("No sanctions found in context")
            return answer
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def parse_sanctions_from_text(text):
    """Parse sanctions table from text into a list of dictionaries."""
    sanctions = []
    lines = text.split("\n")
    header_found = False
    for line in lines:
        if "NOM" in line and "RAISON" in line and "MONTANT" in line:
            header_found = True
            continue
        if header_found and line.strip():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 5:
                sanctions.append({
                    "name": parts[0],
                    "reason": parts[1],
                    "amount": parts[2].replace(" FCFA", "").replace("FCFA", ""),
                    "date": parts[3],
                    "status": parts[4]
                })
    return sanctions if sanctions else None

def count_tokens(text):
    """Count the number of tokens in a text string."""
    if TOKENIZER_AVAILABLE:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return len(tokens)

def clean_json_response(response):
    """Clean an API response to extract valid JSON content."""
    if isinstance(response, bytes):
        try:
            response = response.decode('utf-8')
        except UnicodeDecodeError as e:
            st.error(f"Failed to decode response as UTF-8: {e}")
            return None
    if not isinstance(response, str):
        st.error(f"Response is not a string or bytes: {type(response)}")
        return None
    response = response.strip()
    if not response:
        st.error("Response is empty after stripping whitespace.")
        return None
    response = response.removeprefix('```json').removesuffix('```').strip()
    response = response.removeprefix('```').removesuffix('```').strip()
    if not (response.startswith("{") or response.startswith("[")):
        json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        else:
            st.error(f"Response does not contain valid JSON: {response}")
            return None
    return response

def extract_info_fallback(transcription, meeting_title, date, previous_context=""):
    """Fallback function to extract information using improved string parsing and regex."""
    extracted_data = {
        "presence_list": "Présents: Non spécifié\nAbsents: Non spécifié",
        "agenda_items": "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d'activités\nIV- Faits saillants\nV- Divers",
        "activities_review": [],
        "resolutions_summary": [],
        "sanctions_summary": [],  # Start with empty list
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "rapporteur": "Non spécifié",
        "president": "Non spécifié",
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "date": date,
        "meeting_title": meeting_title
    }

    # Extract presence list
    present_match = re.search(r"(Présents|Présent|Présentes|Présente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    absent_match = re.search(r"(Absents|Absent|Absentes|Absente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    if present_match or absent_match:
        present = present_match.group(2).strip() if present_match else "Non spécifié"
        absent = absent_match.group(2).strip() if absent_match else "Non spécifié"
        extracted_data["presence_list"] = f"Présents: {present}\nAbsents: {absent}"
    else:
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", transcription)
        common_words = {"Réunion", "Projet", "Président", "Rapporteur", "Solde", "Compte", "Ordre", "Agenda"}
        names = [name for name in set(names) if name not in common_words]
        if names:
            extracted_data["presence_list"] = f"Présents: {', '.join(names)}\nAbsents: Non spécifié"

    # Extract agenda items
    agenda_match = re.search(r"(Ordre du jour|Agenda)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if agenda_match:
        agenda_items = agenda_match.group(2).strip()
        items = [item.strip() for item in agenda_items.split("\n") if item.strip()]
        if items:
            numbered_items = []
            for idx, item in enumerate(items, 1):
                if not re.match(r"^[IVXLC]+\-", item):
                    item = f"{to_roman(idx)}- {item}"
                numbered_items.append(item)
            extracted_data["agenda_items"] = "\n".join(numbered_items)

    # Extract start time
    start_time_match = re.search(r"(?:début|commence|commencée)[\s\w]*?(\d{1,2}(?:h\d{2}min|h:\d{2}|\d{2}min))", transcription, re.IGNORECASE)
    if start_time_match:
        extracted_data["start_time"] = start_time_match.group(1).replace("h:", "h").replace("min", "min")

    # Extract duration and calculate end time
    duration_match = re.search(r"(?:durée|dure|duré|lasted)[\s\w]*?(\d{1,2}h(?:\d{1,2}min)?(?:\d{1,2}s)?)", transcription, re.IGNORECASE)
    end_time_match = re.search(r"(?:fin|terminée|terminé|ended)[\s\w]*?(\d{1,2}(?:h\d{2}min|h:\d{2}|\d{2}min))", transcription, re.IGNORECASE)
    if end_time_match:
        extracted_data["end_time"] = end_time_match.group(1).replace("h:", "h").replace("min", "min")
    elif start_time_match and duration_match:
        start_time_str = start_time_match.group(1).replace("h", ":").replace("min", "")
        try:
            start_time = datetime.strptime(start_time_str, "%H:%M")
        except ValueError:
            start_time = datetime.strptime(start_time_str, "%H")
        duration_str = duration_match.group(1)
        hours = int(re.search(r"(\d+)h", duration_str).group(1)) if "h" in duration_str else 0
        minutes = int(re.search(r"(\d+)min", duration_str).group(1)) if "min" in duration_str else 0
        seconds = int(re.search(r"(\d+)s", duration_str).group(1)) if "s" in duration_str else 0
        duration_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        end_time = start_time + duration_delta
        extracted_data["end_time"] = end_time.strftime("%Hh%Mmin")

    # Extract rapporteur and president
    rapporteur_match = re.search(r"(Rapporteur|Rapporteuse)[:\s]*([A-Z][a-z]+(?: [A-Z][a-z]+)?)", transcription, re.IGNORECASE)
    president_match = re.search(r"(Président|Présidente|Prési)[:\s]*([A-Z][a-z]+(?: [A-Z][a-z]+)?)", transcription, re.IGNORECASE)
    if rapporteur_match:
        extracted_data["rapporteur"] = rapporteur_match.group(2)
    if president_match:
        extracted_data["president"] = president_match.group(2)

    # Extract balance
    balance_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d+)", transcription, re.IGNORECASE)
    if balance_match:
        extracted_data["balance_amount"] = balance_match.group(2)
    balance_date_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d{2}/\d{2}/\d{4})", transcription, re.IGNORECASE)
    if balance_date_match:
        extracted_data["balance_date"] = balance_date_match.group(2)

    # Extract activities review
    activities_section = re.search(r"(Revue des activités|Activités de la semaine)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if activities_section:
        activities_text = activities_section.group(2).strip()
        activities_lines = [line.strip() for line in activities_text.split("\n") if line.strip()]
        for line in activities_lines:
            actor_match = re.search(r"^[A-Z][a-z]+|^([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\s,]", line)
            dossier_match = re.search(r"(?:dossier|sur le dossier) ([A-Za-z0-9\s]+)", line, re.IGNORECASE)
            activities_match = re.search(r"(?:activités|menées) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            results_match = re.search(r"(?:résultat|obtenu) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            perspectives_match = re.search(r"(?:perspectives|prévoit de) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            actor = actor_match.group(1) if actor_match else "Non spécifié"
            dossier = dossier_match.group(1).strip() if dossier_match else "Non spécifié"
            activities_desc = activities_match.group(1).strip() if activities_match else "Non spécifié"
            results = results_match.group(1).strip() if results_match else "Non spécifié"
            perspectives = perspectives_match.group(1).strip() if perspectives_match else "Non spécifié"
            extracted_data["activities_review"].append({
                "actor": actor,
                "dossier": dossier,
                "activities": activities_desc,
                "results": results,
                "perspectives": perspectives
            })

    # Extract resolutions
    resolution_section = re.search(r"(Résolution|Resolutions)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if resolution_section:
        resolution_text = resolution_section.group(2).strip()
        resolution_lines = [line.strip() for line in resolution_text.split("\n") if line.strip()]
        for res in resolution_lines:
            responsible_match = re.search(r"(?:par|responsable|attribué à) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", res, re.IGNORECASE)
            deadline_match = re.search(r"(?:d'ici|avant le) (\d{2}/\d{2}/\d{4})", res, re.IGNORECASE)
            responsible = responsible_match.group(1) if responsible_match else "Non spécifié"
            deadline = deadline_match.group(1) if deadline_match else "Non spécifié"
            extracted_data["resolutions_summary"].append({
                "date": date,
                "dossier": "Non spécifié",
                "resolution": res,
                "responsible": responsible,
                "deadline": deadline,
                "execution_date": "",
                "status": "En cours",
                "report_count": "0"
            })

    # Extract sanctions - PRIORITY: Current meeting > Context > Default
    sanction_section = re.search(r"(Sanction|Amende)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if sanction_section:
        sanction_text = sanction_section.group(2).strip()
        sanction_lines = [line.strip() for line in sanction_text.split("\n") if line.strip()]
        for sanc in sanction_lines:
            name_match = re.search(r"^[A-Z][a-z]+|^([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\s,]", sanc)
            amount_match = re.search(r"(\d+)\s*(?:FCFA|XAF)?", sanc)
            reason_match = re.search(r"(?:pour|raison) ([a-zA-Z\s]+)", sanc, re.IGNORECASE)
            name = name_match.group(1) if name_match else "Non spécifié"
            amount = amount_match.group(1) if amount_match else "0"
            reason = reason_match.group(1).strip() if reason_match else sanc
            extracted_data["sanctions_summary"].append({
                "name": name,
                "reason": reason,
                "amount": amount,
                "date": date,
                "status": "Appliquée"
            })
    
    # Use context sanctions if none found in current meeting
    if not extracted_data["sanctions_summary"] and "context_sanctions" in st.session_state:
        sanctions = st.session_state.context_sanctions
        for sanction in sanctions:
            sanction["date"] = date  # Update date to current meeting
        extracted_data["sanctions_summary"] = sanctions
        st.write(f"Using stored context_sanctions: {sanctions}")
    
    # Default if no sanctions found anywhere
    if not extracted_data["sanctions_summary"]:
        extracted_data["sanctions_summary"] = [{
            "name": "Aucune",
            "reason": "Aucune sanction mentionnée",
            "amount": "0",
            "date": date,
            "status": "Non appliquée"
        }]

    return extracted_data

def to_roman(num):
    """Convert an integer to a Roman numeral."""
    roman_numerals = {
        1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
        6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
    }
    return roman_numerals.get(num, str(num))

def extract_info(transcription, meeting_title, date, deepseek_api_key, previous_context=""):
    """Extract key information from the transcription using Deepseek API with previous context."""
    if not transcription or not deepseek_api_key:
        return extract_info_fallback(transcription, meeting_title, date, previous_context)

    # Get relevant context from vector store
    relevant_meetings = vector_store.get_relevant_context(transcription, k=3)
    context_text = "\n\n".join([
        f"Meeting from {meeting['date']}:\n{json.dumps(meeting['extracted_info'], indent=2)}"
        for meeting in relevant_meetings
    ])

    # Format prompt using prompt manager
    formatted_prompt = prompt_manager.format_prompt(
        name="detailed",  # Use the detailed prompt by default
        context=context_text,
        transcript=transcription,
        date=date,
        title=meeting_title
    )

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_api_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": 0.1,
            "max_tokens": 5000
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            st.error(f"Deepseek API error: Status {response.status_code}. Falling back.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        raw_response = response.json()["choices"][0]["message"]["content"].strip()
        cleaned_response = clean_json_response(raw_response)
        if not cleaned_response:
            st.error("Failed to clean JSON response. Falling back.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        extracted_data = json.loads(cleaned_response)
        if "error" in extracted_data:
            st.error(f"API error: {extracted_data['error']}. Falling back.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        # Add meeting metadata
        extracted_data["date"] = date
        extracted_data["meeting_title"] = meeting_title

        # Store the meeting in vector store for future context
        vector_store.add_meeting({
            "transcript": transcription,
            "date": date,
            "title": meeting_title,
            "extracted_info": extracted_data
        })
        vector_store.save(st.session_state.vector_store_dir)

        return extracted_data

    except Exception as e:
        st.error(f"Error extracting info: {e}. Falling back.")
        return extract_info_fallback(transcription, meeting_title, date, previous_context)

def set_cell_background(cell, rgb_color):
    """Set the background color of a table cell."""
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), f"{rgb_color[0]:02X}{rgb_color[1]:02X}{rgb_color[2]:02X}")
    cell._element.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=0.1, bottom=0.1, left=0.1, right=0.1):
    """Set the margins of a table cell."""
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
    """Set the width of the table."""
    table.autofit = False
    table.allow_autofit = False
    table_width = Inches(width_in_inches)
    table.width = table_width
    for row in table.rows:
        for cell in row.cells:
            cell.width = table_width
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def set_column_widths(table, widths_in_inches):
    """Set preferred widths for each column."""
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
    """Add a styled table to the document."""
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
    """Build the Word document from scratch."""
    try:
        doc = Document()

        # Extract presence list and split into present and absent attendees
        presence_list = extracted_info.get("presence_list", "Présents: Non spécifié\nAbsents: Non spécifié")
        present_attendees = []
        absent_attendees = []
        president = extracted_info.get("president", "Non spécifié")
        if "Présents:" in presence_list and "Absents:" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents:"):
                    presents = part.replace("Présents:", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                    if president != "Non spécifié":
                        for i, attendee in enumerate(present_attendees):
                            if attendee.lower() == president.lower():
                                present_attendees[i] = f"{attendee} (Président)"
                                break
                elif part.startswith("Absents:"):
                    absents = part.replace("Absents:", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Non spécifié" else []

        # Process agenda items
        agenda_list = extracted_info.get("agenda_items", "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d'activités\nIV- Faits saillants\nV- Divers").split("\n")
        agenda_list = [item.strip() for item in agenda_list if item.strip()]

        # Add header box
        add_text_in_box(doc, "Direction Recherches et Investissements", bg_color=(192, 192, 192), font_size=16)
        add_styled_paragraph(doc, "COMPTE RENDU DE RÉUNION", bold=True, color=RGBColor(192, 0, 0), alignment=WD_ALIGN_PARAGRAPH.CENTER)
        add_styled_paragraph(doc, extracted_info.get("date", ""), bold=True, color=RGBColor(192, 0, 0), alignment=WD_ALIGN_PARAGRAPH.CENTER)
        add_styled_paragraph(doc, f"Heure de début: {extracted_info.get('start_time', 'Non spécifié')}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        add_styled_paragraph(doc, f"Heure de fin: {extracted_info.get('end_time', 'Non spécifié')}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        rapporteur = extracted_info.get("rapporteur", "Non spécifié")
        if rapporteur != "Non spécifié":
            add_styled_paragraph(doc, f"Rapporteur: {rapporteur}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)

        # Add attendance table
        add_styled_paragraph(doc, "◆ LISTE DE PRÉSENCE", bold=True)
        if present_attendees or absent_attendees:
            max_rows = max(len(present_attendees), len(absent_attendees)) or 1
            attendance_data = [[present_attendees[i] if i < len(present_attendees) else "", absent_attendees[i] if i < len(absent_attendees) else ""] for i in range(max_rows)]
            add_styled_table(doc, max_rows + 1, 2, ["PRÉSENTS", "ABSENTS"], attendance_data, column_widths=[4.5, 4.5], table_width=9.0)
        else:
            add_styled_paragraph(doc, "Aucune présence spécifiée.")

        doc.add_page_break()

        # Add agenda
        add_styled_paragraph(doc, "◆ Ordre du jour", bold=True)
        for item in agenda_list:
            add_styled_paragraph(doc, item)

        doc.add_page_break()

        # Add activities review
        activities = extracted_info.get("activities_review", [{"actor": "Non spécifié", "dossier": "Non spécifié", "activities": "Non spécifié", "results": "Non spécifié", "perspectives": "Non spécifié"}])
        add_styled_paragraph(doc, "REVUE DES ACTIVITÉS", bold=True, color=RGBColor(192, 0, 0))
        activities_data = [[a.get("actor", ""), a.get("dossier", ""), a.get("activities", ""), a.get("results", ""), a.get("perspectives", "")] for a in activities]
        add_styled_table(doc, len(activities) + 1, 5, ["ACTEURS", "DOSSIERS", "ACTIVITÉS", "RÉSULTATS", "PERSPECTIVES"], activities_data, column_widths=[2.0, 2.0, 2.5, 2.0, 2.0], table_width=10.5)

        # Add resolutions
        resolutions = extracted_info.get("resolutions_summary", [{"date": extracted_info.get("date", ""), "dossier": "Non spécifié", "resolution": "Non spécifié", "responsible": "Non spécifié", "deadline": "Non spécifié", "execution_date": "", "status": "En cours", "report_count": "0"}])
        add_styled_paragraph(doc, "RÉCAPITULATIF DES RÉSOLUTIONS", bold=True, color=RGBColor(192, 0, 0))
        resolutions_data = [[r.get("date", ""), r.get("dossier", ""), r.get("resolution", ""), r.get("responsible", ""), r.get("deadline", ""), r.get("execution_date", ""), r.get("status", ""), str(r.get("report_count", ""))] for r in resolutions]
        add_styled_table(doc, len(resolutions) + 1, 8, ["DATE", "DOSSIER", "RÉSOLUTION", "RESP.", "ÉCHÉANCE", "DATE D'EXÉCUTION", "STATUT", "COMPTE RENDU"], resolutions_data, column_widths=[1.5, 1.8, 2.5, 1.2, 1.8, 1.5, 1.2, 1.5], table_width=12.0)

        # Add sanctions
        sanctions = extracted_info.get("sanctions_summary", [{"name": "Aucune", "reason": "Aucune sanction mentionnée", "amount": "0", "date": extracted_info.get("date", ""), "status": "Non appliquée"}])
        st.write(f"Sanctions for document: {sanctions}")
        add_styled_paragraph(doc, "RÉCAPITULATIF DES SANCTIONS", bold=True, color=RGBColor(192, 0, 0))
        sanctions_data = [[s.get("name", ""), s.get("reason", ""), s.get("amount", ""), s.get("date", ""), s.get("status", "")] for s in sanctions]
        add_styled_table(doc, len(sanctions) + 1, 5, ["NOM", "RAISON", "MONTANT (FCFA)", "DATE", "STATUT"], sanctions_data, column_widths=[2.0, 2.5, 2.0, 1.8, 2.2], table_width=10.5)

        # Add balance
        add_styled_paragraph(doc, f"Solde du compte de solidarité DRI (00001-00921711101-10) est de XAF {extracted_info.get('balance_amount', 'Non spécifié')} au {extracted_info.get('balance_date', '')}.")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            with open(tmp.name, "rb") as f:
                docx_data = f.read()
            os.unlink(tmp.name)
        return docx_data

    except Exception as e:
        st.error(f"Error generating document: {e}")
        return None

def main():
    st.title("Meeting Transcription Tool")
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.session_state.mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    st.session_state.deepseek_api_key = st.sidebar.text_input("Deepseek API Key", type="password")
    
    # Add prompt selection to sidebar
    st.sidebar.header("Prompt Selection")
    available_prompts = list(prompt_manager.prompts.keys())
    selected_prompt = st.sidebar.selectbox(
        "Select Prompt Template",
        available_prompts,
        format_func=lambda x: prompt_manager.prompts[x].description
    )
    
    if st.sidebar.checkbox("Show Selected Prompt"):
        st.sidebar.text_area(
            "Prompt Template",
            prompt_manager.get_prompt(selected_prompt).template,
            height=300
        )
    
    st.sidebar.header("Contexte Précédent")
    previous_report = st.sidebar.file_uploader("Télécharger le rapport précédent", type=["pdf", "png", "jpg", "jpeg"])
    if previous_report:
        st.session_state.previous_report = previous_report
        st.session_state.previous_context = ""
        st.sidebar.write("Rapport téléchargé. Posez une question pour extraire le contexte.")
    else:
        st.session_state.previous_report = None
        st.session_state.previous_context = ""
    
    # Test context
    st.sidebar.header("Tester le Contexte")
    question = st.sidebar.text_input("Posez une question sur le rapport précédent :")
    if st.sidebar.button("Poser la Question") and question:
        if not st.session_state.mistral_api_key or not st.session_state.previous_report:
            st.sidebar.error("Fournissez une clé API Mistral et un rapport précédent.")
        else:
            with st.spinner("Extraction du contexte..."):
                context = extract_context_from_report(st.session_state.previous_report, st.session_state.mistral_api_key)
                if context:
                    st.session_state.previous_context = context
                    st.sidebar.text_area("Contexte Extrait", context, height=200)
                    st.sidebar.success("Contexte extrait !")
                else:
                    st.session_state.previous_context = ""
                    st.sidebar.error("Échec de l'extraction.")
            
            with st.spinner("Obtention de la réponse..."):
                answer = answer_question_with_context(question, st.session_state.previous_context, st.session_state.deepseek_api_key)
            st.sidebar.write("**Réponse :**")
            st.sidebar.write(answer)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
    
    with col2:
        st.header("Transcription & Résultat")
        input_method = st.radio("Méthode d'entrée :", ("Télécharger Audio", "Entrer la Transcription"))
        
        if input_method == "Télécharger Audio":
            uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["mp3", "wav", "m4a", "flac"])
            whisper_model = st.selectbox("Modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
            
            if uploaded_file and st.button("Transcrire l'Audio"):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with st.spinner(f"Transcription avec Whisper {whisper_model}..."):
                    transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                    if transcription and not transcription.startswith("Error"):
                        st.session_state.transcription = transcription
                        with st.spinner("Extraction des informations..."):
                            extracted_info = extract_info(st.session_state.transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), st.session_state.deepseek_api_key, st.session_state.get("previous_context", ""))
                            if extracted_info:
                                st.session_state.extracted_info = extracted_info
                                st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        else:
            transcription_input = st.text_area("Entrez la transcription :", height=200)
            if st.button("Soumettre la Transcription") and transcription_input:
                st.session_state.transcription = transcription_input
                with st.spinner("Extraction des informations..."):
                    extracted_info = extract_info(st.session_state.transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), st.session_state.deepseek_api_key, st.session_state.get("previous_context", ""))
                    if extracted_info:
                        st.session_state.extracted_info = extracted_info
                        st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        
        if 'extracted_info' in st.session_state and st.button("Générer et Télécharger le Document"):
            with st.spinner("Génération du document..."):
                docx_data = fill_template_and_generate_docx(st.session_state.extracted_info, meeting_title, meeting_date)
                if docx_data:
                    st.download_button(
                        label="Télécharger le Document",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()