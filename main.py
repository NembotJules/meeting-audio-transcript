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

def extract_info(transcription, meeting_title, date, deepseek_api_key, previous_context=""):
    """Extract key information from the transcription using Deepseek API with previous context."""
    prompt = f"""
    You are an AI assistant specialized in drafting meeting reports. 
    From the following transcription and the previous meeting context (specifically Activity Review, Resolutions Summary), extract key information and return it as structured JSON in English.

    **Previous Meeting Context**:
    {previous_context if previous_context else "No context available."}

    **Current Meeting Transcription**:
    {transcription}

    **Sections to Extract**:
    - **presence_list**: List of present and absent participants as a string (e.g., "Present: Alice, Bob\nAbsent: Charlie"). Identify names mentioned as present or absent. If not found, use "Present: Not specified\nAbsent: Not specified".
    - **agenda_items**: List of agenda items as a string (e.g., "1. Review of minutes\n2. Resolutions"). Deduce discussed or explicitly mentioned items. If not found, use "Not specified".
    - **resolutions_summary**: List of resolutions as an array (list of dictionaries with keys "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count"). "date", "deadline", and "execution_date" in DD/MM/YYYY format. "report_count" as a string (e.g., "0").
    - **sanctions_summary**: List of sanctions as an array (list of dictionaries with keys "name", "reason", "amount", "date", "status"). "date" in DD/MM/YYYY format, "amount" as a string.
    - **start_time**: Meeting start time (format HHhMMmin, e.g., 07h00min). Deduce if mentioned, else use "Not specified".
    - **end_time**: Meeting end time (format HHhMMmin, e.g., 10h34min). Deduce if mentioned, else use "Not specified".
    - **rapporteur**: Name of the meeting rapporteur. Deduce if mentioned, else use "Not specified".
    - **president**: Name of the meeting president. Deduce if mentioned, else use "Not specified".
    - **balance_amount**: DRI Solidarity account balance (e.g., "827540"). Deduce if mentioned, else use "Not specified".
    - **balance_date**: Balance date (format DD/MM/YYYY). Deduce if mentioned, else use provided date: {date}.

    **Instructions**:
    1. For each speaker, identify their dossier(s), dates, resolutions, responsible party, deadline, status, and report count.
    2. If information is missing, use reasonable defaults (e.g., "Not specified" or provided date: {date}).
    3. Ensure JSON is well-formed and dates follow DD/MM/YYYY format.

    Return the result as structured JSON in English.
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
            st.write(f"Raw Deepseek response: {raw_response}")
            try:
                extracted_data = json.loads(raw_response)
                extracted_data["date"] = date
                extracted_data["meeting_title"] = meeting_title  # Add meeting_title for document generation
                return extracted_data
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {e}")
                return None
        else:
            st.error(f"Deepseek API error: Status {response.status_code}, Message: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error extracting information: {e}")
        return None

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

def fill_template_and_generate_docx(extracted_info):
    """Build the Word document from scratch using python-docx"""
    try:
        doc = Document()

        # Extract presence list and split into present and absent attendees
        presence_list = extracted_info.get("presence_list", "Present: Not specified\nAbsent: Not specified")
        present_attendees = []
        absent_attendees = []
        if "Present:" in presence_list and "Absent:" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Present:"):
                    presents = part.replace("Present:", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                elif part.startswith("Absent:"):
                    absents = part.replace("Absent:", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Not specified" else []

        # Process agenda items
        agenda_list = extracted_info.get("agenda_items", "Not specified").split("\n")
        agenda_list = [f"{to_roman(idx)}. {item.strip()}" for idx, item in enumerate(agenda_list, 1) if item.strip() and item != "Not specified"]
        if not agenda_list:
            agenda_list = ["I. Not specified"]

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
            "MEETING NOTES",
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
            f"Start Time: {extracted_info.get('start_time', 'Not specified')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        add_styled_paragraph(
            doc,
            f"End Time: {extracted_info.get('end_time', 'Not specified')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )

        # Add rapporteur and president
        rapporteur = extracted_info.get("rapporteur", "Not specified")
        president = extracted_info.get("president", "Not specified")
        if rapporteur != "Not specified":
            add_styled_paragraph(
                doc,
                f"Rapporteur: {rapporteur}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )
        if president != "Not specified":
            add_styled_paragraph(
                doc,
                f"President of Meeting: {president}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )

        # Add attendance table
        add_styled_paragraph(
            doc,
            "◆ ATTENDANCE LIST",
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
                headers=["PRESENT", "ABSENT"],
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
                "No attendance specified.",
                font_name="Century",
                font_size=12
            )

        # Add agenda items
        add_styled_paragraph(
            doc,
            "◆ Agenda",
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
                "dossier": "Not specified",
                "resolution": "Not specified",
                "responsible": "Not specified",
                "deadline": "Not specified",
                "execution_date": "",
                "status": "In progress",
                "report_count": "0"
            }]
        add_styled_paragraph(
            doc,
            "RESOLUTIONS SUMMARY",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        resolutions_headers = ["DATE", "DOSSIER", "RESOLUTION", "RESP.", "DEADLINE", "EXECUTION DATE", "STATUS", "REPORT COUNT"]
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
                "name": "None",
                "reason": "No sanctions mentioned",
                "amount": "0",
                "date": extracted_info.get("date", ""),
                "status": "Not applied"
            }]
        add_styled_paragraph(
            doc,
            "SANCTIONS SUMMARY",
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        sanctions_headers = ["NAME", "REASON", "AMOUNT (FCFA)", "DATE", "STATUS"]
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
            f"DRI Solidarity account balance (00001-00921711101-10) is XAF {extracted_info.get('balance_amount', 'Not specified')} as of {extracted_info.get('balance_date', '')}.",
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
    
    st.sidebar.header("Previous Context")
    previous_report = st.sidebar.file_uploader("Upload Previous Report (optional)", type=["pdf", "png", "jpg", "jpeg"])
    if previous_report:
        st.session_state.previous_report = previous_report
        st.session_state.previous_context = ""  # Reset context until a question is asked
        st.sidebar.write("Previous report uploaded. Ask a question to extract context.")
    else:
        st.session_state.previous_report = None
        st.session_state.previous_context = ""
    
    # Section to ask questions about the context
    st.sidebar.header("Test the Context")
    question = st.sidebar.text_input("Ask a question about the previous report:")
    if st.sidebar.button("Ask Question") and question:
        if not st.session_state.mistral_api_key:
            st.sidebar.error("Please provide a Mistral API key to extract context.")
        elif not st.session_state.previous_report:
            st.sidebar.error("Please upload a previous report to extract context.")
        else:
            with st.spinner("Extracting context..."):
                context = extract_context_from_report(
                    st.session_state.previous_report, 
                    st.session_state.mistral_api_key
                )
                if context:
                    st.session_state.previous_context = context
                    st.sidebar.text_area("Extracted Context", context, height=200)
                    st.sidebar.success("Context extracted successfully!")
                else:
                    st.session_state.previous_context = ""
                    st.sidebar.error("Failed to extract context. Check the API key or file.")
            
            # Now answer the question
            with st.spinner("Getting answer..."):
                answer = answer_question_with_context(
                    question, 
                    st.session_state.previous_context, 
                    st.session_state.deepseek_api_key
                )
            st.sidebar.write("**Answer:**")
            st.sidebar.write(answer)
    
    # Main app content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Meeting Details")
        meeting_title = st.text_input("Meeting Title", value="Meeting")
        meeting_date = st.date_input("Meeting Date", datetime.now())
    
    with col2:
        st.header("Transcription & Output")
        input_method = st.radio("Choose input method:", ("Upload Audio", "Input Transcript"))
        
        if input_method == "Upload Audio":
            uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])
            whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=1)
            
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if st.button("Transcribe Audio"):
                    with st.spinner(f"Transcribing with Whisper {whisper_model}..."):
                        transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                        if transcription and not transcription.startswith("Error"):
                            st.session_state.transcription = transcription
                            st.text_area("Transcription", transcription, height=200)
        else:
            transcription_input = st.text_area("Enter the meeting transcript:", height=200)
            if st.button("Submit Transcript") and transcription_input:
                st.session_state.transcription = transcription_input
                st.text_area("Transcription", transcription_input, height=200)
        
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
                    if extracted_info:
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