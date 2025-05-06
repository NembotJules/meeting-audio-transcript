
import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docxtpl import DocxTemplate
import warnings
import torch
import torchaudio
import json

# Suppression des avertissements pour un affichage plus propre
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Outil de Transcription de Réunion",
    page_icon=":microphone:",
    layout="wide"
)

# Path to the Word template with all Jinja placeholders
TEMPLATE_PATH = "Template_reunion_full_placeholders.docx"

def transcribe_audio(audio_file, file_extension, model_size="base"):
    """Transcribe the uploaded audio file to text using the Whisper model."""
    model_id_mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
    }
    model_id = model_id_mapping.get(model_size, "openai/whisper-base")
    transcriber = pipeline("automatic-speech-recognition", model=model_id)

    # Save to temp file so torchaudio can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    try:
        waveform, sample_rate = torchaudio.load(tmp_path, backend="ffmpeg")
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        result = transcriber(
            {"raw": waveform[0].numpy(), "sampling_rate": 16000},
            chunk_length_s=30,
            stride_length_s=5
        )
        return result["text"]
    finally:
        os.unlink(tmp_path)

def extract_info(transcription, meeting_title, date, attendees, api_key, action_items=None):
    """Extract structured info from the transcription via Deepseek API."""
    if action_items is None:
        action_items = []
    action_items_text = "\n".join(f"{i+1}. {itm}" for i, itm in enumerate(action_items)) or "Aucun point d'action."

    prompt = f"""
Vous êtes un expert en rédaction de comptes rendus. À partir de la transcription :
- Titre : {meeting_title}
- Date : {date}
- Participants fournis : {attendees}
- Points d'action :
{action_items_text}

Transcription :
{transcription}

Retournez un JSON français avec les clés :
"date", "start_time", "end_time", 
"presence_list", "agenda_items", 
"resolutions_summary", "sanctions_summary",
"balance_amount", "balance_date".
"""

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
    resp = requests.post("https://api.deepseek.com/v1/chat/completions",
                         headers=headers, json=payload)
    if resp.status_code != 200:
        st.error(f"Deepseek API Error {resp.status_code}: {resp.text}")
        return None

    content = resp.json()["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        return None

def extract_info_fallback(transcription, meeting_title, date, attendees, action_items=None):
    """Fallback defaults when Deepseek fails."""
    if action_items is None:
        action_items = []
    return {
        "date": date,
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "presence_list": f"Présents : {attendees}\nAbsents : ",
        "agenda_items": ["Non spécifié"],
        "resolutions_summary": [{
            "date": date,
            "dossier": meeting_title,
            "resolution": "Non spécifié",
            "responsible": "Non spécifié",
            "deadline": "Non spécifié",
            "execution_date": "",
            "status": "En cours",
            "report_count": "00"
        }],
        "sanctions_summary": [{
            "name": "Aucun",
            "reason": "Aucune sanction",
            "amount": "0",
            "date": date,
            "status": "Non appliqué"
        }],
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "action_items": action_items,
        "transcription": transcription
    }

def parse_presence_list(presence_list_str):
    """Convert "Présents : A,B\nAbsents : C" into two Python lists."""
    attendees, absentees = [], []
    for line in presence_list_str.splitlines():
        key, _, names = line.partition(":")
        names = [n.strip() for n in names.split(",") if n.strip()]
        if key.strip().lower().startswith("présent"):
            attendees = names
        if key.strip().lower().startswith("absent"):
            absentees = names
    return attendees, absentees

def fill_template_and_generate_docx(extracted, template_path):
    """Render the Jinja template and return .docx bytes."""
    tpl = DocxTemplate(template_path)
    attendees, absentees = parse_presence_list(extracted.get("presence_list", ""))
    context = {
        "date": extracted["date"],
        "start_time": extracted["start_time"],
        "end_time": extracted["end_time"],
        "attendees": attendees,
        "absentees": absentees,
        "agenda_items": extracted.get("agenda_items", []),
        "resolutions": extracted.get("resolutions_summary", []),
        "sanctions": extracted.get("sanctions_summary", []),
        "balance_amount": extracted.get("balance_amount", ""),
        "balance_date": extracted.get("balance_date", "")
    }
    tpl.render(context)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tpl.save(tmp.name)
        result = open(tmp.name, "rb").read()
    os.unlink(tmp.name)
    return result

def main():
    st.title("Outil de Transcription Audio de Réunion")

    # Sidebar inputs
    with st.sidebar:
        st.header("Source de la Transcription")
        method = st.radio("Méthode :", ["Fichier audio", "Transcription manuelle"])
        if method == "Fichier audio":
            uploaded = st.file_uploader("Choisir un audio", type=["wav","mp3","m4a","flac"])
        else:
            uploaded = None
            manual = st.text_area("Collez votre transcription :", height=200)
        whisper_model = st.selectbox("Modèle Whisper", ["tiny","base","small","medium","large"], index=1)
        api_key = st.text_input("Clé API Deepseek", type="password")

    # Meeting details
    st.header("Détails de la Réunion")
    title = st.text_input("Titre", value="Réunion")
    date = st.date_input("Date", datetime.now())
    attendees_str = st.text_area("Participants (virgules)", value="")
    st.subheader("Points d'Action")
    if "action_items" not in st.session_state:
        st.session_state.action_items = [""]
    for i, itm in enumerate(st.session_state.action_items):
        st.session_state.action_items[i] = st.text_input(f"Point {i+1}", itm, key=f"ai{i}")
    if st.button("Ajouter un point"):
        st.session_state.action_items.append("")
        st.experimental_rerun()

    # Transcription / Formatting
    st.header("Transcription & Sortie")
    transcription = None
    if method == "Fichier audio" and uploaded:
        if st.button("Transcrire"):
            with st.spinner("Transcription en cours..."):
                transcription = transcribe_audio(uploaded, uploaded.name.split(".")[-1], whisper_model)
                st.session_state.transcription = transcription
    elif method != "Fichier audio" and manual:
        transcription = manual

    transcription = transcription or st.session_state.get("transcription")
    if transcription:
        st.subheader("Transcription")
        transcription = st.text_area("Éditez si besoin :", transcription, height=200)
        if st.button("Formater les Notes de Réunion"):
            action_items = [a for a in st.session_state.action_items if a.strip()]
            if api_key:
                extracted = extract_info(transcription, title, date.strftime("%d/%m/%Y"), attendees_str, api_key, action_items)
                if extracted is None:
                    extracted = extract_info_fallback(transcription, title, date.strftime("%d/%m/%Y"), attendees_str, action_items)
            else:
                extracted = extract_info_fallback(transcription, title, date.strftime("%d/%m/%Y"), attendees_str, action_items)

            st.subheader("Informations Extraites")
            st.text_area("JSON brut :", json.dumps(extracted, ensure_ascii=False, indent=2), height=300)

            with st.spinner("Génération du document Word..."):
                docx_bytes = fill_template_and_generate_docx(extracted, TEMPLATE_PATH)

            st.download_button(
                "Télécharger le CR",
                data=docx_bytes,
                file_name=f"{title}_{date.strftime('%Y-%m-%d')}_CR.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if __name__ == "__main__":
    main()
