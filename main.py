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

st.set_page_config(page_title="Outil de Transcription de Réunion avec Template", page_icon=":microphone:", layout="wide")

# --- FONCTIONS ------------------------------------------------------------------

def transcribe_audio(audio_file, file_extension, model_size="base"):
    """Transcribe the uploaded audio file to text using the Whisper model"""
    model_id_mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
    }
    model_id = model_id_mapping.get(model_size, "openai/whisper-base")
    transcriber = pipeline("automatic-speech-recognition", model=model_id)
    # Save to temporary file for torchaudio
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name
    # Load and resample
    waveform, sr = torchaudio.load(temp_audio_path, backend="ffmpeg")
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    result = transcriber({"raw": waveform[0].numpy(), "sampling_rate": 16000},
                         chunk_length_s=30, stride_length_s=5)
    os.unlink(temp_audio_path)
    return result.get("text", "")


def extract_info(transcription, meeting_title, date, attendees, api_key):
    """Extract key information from the transcription using Deepseek API"""
    prompt = f"""
    Vous êtes un expert en rédaction de comptes rendus de réunion. À partir de la transcription suivante, extrayez et structurez les informations suivantes pour remplir un modèle de compte rendu de réunion. Retournez les informations sous forme de JSON avec les clés suivantes :

    - "date" : La date de la réunion (format DD/MM/YYYY, par défaut {date}).
    - "start_time" : L'heure de début de la réunion (format HHhMMmin).
    - "end_time" : L'heure de fin de la réunion (format HHhMMmin).
    - "presence_list" : Liste des participants présents et absents.
    - "agenda_items" : Liste des points discutés.
    - "resolutions_summary" : Liste de résolutions (liste de dictionnaires avec clés "date","dossier","resolution","responsible","deadline","execution_date","status","report_count").
    - "sanctions_summary" : Liste de sanctions (liste de dictionnaires avec clés "name","reason","amount","date","status").
    - "balance_amount" : Le solde du compte DRI Solidarité.
    - "balance_date" : La date du solde (format DD/MM/YYYY).

    Détails de la Réunion :
    - Titre : {meeting_title}
    - Date par défaut : {date}
    - Participants : {attendees}

    Transcription :
    {transcription}

    Retournez le résultat sous forme de JSON structuré, en français. Si une information n'est pas trouvée, utilisez "Non spécifié" ou la date fournie.
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
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        raw = response.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            st.error("Impossible de parser la réponse Deepseek en JSON.")
            return None
    else:
        st.error(f"Erreur Deepseek: {response.status_code} {response.text}")
        return None


def extract_info_fallback(transcription, meeting_title, date, attendees,
                          start_time="Non spécifié", end_time="Non spécifié",
                          agenda_items=None, balance_amount="Non spécifié",
                          balance_date=None):
    """Fallback if Deepseek API fails"""
    if agenda_items is None:
        agenda_items = []
    if balance_date is None:
        balance_date = date

    agenda_formatted = [f"{idx}. {item}" for idx, item in enumerate(agenda_items, 1)]
    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "presence_list": attendees or "Non spécifié",
        "agenda_items": agenda_formatted,
        "resolutions_summary": [],
        "sanctions_summary": [],
        "balance_amount": balance_amount,
        "balance_date": balance_date
    }


def generate_docx_from_template(template_path, context):
    """Load a .docx template with Jinja placeholders and render it."""
    doc = DocxTemplate(template_path)
    doc.render(context)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        data = open(tmp.name, 'rb').read()
    os.unlink(tmp.name)
    return data

# --- APPLICATION STREAMLIT ----------------------------------------------------

def main():
    st.title("Transcription & Génération de Compte-rendu")

    # Sidebar: Source + API + Template
    with st.sidebar:
        st.header("Source de la Transcription")
        method = st.radio("Méthode :", ("Fichier audio", "Saisie manuelle"))
        if method == "Fichier audio":
            audio_file = st.file_uploader("Audio (wav/mp3/m4a/flac)", type=["wav","mp3","m4a","flac"])
            whisper_model = st.selectbox("Modèle Whisper", ["tiny","base","small","medium","large"], index=1)
        else:
            audio_file = None
            manual_text = st.text_area("Transcription manuelle", height=200)
        
        st.header("Deepseek API")
        api_key = st.text_input("Clé API Deepseek", type="password")

        st.header("Template Word (.docx)")
        template_file = st.file_uploader("Charger template Jinja2", type=["docx"] )

    # Meeting details
    meeting_title = st.text_input("Titre Réunion", "Réunion")
    meeting_date = st.date_input("Date Réunion", datetime.now())
    start_time = st.text_input("Heure début (HHhMMmin)", "07h00min")
    end_time = st.text_input("Heure fin (HHhMMmin)", "10h34min")
    attendees = st.text_area("Participants (virgules)")

    st.subheader("Ordre du jour")
    if 'agenda' not in st.session_state:
        st.session_state.agenda = [""]
    for i in range(len(st.session_state.agenda)):
        st.session_state.agenda[i] = st.text_input(f"Point {i+1}", st.session_state.agenda[i], key=f"agenda_{i}")
    if st.button("Ajouter un point"):
        st.session_state.agenda.append("")
        return  # stop to trigger rerun with updated state
    agenda_items = [i for i in st.session_state.agenda if i.strip()]

    balance_amount = st.text_input("Solde (XAF)", "682040")
    balance_date = st.date_input("Date solde", meeting_date)

    # Transcription
    transcription = None
    if method == "Fichier audio" and audio_file:
        if st.button("Transcrire"):    
            with st.spinner("Transcription Whisper..."):
                transcription = transcribe_audio(audio_file, audio_file.name.split('.')[-1], whisper_model)
            st.success("Transcription terminée!")
    elif method == "Saisie manuelle" and manual_text:
        transcription = manual_text
        st.success("Transcription chargée!")

    if transcription:
        st.subheader("Transcription")
        transcription = st.text_area("Éditez si besoin :", transcription, height=200)

        if st.button("Générer Compte-rendu (.docx)"):
            # Extraction info
            if api_key:
                info = extract_info(transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), attendees, api_key)
                if not info:
                    info = extract_info_fallback(transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), attendees,
                                                  start_time, end_time, agenda_items, balance_amount,
                                                  balance_date.strftime("%d/%m/%Y"))
                # Override
                info.update({
                    "start_time": start_time,
                    "end_time": end_time,
                    "agenda_items": agenda_items,
                    "balance_amount": balance_amount,
                    "balance_date": balance_date.strftime("%d/%m/%Y")
                })
            else:
                st.warning("Pas de clé API Deepseek, mode fallback")
                info = extract_info_fallback(transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), attendees,
                                              start_time, end_time, agenda_items, balance_amount,
                                              balance_date.strftime("%d/%m/%Y"))

            if template_file:
                # Save template
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmpf:
                    tmpf.write(template_file.getvalue())
                    template_path = tmpf.name
                # Build context
                context = {
                    "date": info.get("date"),
                    "start_time": info.get("start_time"),
                    "end_time": info.get("end_time"),
                    "presence_list": info.get("presence_list"),
                    "agenda": info.get("agenda_items", []),
                    "resolutions": info.get("resolutions_summary", []),
                    "sanctions": info.get("sanctions_summary", []),
                    "account_balance": info.get("balance_amount"),
                    "balance_date": info.get("balance_date")
                }
                # Generate
                docx_bytes = generate_docx_from_template(template_path, context)
                os.unlink(template_path)
                # Download
                st.download_button(
                    "Télécharger Compte-rendu", data=docx_bytes,
                    file_name=f"CR_{meeting_date.strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.error("Veuillez charger un template .docx")

if __name__ == '__main__':
    main()
