import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docx import Document
from docx.oxml.ns import qn
from docx.shared import RGBColor, Pt
import warnings
import torch
import torchaudio
import json

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

def extract_info(transcription, meeting_title, date, attendees, api_key, action_items=None):
    """Extract key information from the transcription using Deepseek API"""
    if action_items is None:
        action_items = []
    
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"
    else:
        action_items_text = "Aucun point d'action n'a été enregistré."

    prompt = f"""
    Vous êtes un expert en rédaction de comptes rendus de réunion. À partir de la transcription suivante, extrayez et structurez les informations suivantes pour remplir un modèle de compte rendu de réunion. Retournez les informations sous forme de JSON avec les clés suivantes :

    - "date" : La date de la réunion (format DD/MM/YYYY, par défaut {date}).
    - "start_time" : L'heure de début de la réunion (format HHhMMmin, par exemple 07h00min).
    - "end_time" : L'heure de fin de la réunion (format HHhMMmin, par exemple 10h34min).
    - "presence_list" : Liste des participants présents et absents (chaîne de texte, par exemple "Présents : Alice, Bob\nAbsents : Charlie").
    - "agenda_items" : Liste des points discutés (chaîne de texte avec des sauts de ligne, par exemple "1. Point 1\n2. Point 2").
    - "resolutions_summary" : Liste de résolutions sous forme de tableau (liste de dictionnaires avec les clés "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count").
    - "sanctions_summary" : Liste de sanctions sous forme de tableau (liste de dictionnaires avec les clés "name", "reason", "amount", "date", "status").
    - "balance_amount" : Le solde du compte DRI Solidarité (chaîne de texte, par exemple "682040").
    - "balance_date" : La date du solde (format DD/MM/YYYY, par exemple "06/02/2025").

    Détails de la Réunion :
    - Titre : {meeting_title}
    - Date par défaut : {date}
    - Participants fournis : {attendees}
    - Points d'action :
    {action_items_text}
    
    Transcription :
    {transcription}
    
    Retournez le résultat sous forme de JSON structuré, en français. Si une information n'est pas trouvée dans la transcription, utilisez des valeurs par défaut raisonnables (par exemple, "Non spécifié" ou la date fournie). Assurez-vous que le JSON est bien formé.
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
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                st.error(f"Erreur lors du parsing JSON : {e}")
                return None
        else:
            st.error(f"Erreur API Deepseek : Statut {response.status_code}, Message : {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des informations : {e}")
        return None

def extract_info_fallback(transcription, meeting_title, date, attendees, action_items=None, start_time="Non spécifié", end_time="Non spécifié", agenda_items=None):
    """Fallback mode for structuring information if Deepseek API fails"""
    if action_items is None:
        action_items = []
    if agenda_items is None:
        agenda_items = ["Non spécifié dans la transcription."]
    
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"
    else:
        action_items_text = "Aucun point d'action n'a été enregistré."
    
    agenda_text = "\n".join([f"{idx}. {item}" for idx, item in enumerate(agenda_items, 1)])
    
    return {
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "presence_list": attendees if attendees else "Non spécifié",
        "agenda_items": agenda_text,
        "resolutions_summary": [
            {
                "date": date,
                "dossier": meeting_title,
                "resolution": "Non spécifié",
                "responsible": "Non spécifié",
                "deadline": "Non spécifié",
                "execution_date": "",
                "status": "En cours",
                "report_count": "00"
            }
        ],
        "sanctions_summary": [
            {
                "name": "Aucun",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": date,
                "status": "Non appliqué"
            }
        ],
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "action_items": action_items_text,
        "transcription": transcription
    }

def generate_docx(extracted_info):
    """Generate a Word document from scratch using the extracted information with black and red theme"""
    try:
        # Create a new Word document
        doc = Document()
        
        # Add heading: Direction Recherches et Investissements
        heading1 = doc.add_heading("Direction Recherches et Investissements", level=1)
        for run in heading1.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        
        # Add subheading: COMPTE RENDU RÉUNION HEBDOMADAIRE
        heading2 = doc.add_heading("COMPTE RENDU RÉUNION HEBDOMADAIRE", level=2)
        for run in heading2.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        
        # Add date
        doc.add_paragraph(extracted_info["date"])
        
        # Add start time
        doc.add_paragraph(f"Heure début : {extracted_info['start_time']}")
        
        # Add end time
        doc.add_paragraph(f"Heure de fin : {extracted_info['end_time']}")
        
        # Add presence/absence table
        heading_presence = doc.add_heading("LISTE DE PRÉSENCE / ABSENCE :", level=3)
        for run in heading_presence.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        
        # Parse presence_list into Présents and Absents
        presence_list = extracted_info["presence_list"]
        presents = "Non spécifié"
        absents = "Non spécifié"
        if "Présents :" in presence_list and "Absents :" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents :"):
                    presents = part.replace("Présents :", "").strip()
                elif part.startswith("Absents :"):
                    absents = part.replace("Absents :", "").strip()
        else:
            # If not in the expected format, assume the entire string is the list of presents
            presents = presence_list if presence_list != "Non spécifié" else "Non spécifié"
        
        # Create a 2-column table for presence/absence
        table = doc.add_table(rows=2, cols=2)
        table.style = "Table Grid"
        # Add table headers with red text
        headers = ["Présence", "Absence"]
        for j, header in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = header
            for run in cell.paragraphs[0].runs:
                run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        # Fill the table
        table.cell(1, 0).text = presents
        table.cell(1, 1).text = absents
        
        # Add agenda items
        heading_agenda = doc.add_heading("ORDRE DU JOUR :", level=3)
        for run in heading_agenda.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        # Split agenda_items into lines
        agenda_lines = extracted_info["agenda_items"].split("\n")
        for line in agenda_lines:
            doc.add_paragraph(line)
        
        # Add resolutions table
        heading_resolutions = doc.add_heading("RÉCAPITULATIF DES RÉSOLUTIONS", level=3)
        for run in heading_resolutions.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
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
                "report_count": "00"
            }]
        table = doc.add_table(rows=len(resolutions) + 1, cols=8)
        table.style = "Table Grid"
        # Add table headers with red text
        headers = ["DATE", "DOSSIERS", "RÉSOLUTIONS", "RESP.", "DÉLAI D'EXÉCUTION", "DATE D'EXÉCUTION", "STATUT", "NBR DE REPORT"]
        for j, header in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = header
            for run in cell.paragraphs[0].runs:
                run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        # Fill table rows with black text
        for row_idx, resolution in enumerate(resolutions, 1):
            table.cell(row_idx, 0).text = resolution.get("date", "Non spécifié")
            table.cell(row_idx, 1).text = resolution.get("dossier", "Non spécifié")
            table.cell(row_idx, 2).text = resolution.get("resolution", "Non spécifié")
            table.cell(row_idx, 3).text = resolution.get("responsible", "Non spécifié")
            table.cell(row_idx, 4).text = resolution.get("deadline", "Non spécifié")
            table.cell(row_idx, 5).text = resolution.get("execution_date", "")
            table.cell(row_idx, 6).text = resolution.get("status", "En cours")
            table.cell(row_idx, 7).text = resolution.get("report_count", "00")
        
        # Add sanctions table
        heading_sanctions = doc.add_heading("RÉCAPITULATIF DES SANCTIONS", level=3)
        for run in heading_sanctions.runs:
            run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        sanctions = extracted_info.get("sanctions_summary", [])
        if not sanctions:
            sanctions = [{
                "name": "Aucun",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info["date"],
                "status": "Non appliqué"
            }]
        table = doc.add_table(rows=len(sanctions) + 1, cols=5)
        table.style = "Table Grid"
        # Add table headers with red text
        headers = ["NOM", "MOTIF", "MONTANT (FCFA)", "DATE", "STATUT"]
        for j, header in enumerate(headers):
            cell = table.cell(0, j)
            cell.text = header
            for run in cell.paragraphs[0].runs:
                run.font.color.rgb = RGBColor(255, 0, 0)  # Red color
        # Fill table rows with black text
        for row_idx, sanction in enumerate(sanctions, 1):
            table.cell(row_idx, 0).text = sanction.get("name", "Aucun")
            table.cell(row_idx, 1).text = sanction.get("reason", "Aucune sanction mentionnée")
            table.cell(row_idx, 2).text = sanction.get("amount", "0")
            table.cell(row_idx, 3).text = sanction.get("date", extracted_info["date"])
            table.cell(row_idx, 4).text = sanction.get("status", "Non appliqué")
        
        # Add spacing after the sanctions table
        doc.add_paragraph("")  # Blank paragraph for spacing
        doc.add_paragraph("")  # Additional blank paragraph for more spacing
        
        # Add balance information
        balance_para = doc.add_paragraph(
            f"Le solde du compte DRI Solidarité (00001-00921711101-10) est de XAF {extracted_info['balance_amount']} au {extracted_info['balance_date']}."
        )
        for run in balance_para.runs:
            run.font.color.rgb = RGBColor(0, 0, 0)  # Black color
        
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
        start_time = st.text_input("Heure de début (format HHhMMmin, ex: 07h00min)", value="07h00min")
        end_time = st.text_input("Heure de fin (format HHhMMmin, ex: 10h34min)", value="10h34min")
        attendees = st.text_area("Participants (séparés par des virgules)")
        
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
        
        st.subheader("Points d'Action")
        action_items_container = st.container()
        if 'action_items' not in st.session_state:
            st.session_state.action_items = [""]
        
        with action_items_container:
            new_action_items = []
            for i, item in enumerate(st.session_state.action_items):
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    new_item = st.text_input(f"Point d'Action {i+1}", item, key=f"action_item_{i}")
                with cols[1]:
                    if st.button("𝗫", key=f"del_action_{i}"):
                        pass
                    else:
                        new_action_items.append(new_item)
            st.session_state.action_items = new_action_items if new_action_items else [""]
        
        if st.button("Ajouter un Point d'Action"):
            st.session_state.action_items.append("")
            st.rerun()
    
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
                action_items = [item for item in st.session_state.action_items if item.strip()]
                agenda_items = [item for item in st.session_state.agenda_items if item.strip()]
                
                if st.session_state.api_key:
                    with st.spinner("Extraction des informations avec Deepseek..."):
                        extracted_info = extract_info(
                            edited_transcription,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y"),
                            attendees,
                            st.session_state.api_key,
                            action_items
                        )
                        if not extracted_info:
                            extracted_info = extract_info_fallback(
                                edited_transcription,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                action_items,
                                start_time,
                                end_time,
                                agenda_items
                            )
                        else:
                            # Override with user inputs
                            extracted_info["start_time"] = start_time
                            extracted_info["end_time"] = end_time
                            extracted_info["agenda_items"] = "\n".join([f"{idx}. {item}" for idx, item in enumerate(agenda_items, 1)]) if agenda_items else "Non spécifié"
                else:
                    st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                    extracted_info = extract_info_fallback(
                        edited_transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        attendees,
                        action_items,
                        start_time,
                        end_time,
                        agenda_items
                    )
                
                if extracted_info:
                    st.session_state.extracted_info = extracted_info
                    st.subheader("Informations Extraites")
                    st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
                    
                    with st.spinner("Génération du document Word..."):
                        docx_data = generate_docx(extracted_info)
                    
                    if docx_data:
                        st.download_button(
                            label="Télécharger les Notes de Réunion",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
        
        elif hasattr(st.session_state, 'extracted_info') and DOCX_AVAILABLE:
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(st.session_state.extracted_info, indent=2, ensure_ascii=False), height=300)
            
            with st.spinner("Génération du document Word..."):
                docx_data = generate_docx(st.session_state.extracted_info)
            
            if docx_data:
                st.download_button(
                    label="Télécharger les Notes de Réunion",
                    data=docx_data,
                    file_name=f"{meeting_title}_{datetime.now().strftime('%Y-%m-%d')}_notes.docx",
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
        start_time = st.text_input("Heure de début (format HHhMMmin, ex: 07h00min)", value="07h00min")
        end_time = st.text_input("Heure de fin (format HHhMMmin, ex: 10h34min)", value="10h34min")
        attendees = st.text_area("Participants (séparés par des virgules)")
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = extract_info_fallback(
                transcription,
                meeting_title,
                meeting_date.strftime("%d/%m/%Y"),
                attendees,
                start_time=start_time,
                end_time=end_time
            )
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
            
            try:
                docx_data = generate_docx(extracted_info)
                if docx_data:
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.warning(f"Erreur lors de la génération du document: {e}")