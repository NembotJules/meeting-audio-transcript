import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docx import Document
import warnings

# Suppression des avertissements pour un affichage plus propre
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Outil de Transcription de Réunion", page_icon=":microphone:", layout="wide")

def transcribe_audio(audio_file, file_extension):
    """Transcrit le fichier audio téléchargé en texte en utilisant le modèle Whisper"""
    try:
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
            temp_audio.write(audio_file.getvalue())
            temp_audio_path = temp_audio.name
        result = transcriber(temp_audio_path, chunk_length_s=30, stride_length_s=5)
        os.unlink(temp_audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Erreur lors de la transcription audio: {e}")
        return f"Erreur lors de la transcription audio: {e}"

def extract_info(transcription, meeting_title, date, attendees, api_key, action_items=None):
    """Extrait les informations clés de la transcription en utilisant l'API Deepseek"""
    if action_items is None:
        action_items = []
    
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"
    else:
        action_items_text = "Aucun point d'action n'a été enregistré."

    prompt = f"""
    Vous êtes un expert en rédaction de comptes rendus de réunion. À partir de la transcription suivante, extrayez et structurez les informations suivantes en sections claires :
    - Présence : Liste des participants présents et absents.
    - Ordre du jour : Points discutés pendant la réunion.
    - Résolutions : Décisions prises, responsables assignés, et délais.
    - Sanctions : Sanctions ou pénalités appliquées, si mentionnées.
    - Informations financières : Soldes, montants, ou autres données financières.
    
    Détails de la Réunion :
    - Titre : {meeting_title}
    - Date : {date}
    - Participants : {attendees}
    - Points d'action :
    {action_items_text}
    
    Transcription :
    {transcription}
    
    Retournez les informations structurées en sections avec des en-têtes clairs, en français.
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
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            st.error(f"Erreur lors de l'extraction des informations: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des informations: {e}")
        return None

def extract_info_fallback(transcription, meeting_title, date, attendees, action_items=None):
    """Mode de secours pour structurer les informations si l'API Deepseek échoue"""
    if action_items is None:
        action_items = []
    
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"
    else:
        action_items_text = "Aucun point d'action n'a été enregistré."
    
    return f"""
## Présence
{attendees}

## Ordre du jour
Non spécifié dans la transcription.

## Résolutions
| DATE | DOSSIERS | RÉSOLUTIONS | RESP. | DÉLAI D'EXÉCUTION | DATE D'EXÉCUTION | STATUT | NBR DE REPORT |
| ---- | -------- | ----------- | ----- | ----------------- | ---------------- | ------ | ------------- |
| {date} | {meeting_title} | Non spécifié | Non spécifié | Non spécifié | Non spécifié | En cours | 00 |

## Sanctions
Aucune sanction mentionnée.

## Informations financières
Aucune information financière mentionnée.

## Points d'action
{action_items_text}

## Transcription
{transcription}
"""

def generate_word_document(extracted_info, meeting_title, date):
    """Génère un document Word à partir des informations extraites"""
    doc = Document()
    doc.add_heading(f"Compte Rendu de Réunion - {meeting_title}", 0)
    doc.add_paragraph(f"Date : {date}")
    
    sections = extracted_info.split("\n\n")
    for section in sections:
        lines = section.split("\n")
        if lines and lines[0].startswith("## "):
            heading = lines[0].replace("## ", "").strip()
            doc.add_heading(heading, level=1)
            content = "\n".join(lines[1:]).strip()
            if heading == "Résolutions" and content.startswith("|"):
                # Gérer le tableau des résolutions
                table_data = [row.split("|")[1:-1] for row in content.split("\n") if row.startswith("|")]
                table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
                table.style = "Table Grid"
                for i, row in enumerate(table_data):
                    for j, cell in enumerate(row):
                        table.cell(i, j).text = cell.strip()
            else:
                doc.add_paragraph(content)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        with open(tmp.name, "rb") as f:
            docx_data = f.read()
        os.unlink(tmp.name)
    return docx_data

def main():
    st.title("Outil de Transcription Audio de Réunion")
    
    # Vérification des dépendances
    try:
        from transformers import pipeline
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        st.warning("""
        ⚠️ Les dépendances nécessaires (transformers, torch) ne sont pas installées.
        Exécutez : `pip install transformers torch`
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
    
    # Initialisation de l'état
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    
    # Barre latérale pour les options
    with st.sidebar:
        st.header("Télécharger un Fichier Audio")
        uploaded_file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "m4a", "flac"])
        
        st.header("Options de Transcription")
        whisper_model = st.selectbox(
            "Taille du Modèle Whisper",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Les modèles plus grands sont plus précis mais plus lents"
        )
        
        st.header("Paramètres API Deepseek")
        api_key = st.text_input("Clé API Deepseek", value=st.session_state.api_key, type="password")
        st.session_state.api_key = api_key
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"Fichier téléchargé: {uploaded_file.name}")
            transcribe_button = st.button("Transcrire l'Audio")
    
    # Contenu principal
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Détails de la Réunion")
            meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
            meeting_date = st.date_input("Date de la Réunion", datetime.now())
            attendees = st.text_area("Participants (séparés par des virgules)")
            
            st.subheader("Points d'Action")
            action_items_container = st.container()
            if 'action_items' not in st.session_state:
                st.session_state.action_items = [""]
            
            with action_items_container:
                new_action_items = []
                for i, item in enumerate(st.session_state.action_items):
                    cols = st.columns([0.9, 0.1])
                    with cols[0]:
                        new_item = st.text_input(f"Point {i+1}", item, key=f"item_{i}")
                    with cols[1]:
                        if st.button("𝗫", key=f"del_{i}"):
                            pass
                        else:
                            new_action_items.append(new_item)
                st.session_state.action_items = new_action_items if new_action_items else [""]
            
            if st.button("Ajouter un Point d'Action"):
                st.session_state.action_items.append("")
                st.rerun()
        
        with col2:
            st.header("Transcription & Sortie")
            
            if transcribe_button and WHISPER_AVAILABLE:
                with st.spinner(f"Transcription audio avec le modèle Whisper {whisper_model}..."):
                    transcription = transcribe_audio(uploaded_file, file_extension)
                
                if transcription and not transcription.startswith("Erreur"):
                    st.success("Transcription terminée!")
                    st.session_state.transcription = transcription
                    
                    st.subheader("Transcription Brute")
                    st.text_area("Modifier si nécessaire:", transcription, height=200, key="edited_transcription")
                    
                    if st.button("Formater les Notes de Réunion") and DOCX_AVAILABLE:
                        edited_transcription = st.session_state.get("edited_transcription", transcription)
                        action_items = [item for item in st.session_state.action_items if item.strip()]
                        
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
                        else:
                            st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                            extracted_info = extract_info_fallback(
                                edited_transcription,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                action_items
                            )
                        
                        if extracted_info:
                            st.session_state.extracted_info = extracted_info
                            st.subheader("Informations Extraites")
                            st.text_area("Aperçu:", extracted_info, height=300)
                            
                            with st.spinner("Génération du document Word..."):
                                docx_data = generate_word_document(
                                    extracted_info,
                                    meeting_title,
                                    meeting_date.strftime("%d/%m/%Y")
                                )
                            
                            st.download_button(
                                label="Télécharger les Notes de Réunion",
                                data=docx_data,
                                file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
            
            elif hasattr(st.session_state, 'transcription') and WHISPER_AVAILABLE:
                st.subheader("Transcription Brute")
                st.text_area("Modifier si nécessaire:", st.session_state.transcription, height=200, key="edited_transcription")
                
                if st.button("Formater les Notes de Réunion") and DOCX_AVAILABLE:
                    edited_transcription = st.session_state.get("edited_transcription", st.session_state.transcription)
                    action_items = [item for item in st.session_state.action_items if item.strip()]
                    
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
                    else:
                        st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                        extracted_info = extract_info_fallback(
                            edited_transcription,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y"),
                            attendees,
                            action_items
                        )
                    
                    if extracted_info:
                        st.session_state.extracted_info = extracted_info
                        st.subheader("Informations Extraites")
                        st.text_area("Aperçu:", extracted_info, height=300)
                        
                        with st.spinner("Génération du document Word..."):
                            docx_data = generate_word_document(
                                extracted_info,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y")
                            )
                        
                        st.download_button(
                            label="Télécharger les Notes de Réunion",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
            
            elif hasattr(st.session_state, 'extracted_info') and DOCX_AVAILABLE:
                st.subheader("Informations Extraites")
                st.text_area("Aperçu:", st.session_state.extracted_info, height=300)
                
                with st.spinner("Génération du document Word..."):
                    docx_data = generate_word_document(
                        st.session_state.extracted_info,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y")
                    )
                
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
           pip install streamlit transformers torch python-docx requests
           ```
        2. Pour Streamlit Cloud, assurez-vous d'avoir un fichier `requirements.txt` :
           ```
           streamlit>=1.24.0
           transformers>=4.30.0
           torch>=2.0.1
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
        attendees = st.text_area("Participants (séparés par des virgules)")
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = f"""
## Présence
{attendees}

## Ordre du jour
Non spécifié.

## Résolutions
| DATE | DOSSIERS | RÉSOLUTIONS | RESP. | DÉLAI D'EXÉCUTION | DATE D'EXÉCUTION | STATUT | NBR DE REPORT |
| ---- | -------- | ----------- | ----- | ----------------- | ---------------- | ------ | ------------- |
| {meeting_date.strftime("%d/%m/%Y")} | {meeting_title} | Non spécifié | Non spécifié | Non spécifié | Non spécifié | En cours | 00 |

## Sanctions
Aucune sanction mentionnée.

## Informations financières
Aucune information financière mentionnée.

## Transcription
{transcription}
"""
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", extracted_info, height=300)
            
            try:
                from docx import Document
                docx_data = generate_word_document(extracted_info, meeting_title, meeting_date.strftime("%d/%m/%Y"))
                st.download_button(
                    label="Télécharger les Notes de Réunion",
                    data=docx_data,
                    file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except ImportError:
                st.warning("python-docx n'est pas installé. Téléchargement impossible.")