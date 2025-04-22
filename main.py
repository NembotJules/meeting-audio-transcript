import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa


st.set_page_config(page_title="Outil de Transcription de Réunion", page_icon=":microphone:", layout="wide")


def transcribe_audio(audio_file, file_extension, model_size="base"):
    """Transcrit le fichier audio téléchargé en texte en utilisant le modèle Whisper de Hugging Face"""

    # Création d'un fichier temporaire avec la bonne extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name

    try:
        # Correspondance entre la taille du modèle et l'ID du modèle Hugging Face
        model_id_mapping = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large": "openai/whisper-large-v3",
        }
        model_id = model_id_mapping.get(model_size, "openai/whisper-base")
        
        # Chargement de l'audio
        speech_array, sampling_rate = librosa.load(temp_audio_path, sr=16000)
        
        # Chargement du processeur et du modèle depuis Hugging Face
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        # Utilisation du GPU si disponible
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Traitement de l'audio
        input_features = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        # Génération des IDs de tokens
        predicted_ids = model.generate(input_features)
        
        # Décodage des IDs de tokens en texte
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription

    except Exception as e:
        st.error(f"Erreur lors de la transcription audio: {e}")
        return "Erreur lors de la transcription audio"

    finally:
        # Nettoyage du fichier temporaire
        os.unlink(temp_audio_path)


def format_meeting_notes_with_llm(transcript, meeting_title, date, attendees, template, api_key, action_items=None):
    """Formate la transcription en notes de réunion en utilisant l'API Deepseek"""

    if action_items is None:
        action_items = []

    # Préparation des points d'action sous forme de texte s'ils existent
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"
    else: 
        action_items_text = "Aucun point d'action n'a été enregistré pendant la réunion."

    # L'invite pour le LLM
    prompt = f"""
    Vous êtes un professionnel du formatage des notes de réunion. Formatez la transcription de réunion suivante selon le modèle fourni.
    
    Détails de la Réunion:
    - Titre de la Réunion: {meeting_title}
    - Date: {date}
    - Participants: {attendees}
    - Points d'Action:
    {action_items_text}

    Transcription de la Réunion: 
    {transcript}

    Modèle de Réunion:
    {template}

   Veuillez formater la transcription de la réunion selon ce modèle, en la rendant professionnelle et bien organisée.
   Travaillez en français.
"""
    
    try: 
        # Appel de l'API Deepseek
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1, 
            "max_tokens": 4000
            
        }

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            formatted_notes = result["choices"][0]["message"]["content"].strip()
            return formatted_notes
        else: 
            st.error(f"Erreur lors du formatage des notes de réunion: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"Erreur lors du formatage des notes: {e}")
        return format_meeting_notes_fallback(transcript, meeting_title, date, attendees, template, action_items)
    

def format_meeting_notes_fallback(transcript, meeting_title, date, attendees, template, action_items=None):
    """Formateur de secours si l'appel LLM échoue"""

    # Création d'un tableau pour la réunion
    meeting_notes = f"""
    | DATE | DOSSIERS | RÉSOLUTIONS | RESP. | DÉLAI D'EXÉCUTION | DATE D'EXÉCUTION | STATUT | NBR DE REPORT |
    | ---- | -------- | ----------- | ----- | ----------------- | ---------------- | ------ | ------------- |
    | {date} | {meeting_title} | | | | | En cours | 00 |
    """

    # Ajout des points d'action s'ils existent
    meeting_notes += "\n\n## Points d'Action:\n"
    if action_items:
       for idx, item in enumerate(action_items, 1):
           meeting_notes += f"{idx}. {item}\n"
    else: 
        meeting_notes += "Aucun point d'action n'a été enregistré pendant la réunion."
    
    # Ajout de la transcription
    meeting_notes += f"\n\n## Transcription:\n{transcript}"

    return meeting_notes


def main():
    st.title("Outil de Transcription Audio de Réunion")
    
    # Initialisation de l'état pour la clé API et le modèle
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'template' not in st.session_state:
        st.session_state.template = """
| DATE | DOSSIERS | RÉSOLUTIONS | RESP. | DÉLAI D'EXÉCUTION | DATE D'EXÉCUTION | STATUT | NBR DE REPORT |
| ---- | -------- | ----------- | ----- | ----------------- | ---------------- | ------ | ------------- |
| [DATE] | [TITRE DE LA RÉUNION] | [POINTS CLÉS DISCUTÉS] | [RESPONSABLE] | [DÉLAI] | | En cours | 00 |

## Points d'Action:
[LISTE DES POINTS D'ACTION AVEC LES PERSONNES RESPONSABLES]

## Prochaines Étapes:
[ACTIONS DE SUIVI OU PROCHAINE RÉUNION]
"""
    
    # Barre latérale pour le téléchargement de fichiers et les options
    with st.sidebar:
        st.header("Télécharger un Fichier Audio")
        uploaded_file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "m4a", "flac"])
        
        # Options du modèle
        st.header("Options de Transcription")
        whisper_model = st.selectbox(
            "Taille du Modèle Whisper",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Les modèles plus grands sont plus précis mais plus lents"
        )
        
        # Paramètres API
        st.header("Paramètres API Deepseek")
        api_key = st.text_input("Clé API Deepseek", 
                                value=st.session_state.api_key, 
                                type="password",
                                help="Entrez votre clé API Deepseek")
        # Sauvegarde de la clé API dans l'état de session
        st.session_state.api_key = api_key
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"Fichier téléchargé: {uploaded_file.name}")
            
            # Ajout d'un bouton pour démarrer la transcription
            transcribe_button = st.button("Transcrire l'Audio")
    
    # Zone de contenu principal avec deux colonnes
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Détails de la Réunion")
            meeting_title = st.text_input("Titre de la Réunion")
            meeting_date = st.date_input("Date de la Réunion", datetime.now())
            attendees = st.text_area("Participants (séparés par des virgules)")
            
            # Personnalisation du modèle
            st.subheader("Modèle de Notes de Réunion")
            template = st.text_area("Personnaliser le Modèle", 
                                   value=st.session_state.template, 
                                   height=250)
            # Sauvegarde du modèle dans l'état de session
            st.session_state.template = template
            
            # Conteneur pour les points d'action dynamiques
            st.subheader("Points d'Action")
            action_items_container = st.container()
            
            # Initialisation de l'état de session pour les points d'action si ce n'est pas déjà fait
            if 'action_items' not in st.session_state:
                st.session_state.action_items = [""]
                
            # Affichage de tous les points d'action actuels
            with action_items_container:
                new_action_items = []
                
                for i, item in enumerate(st.session_state.action_items):
                    # Pour chaque point d'action, création d'une ligne avec une entrée de texte et un bouton de suppression
                    cols = st.columns([0.9, 0.1])
                    with cols[0]:
                        new_item = st.text_input(f"Point {i+1}", item, key=f"item_{i}")
                    with cols[1]:
                        if st.button("𝗫", key=f"del_{i}"):
                            pass  # Nous gérerons la suppression en n'ajoutant pas à la nouvelle liste
                        else:
                            new_action_items.append(new_item)
                
                # Mise à jour de l'état de session avec la liste filtrée (gère les suppressions)
                st.session_state.action_items = new_action_items if new_action_items else [""]
                
            # Bouton pour ajouter un nouveau point d'action
            if st.button("Ajouter un Point d'Action"):
                st.session_state.action_items.append("")
                st.rerun()  # Force le rafraîchissement pour afficher le nouveau champ
        
        with col2:
            st.header("Transcription & Sortie")
            
            if transcribe_button:
                with st.spinner(f"Transcription audio avec le modèle Whisper {whisper_model} en cours..."):
                    transcript = transcribe_audio(uploaded_file, file_extension, whisper_model)
                
                if transcript:
                    st.success("Transcription terminée!")
                    
                    # Stockage de la transcription dans l'état de session
                    st.session_state.transcript = transcript
                    
                    # Affichage de la transcription
                    st.subheader("Transcription Brute")
                    st.text_area("Modifier si nécessaire:", transcript, height=200, key="edited_transcript")
                    
                    # Bouton de formatage des notes
                    if st.button("Formater les Notes de Réunion"):
                        # Utilisation de la transcription modifiée
                        edited_transcript = st.session_state.get("edited_transcript", transcript)
                        
                        # Filtrage des points d'action vides
                        action_items = [item for item in st.session_state.action_items if item.strip()]
                        
                        if st.session_state.api_key:
                            with st.spinner("Formatage avec Deepseek LLM..."):
                                # Formatage avec le LLM
                                formatted_notes = format_meeting_notes_with_llm(
                                    edited_transcript,
                                    meeting_title,
                                    meeting_date.strftime("%d/%m/%Y"),
                                    attendees,
                                    st.session_state.template,
                                    st.session_state.api_key,
                                    action_items
                                )
                        else:
                            st.warning("Aucune clé API Deepseek fournie. Utilisation du formateur de secours.")
                            # Formatage avec le formateur de secours
                            formatted_notes = format_meeting_notes_fallback(
                                edited_transcript,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                st.session_state.template,
                                action_items
                            )
                        
                        # Stockage des notes formatées dans l'état de session
                        st.session_state.formatted_notes = formatted_notes
                        
                        # Affichage des notes formatées
                        st.subheader("Notes de Réunion Formatées")
                        st.text_area("Aperçu:", formatted_notes, height=300)
                        
                        # Création d'un bouton de téléchargement pour les notes formatées
                        st.download_button(
                            label="Télécharger les Notes de Réunion",
                            data=formatted_notes,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.md",
                            mime="text/markdown"
                        )
            
            # Si nous avons déjà transcrit, affichage des résultats
            elif hasattr(st.session_state, 'transcript'):
                # Affichage de la transcription
                st.subheader("Transcription Brute")
                st.text_area("Modifier si nécessaire:", st.session_state.transcript, height=200, key="edited_transcript")
                
                # Bouton de formatage des notes
                if st.button("Formater les Notes de Réunion"):
                    # Utilisation de la transcription modifiée
                    edited_transcript = st.session_state.get("edited_transcript", st.session_state.transcript)
                    
                    # Filtrage des points d'action vides
                    action_items = [item for item in st.session_state.action_items if item.strip()]
                    
                    if st.session_state.api_key:
                        with st.spinner("Formatage avec Deepseek LLM..."):
                            # Formatage avec le LLM
                            formatted_notes = format_meeting_notes_with_llm(
                                edited_transcript,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                st.session_state.template,
                                st.session_state.api_key,
                                action_items
                            )
                    else:
                        st.warning("Aucune clé API Deepseek fournie. Utilisation du formateur de secours.")
                        # Formatage avec le formateur de secours
                        formatted_notes = format_meeting_notes_fallback(
                            edited_transcript,
                            meeting_title,
                            meeting_date.strftime("%d/%m/%Y"),
                            attendees,
                            st.session_state.template,
                            action_items
                        )
                    
                    # Stockage des notes formatées dans l'état de session
                    st.session_state.formatted_notes = formatted_notes
                    
                    # Affichage des notes formatées
                    st.subheader("Notes de Réunion Formatées")
                    st.text_area("Aperçu:", formatted_notes, height=300)
                    
                    # Création d'un bouton de téléchargement pour les notes formatées
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=formatted_notes,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.md",
                        mime="text/markdown"
                    )
            
            # Si nous avons déjà des notes formatées, les afficher
            elif hasattr(st.session_state, 'formatted_notes'):
                st.subheader("Notes de Réunion Formatées")
                st.text_area("Aperçu:", st.session_state.formatted_notes, height=300)
                
                # Création d'un bouton de téléchargement pour les notes formatées
                st.download_button(
                    label="Télécharger les Notes de Réunion",
                    data=st.session_state.formatted_notes,
                    file_name=f"notes_reunion_{datetime.now().strftime('%Y-%m-%d')}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()