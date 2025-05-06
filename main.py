import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docxtpl import DocxTemplate
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import warnings
import torch
import torchaudio
import json

# Suppression des avertissements pour un affichage plus propre
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Outil de Transcription de Réunion", page_icon=":microphone:", layout="wide")

# Path to the Word template
TEMPLATE_PATH = "Template_reunion.docx"

def transcribe_audio(audio_file, file_extension, model_size="base"):
    """Transcrit le fichier audio téléchargé en texte en utilisant le modèle Whisper"""
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
            response_text = response.json()["choices"][0]["message"]["content"].strip()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                st.error(f"Erreur lors du parsing de la réponse Deepseek en JSON: {e}. Utilisation du mode de secours.")
                return None
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
    
    return {
        "date": date,
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "presence_list": attendees if attendees else "Non spécifié",
        "agenda_items": "Non spécifié dans la transcription.",
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

def fill_template_and_generate_docx(extracted_info, template_path):
    """Remplit le modèle Word et génère un document téléchargeable"""
    try:
        # Vérifier si le modèle existe
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Le fichier modèle {template_path} n'est pas trouvé.")
        
        # Charger le modèle avec docxtpl
        doc = DocxTemplate(template_path)
        
        # Préparer les données pour le modèle
        context = {
            "date": extracted_info["date"],
            "start_time": extracted_info["start_time"],
            "end_time": extracted_info["end_time"],
            "presence_list": extracted_info["presence_list"],
            "agenda_items": extracted_info["agenda_items"],
            # Les tableaux seront traités après, donc on passe des chaînes vides ici
            "resolutions_summary": "",
            "sanctions_summary": "",
            "balance_amount": extracted_info["balance_amount"],
            "balance_date": extracted_info["balance_date"]
        }
        
        # Remplir les placeholders simples
        doc.render(context)
        
        # Sauvegarder temporairement le document rempli
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            tmp_path = tmp.name
        
        # Charger le document avec python-docx pour ajouter les tableaux
        docx_doc = Document(tmp_path)
        
        # Trouver les paragraphes correspondant aux sections "RÉCAPITULATIF DES RÉSOLUTIONS" et "RÉCAPITULATIF DES SANCTIONS"
        resolutions_found = False
        sanctions_found = False
        for i, para in enumerate(docx_doc.paragraphs):
            if "RÉCAPITULATIF DES RÉSOLUTIONS" in para.text:
                resolutions_found = True
                # Ajouter le tableau des résolutions
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
                table = docx_doc.add_table(rows=len(resolutions) + 1, cols=8)
                table.style = "Table Grid"
                # En-têtes du tableau
                headers = ["DATE", "DOSSIERS", "RÉSOLUTIONS", "RESP.", "DÉLAI D'EXÉCUTION", "DATE D'EXÉCUTION", "STATUT", "NBR DE REPORT"]
                for j, header in enumerate(headers):
                    table.cell(0, j).text = header
                # Remplir les données
                for row_idx, resolution in enumerate(resolutions, 1):
                    table.cell(row_idx, 0).text = resolution.get("date", "Non spécifié")
                    table.cell(row_idx, 1).text = resolution.get("dossier", "Non spécifié")
                    table.cell(row_idx, 2).text = resolution.get("resolution", "Non spécifié")
                    table.cell(row_idx, 3).text = resolution.get("responsible", "Non spécifié")
                    table.cell(row_idx, 4).text = resolution.get("deadline", "Non spécifié")
                    table.cell(row_idx, 5).text = resolution.get("execution_date", "")
                    table.cell(row_idx, 6).text = resolution.get("status", "En cours")
                    table.cell(row_idx, 7).text = resolution.get("report_count", "00")
                # Déplacer le tableau après le paragraphe
                para._element.addnext(table._element)
                para.text = para.text.replace("{{resolutions_summary}}", "")
            
            if "RÉCAPITULATIF DES SANCTIONS" in para.text:
                sanctions_found = True
                # Ajouter le tableau des sanctions
                sanctions = extracted_info.get("sanctions_summary", [])
                if not sanctions:
                    sanctions = [{
                        "name": "Aucun",
                        "reason": "Aucune sanction mentionnée",
                        "amount": "0",
                        "date": extracted_info["date"],
                        "status": "Non appliqué"
                    }]
                table = docx_doc.add_table(rows=len(sanctions) + 1, cols=5)
                table.style = "Table Grid"
                # En-têtes du tableau
                headers = ["NOM", "MOTIF", "MONTANT (FCFA)", "DATE", "STATUT"]
                for j, header in enumerate(headers):
                    table.cell(0, j).text = header
                # Remplir les données
                for row_idx, sanction in enumerate(sanctions, 1):
                    table.cell(row_idx, 0).text = sanction.get("name", "Aucun")
                    table.cell(row_idx, 1).text = sanction.get("reason", "Aucune sanction mentionnée")
                    table.cell(row_idx, 2).text = sanction.get("amount", "0")
                    table.cell(row_idx, 3).text = sanction.get("date", extracted_info["date"])
                    table.cell(row_idx, 4).text = sanction.get("status", "Non appliqué")
                # Déplacer le tableau après le paragraphe
                para._element.addnext(table._element)
                para.text = para.text.replace("{{sanctions_summary}}", "")
        
        # Si les sections n'ont pas été trouvées, ajouter les tableaux à la fin
        if not resolutions_found:
            docx_doc.add_heading("RÉCAPITULATIF DES RÉSOLUTIONS", level=1)
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
            table = docx_doc.add_table(rows=len(resolutions) + 1, cols=8)
            table.style = "Table Grid"
            headers = ["DATE", "DOSSIERS", "RÉSOLUTIONS", "RESP.", "DÉLAI D'EXÉCUTION", "DATE D'EXÉCUTION", "STATUT", "NBR DE REPORT"]
            for j, header in enumerate(headers):
                table.cell(0, j).text = header
            for row_idx, resolution in enumerate(resolutions, 1):
                table.cell(row_idx, 0).text = resolution.get("date", "Non spécifié")
                table.cell(row_idx, 1).text = resolution.get("dossier", "Non spécifié")
                table.cell(row_idx, 2).text = resolution.get("resolution", "Non spécifié")
                table.cell(row_idx, 3).text = resolution.get("responsible", "Non spécifié")
                table.cell(row_idx, 4).text = resolution.get("deadline", "Non spécifié")
                table.cell(row_idx, 5).text = resolution.get("execution_date", "")
                table.cell(row_idx, 6).text = resolution.get("status", "En cours")
                table.cell(row_idx, 7).text = resolution.get("report_count", "00")
        
        if not sanctions_found:
            docx_doc.add_heading("RÉCAPITULATIF DES SANCTIONS", level=1)
            sanctions = extracted_info.get("sanctions_summary", [])
            if not sanctions:
                sanctions = [{
                    "name": "Aucun",
                    "reason": "Aucune sanction mentionnée",
                    "amount": "0",
                    "date": extracted_info["date"],
                    "status": "Non appliqué"
                }]
            table = docx_doc.add_table(rows=len(sanctions) + 1, cols=5)
            table.style = "Table Grid"
            headers = ["NOM", "MOTIF", "MONTANT (FCFA)", "DATE", "STATUT"]
            for j, header in enumerate(headers):
                table.cell(0, j).text = header
            for row_idx, sanction in enumerate(sanctions, 1):
                table.cell(row_idx, 0).text = sanction.get("name", "Aucun")
                table.cell(row_idx, 1).text = sanction.get("reason", "Aucune sanction mentionnée")
                table.cell(row_idx, 2).text = sanction.get("amount", "0")
                table.cell(row_idx, 3).text = sanction.get("date", extracted_info["date"])
                table.cell(row_idx, 4).text = sanction.get("status", "Non appliqué")
        
        # Sauvegarder le document final
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as final_tmp:
            docx_doc.save(final_tmp.name)
            with open(final_tmp.name, "rb") as f:
                docx_data = f.read()
            os.unlink(final_tmp.name)
        
        # Nettoyer le fichier temporaire initial
        os.unlink(tmp_path)
        
        return docx_data
    
    except Exception as e:
        st.error(f"Erreur lors de la génération du document Word: {e}")
        st.write("""
        Assurez-vous que :
        - Le fichier Template_reunion.docx utilise la syntaxe {{placeholder}} pour les placeholders (par exemple, {{date}} au lieu de [Date]).
        - Le fichier n'est pas ouvert dans une autre application.
        - Vous avez les permissions nécessaires pour écrire des fichiers temporaires.
        """)
        return None

def main():
    st.title("Outil de Transcription Audio de Réunion")
    
    # Vérification des dépendances
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
        from docxtpl import DocxTemplate
        from docx import Document
        DOCX_AVAILABLE = True
    except ImportError:
        DOCX_AVAILABLE = False
        st.warning("""
        ⚠️ Les bibliothèques docxtpl et python-docx ne sont pas installées.
        Exécutez : `pip install docxtpl python-docx`
        """)
    
    # Vérifier si le modèle existe
    if not os.path.exists(TEMPLATE_PATH):
        st.error(f"Le modèle Word {TEMPLATE_PATH} n'est pas trouvé. Veuillez placer le fichier dans le même répertoire que le script.")
    
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
                    transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                
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
                                if not extracted_info:
                                    extracted_info = extract_info_fallback(
                                        edited_transcription,
                                        meeting_title,
                                        meeting_date.strftime("%d/%m/%Y"),
                                        attendees,
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
                            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
                            
                            with st.spinner("Génération du document Word..."):
                                docx_data = fill_template_and_generate_docx(extracted_info, TEMPLATE_PATH)
                            
                            if docx_data:
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
                            if not extracted_info:
                                extracted_info = extract_info_fallback(
                                    edited_transcription,
                                    meeting_title,
                                    meeting_date.strftime("%d/%m/%Y"),
                                    attendees,
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
                        st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
                        
                        with st.spinner("Génération du document Word..."):
                            docx_data = fill_template_and_generate_docx(extracted_info, TEMPLATE_PATH)
                        
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
                    docx_data = fill_template_and_generate_docx(st.session_state.extracted_info, TEMPLATE_PATH)
                
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
           pip install streamlit transformers torch torchaudio docxtpl python-docx requests
           ```
        2. Pour Streamlit Cloud, assurez-vous d'avoir un fichier `requirements.txt` :
           ```
           streamlit>=1.24.0
           transformers>=4.30.0
           torch>=2.0.1
           torchaudio>=2.0.2
           docxtpl>=0.16.7
           python-docx>=0.8.11
           requests>=2.28.0
           ```
        3. Installez ffmpeg pour les fichiers .m4a :
           - Sur Ubuntu : `sudo apt-get install ffmpeg`
           - Sur macOS : `brew install ffmpeg`
           - Sur Windows : Téléchargez depuis https://ffmpeg.org/download.html
        4. Assurez-vous que le modèle Word (Template_reunion.docx) est dans le même répertoire que le script et utilise la syntaxe {{placeholder}} pour les placeholders (par exemple, {{date}} au lieu de [Date]).
        5. Vérifiez que le fichier modèle n'est pas ouvert dans une autre application pendant l'exécution.
        """)
        
        st.title("Mode Secours")
        st.warning("Application en mode limité. La transcription audio n'est pas disponible.")
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
        attendees = st.text_area("Participants (séparés par des virgules)")
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = extract_info_fallback(
                transcription,
                meeting_title,
                meeting_date.strftime("%d/%m/%Y"),
                attendees
            )
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
            
            try:
                docx_data = fill_template_and_generate_docx(extracted_info, TEMPLATE_PATH)
                if docx_data:
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.warning(f"Erreur lors de la génération du document: {e}")