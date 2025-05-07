import streamlit as st
import os
import tempfile
from datetime import datetime
import requests
from transformers import pipeline
from docxtpl import DocxTemplate
from docx import Document
from docx.shared import RGBColor
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

def extract_info(transcription, meeting_title, date, attendees, api_key):
    """Extract key information from the transcription using Deepseek API"""
    prompt = f"""
    Vous êtes un expert en rédaction de comptes rendus de réunion. À partir de la transcription suivante, extrayez et structurez les informations suivantes pour remplir un modèle de compte rendu de réunion. Retournez les informations sous forme de JSON avec les clés suivantes :

    - "date" : La date de la réunion (format DD/MM/YYYY, par défaut {date}).
    - "start_time" : L'heure de début de la réunion (format HHhMMmin, par exemple 07h00min).
    - "end_time" : L'heure de fin de la réunion (format HHhMMmin, par exemple 10h34min).
    - "presence_list" : Liste des participants présents et absents (chaîne de texte, par exemple "Présents : Alice, Bob\nAbsents : Charlie").
    - "agenda_items" : Liste des points discutés (chaîne de texte avec des sauts de ligne, par exemple "1. Point 1\n2. Point 2").
    - "resolutions_summary" : Liste de résolutions sous forme de tableau (liste de dictionnaires avec les clés "date", "dossier", "resolution", "responsible", "deadline", "execution_date", "status", "report_count"). Le champ "dossier" doit refléter le sujet spécifique de la résolution (par exemple, "Planification Projet X", "Budget Q2"), et non le titre général de la réunion.
    - "sanctions_summary" : Liste de sanctions sous forme de tableau (liste de dictionnaires avec les clés "name", "reason", "amount", "date", "status").
    - "balance_amount" : Le solde du compte DRI Solidarité (chaîne de texte, par exemple "682040").
    - "balance_date" : La date du solde (format DD/MM/YYYY, par exemple "06/02/2025").

    Détails de la Réunion :
    - Titre : {meeting_title}
    - Date par défaut : {date}
    - Participants fournis : {attendees}
    
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

def extract_info_fallback(transcription, meeting_title, date, attendees, start_time="Non spécifié", end_time="Non spécifié", agenda_items=None, balance_amount="Non spécifié", balance_date=None):
    """Fallback mode for structuring information if Deepseek API fails"""
    if agenda_items is None:
        agenda_items = ["Non spécifié dans la transcription."]
    if balance_date is None:
        balance_date = date
    
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
                "dossier": "Planification Projet",
                "resolution": "Non spécifié",
                "responsible": "Non spécifié",
                "deadline": "Non spécifié",
                "execution_date": "",
                "status": "En cours",
                "report_count": "00"
            },
            {
                "date": date,
                "dossier": "Budget Q2",
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

def fill_template_and_generate_docx(extracted_info):
    """Fill the Word template with extracted info using a hybrid approach"""
    try:
        # Load the template
        template_path = "Template_reunion (1).docx"
        if not os.path.exists(template_path):
            st.error(f"Le fichier de modèle '{template_path}' n'existe pas. Assurez-vous qu'il est dans le répertoire de travail.")
            return None
        
        doc = DocxTemplate(template_path)
        
        # Parse presence_list into lists of present and absent attendees
        presence_list = extracted_info["presence_list"]
        present_attendees = []
        absent_attendees = []
        if "Présents :" in presence_list and "Absents :" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents :"):
                    presents = part.replace("Présents :", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                elif part.startswith("Absents :"):
                    absents = part.replace("Absents :", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Non spécifié" else ["Non spécifié"]
        
        # Prepare agenda items as a list
        agenda_list = extracted_info["agenda_items"].split("\n") if extracted_info["agenda_items"] else ["Non spécifié"]
        agenda_list = [f"{to_roman(idx)}. {item.strip()}" for idx, item in enumerate(agenda_list, 1) if item.strip()]
        
        # Prepare the context for docxtpl (non-table placeholders only)
        context = {
            "date": extracted_info["date"],
            "start_time": extracted_info["start_time"],
            "end_time": extracted_info["end_time"],
            "account_balance": extracted_info["balance_amount"],
            "balance_date": extracted_info["balance_date"],
        }
        
        # Log the context for debugging
        st.write("Contexte pour le modèle (docxtpl) :", context)
        
        # Render the template with docxtpl (for non-table placeholders)
        doc.render(context)
        
        # Save the rendered document to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            # Load the rendered document with python-docx to populate tables
            rendered_doc = Document(tmp.name)
        
        # Populate the attendance table
        for table in rendered_doc.tables:
            if len(table.rows) > 0 and len(table.columns) == 2:
                if "PRÉSENCES" in table.cell(0, 0).text and "ABSENCES" in table.cell(0, 1).text:
                    row_index = 1  # Start after header
                    for present in present_attendees:
                        if present and present != "Non spécifié":
                            if row_index < len(table.rows):
                                table.cell(row_index, 0).text = present
                            else:
                                new_row = table.add_row()
                                new_row.cells[0].text = present
                            row_index += 1
                    
                    row_index = 1  # Reset for absents
                    for absent in absent_attendees:
                        if absent and absent != "Non spécifié":
                            if row_index < len(table.rows):
                                table.cell(row_index, 1).text = absent
                            else:
                                new_row = table.add_row()
                                new_row.cells[1].text = absent
                            row_index += 1
                    break
        
        # Populate the agenda items
        agenda_inserted = False
        for i, paragraph in enumerate(rendered_doc.paragraphs):
            if "ORDRE DU JOUR :" in paragraph.text:
                for j, item in enumerate(agenda_list):
                    if j == 0:
                        # Replace the first agenda item paragraph
                        rendered_doc.paragraphs[i + 1].text = item
                    else:
                        # Insert new paragraphs for additional items
                        new_paragraph = rendered_doc.paragraphs[i + 1]._element
                        new_p = new_paragraph.getparent().insert(new_paragraph.getparent().index(new_paragraph) + j, new_paragraph.__class__())
                        new_p.text = item
                agenda_inserted = True
                break
        if not agenda_inserted:
            st.warning("Section 'ORDRE DU JOUR :' non trouvée dans le modèle.")
        
        # Populate the resolutions table
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
        
        for table in rendered_doc.tables:
            if len(table.rows) > 0 and len(table.columns) >= 8:
                cell_text = ' '.join([cell.text for row in table.rows for cell in row.cells])
                if "NBR DE REPORT" in cell_text:
                    for i, resolution in enumerate(resolutions):
                        row_index = i + 1  # Skip header
                        if row_index < len(table.rows):
                            row = table.rows[row_index]
                        else:
                            row = table.add_row()
                        if len(row.cells) >= 8:
                            row.cells[0].text = resolution.get("date", "")
                            row.cells[1].text = resolution.get("dossier", "")
                            row.cells[2].text = resolution.get("resolution", "")
                            row.cells[3].text = resolution.get("responsible", "")
                            row.cells[4].text = resolution.get("deadline", "")
                            row.cells[5].text = resolution.get("execution_date", "")
                            row.cells[6].text = resolution.get("status", "")
                            row.cells[7].text = str(resolution.get("report_count", ""))
                    break
        
        # Populate the sanctions table
        sanctions = extracted_info.get("sanctions_summary", [])
        if not sanctions:
            sanctions = [{
                "name": "Aucun",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info["date"],
                "status": "Non appliqué"
            }]
        
        for table in rendered_doc.tables:
            if len(table.rows) > 0 and len(table.columns) >= 5:
                headers = [cell.text.strip() for cell in table.rows[0].cells]
                if "NOM" in ' '.join(headers) and "MOTIF" in ' '.join(headers):
                    for i, sanction in enumerate(sanctions):
                        row_index = i + 1  # Skip header
                        if row_index < len(table.rows):
                            row = table.rows[row_index]
                        else:
                            row = table.add_row()
                        if len(row.cells) >= 5:
                            row.cells[0].text = sanction.get("name", "")
                            row.cells[1].text = sanction.get("reason", "")
                            row.cells[2].text = sanction.get("amount", "")
                            row.cells[3].text = sanction.get("date", "")
                            row.cells[4].text = sanction.get("status", "")
                    break
        
        # Save the final document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as final_tmp:
            rendered_doc.save(final_tmp.name)
            with open(final_tmp.name, "rb") as f:
                docx_data = f.read()
            os.unlink(final_tmp.name)
        
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
        from docxtpl import DocxTemplate
        from docx import Document
        DOCX_AVAILABLE = True
    except ImportError:
        DOCX_AVAILABLE = False
        st.warning("""
        ⚠️ Les bibliothèques docxtpl et python-docx ne sont pas installées.
        Exécutez : `pip install docxtpl python-docx`
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
        
        st.subheader("Solde du Compte DRI Solidarité")
        balance_amount = st.text_input("Solde (en XAF, ex: 682040)", value="682040")
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
                            st.session_state.api_key
                        )
                        if not extracted_info:
                            extracted_info = extract_info_fallback(
                                edited_transcription,
                                meeting_title,
                                meeting_date.strftime("%d/%m/%Y"),
                                attendees,
                                start_time,
                                end_time,
                                agenda_items,
                                balance_amount,
                                balance_date.strftime("%d/%m/%Y")
                            )
                        else:
                            # Override with user inputs
                            extracted_info["start_time"] = start_time
                            extracted_info["end_time"] = end_time
                            extracted_info["agenda_items"] = "\n".join([f"{idx}. {item}" for idx, item in enumerate(agenda_items, 1)]) if agenda_items else "Non spécifié"
                            extracted_info["balance_amount"] = balance_amount
                            extracted_info["balance_date"] = balance_date.strftime("%d/%m/%Y")
                else:
                    st.warning("Aucune clé API Deepseek fournie. Utilisation du mode de secours.")
                    extracted_info = extract_info_fallback(
                        edited_transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        attendees,
                        start_time,
                        end_time,
                        agenda_items,
                        balance_amount,
                        balance_date.strftime("%d/%m/%Y")
                    )
                
                if extracted_info:
                    st.session_state.extracted_info = extracted_info
                    st.subheader("Informations Extraites")
                    st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
                    
                    with st.spinner("Génération du document Word..."):
                        docx_data = fill_template_and_generate_docx(extracted_info)
                    
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
                docx_data = fill_template_and_generate_docx(st.session_state.extracted_info)
            
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
           docxtpl>=0.16.0
           python-docx>=0.8.11
           requests>=2.28.0
           ```
        3. Installez ffmpeg pour les fichiers .m4a :
           - Sur Ubuntu : `sudo apt-get install ffmpeg`
           - Sur macOS : `brew install ffmpeg`
           - Sur Windows : Téléchargez depuis https://ffmpeg.org/download.html
        4. Assurez-vous que le fichier de modèle 'Template_reunion (1).docx' est dans le répertoire de travail.
        5. Vérifiez que le modèle contient les sections suivantes avec les en-têtes de tableau corrects :
           - Tableau de présence avec les colonnes "PRÉSENCES" et "ABSENCES".
           - Section "ORDRE DU JOUR :" suivie de paragraphes pour les éléments de l'agenda.
           - Tableau des résolutions avec les colonnes incluant "NBR DE REPORT".
           - Tableau des sanctions avec les colonnes incluant "NOM" et "MOTIF".
        """)
        
        st.title("Mode Secours")
        st.warning("Application en mode limité. La transcription audio n'est pas disponible.")
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
        start_time = st.text_input("Heure de début (format HHhMMmin, ex: 07h00min)", value="07h00min")
        end_time = st.text_input("Heure de fin (format HHhMMmin, ex: 10h34min)", value="10h34min")
        attendees = st.text_area("Participants (séparés par des virgules)")
        balance_amount = st.text_input("Solde du compte DRI Solidarité (en XAF, ex: 682040)", value="682040")
        balance_date = st.date_input("Date du solde", value=meeting_date)
        transcription = st.text_area("Transcription (saisie manuelle)", height=300)
        
        if st.button("Formater les Notes de Réunion"):
            extracted_info = extract_info_fallback(
                transcription,
                meeting_title,
                meeting_date.strftime("%d/%m/%Y"),
                attendees,
                start_time=start_time,
                end_time=end_time,
                balance_amount=balance_amount,
                balance_date=balance_date.strftime("%d/%m/%Y")
            )
            st.subheader("Informations Extraites")
            st.text_area("Aperçu:", json.dumps(extracted_info, indent=2, ensure_ascii=False), height=300)
            
            try:
                docx_data = fill_template_and_generate_docx(extracted_info)
                if docx_data:
                    st.download_button(
                        label="Télécharger les Notes de Réunion",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            except Exception as e:
                st.warning(f"Erreur lors de la génération du document: {e}")