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

# Try to import tiktoken for accurate token counting; fall back to simple method if unavailable
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

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

def count_tokens(text):
    """Count the number of tokens in a text string."""
    if TOKENIZER_AVAILABLE:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    else:
        # Fallback: Simple token count based on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return len(tokens)

def clean_json_response(response):
    """
    Clean an API response to extract valid JSON content.
    Returns the cleaned response as a string, or None if cleaning fails.
    """
    # Handle bytes input
    if isinstance(response, bytes):
        try:
            response = response.decode('utf-8')
        except UnicodeDecodeError as e:
            st.error(f"Failed to decode response as UTF-8: {e}")
            return None

    # Ensure the response is a string
    if not isinstance(response, str):
        st.error(f"Response is not a string or bytes: {type(response)}")
        return None

    # Log the raw response for debugging
    st.write(f"Raw API response before cleaning: {repr(response)}")

    # Remove leading/trailing whitespace
    response = response.strip()

    # Handle empty response
    if not response:
        st.error("Response is empty after stripping whitespace.")
        return None

    # Remove Markdown code fences (e.g., ```json ... ``` or ``` ... ```)
    response = response.removeprefix('```json').removesuffix('```').strip()
    response = response.removeprefix('```').removesuffix('```').strip()

    # Check if the response looks like JSON (starts with { or [)
    if not (response.startswith("{") or response.startswith("[")):
        # Try to extract JSON from a larger text (e.g., "Here is the JSON: {...}")
        json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        else:
            st.error(f"Response does not contain valid JSON: {response}")
            return None

    # Log the cleaned response
    st.write(f"Cleaned response: {repr(response)}")
    return response

def extract_info_fallback(transcription, meeting_title, date, previous_context=""):
    """Fallback function to extract information using improved string parsing and regex."""
    extracted_data = {
        "presence_list": "Présents: Non spécifié\nAbsents: Non spécifié",
        "agenda_items": "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers",
        "resolutions_summary": [],
        "sanctions_summary": [],
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "rapporteur": "Non spécifié",
        "president": "Non spécifié",
        "balance_amount": "Non spécifié",
        "balance_date": date,
        "date": date,
        "meeting_title": meeting_title
    }

    # Extract presence list (French keywords)
    present_match = re.search(r"(Présents|Présent|Présentes|Présente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    absent_match = re.search(r"(Absents|Absent|Absentes|Absente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    if present_match or absent_match:
        present = present_match.group(2).strip() if present_match else "Non spécifié"
        absent = absent_match.group(2).strip() if absent_match else "Non spécifié"
        extracted_data["presence_list"] = f"Présents: {present}\nAbsents: {absent}"
    else:
        # Fallback to infer presence from names mentioned in the transcript
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", transcription)
        # Filter out common words that might be mistaken as names
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
            # Add Roman numerals if not present
            numbered_items = []
            for idx, item in enumerate(items, 1):
                if not re.match(r"^[IVXLC]+\-", item):
                    item = f"{to_roman(idx)}- {item}"
                numbered_items.append(item)
            extracted_data["agenda_items"] = "\n".join(numbered_items)

    # Extract start time
    start_time_pattern = r"(?:début|commence|commencée)[\s\w]*?(\d{1,2}(?:h\d{2}min|h:\d{2}|\d{2}min))"
    start_time_match = re.search(start_time_pattern, transcription, re.IGNORECASE)
    if start_time_match:
        extracted_data["start_time"] = start_time_match.group(1).replace("h:", "h").replace("min", "min")

    # Extract duration and calculate end time if end time is not specified
    duration_pattern = r"(?:durée|dure|duré|lasted)[\s\w]*?(\d{1,2}h(?:\d{1,2}min)?(?:\d{1,2}s)?)"
    duration_match = re.search(duration_pattern, transcription, re.IGNORECASE)
    end_time_pattern = r"(?:fin|terminée|terminé|ended)[\s\w]*?(\d{1,2}(?:h\d{2}min|h:\d{2}|\d{2}min))"
    end_time_match = re.search(end_time_pattern, transcription, re.IGNORECASE)

    if end_time_match:
        extracted_data["end_time"] = end_time_match.group(1).replace("h:", "h").replace("min", "min")
    elif start_time_match and duration_match:
        # Parse start time
        start_time_str = start_time_match.group(1).replace("h", ":").replace("min", "")
        try:
            start_time = datetime.strptime(start_time_str, "%H:%M")
        except ValueError:
            start_time = datetime.strptime(start_time_str, "%H")

        # Parse duration
        duration_str = duration_match.group(1)
        hours = minutes = seconds = 0
        if "h" in duration_str:
            hours = int(re.search(r"(\d+)h", duration_str).group(1))
        if "min" in duration_str:
            minutes = int(re.search(r"(\d+)min", duration_str).group(1))
        if "s" in duration_str:
            seconds = int(re.search(r"(\d+)s", duration_str).group(1))

        # Calculate end time
        duration_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        end_time = start_time + duration_delta
        extracted_data["end_time"] = end_time.strftime("%Hh%Mmin")

    # Extract rapporteur and president
    rapporteur_match = re.search(r"(Rapporteur|Rapporteuse)[:\s]*([A-Z][a-z]+)", transcription, re.IGNORECASE)
    president_match = re.search(r"(Président|Présidente|Prési)[:\s]*([A-Z][a-z]+)", transcription, re.IGNORECASE)
    if rapporteur_match:
        extracted_data["rapporteur"] = rapporteur_match.group(2)
    if president_match:
        extracted_data["president"] = president_match.group(2)

    # Extract balance amount
    balance_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d+)", transcription, re.IGNORECASE)
    if balance_match:
        extracted_data["balance_amount"] = balance_match.group(2)

    # Extract balance date
    balance_date_match = re.search(r"(solde|compte|balance)[\s\w]*?(\d{2}/\d{2}/\d{4})", transcription, re.IGNORECASE)
    if balance_date_match:
        extracted_data["balance_date"] = balance_date_match.group(2)

    # Extract resolutions (improved)
    resolution_section = re.search(r"(Résolution|Resolutions)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if resolution_section:
        resolution_text = resolution_section.group(2).strip()
        # Split into individual resolutions if multiple are mentioned
        resolution_lines = [line.strip() for line in resolution_text.split("\n") if line.strip()]
        resolutions = []
        for res in resolution_lines:
            # Try to extract responsible person and deadline
            responsible_match = re.search(r"(?:par|responsable|attribué à) ([A-Z][a-z]+)", res, re.IGNORECASE)
            deadline_match = re.search(r"(?:d'ici|avant le) (\d{2}/\d{2}/\d{4})", res, re.IGNORECASE)
            responsible = responsible_match.group(1) if responsible_match else "Non spécifié"
            deadline = deadline_match.group(1) if deadline_match else "Non spécifié"
            resolutions.append({
                "date": date,
                "dossier": "Non spécifié",
                "resolution": res,
                "responsible": responsible,
                "deadline": deadline,
                "execution_date": "",
                "status": "En cours",
                "report_count": "0"
            })
        extracted_data["resolutions_summary"] = resolutions

    # Extract sanctions (improved)
    sanction_section = re.search(r"(Sanction|Amende)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if sanction_section:
        sanction_text = sanction_section.group(2).strip()
        sanction_lines = [line.strip() for line in sanction_text.split("\n") if line.strip()]
        sanctions = []
        for sanc in sanction_lines:
            name_match = re.search(r"^[A-Z][a-z]+|^([A-Z][a-z]+)[\s,]", sanc)
            amount_match = re.search(r"(\d+)\s*(?:FCFA|XAF)?", sanc)
            reason_match = re.search(r"(?:pour|raison) ([a-zA-Z\s]+)", sanc, re.IGNORECASE)
            name = name_match.group(1) if name_match else "Non spécifié"
            amount = amount_match.group(1) if amount_match else "0"
            reason = reason_match.group(1).strip() if reason_match else sanc
            sanctions.append({
                "name": name,
                "reason": reason,
                "amount": amount,
                "date": date,
                "status": "Appliquée"
            })
        extracted_data["sanctions_summary"] = sanctions
    else:
        # If no sanctions found in transcript, the API will handle extraction from context
        extracted_data["sanctions_summary"] = []

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

    prompt = f"""
    Vous êtes un assistant IA chargé d'extraire des informations clés d'un transcript de réunion en français pour une institution bancaire. Votre tâche est de produire un JSON structuré avec des clés en anglais. Vous devez TOUJOURS retourner un JSON valide, même en cas d'échec.

    **Contexte de la réunion précédente** :
    {previous_context if previous_context else "Aucun contexte disponible."}

    **Transcript de la réunion** :
    {transcription}

    **Instructions** :
    - Extraire les informations suivantes et les structurer dans un objet JSON avec les clés en anglais.
    - Si une information est manquante, utilisez les valeurs par défaut indiquées.
    - Assurez-vous que toutes les dates sont au format JJ/MM/AAAA (par exemple, "15/05/2025").
    - Les clés du JSON doivent être en anglais, mais les valeurs doivent refléter le texte original (en français).
    - Retournez uniquement un JSON valide. Si vous échouez, retournez {{"error": "Impossible de traiter le transcript"}}.
    - Ne produisez AUCUN texte en dehors du JSON, pas de commentaires ni d'explications.

    **Informations à extraire** :

    1. **presence_list** : Liste des présents et absents sous forme de chaîne (ex. "Présents: Alice, Bob\nAbsents: Charlie").
       - **Présents** : Identifiez les participants mentionnés comme présents (mots-clés : "Présents", "Présent") ou ayant pris la parole (ex. "Alice a dit…").
       - **Absents** : Recherchez les mentions explicites (mots-clés : "Absents", "Absent"). Sinon, indiquez "Absents: Non spécifié".
       - Si aucune info, indiquez : "Présents: Non spécifié\nAbsents: Non spécifié".

    2. **agenda_items** : Liste des points de l'ordre du jour sous forme de chaîne (ex. "I- Revue\nII- Résolutions").
       - Recherchez "Ordre du jour" ou "Agenda".
       - Sinon, déduisez à partir des sujets discutés.
       - Par défaut : "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers".

    3. **president** : Président de séance (mots-clés : "Président", "Présidente", "Prési").
       - Par défaut : "Non spécifié".

    4. **rapporteur** : Rapporteur (mots-clés : "Rapporteur", "Rapporteuse").
       - Par défaut : "Non spécifié".

    5. **start_time** et **end_time** : Heure de début et fin (format HHhMMmin, ex. "10h00min").
       - Par défaut : "Non spécifié".

    6. **balance_amount** : Solde du compte (mots-clés : "solde", "compte", "balance").
       - Par défaut : "Non spécifié".

    7. **balance_date** : Date du solde (format JJ/MM/AAAA).
       - Par défaut : {date}.

    8. **resolutions_summary** : Liste de dictionnaires pour les résolutions.
       - Clés : "date" (JJ/MM/AAAA), "dossier", "resolution", "responsible", "deadline" (JJ/MM/AAAA), "execution_date" (JJ/MM/AAAA), "status", "report_count".
       - Par défaut : "dossier": "Non spécifié", "responsible": "Non spécifié", "deadline": "Non spécifié", "execution_date": "", "status": "En cours", "report_count": "0".
       - Recherchez les mentions sous "Résolution" ou "Resolutions" dans le transcript.

    9. **sanctions_summary** : Liste de dictionnaires pour les sanctions.
       - Clés : "name", "reason", "amount", "date" (JJ/MM/AAAA), "status".
       - Par défaut : "name": "Aucune", "reason": "Aucune sanction mentionnée", "amount": "0", "status": "Non appliquée".
       - Recherchez les mentions sous "Sanction" ou "Amende" dans le transcript.
       - **Si aucune sanction n'est trouvée dans le transcript**, extrayez les sanctions du **contexte de la réunion précédente** (fourni ci-dessus) sous la section "RÉCAPITULATIF DES SANCTIONS".
         - Dans le contexte, les sanctions sont présentées dans un tableau avec les colonnes : NOM | RAISON | MONTANT (FCFA) | DATE | STATUT.
         - Mappez ces colonnes aux clés : "name" (NOM), "reason" (RAISON), "amount" (MONTANT, sans "FCFA"), "date" (utilisez la date de la réunion actuelle : {date}), "status" (STATUT).
         - Ignorez la ligne d'en-tête du tableau (NOM | RAISON | etc.) et extrayez uniquement les lignes de données.
         - Si aucune sanction n'est trouvée dans le contexte, utilisez les valeurs par défaut.

    **Exemple de sortie** :
    {{"presence_list": "Présents: Alice, Bob\nAbsents: Charlie", "agenda_items": "I- Revue\nII- Résolutions", "president": "Alice", "rapporteur": "Bob", "start_time": "10h00min", "end_time": "11h00min", "balance_amount": "827540", "balance_date": "15/05/2025", "resolutions_summary": [{{"date": "15/05/2025", "dossier": "Projet X", "resolution": "Finaliser le rapport", "responsible": "Alice", "deadline": "20/05/2025", "execution_date": "", "status": "En cours", "report_count": "0"}}], "sanctions_summary": [{{"name": "Charlie", "reason": "Retard", "amount": "5000", "date": "15/05/2025", "status": "Appliquée"}}]}}
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
            "max_tokens": 5000
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code != 200:
            st.error(f"Deepseek API error: Status {response.status_code}, Message: {response.text}. Falling back to basic extraction.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        # Log the full response for debugging
        full_response = response.json()
        st.write(f"Full Deepseek response: {json.dumps(full_response, indent=2)}")
        
        # Extract the content
        raw_response = full_response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # Clean the response
        cleaned_response = clean_json_response(raw_response)
        if cleaned_response is None:
            st.error("Failed to clean API response into valid JSON. Falling back to basic extraction.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        # Count tokens in the API response and the cleaned response
        api_tokens = full_response.get("usage", {}).get("completion_tokens", 0)
        displayed_tokens = count_tokens(cleaned_response)
        token_difference = api_tokens - displayed_tokens
        st.write(f"Token Count - API Response: {api_tokens}, Displayed Response: {displayed_tokens}, Difference: {token_difference}")
        if token_difference > 0:
            st.warning("The displayed response has fewer tokens than the API response, indicating possible truncation during cleaning.")

        # Attempt to parse the cleaned response as JSON
        extracted_data = json.loads(cleaned_response)
        
        # Check if the response contains an error key
        if "error" in extracted_data:
            st.error(f"Deepseek API error: {extracted_data['error']}. Falling back to basic extraction.")
            return extract_info_fallback(transcription, meeting_title, date, previous_context)

        # If end_time is "Non spécifié" but start_time and duration are available, calculate end_time
        if extracted_data.get("end_time") == "Non spécifié":
            start_time_str = extracted_data.get("start_time", "Non spécifié")
            duration_match = re.search(r"(?:durée|dure|duré|lasted)[\s\w]*?(\d{1,2}h(?:\d{1,2}min)?(?:\d{1,2}s)?)", transcription, re.IGNORECASE)
            if start_time_str != "Non spécifié" and duration_match:
                # Parse start time
                start_time_cleaned = start_time_str.replace("h", ":").replace("min", "")
                try:
                    start_time = datetime.strptime(start_time_cleaned, "%H:%M")
                except ValueError:
                    start_time = datetime.strptime(start_time_cleaned, "%H")

                # Parse duration
                duration_str = duration_match.group(1)
                hours = minutes = seconds = 0
                if "h" in duration_str:
                    hours = int(re.search(r"(\d+)h", duration_str).group(1))
                if "min" in duration_str:
                    minutes = int(re.search(r"(\d+)min", duration_str).group(1))
                if "s" in duration_str:
                    seconds = int(re.search(r"(\d+)s", duration_str).group(1))

                # Calculate end time
                duration_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                end_time = start_time + duration_delta
                extracted_data["end_time"] = end_time.strftime("%Hh%Mmin")
        
        # Add meeting metadata
        extracted_data["date"] = date
        extracted_data["meeting_title"] = meeting_title
        return extracted_data

    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON after cleaning: {e}. Cleaned response: {repr(cleaned_response)}. Falling back to basic extraction.")
        return extract_info_fallback(transcription, meeting_title, date, previous_context)
    except Exception as e:
        st.error(f"Error extracting information: {e}. Falling back to basic extraction.")
        return extract_info_fallback(transcription, meeting_title, date, previous_context)

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

def fill_template_and_generate_docx(extracted_info, meeting_title, meeting_date):
    """Build the Word document from scratch using python-docx and return the file data for download."""
    try:
        doc = Document()

        # Extract presence list and split into present and absent attendees
        presence_list = extracted_info.get("presence_list", "Présents: Non spécifié\nAbsents: Non spécifié")
        present_attendees = []
        absent_attendees = []
        if "Présents:" in presence_list and "Absents:" in presence_list:
            parts = presence_list.split("\n")
            for part in parts:
                if part.startswith("Présents:"):
                    presents = part.replace("Présents:", "").strip()
                    present_attendees = [name.strip() for name in presents.split(",") if name.strip()]
                elif part.startswith("Absents:"):
                    absents = part.replace("Absents:", "").strip()
                    absent_attendees = [name.strip() for name in absents.split(",") if name.strip()]
        else:
            present_attendees = [name.strip() for name in presence_list.split(",") if name.strip()] if presence_list != "Non spécifié" else []

        # Process agenda items
        agenda_list = extracted_info.get("agenda_items", "I- Relecture du compte rendu et adoption\nII- Récapitulatif des résolutions et sanctions\nIII- Revue d’activités\nIV- Faits saillants\nV- Divers").split("\n")
        agenda_list = [item.strip() for item in agenda_list if item.strip()]

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
            "COMPTE RENDU DE RÉUNION",
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
            f"Heure de début: {extracted_info.get('start_time', 'Non spécifié')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )
        add_styled_paragraph(
            doc,
            f"Heure de fin: {extracted_info.get('end_time', 'Non spécifié')}",
            font_name="Century",
            font_size=12,
            bold=True,
            alignment=WD_ALIGN_PARAGRAPH.CENTER
        )

        # Add rapporteur and president
        rapporteur = extracted_info.get("rapporteur", "Non spécifié")
        president = extracted_info.get("president", "Non spécifié")
        if rapporteur != "Non spécifié":
            add_styled_paragraph(
                doc,
                f"Rapporteur: {rapporteur}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )
        if president != "Non spécifié":
            add_styled_paragraph(
                doc,
                f"Président de la réunion: {president}",
                font_name="Century",
                font_size=12,
                bold=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER
            )

        # Add attendance table (no page break, back to original position)
        add_styled_paragraph(
            doc,
            "◆ LISTE DE PRÉSENCE",
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
            
            attendance_column_widths = [4.5, 4.5]
            add_styled_table(
                doc,
                rows=max_rows + 1,
                cols=2,
                headers=["PRÉSENTS", "ABSENTS"],
                data=attendance_data,
                header_bg_color=(0, 0, 0),
                header_text_color=(255, 255, 255),
                alt_row_bg_color=(192, 192, 192),
                column_widths=attendance_column_widths,
                table_width=9.0
            )
        else:
            add_styled_paragraph(
                doc,
                "Aucune présence spécifiée.",
                font_name="Century",
                font_size=12
            )

        # Add page break before "Ordre du jour"
        doc.add_page_break()

        # Add agenda items
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

        # Add page break after "Ordre du jour" to isolate it
        doc.add_page_break()

        # Add resolutions summary
        resolutions = extracted_info.get("resolutions_summary", [])
        if not resolutions:
            resolutions = [{
                "date": extracted_info.get("date", ""),
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
            "RÉCAPITULATIF DES RÉSOLUTIONS",  # Updated from "RÉSUMÉ DES RÉSOLUTIONS"
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        resolutions_headers = ["DATE", "DOSSIER", "RÉSOLUTION", "RESP.", "ÉCHÉANCE", "DATE D'EXÉCUTION", "STATUT", "COMPTE RENDU"]
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
        resolutions_column_widths = [1.5, 1.8, 2.5, 1.2, 1.8, 1.5, 1.2, 1.5]
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
            table_width=12.0
        )

        # Add sanctions summary
        sanctions = extracted_info.get("sanctions_summary", [])
        if not sanctions:
            sanctions = [{
                "name": "Aucune",
                "reason": "Aucune sanction mentionnée",
                "amount": "0",
                "date": extracted_info.get("date", ""),
                "status": "Non appliquée"
            }]
        add_styled_paragraph(
            doc,
            "RÉCAPITULATIF DES SANCTIONS",  # Updated from "RÉSUMÉ DES SANCTIONS"
            font_name="Century",
            font_size=12,
            bold=True,
            color=RGBColor(192, 0, 0)
        )
        sanctions_headers = ["NOM", "RAISON", "MONTANT (FCFA)", "DATE", "STATUT"]
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
        sanctions_column_widths = [2.0, 2.5, 2.0, 1.8, 2.2]
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
            table_width=10.5
        )

        # Add a blank paragraph for spacing
        doc.add_paragraph("")

        # Add balance information
        add_styled_paragraph(
            doc,
            f"Solde du compte de solidarité DRI (00001-00921711101-10) est de XAF {extracted_info.get('balance_amount', 'Non spécifié')} au {extracted_info.get('balance_date', '')}.",
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
    
    st.sidebar.header("Contexte Précédent")
    previous_report = st.sidebar.file_uploader("Télécharger le rapport précédent (optionnel)", type=["pdf", "png", "jpg", "jpeg"])
    if previous_report:
        st.session_state.previous_report = previous_report
        st.session_state.previous_context = ""  # Reset context until a question is asked
        st.sidebar.write("Rapport précédent téléchargé. Posez une question pour extraire le contexte.")
    else:
        st.session_state.previous_report = None
        st.session_state.previous_context = ""
    
    # Section to ask questions about the context
    st.sidebar.header("Tester le Contexte")
    question = st.sidebar.text_input("Posez une question sur le rapport précédent :")
    if st.sidebar.button("Poser la Question") and question:
        if not st.session_state.mistral_api_key:
            st.sidebar.error("Veuillez fournir une clé API Mistral pour extraire le contexte.")
        elif not st.session_state.previous_report:
            st.sidebar.error("Veuillez télécharger un rapport précédent pour extraire le contexte.")
        else:
            with st.spinner("Extraction du contexte..."):
                context = extract_context_from_report(
                    st.session_state.previous_report, 
                    st.session_state.mistral_api_key
                )
                if context:
                    st.session_state.previous_context = context
                    st.sidebar.text_area("Contexte Extrait", context, height=200)
                    st.sidebar.success("Contexte extrait avec succès !")
                else:
                    st.session_state.previous_context = ""
                    st.sidebar.error("Échec de l'extraction du contexte. Vérifiez la clé API ou le fichier.")
            
            # Now answer the question
            with st.spinner("Obtention de la réponse..."):
                answer = answer_question_with_context(
                    question, 
                    st.session_state.previous_context, 
                    st.session_state.deepseek_api_key
                )
            st.sidebar.write("**Réponse :**")
            st.sidebar.write(answer)
    
    # Main app content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Détails de la Réunion")
        meeting_title = st.text_input("Titre de la Réunion", value="Réunion")
        meeting_date = st.date_input("Date de la Réunion", datetime.now())
    
    with col2:
        st.header("Transcription & Résultat")
        input_method = st.radio("Choisissez la méthode d'entrée :", ("Télécharger Audio", "Entrer la Transcription"))
        
        if input_method == "Télécharger Audio":
            uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["mp3", "wav", "m4a", "flac"])
            whisper_model = st.selectbox("Modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
            
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if st.button("Transcrire l'Audio"):
                    with st.spinner(f"Transcription avec Whisper {whisper_model}..."):
                        transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                        if transcription and not transcription.startswith("Error"):
                            st.session_state.transcription = transcription
                            # Automatically extract information after transcription
                            with st.spinner("Extraction des informations..."):
                                extracted_info = extract_info(
                                    st.session_state.transcription,
                                    meeting_title,
                                    meeting_date.strftime("%d/%m/%Y"),
                                    st.session_state.deepseek_api_key,
                                    st.session_state.get("previous_context", "")
                                )
                                if extracted_info:
                                    st.session_state.extracted_info = extracted_info
                                    st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        else:
            transcription_input = st.text_area("Entrez la transcription de la réunion :", height=200)
            if st.button("Soumettre la Transcription") and transcription_input:
                st.session_state.transcription = transcription_input
                # Automatically extract information after submission
                with st.spinner("Extraction des informations..."):
                    extracted_info = extract_info(
                        st.session_state.transcription,
                        meeting_title,
                        meeting_date.strftime("%d/%m/%Y"),
                        st.session_state.deepseek_api_key,
                        st.session_state.get("previous_context", "")
                    )
                    if extracted_info:
                        st.session_state.extracted_info = extracted_info
                        st.text_area("Informations Extraites", json.dumps(extracted_info, indent=2), height=300)
        
        if 'extracted_info' in st.session_state:
            if st.button("Générer et Télécharger le Document"):
                with st.spinner("Génération du document..."):
                    docx_data = fill_template_and_generate_docx(
                        st.session_state.extracted_info,
                        meeting_title,
                        meeting_date
                    )
                    if docx_data:
                        st.download_button(
                            label="Télécharger le Document",
                            data=docx_data,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download-button"
                        )

if __name__ == "__main__":
    main()