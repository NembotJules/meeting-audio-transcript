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
    # Define expected team members (real names from historical data)
    expected_members = [
        "Grace Divine", "Vladimir SOUA", "Gael KIAMPI", "Emmanuel TEINGA",
        "Francis KAMSU", "Jordan KAMSU-KOM", "Loïc KAMENI", "Christian DJIMELI",
        "Daniel BAYECK", "Brice DZANGUE", "Sherelle KANA", "Jules NEMBOT",
        "Nour MAHAMAT", "Franklin TANDJA", "Marcellin SEUJIP", "Divine NDE",
        "Brian ELLA ELLA", "Amelin EPOH", "Franklin YOUMBI", "Cédric DONFACK",
        "Wilfried DOPGANG", "Ismaël POUNGOUM", "Éric BEIDI", "Boris ON MAKONG",
        "Charlène GHOMSI"
    ]
    
    # Use the new structure matching historical processor with proper French defaults
    extracted_data = {
        "meeting_metadata": {
            "date": date,
            "title": meeting_title
        },
        "attendance": {
            "present": [],
            "absent": []
        },
        "agenda_items": [
            "I- Relecture du Compte Rendu",
            "II- Récapitulatif des Résolutions et des Sanctions", 
            "III- Revue d'activités",
            "IV- Faits Saillants",
            "V- Divers"
        ],
        "activities_review": [],
        "resolutions_summary": [],
        "sanctions_summary": [],
        "key_highlights": [],
        "miscellaneous": [],
        # Additional fields for document generation
        "start_time": "Non spécifié",
        "end_time": "Non spécifié",
        "rapporteur": "Non spécifié",
        "president": "Non spécifié",
        "balance_amount": "Non spécifié",
        "balance_date": date
    }

    # Extract presence list
    present_match = re.search(r"(Présents|Présent|Présentes|Présente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    absent_match = re.search(r"(Absents|Absent|Absentes|Absente)[:\s]*([^\n]+)", transcription, re.IGNORECASE)
    if present_match or absent_match:
        present = present_match.group(2).strip() if present_match else ""
        absent = absent_match.group(2).strip() if absent_match else ""
        extracted_data["attendance"]["present"] = [name.strip() for name in present.split(",") if name.strip()] if present else []
        extracted_data["attendance"]["absent"] = [name.strip() for name in absent.split(",") if name.strip()] if absent else []
    else:
        names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", transcription)
        common_words = {"Réunion", "Projet", "Président", "Rapporteur", "Solde", "Compte", "Ordre", "Agenda"}
        names = [name for name in set(names) if name not in common_words]
        if names:
            extracted_data["attendance"]["present"] = names
            extracted_data["attendance"]["absent"] = []

    # Extract agenda items - only if explicitly mentioned, otherwise keep defaults
    agenda_match = re.search(r"(Ordre du jour|Agenda)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    if agenda_match:
        agenda_items = agenda_match.group(2).strip()
        # Only replace default if we have clear agenda items
        items = [item.strip() for item in agenda_items.split("\n") if item.strip() and len(item.strip()) > 5]
        if items and len(items) >= 3:  # Only if we have substantial agenda items
            numbered_items = []
            for idx, item in enumerate(items, 1):
                if not re.match(r"^[IVXLC]+\-", item):
                    item = f"{to_roman(idx)}- {item}"
                numbered_items.append(item)
            extracted_data["agenda_items"] = numbered_items

    # Extract start time with improved patterns
    start_time_patterns = [
        r"(?:début|commence|commencée|démarré|start)[\s\w]*?(\d{1,2}[h:]\d{2})",
        r"(?:début|commence|commencée|démarré|start)[\s\w]*?(\d{1,2}h\d{2}min)",
        r"(?:début|commence|commencée|démarré|start)[\s\w]*?(\d{1,2}h)",
        r"(?:à|vers|around)\s*(\d{1,2}[h:]\d{2})",
        r"(?:à|vers|around)\s*(\d{1,2}h)",
        r"(\d{1,2}[h:]\d{2})[\s\w]*(?:début|commence|start)",
        r"(\d{1,2}h\d{2}min)[\s\w]*(?:début|commence|start)",
        r"(\d{1,2}h)[\s\w]*(?:début|commence|start)"
    ]
    
    for pattern in start_time_patterns:
        start_time_match = re.search(pattern, transcription, re.IGNORECASE)
        if start_time_match:
            time_str = start_time_match.group(1)
            # Normalize format
            time_str = time_str.replace(":", "h").replace("min", "min")
            if not time_str.endswith("h") and not time_str.endswith("min") and "h" in time_str:
                if not time_str.endswith("min"):
                    time_str += "min"
            extracted_data["start_time"] = time_str
            break

    # Extract end time with improved patterns
    end_time_patterns = [
        r"(?:fin|terminée|terminé|ended|fini)[\s\w]*?(\d{1,2}[h:]\d{2})",
        r"(?:fin|terminée|terminé|ended|fini)[\s\w]*?(\d{1,2}h\d{2}min)",
        r"(?:fin|terminée|terminé|ended|fini)[\s\w]*?(\d{1,2}h)",
        r"(?:jusqu'à|until|vers|around)[\s\w]*?(\d{1,2}[h:]\d{2})",
        r"(?:jusqu'à|until|vers|around)[\s\w]*?(\d{1,2}h)",
        r"(\d{1,2}[h:]\d{2})[\s\w]*(?:fin|terminée|end)",
        r"(\d{1,2}h\d{2}min)[\s\w]*(?:fin|terminée|end)",
        r"(\d{1,2}h)[\s\w]*(?:fin|terminée|end)"
    ]
    
    for pattern in end_time_patterns:
        end_time_match = re.search(pattern, transcription, re.IGNORECASE)
        if end_time_match:
            time_str = end_time_match.group(1)
            # Normalize format
            time_str = time_str.replace(":", "h").replace("min", "min")
            if not time_str.endswith("h") and not time_str.endswith("min") and "h" in time_str:
                if not time_str.endswith("min"):
                    time_str += "min"
            extracted_data["end_time"] = time_str
            break

    # Extract duration and calculate end time if we have start time but no end time
    if extracted_data["start_time"] != "Non spécifié" and extracted_data["end_time"] == "Non spécifié":
        duration_patterns = [
            r"(?:durée|dure|duré|lasted|pendant)[\s\w]*?(\d{1,2}h(?:\d{1,2}min)?)",
            r"(?:durée|dure|duré|lasted|pendant)[\s\w]*?(\d{1,2}h)",
            r"(?:durée|dure|duré|lasted|pendant)[\s\w]*?(\d{1,2}min)",
            r"(?:durée|dure|duré|lasted|pendant)[\s\w]*?(\d{1,2}\s*heures?)"
        ]
        
        for pattern in duration_patterns:
            duration_match = re.search(pattern, transcription, re.IGNORECASE)
            if duration_match:
                try:
                    start_time_str = extracted_data["start_time"].replace("h", ":").replace("min", "")
                    if ":" not in start_time_str:
                        start_time_str += ":00"
                    start_time = datetime.strptime(start_time_str, "%H:%M")
                    
                    duration_str = duration_match.group(1)
                    hours = 0
                    minutes = 0
                    
                    if "h" in duration_str:
                        hours_match = re.search(r"(\d+)h", duration_str)
                        if hours_match:
                            hours = int(hours_match.group(1))
                    
                    if "min" in duration_str:
                        minutes_match = re.search(r"(\d+)min", duration_str)
                        if minutes_match:
                            minutes = int(minutes_match.group(1))
                    elif "heures" in duration_str or "heure" in duration_str:
                        hours_match = re.search(r"(\d+)", duration_str)
                        if hours_match:
                            hours = int(hours_match.group(1))
                    
                    duration_delta = timedelta(hours=hours, minutes=minutes)
                    end_time = start_time + duration_delta
                    extracted_data["end_time"] = end_time.strftime("%Hh%Mmin")
                    break
                except (ValueError, AttributeError) as e:
                    continue

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

    # Extract activities review - ensure ALL expected members have entries, but preserve multiple activities per person
    activities_section = re.search(r"(Revue des activités|Activités de la semaine)[:\s]*([\s\S]*?)(?=\n[A-Z]+:|\Z)", transcription, re.IGNORECASE)
    all_activities = []
    
    if activities_section:
        activities_text = activities_section.group(2).strip()
        activities_lines = [line.strip() for line in activities_text.split("\n") if line.strip()]
        
        for line in activities_lines:
            actor_match = re.search(r"^[A-Z][a-z]+|^([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\s,]", line)
            dossier_match = re.search(r"(?:dossier|sur le dossier) ([A-Za-z0-9\s]+)", line, re.IGNORECASE)
            activities_match = re.search(r"(?:activités|menées) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            results_match = re.search(r"(?:résultat|obtenu) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            perspectives_match = re.search(r"(?:perspectives|prévoit de) ([A-Za-z\s,]+)", line, re.IGNORECASE)
            
            if actor_match:
                actor = actor_match.group(1) if actor_match.group(1) else actor_match.group(0)
                # Add each activity as a separate entry (don't overwrite previous activities for same actor)
                all_activities.append({
                    "actor": actor,
                    "dossier": dossier_match.group(1).strip() if dossier_match else "Non spécifié",
                    "activities": activities_match.group(1).strip() if activities_match else "Non spécifié",
                    "results": results_match.group(1).strip() if results_match else "Non spécifié",
                    "perspectives": perspectives_match.group(1).strip() if perspectives_match else "Non spécifié"
                })
    
    # Ensure ALL expected members have at least one activity entry
    existing_actors = {activity.get("actor", "") for activity in all_activities}
    for member in expected_members:
        if member not in existing_actors:
            all_activities.append({
                "actor": member,
                "dossier": "Non spécifié",
                "activities": "RAS",
                "results": "RAS", 
                "perspectives": "RAS"
            })

    extracted_data["activities_review"] = all_activities

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
                "status": "En cours"
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
        st.info(f"📋 Using stored context sanctions: {len(sanctions)} sanctions")
    
    # Default if no sanctions found anywhere
    if not extracted_data["sanctions_summary"]:
        extracted_data["sanctions_summary"] = [{
            "name": "Aucune",
            "reason": "Aucune sanction mentionnée",
            "amount": "0",
            "date": date,
            "status": "Non appliquée"
        }]

    # Apply missing data from history (excluding current meeting)
    extracted_data = smart_historical_data_filling(extracted_data, date, allow_circular=False)

    return extracted_data

def to_roman(num):
    """Convert an integer to a Roman numeral."""
    roman_numerals = {
        1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
        6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"
    }
    return roman_numerals.get(num, str(num))

def load_historical_meetings(context_dir="processed_meetings", max_meetings=3, exclude_date=None):
    """Load the most recent historical meetings for context and missing data, excluding current meeting date."""
    try:
        if not os.path.exists(context_dir):
            return []
        
        # Get all JSON files in the context directory
        json_files = []
        for file in os.listdir(context_dir):
            if file.endswith('.json'):
                filepath = os.path.join(context_dir, file)
                
                # Skip files that match the current meeting date to avoid circular context
                if exclude_date:
                    # Check if filename contains the exclude_date
                    exclude_date_formatted = exclude_date.replace("/", "-")
                    if exclude_date_formatted in file:
                        st.info(f"🚫 Excluding current meeting {file} from historical context to avoid circular reference")
                        continue
                
                # Get file modification time for sorting
                mtime = os.path.getmtime(filepath)
                json_files.append((mtime, filepath, file))
        
        if not json_files:
            return []
        
        # Sort by modification time (most recent first) and take the last max_meetings
        json_files.sort(key=lambda x: x[0], reverse=True)
        recent_files = json_files[:max_meetings]
        
        meetings = []
        for mtime, filepath, filename in recent_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    meeting_data = json.load(f)
                meetings.append(meeting_data)
            except Exception as e:
                st.warning(f"Error loading {filename}: {str(e)}")
                continue
        
        return meetings
        
    except Exception as e:
        st.warning(f"Error loading historical meetings: {str(e)}")
        return []

def get_missing_data_from_history(extracted_data, historical_meetings):
    """Fill in missing data from historical meetings."""
    if not historical_meetings:
        return extracted_data
    
    # Use the most recent meeting as the primary source for missing data
    latest_meeting = historical_meetings[0]
    
    # More aggressive sanctions handling - check if sanctions are missing or just default
    current_sanctions = extracted_data.get("sanctions_summary", [])
    has_real_sanctions = any(s.get("name", "") not in ["Aucune", "", "Non spécifié"] for s in current_sanctions)
    
    if not has_real_sanctions:  # No real sanctions found in current meeting
        latest_sanctions = latest_meeting.get("sanctions_summary", [])
        historical_real_sanctions = [s for s in latest_sanctions if s.get("name", "") not in ["Aucune", "", "Non spécifié"]]
        
        if historical_real_sanctions:
            # Update dates to current meeting and copy sanctions
            updated_sanctions = []
            for sanction in historical_real_sanctions:
                updated_sanction = sanction.copy()
                updated_sanction["date"] = extracted_data["meeting_metadata"]["date"]
                updated_sanctions.append(updated_sanction)
            
            extracted_data["sanctions_summary"] = updated_sanctions
            st.info(f"📋 Pulled {len(updated_sanctions)} sanctions from previous meeting ({latest_meeting['meeting_metadata']['date']})")
            
            # Log what sanctions were pulled
            sanction_names = [s.get("name", "") for s in updated_sanctions]
            st.write(f"Sanctions carried forward: {', '.join(sanction_names)}")
    
    # If no activities, try to get perspectives from previous meeting as starting point
    if not extracted_data.get("activities_review") or len(extracted_data["activities_review"]) == 0:
        latest_activities = latest_meeting.get("activities_review", [])
        if latest_activities:
            # Convert perspectives to new activities
            new_activities = []
            for activity in latest_activities:
                perspectives = activity.get("perspectives", "")
                if perspectives and perspectives not in ["RAS", "Aucune", "", "À définir", "Non spécifié"]:
                    new_activities.append({
                        "actor": activity.get("actor", ""),
                        "dossier": activity.get("dossier", ""),
                        "activities": f"Continuation: {perspectives}",
                        "results": "À déterminer",
                        "perspectives": "À définir"
                    })
            if new_activities:
                extracted_data["activities_review"] = new_activities
                st.info(f"📋 Generated {len(new_activities)} activity continuations from previous meeting")
    
    return extracted_data

def extract_info(transcription, meeting_title, date, deepseek_api_key, previous_context="", test_mode=False):
    """Extract key information from the transcription using Deepseek API with historical context."""
    if not transcription or not deepseek_api_key:
        return extract_info_fallback(transcription, meeting_title, date, previous_context)

    # Load historical meetings for context (allow circular reference in test mode)
    exclude_date = None if test_mode else date
    historical_meetings = load_historical_meetings(exclude_date=exclude_date)
    
    if test_mode and historical_meetings:
        st.info(f"🧪 Test mode: Loaded {len(historical_meetings)} meetings including current for perfect data completion")
    
    # Create comprehensive historical context with full JSON content
    historical_context = ""
    if historical_meetings:
        historical_context = "\n\n=== COMPLETE HISTORICAL MEETING DATA ===\n"
        historical_context += "Use this data to understand continuity, ongoing activities, pending resolutions, and current sanctions.\n"
        historical_context += "Pay special attention to sanctions_summary for ongoing sanctions that should continue.\n\n"
        
        for i, meeting in enumerate(historical_meetings[:2]):  # Use last 2 meetings for full context
            historical_context += f"=== MEETING {i+1}: {meeting['meeting_metadata']['date']} ===\n"
            
            # Include attendance for continuity
            if meeting.get("attendance"):
                historical_context += f"Attendance:\n"
                historical_context += f"- Present: {', '.join(meeting['attendance'].get('present', []))}\n"
                historical_context += f"- Absent: {', '.join(meeting['attendance'].get('absent', []))}\n\n"
            
            # Include activities with full detail
            if meeting.get("activities_review"):
                historical_context += f"Activities Review:\n"
                for activity in meeting["activities_review"]:
                    historical_context += f"- Actor: {activity.get('actor', '')}\n"
                    historical_context += f"  Dossier: {activity.get('dossier', '')}\n"
                    historical_context += f"  Activities: {activity.get('activities', '')}\n"
                    historical_context += f"  Results: {activity.get('results', '')}\n"
                    historical_context += f"  Perspectives: {activity.get('perspectives', '')}\n\n"
            
            # Include resolutions with full detail
            if meeting.get("resolutions_summary"):
                historical_context += f"Resolutions:\n"
                for resolution in meeting["resolutions_summary"]:
                    historical_context += f"- Date: {resolution.get('date', '')}\n"
                    historical_context += f"  Dossier: {resolution.get('dossier', '')}\n"
                    historical_context += f"  Resolution: {resolution.get('resolution', '')}\n"
                    historical_context += f"  Responsible: {resolution.get('responsible', '')}\n"
                    historical_context += f"  Deadline: {resolution.get('deadline', '')}\n"
                    historical_context += f"  Status: {resolution.get('status', '')}\n\n"
            
            # CRITICALLY IMPORTANT: Include sanctions with full detail
            if meeting.get("sanctions_summary"):
                historical_context += f"SANCTIONS (IMPORTANT - these may continue to next meeting):\n"
                for sanction in meeting["sanctions_summary"]:
                    if sanction.get("name", "") != "Aucune":  # Only include real sanctions
                        historical_context += f"- Name: {sanction.get('name', '')}\n"
                        historical_context += f"  Reason: {sanction.get('reason', '')}\n"
                        historical_context += f"  Amount: {sanction.get('amount', '')} FCFA\n"
                        historical_context += f"  Date: {sanction.get('date', '')}\n"
                        historical_context += f"  Status: {sanction.get('status', '')}\n\n"
            
            # Include key highlights and miscellaneous
            if meeting.get("key_highlights"):
                historical_context += f"Key Highlights: {', '.join(meeting['key_highlights'])}\n"
            if meeting.get("miscellaneous"):
                historical_context += f"Miscellaneous: {', '.join(meeting['miscellaneous'])}\n"
            
            historical_context += "\n" + "="*60 + "\n\n"

    # Create a structured prompt matching historical processor format with proper French defaults
    prompt = f"""
    You are extracting information from a meeting transcript. Use the historical context below to understand continuity and ongoing items.

    {historical_context}

    Context from uploaded document:
    {previous_context}

    Current meeting transcript:
    {transcription}

    Meeting Date: {date}
    Meeting Title: {meeting_title}

    IMPORTANT EXTRACTION RULES:
    1. AGENDA ITEMS: Unless the transcript explicitly mentions different agenda items, use these French defaults:
       - "I- Relecture du Compte Rendu"
       - "II- Récapitulatif des Résolutions et des Sanctions"
       - "III- Revue d'activités"
       - "IV- Faits Saillants"
       - "V- Divers"
    
    2. ACTIVITIES REVIEW: Create entries for ALL team members, not just those mentioned:
       - Grace Divine, Vladimir SOUA, Gael KIAMPI, Emmanuel TEINGA
       - Francis KAMSU, Jordan KAMSU-KOM, Loïc KAMENI, Christian DJIMELI
       - Daniel BAYECK, Brice DZANGUE, Sherelle KANA, Jules NEMBOT
       - Nour MAHAMAT, Franklin TANDJA, Marcellin SEUJIP, Divine NDE
       - Brian ELLA ELLA, Amelin EPOH, Franklin YOUMBI, Cédric DONFACK
       - Wilfried DOPGANG, Ismaël POUNGOUM, Éric BEIDI, Boris ON MAKONG, Charlène GHOMSI
       - If not mentioned, use "RAS" for activities, results, and perspectives
    
    3. If the current transcript mentions ongoing sanctions or doesn't specify new sanctions, use the sanctions from the historical context with updated dates.
    4. For activities, check if they continue from previous meeting perspectives.
    5. For resolutions, check if they reference previous resolutions.
    6. Maintain continuity with historical data where appropriate.

    Extract the following information and return as JSON in this EXACT structure:
    {{
        "meeting_metadata": {{
            "date": "{date}",
            "title": "{meeting_title}"
        }},
        "attendance": {{
            "present": ["list of present attendees"],
            "absent": ["list of absent attendees"]
        }},
        "agenda_items": ["I- Relecture du Compte Rendu", "II- Récapitulatif des Résolutions et des Sanctions", "III- Revue d'activités", "IV- Faits Saillants", "V- Divers"],
        "activities_review": [
            {{
                "actor": "person name",
                "dossier": "project/file name",
                "activities": "activities performed",
                "results": "results obtained",
                "perspectives": "future plans"
            }}
        ],
        "resolutions_summary": [
            {{
                "date": "{date}",
                "dossier": "project name",
                "resolution": "resolution text",
                "responsible": "person responsible",
                "deadline": "deadline date",
                "execution_date": "",
                "status": "En cours"
            }}
        ],
        "sanctions_summary": [
            {{
                "name": "person name",
                "reason": "reason for sanction",
                "amount": "amount in FCFA",
                "date": "{date}",
                "status": "status"
            }}
        ],
        "key_highlights": ["highlight 1", "highlight 2"],
        "miscellaneous": ["misc item 1", "misc item 2"]
    }}

    Additional fields for document generation:
    - start_time: meeting start time
    - end_time: meeting end time
    - rapporteur: meeting rapporteur
    - president: meeting president
    - balance_amount: account balance
    - balance_date: balance date
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
            "max_tokens": 8000  # Increased to handle larger context
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

        # Fill in missing data from historical meetings (as a backup) - using smart filling
        extracted_data = smart_historical_data_filling(extracted_data, date, allow_circular=False)

        # Add meeting metadata if not present
        if "meeting_metadata" not in extracted_data:
            extracted_data["meeting_metadata"] = {"date": date, "title": meeting_title}

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
    """Build the Word document from scratch using the new JSON structure."""
    try:
        # Ensure complete member data and proper structure
        extracted_info = ensure_complete_member_data(extracted_info)
        
        doc = Document()

        # Handle new structure - attendance is now a dict with present/absent arrays
        attendance = extracted_info.get("attendance", {"present": [], "absent": []})
        present_attendees = attendance.get("present", [])
        absent_attendees = attendance.get("absent", [])
        
        # Get president info
        president = extracted_info.get("president", "Non spécifié")
        if president != "Non spécifié" and present_attendees:
            for i, attendee in enumerate(present_attendees):
                if attendee.lower() == president.lower():
                    present_attendees[i] = f"{attendee} (Président)"
                    break

        # Handle agenda items - should be a list now (guaranteed to be French)
        agenda_items = extracted_info.get("agenda_items", [])
        if isinstance(agenda_items, str):
            # Fallback for old format
            agenda_list = [item.strip() for item in agenda_items.split("\n") if item.strip()]
        else:
            agenda_list = agenda_items

        # Add header box
        add_text_in_box(doc, "Direction Recherches et Investissements", bg_color=(192, 192, 192), font_size=16)
        add_styled_paragraph(doc, "COMPTE RENDU DE RÉUNION", bold=True, color=RGBColor(192, 0, 0), alignment=WD_ALIGN_PARAGRAPH.CENTER)
        
        # Use meeting_metadata if available
        meeting_metadata = extracted_info.get("meeting_metadata", {})
        display_date = meeting_metadata.get("date", extracted_info.get("date", ""))
        add_styled_paragraph(doc, display_date, bold=True, color=RGBColor(192, 0, 0), alignment=WD_ALIGN_PARAGRAPH.CENTER)
        
        add_styled_paragraph(doc, f"Heure de début: {extracted_info.get('start_time', 'Non spécifié')}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        add_styled_paragraph(doc, f"Heure de fin: {extracted_info.get('end_time', 'Non spécifié')}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        rapporteur = extracted_info.get("rapporteur", "Non spécifié")
        if rapporteur != "Non spécifié":
            add_styled_paragraph(doc, f"Rapporteur: {rapporteur}", bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER)

        # Add attendance table with proper styling
        add_styled_paragraph(doc, "◆ LISTE DE PRÉSENCE", bold=True)
        if present_attendees or absent_attendees:
            max_rows = max(len(present_attendees), len(absent_attendees)) or 1
            
            # Create table for attendance
            attendance_table = doc.add_table(rows=max_rows + 1, cols=2)
            try:
                attendance_table.style = "Table Grid"
            except KeyError:
                pass
            
            set_table_width(attendance_table, 9.0)
            set_column_widths(attendance_table, [4.5, 4.5])
            
            # Add headers with red background
            headers = [f"Présents ({len(present_attendees)})", f"Absents ({len(absent_attendees)})"]
            for j, header in enumerate(headers):
                cell = attendance_table.cell(0, j)
                cell.text = header
                run = cell.paragraphs[0].runs[0]
                run.font.name = "Century"
                run.font.size = Pt(12)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, (200, 0, 0))  # Red background
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add data rows with alternating colors
            attendance_data = [[present_attendees[i] if i < len(present_attendees) else "", 
                              absent_attendees[i] if i < len(absent_attendees) else ""] for i in range(max_rows)]
            
            for i, row_data in enumerate(attendance_data):
                row = attendance_table.rows[i + 1]
                
                # Add alternating row colors
                if i % 2 == 0:
                    row_color = (240, 240, 240)  # Light gray
                else:
                    row_color = (255, 255, 255)  # White
                
                for j, cell_text in enumerate(row_data):
                    cell = row.cells[j]
                    cell.text = cell_text
                    run = cell.paragraphs[0].runs[0]
                    run.font.name = "Century"
                    run.font.size = Pt(11)
                    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                    set_cell_background(cell, row_color)
        else:
            add_styled_paragraph(doc, "Aucune présence spécifiée.")

        doc.add_page_break()

        # Add agenda with improved styling
        add_styled_paragraph(doc, "ORDRE DU JOUR :", bold=True, font_size=14, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        doc.add_paragraph()  # Add spacing
        
        if agenda_list:
            for item in agenda_list:
                # Clean formatting for agenda items
                p = doc.add_paragraph()
                p.style = 'List Number'
                run = p.add_run(item)
                run.font.name = "Century"
                run.font.size = Pt(12)
        else:
            # Fallback to French defaults
            french_agenda = [
                "Relecture du compte rendu et adoption",
                "Récapitulatif des résolutions et sanctions",
                "Revue d'activités",
                "Faits saillants",
                "Divers"
            ]
            for i, item in enumerate(french_agenda, 1):
                p = doc.add_paragraph()
                p.style = 'List Number'
                
                # Add Roman numeral
                roman_num = ["I", "II", "III", "IV", "V"][i-1]
                run = p.add_run(f"{roman_num}- {item}")
                run.font.name = "Century"
                run.font.size = Pt(12)

        doc.add_page_break()

        # Add section I header
        add_styled_paragraph(doc, "I. Relecture du Compte-rendu", bold=True, font_size=14, color=RGBColor(0, 0, 0))
        add_styled_paragraph(doc, "Le compte rendu précédent n'a pas été adopté et validé.", font_size=12)
        doc.add_paragraph()  # Add spacing
        
        # Add section II header  
        add_styled_paragraph(doc, "II. Récapitulatif des résolutions et sanctions", bold=True, font_size=14, color=RGBColor(0, 0, 0))
        add_styled_paragraph(doc, "Les tableaux des résolutions et des sanctions ont été examinés et mis à jour.", font_size=12)
        
        # Add resolutions (guaranteed to have at least one entry)
        resolutions = extracted_info.get("resolutions_summary", [])
        st.info(f"📊 Generating resolutions table with {len(resolutions)} entries")
        
        add_styled_paragraph(doc, "RÉCAPITULATIF DES RÉSOLUTIONS", bold=True, color=RGBColor(192, 0, 0))
        resolutions_data = [[r.get("date", ""), r.get("dossier", ""), r.get("resolution", ""), r.get("responsible", ""), r.get("deadline", ""), r.get("execution_date", ""), r.get("status", ""), str(r.get("report_count", "0"))] for r in resolutions]
        add_styled_table(doc, len(resolutions) + 1, 8, ["DATE", "DOSSIER", "RÉSOLUTION", "RESP.", "ÉCHÉANCE", "DATE D'EXÉCUTION", "STATUT", "COMPTE RENDU"], resolutions_data, column_widths=[1.5, 1.8, 2.5, 1.2, 1.8, 1.5, 1.2, 1.5], table_width=12.0)

        # Add sanctions (guaranteed to have at least one entry)
        sanctions = extracted_info.get("sanctions_summary", [])
        st.info(f"📊 Generating sanctions table with {len(sanctions)} entries")
        
        add_styled_paragraph(doc, "RÉCAPITULATIF DES SANCTIONS", bold=True, color=RGBColor(192, 0, 0))
        sanctions_data = [[s.get("name", ""), s.get("reason", ""), str(s.get("amount", "")), s.get("date", ""), s.get("status", "")] for s in sanctions]
        add_styled_table(doc, len(sanctions) + 1, 5, ["NOM", "RAISON", "MONTANT (FCFA)", "DATE", "STATUT"], sanctions_data, column_widths=[2.0, 2.5, 2.0, 1.8, 2.2], table_width=10.5)

        doc.add_page_break()
        
        # Add section III header
        add_styled_paragraph(doc, "III. Revue d'activités", bold=True, font_size=14, color=RGBColor(0, 0, 0))
        doc.add_paragraph()  # Add spacing

        # Add activities review (organized by department with headers)
        raw_activities = extracted_info.get("activities_review", [])
        organized_activities = organize_activities_by_department(raw_activities)
        
        st.info(f"📊 Generating activity table organized by departments: {len([a for a in organized_activities if a.get('type') == 'activity'])} activities across {len([a for a in organized_activities if a.get('type') == 'header'])} departments")
        
        add_activities_table_with_departments(doc, organized_activities)

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

def ensure_complete_member_data(extracted_info):
    """Ensure proper data structure and formatting, but only include members who actually have activities."""
    
    # Only work with members who actually have activities - don't force all 25 members
    activities = extracted_info.get("activities_review", [])
    
    # Filter out empty or invalid activities
    valid_activities = []
    for activity in activities:
        actor = activity.get("actor", "").strip()
        if actor and actor not in ["Non spécifié", "", "RAS"]:
            # Only include if they have meaningful activities
            activities_text = activity.get("activities", "").strip()
            if activities_text and activities_text not in ["Non spécifié", "", "RAS"]:
                valid_activities.append(activity)
    
    extracted_info["activities_review"] = valid_activities
    
    # Ensure resolutions_summary has at least one entry
    if not extracted_info.get("resolutions_summary"):
        extracted_info["resolutions_summary"] = [{
            "date": extracted_info["meeting_metadata"]["date"],
            "dossier": "Non spécifié",
            "resolution": "Aucune résolution spécifique mentionnée",
            "responsible": "Non spécifié",
            "deadline": "Non spécifié",
            "execution_date": "",
            "status": "En cours"
        }]
    
    # Ensure sanctions_summary has at least one entry
    if not extracted_info.get("sanctions_summary") or all(s.get("name") == "Aucune" for s in extracted_info.get("sanctions_summary", [])):
        # Try to load from historical context if available
        historical_meetings = load_historical_meetings(exclude_date=extracted_info["meeting_metadata"]["date"])
        sanctions_found = False
        
        if historical_meetings:
            for meeting in historical_meetings:
                sanctions = meeting.get("sanctions_summary", [])
                real_sanctions = [s for s in sanctions if s.get("name", "") not in ["Aucune", "", "Non spécifié"]]
                if real_sanctions:
                    # Update dates and carry forward sanctions
                    updated_sanctions = []
                    for sanction in real_sanctions:
                        updated_sanction = sanction.copy()
                        updated_sanction["date"] = extracted_info["meeting_metadata"]["date"]
                        updated_sanctions.append(updated_sanction)
                    extracted_info["sanctions_summary"] = updated_sanctions
                    sanctions_found = True
                    st.info(f"📋 Carried forward {len(updated_sanctions)} sanctions from historical meeting")
                    break
        
        # If no historical sanctions, keep default
        if not sanctions_found:
            extracted_info["sanctions_summary"] = [{
                "name": "Aucune",
                "reason": "Aucune sanction mentionnée", 
                "amount": "0",
                "date": extracted_info["meeting_metadata"]["date"],
                "status": "Non appliquée"
            }]
    
    # Ensure agenda items are in French
    if not extracted_info.get("agenda_items") or any("english" in item.lower() for item in extracted_info.get("agenda_items", [])):
        extracted_info["agenda_items"] = [
            "I- Relecture du Compte Rendu",
            "II- Récapitulatif des Résolutions et des Sanctions",
            "III- Revue d'activités", 
            "IV- Faits Saillants",
            "V- Divers"
        ]
    
    return extracted_info

def show_transcript_quality_tips():
    """Display transcript quality improvement tips."""
    with st.sidebar.expander("🎯 Transcript Quality Tips", expanded=False):
        st.markdown("""
        **Poor Transcript Issues:**
        - Teams transcripts are often incomplete
        - Missing speaker identification  
        - Poor French language recognition
        - Incomplete sentences and context
        
        **Better Alternatives:**
        1. **Use Whisper transcription** (built into this app)
           - Upload audio file instead of using Teams transcript
           - Choose "medium" or "large" model for better French
        
        2. **Pre-process your audio:**
           - Clear background noise
           - Ensure good microphone quality
           - Speakers should speak clearly
        
        3. **Manual cleanup:**
           - Review and edit transcripts before processing
           - Add speaker names: "Grace: ..." 
           - Fix obvious errors
        
        4. **Use structured format:**
           - Mention agenda items clearly
           - State resolutions explicitly
           - Name speakers for activities
        """)

def smart_historical_data_filling(extracted_data, date, allow_circular=False):
    """
    Intelligently fill missing data using historical meetings as memory.
    This function allows using the current meeting's JSON if it exists to test perfect generation.
    """
    try:
        # Define department structure for member identification
        department_members = {
            "DEPARTEMENT INVESTISSEMENT": ["Grace Divine", "Vladimir SOUA", "Nour MAHAMAT", "Eric BEIDI"],
            "DEPARTEMENT PROJET": ["Marcellin SEUJIP", "Franklin TANDJA"],
            "DEPARTEMENT IA": ["Emmanuel TEINGA", "Sherelle KANA", "Jules NEMBOT", "Brice DZANGUE"],
            "DEPARTEMENT INNOVATION": ["Jordan KAMSU-KOM", "Christian DJIMELI", "Daniel BAYECK", "Brian ELLA ELLA"],
            "DEPARTEMENT ETUDE": ["Gael KIAMPI", "Francis KAMSU", "Loïc KAMENI"]
        }
        
        # Flatten to get all known team members
        all_known_members = []
        for dept_members in department_members.values():
            all_known_members.extend(dept_members)
        
        # Load historical meetings (include current if allow_circular=True for testing)
        exclude_date = None if allow_circular else date
        historical_meetings = load_historical_meetings(exclude_date=exclude_date)
        
        if not historical_meetings:
            return extracted_data
        
        st.info(f"🧠 Using historical memory from {len(historical_meetings)} meetings to fill gaps...")
        
        # Use the most recent meeting as primary memory source
        latest_meeting = historical_meetings[0]
        
        # Fill missing activity data using historical perspectives as current activities
        activities = extracted_data.get("activities_review", [])
        
        # Check if we have activities but they might be incomplete (check for members who had activities before)
        existing_actors = {activity.get("actor", "") for activity in activities}
        
        # Only check for missing members who had activities in historical meetings
        historical_activities = latest_meeting.get("activities_review", [])
        historically_active_members = {activity.get("actor", "") for activity in historical_activities 
                                     if activity.get("perspectives", "") not in ["RAS", "Non spécifié", ""]}
        
        missing_members = [member for member in historically_active_members 
                          if member in all_known_members and member not in existing_actors]
        
        if missing_members:
            st.info(f"📋 Filling {len(missing_members)} missing members who had activities in historical meetings...")
            
            # Use historical data to fill missing member activities
            historical_actors = {}
            
            # Group historical activities by actor (allowing multiple activities per person)
            for activity in historical_activities:
                actor = activity.get("actor", "")
                if actor not in historical_actors:
                    historical_actors[actor] = []
                historical_actors[actor].append(activity)
            
            # Add missing members using their historical data
            for member in missing_members:
                if member in historical_actors:
                    # Add all historical activities for this member
                    for historical_activity in historical_actors[member]:
                        perspectives = historical_activity.get("perspectives", "")
                        
                        if perspectives and perspectives not in ["RAS", "Non spécifié", ""]:
                            # Use perspectives as current activities
                            activities.append({
                                "actor": member,
                                "dossier": historical_activity.get("dossier", "Non spécifié"),
                                "activities": f"Continuation: {perspectives}",
                                "results": "En cours",
                                "perspectives": "À définir selon avancement"
                            })
                            st.write(f"   • {member} ({historical_activity.get('dossier', '')}): Continued from '{perspectives[:50]}...'")
                        else:
                            # Use the last known activity only if it had meaningful content
                            last_activities = historical_activity.get("activities", "")
                            if last_activities and last_activities not in ["RAS", "Non spécifié", ""]:
                                activities.append({
                                    "actor": member,
                                    "dossier": historical_activity.get("dossier", "Non spécifié"),
                                    "activities": last_activities,
                                    "results": "RAS",
                                    "perspectives": "RAS"
                                })
            
            extracted_data["activities_review"] = activities
        
        # Fill missing resolutions using historical data
        resolutions = extracted_data.get("resolutions_summary", [])
        if len(resolutions) == 0:
            st.info("📋 Filling missing resolutions from historical memory...")
            historical_resolutions = latest_meeting.get("resolutions_summary", [])
            
            # Get ongoing resolutions (status != "Exécuté")
            ongoing_resolutions = []
            for resolution in historical_resolutions:
                if resolution.get("status", "") != "Exécuté":
                    # Update date to current meeting
                    updated_resolution = resolution.copy()
                    updated_resolution["date"] = date
                    ongoing_resolutions.append(updated_resolution)
            
            if ongoing_resolutions:
                extracted_data["resolutions_summary"] = ongoing_resolutions
                st.write(f"   • Carried forward {len(ongoing_resolutions)} ongoing resolutions")
            else:
                # Create default resolution entry
                extracted_data["resolutions_summary"] = [{
                    "date": date,
                    "dossier": "Suivi général",
                    "resolution": "Suivi des activités en cours selon perspectives définies",
                    "responsible": "Tous les membres",
                    "deadline": "Prochaine réunion",
                    "execution_date": "",
                    "status": "En cours"
                }]
        
        # Fill start_time and end_time if missing
        if extracted_data.get("start_time") == "Non spécifié":
            # Try to use common meeting times or historical data
            historical_start = latest_meeting.get("start_time", "Non spécifié")
            if historical_start != "Non spécifié":
                extracted_data["start_time"] = historical_start
                st.write(f"   • Using historical start time: {historical_start}")
            else:
                extracted_data["start_time"] = "09h00min"  # Default meeting start time
                st.write(f"   • Using default start time: 09h00min")
        
        if extracted_data.get("end_time") == "Non spécifié":
            historical_end = latest_meeting.get("end_time", "Non spécifié")
            if historical_end != "Non spécifié":
                extracted_data["end_time"] = historical_end
                st.write(f"   • Using historical end time: {historical_end}")
            else:
                # Calculate end time based on start time + typical meeting duration
                try:
                    start_time_str = extracted_data["start_time"].replace("h", ":").replace("min", "")
                    if ":" not in start_time_str:
                        start_time_str += ":00"
                    start_time = datetime.strptime(start_time_str, "%H:%M")
                    end_time = start_time + timedelta(hours=2)  # 2-hour default duration
                    extracted_data["end_time"] = end_time.strftime("%Hh%Mmin")
                    st.write(f"   • Calculated end time: {extracted_data['end_time']}")
                except ValueError:
                    extracted_data["end_time"] = "11h00min"  # Default end time
                    st.write(f"   • Using default end time: 11h00min")
        
        return extracted_data
        
    except Exception as e:
        st.warning(f"Error in smart historical data filling: {str(e)}")
        return extracted_data

def test_perfect_json_generation(date="16/05/2025"):
    """Test function to load existing perfect JSON and generate document."""
    try:
        import json
        import os
        
        # Find the JSON file for the date
        date_formatted = date.replace("/", "-")
        json_file = None
        
        if os.path.exists("processed_meetings"):
            for file in os.listdir("processed_meetings"):
                if date_formatted in file and file.endswith('.json'):
                    json_file = os.path.join("processed_meetings", file)
                    break
        
        if not json_file:
            st.error(f"No JSON file found for date {date}")
            return None
        
        # Load the JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            extracted_info = json.load(f)
        
        st.success(f"✅ Loaded perfect JSON from {json_file}")
        
        # Ensure complete member data
        extracted_info = ensure_complete_member_data(extracted_info)
        
        # Apply smart historical filling with circular reference allowed
        extracted_info = smart_historical_data_filling(extracted_info, date, allow_circular=True)
        
        st.info(f"📊 Final data: {len(extracted_info.get('activities_review', []))} activities, {len(extracted_info.get('resolutions_summary', []))} resolutions, {len(extracted_info.get('sanctions_summary', []))} sanctions")
        
        return extracted_info
        
    except Exception as e:
        st.error(f"Error loading perfect JSON: {str(e)}")
        return None

def group_activities_by_person_and_dossier(activities_list):
    """
    Group activities by person and then by dossier, allowing multiple activities per person/dossier.
    Returns a flattened list suitable for table generation with proper grouping.
    """
    if not activities_list:
        return []
    
    # Group activities by actor
    actor_groups = {}
    for activity in activities_list:
        actor = activity.get("actor", "").strip()
        if not actor or actor == "Non spécifié":
            continue
            
        if actor not in actor_groups:
            actor_groups[actor] = {}
        
        dossier = activity.get("dossier", "Non spécifié").strip()
        if not dossier:
            dossier = "Non spécifié"
            
        if dossier not in actor_groups[actor]:
            actor_groups[actor][dossier] = []
            
        actor_groups[actor][dossier].append(activity)
    
    # Convert to flattened table format
    flattened_activities = []
    
    # Get expected team members to ensure complete coverage
    expected_members = [
        "Grace Divine", "Vladimir SOUA", "Gael KIAMPI", "Emmanuel TEINGA",
        "Francis KAMSU", "Jordan KAMSU-KOM", "Loïc KAMENI", "Christian DJIMELI",
        "Daniel BAYECK", "Brice DZANGUE", "Sherelle KANA", "Jules NEMBOT",
        "Nour MAHAMAT", "Franklin TANDJA", "Marcellin SEUJIP", "Divine NDE",
        "Brian ELLA ELLA", "Amelin EPOH", "Franklin YOUMBI", "Cédric DONFACK",
        "Wilfried DOPGANG", "Ismaël POUNGOUM", "Éric BEIDI", "Boris ON MAKONG",
        "Charlène GHOMSI"
    ]
    
    for member in expected_members:
        if member in actor_groups:
            dossiers = actor_groups[member]
            first_dossier = True
            
            for dossier_name, dossier_activities in dossiers.items():
                if len(dossier_activities) == 1:
                    # Single activity for this dossier
                    activity = dossier_activities[0]
                    flattened_activities.append({
                        "actor": member if first_dossier else "",  # Only show name on first row
                        "dossier": dossier_name,
                        "activities": activity.get("activities", "RAS"),
                        "results": activity.get("results", "RAS"),
                        "perspectives": activity.get("perspectives", "RAS")
                    })
                    first_dossier = False
                else:
                    # Multiple activities for this dossier - combine them
                    combined_activities = []
                    combined_results = []
                    combined_perspectives = []
                    
                    for activity in dossier_activities:
                        act = activity.get("activities", "").strip()
                        res = activity.get("results", "").strip() 
                        per = activity.get("perspectives", "").strip()
                        
                        if act and act != "RAS" and act != "Non spécifié":
                            combined_activities.append(f"• {act}")
                        if res and res != "RAS" and res != "Non spécifié":
                            combined_results.append(f"• {res}")
                        if per and per != "RAS" and per != "Non spécifié":
                            combined_perspectives.append(f"• {per}")
                    
                    flattened_activities.append({
                        "actor": member if first_dossier else "",  # Only show name on first row
                        "dossier": dossier_name,
                        "activities": "\n".join(combined_activities) if combined_activities else "RAS",
                        "results": "\n".join(combined_results) if combined_results else "RAS", 
                        "perspectives": "\n".join(combined_perspectives) if combined_perspectives else "RAS"
                    })
                    first_dossier = False
        else:
            # Member not found in activities - add default entry
            flattened_activities.append({
                "actor": member,
                "dossier": "Non spécifié",
                "activities": "RAS",
                "results": "RAS",
                "perspectives": "RAS"
            })
    
    return flattened_activities

def organize_activities_by_department(activities_list):
    """
    Organize activities by department as shown in the real meeting documents.
    Returns activities grouped by department with headers.
    """
    if not activities_list:
        return []
    
    # Define department structure based on the screenshots
    departments = {
        "DEPARTEMENT INVESTISSEMENT": ["Grace Divine", "Vladimir SOUA", "Nour MAHAMAT", "Eric BEIDI"],
        "DEPARTEMENT PROJET": ["Marcellin SEUJIP", "Franklin TANDJA"],
        "DEPARTEMENT IA": ["Emmanuel TEINGA", "Sherelle KANA", "Jules NEMBOT", "Brice DZANGUE"],
        "DEPARTEMENT INNOVATION": ["Jordan KAMSU-KOM", "Christian DJIMELI", "Daniel BAYECK", "Brian ELLA ELLA"],
        "DEPARTEMENT ETUDE": ["Gael KIAMPI", "Francis KAMSU", "Loïc KAMENI"]
    }
    
    # Group activities by actor first
    actor_activities = {}
    for activity in activities_list:
        actor = activity.get("actor", "").strip()
        if not actor or actor == "Non spécifié":
            continue
            
        if actor not in actor_activities:
            actor_activities[actor] = {}
        
        dossier = activity.get("dossier", "Non spécifié").strip()
        if not dossier:
            dossier = "Non spécifié"
            
        if dossier not in actor_activities[actor]:
            actor_activities[actor][dossier] = []
            
        actor_activities[actor][dossier].append(activity)
    
    # Organize by departments
    organized_activities = []
    
    for dept_name, dept_members in departments.items():
        dept_has_activities = False
        dept_activities = []
        
        for member in dept_members:
            if member in actor_activities:
                dept_has_activities = True
                dossiers = actor_activities[member]
                first_dossier = True
                
                for dossier_name, dossier_activities in dossiers.items():
                    if len(dossier_activities) == 1:
                        # Single activity for this dossier
                        activity = dossier_activities[0]
                        dept_activities.append({
                            "type": "activity",
                            "actor": member if first_dossier else "",
                            "dossier": dossier_name,
                            "activities": activity.get("activities", "RAS"),
                            "results": activity.get("results", "RAS"),
                            "perspectives": activity.get("perspectives", "RAS")
                        })
                        first_dossier = False
                    else:
                        # Multiple activities for this dossier - combine them
                        combined_activities = []
                        combined_results = []
                        combined_perspectives = []
                        
                        for activity in dossier_activities:
                            act = activity.get("activities", "").strip()
                            res = activity.get("results", "").strip() 
                            per = activity.get("perspectives", "").strip()
                            
                            if act and act != "RAS" and act != "Non spécifié":
                                combined_activities.append(f"• {act}")
                            if res and res != "RAS" and res != "Non spécifié":
                                combined_results.append(f"• {res}")
                            if per and per != "RAS" and per != "Non spécifié":
                                combined_perspectives.append(f"• {per}")
                        
                        dept_activities.append({
                            "type": "activity",
                            "actor": member if first_dossier else "",
                            "dossier": dossier_name,
                            "activities": "\n".join(combined_activities) if combined_activities else "RAS",
                            "results": "\n".join(combined_results) if combined_results else "RAS", 
                            "perspectives": "\n".join(combined_perspectives) if combined_perspectives else "RAS"
                        })
                        first_dossier = False
        
        # Add department header and activities if department has any activities
        if dept_has_activities:
            # Add department header
            organized_activities.append({
                "type": "header",
                "department": dept_name,
                "actor": "",
                "dossier": "",
                "activities": "",
                "results": "",
                "perspectives": ""
            })
            # Add department activities
            organized_activities.extend(dept_activities)
    
    return organized_activities

def add_activities_table_with_departments(doc, organized_activities, table_width=10.5):
    """Add activities table with department headers and proper styling."""
    if not organized_activities:
        return None
    
    # Define department colors based on screenshots
    dept_colors = {
        "DEPARTEMENT INVESTISSEMENT": (255, 0, 0),      # Red
        "DEPARTEMENT PROJET": (0, 128, 0),              # Green  
        "DEPARTEMENT IA": (128, 0, 128),                # Purple
        "DEPARTEMENT INNOVATION": (0, 0, 255),          # Blue
        "DEPARTEMENT ETUDE": (255, 165, 0)              # Orange
    }
    
    # Count total rows needed (headers + activities)
    total_rows = len(organized_activities) + 1  # +1 for main header
    
    # Create table
    table = doc.add_table(rows=total_rows, cols=5)
    try:
        table.style = "Table Grid"
    except KeyError:
        pass  # Use default style if Table Grid not available
    
    set_table_width(table, table_width)
    set_column_widths(table, [2.0, 2.0, 2.5, 2.0, 2.0])
    
    # Add main header row
    headers = ["ACTEURS", "DOSSIERS", "ACTIVITÉS", "RÉSULTATS", "PERSPECTIVES"]
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        run = cell.paragraphs[0].runs[0]
        run.font.name = "Century"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, (0, 0, 0))  # Black header
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add data rows
    current_row = 1
    for item in organized_activities:
        row = table.rows[current_row]
        
        if item.get("type") == "header":
            # Department header row
            dept_name = item.get("department", "")
            
            # Merge all cells in this row for department header
            merged_cell = row.cells[0]
            for i in range(1, 5):
                merged_cell.merge(row.cells[i])
            
            merged_cell.text = dept_name
            run = merged_cell.paragraphs[0].runs[0]
            run.font.name = "Century"
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            
            # Set department color
            dept_color = dept_colors.get(dept_name, (128, 128, 128))  # Default gray
            set_cell_background(merged_cell, dept_color)
            merged_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            merged_cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            
        else:
            # Regular activity row
            data = [
                item.get("actor", ""),
                item.get("dossier", ""),
                item.get("activities", ""),
                item.get("results", ""),
                item.get("perspectives", "")
            ]
            
            for j, cell_text in enumerate(data):
                cell = row.cells[j]
                cell.text = cell_text
                run = cell.paragraphs[0].runs[0]
                run.font.name = "Century"
                run.font.size = Pt(12)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                
                # Add alternating row colors for readability
                if current_row % 2 == 0:
                    set_cell_background(cell, (240, 240, 240))  # Light gray
        
        current_row += 1
    
    return table

def show_meeting_guidance():
    """Display comprehensive meeting guidance for better transcription and data extraction."""
    with st.sidebar.expander("📋 Meeting Structure Guide", expanded=False):
        st.markdown("""
        ### 🎯 For Better Data Extraction
        
        **📢 Meeting Opening (Speaker should say):**
        ```
        "Bonjour, nous commençons la réunion à [HEURE]
        Présents: [List all present members clearly]
        Absents: [List absent members with reasons]
        Rapporteur: [Name]
        Président: [Name]"
        ```
        
        **📋 Activity Review Structure:**
        ```
        Department by department:
        "DEPARTEMENT INVESTISSEMENT:
        - Grace Divine: Dossier [NAME], Activités [DETAILS], 
          Résultats [DETAILS], Perspectives [DETAILS]
        - Vladimir SOUA: ..."
        ```
        
        **📜 Resolutions (Speaker should say):**
        ```
        "RESOLUTION: [Clear description]
        Responsable: [Person name]
        Échéance: [Date DD/MM/YYYY]
        Statut: En cours/Exécuté"
        ```
        
        **💰 Sanctions (If any):**
        ```
        "SANCTION: [Person] pour [Reason] 
        Montant: [Amount] FCFA
        Date: [DD/MM/YYYY]"
        ```
        
        **⏰ Meeting End:**
        ```
        "La réunion se termine à [HEURE]
        Solde du compte: [Amount] FCFA au [Date]"
        ```
        """)

    with st.sidebar.expander("🎙️ Audio Quality Tips", expanded=False):
        st.markdown("""
        ### 🔊 Recording Best Practices
        
        **Before Meeting:**
        - Use good microphone (not laptop mic)
        - Test audio levels
        - Quiet room, minimal background noise
        - Position mic centrally
        
        **During Meeting:**
        - Speak clearly and slowly
        - State your name before speaking
        - Pause between topics
        - Repeat important information
        - Spell out complex terms
        
        **For Virtual Meetings:**
        - Use "Record" feature in Teams/Zoom
        - Ask participants to mute when not speaking
        - Use headphones to reduce echo
        - Enable auto-transcription if available
        
        **Post-Meeting:**
        - Review transcript before uploading
        - Fix obvious errors manually
        - Add missing speaker names
        - Clarify unclear sections
        """)

    with st.sidebar.expander("🏗️ Meeting Template", expanded=False):
        st.markdown("""
        ### 📝 Suggested Meeting Flow
        
        **1. Opening (5 min)**
        - Time, attendance, roles
        
        **2. Previous Minutes Review (10 min)**
        - Quick validation
        
        **3. Resolutions & Sanctions Review (10 min)**
        - Update status of previous items
        
        **4. Activity Review by Department (30 min)**
        - INVESTISSEMENT → PROJET → IA → INNOVATION → ETUDE
        - Each person: Dossier, Activities, Results, Perspectives
        
        **5. Key Highlights (10 min)**
        - Important announcements
        
        **6. Miscellaneous (10 min)**
        - Other topics
        
        **7. Closing (5 min)**
        - End time, account balance
        """)

def main():
    st.title("Meeting Transcription Tool")
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.session_state.mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")
    st.session_state.deepseek_api_key = st.sidebar.text_input("Deepseek API Key", type="password")
    
    # Show transcript quality tips
    show_transcript_quality_tips()
    
    # Show meeting guidance for better data extraction
    show_meeting_guidance()
    
    # Show historical context information
    st.sidebar.header("📊 Historical Context")
    historical_meetings = load_historical_meetings()
    if historical_meetings:
        st.sidebar.success(f"✅ {len(historical_meetings)} historical meetings loaded")
        
        # Show recent meetings
        for i, meeting in enumerate(historical_meetings):
            date = meeting.get('meeting_metadata', {}).get('date', 'Unknown')
            st.sidebar.write(f"📅 Meeting {i+1}: {date}")
            
            # Show sanctions count
            sanctions = meeting.get('sanctions_summary', [])
            real_sanctions = [s for s in sanctions if s.get('name', '') not in ['Aucune', '', 'Non spécifié']]
            if real_sanctions:
                st.sidebar.write(f"   🚨 {len(real_sanctions)} sanctions available")
                with st.sidebar.expander(f"View sanctions from {date}"):
                    for sanction in real_sanctions[:5]:  # Show first 5
                        st.write(f"• {sanction.get('name', '')}: {sanction.get('reason', '')} ({sanction.get('amount', '')} FCFA)")
            else:
                st.sidebar.write(f"   ✅ No active sanctions")
    else:
        st.sidebar.warning("⚠️ No historical meetings found")
        st.sidebar.info("💡 Use the Historical Processor app to process past meeting documents first.")
    
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
        
        # Add warning about circular reference
        formatted_date = meeting_date.strftime("%d/%m/%Y")
        st.info(f"📅 Meeting date: {formatted_date}")
        
        # Check if meeting already exists and offer testing mode
        meeting_exists = os.path.exists("processed_meetings") and any(formatted_date.replace("/", "-") in f for f in os.listdir("processed_meetings") if f.endswith('.json'))
        
        if meeting_exists:
            st.warning(f"⚠️ A meeting for {formatted_date} already exists in historical data.")
            
            # Add option for testing perfect JSON generation
            test_mode = st.checkbox(
                "🧪 **Test Mode**: Use existing JSON as perfect data source",
                help="Enable this to test document generation using the existing perfect JSON data. This allows circular reference for testing purposes."
            )
            
            if test_mode:
                st.success("✅ Test mode enabled! Will use existing JSON data to fill gaps for perfect document generation.")
                st.session_state.test_mode = True
                
                # Add test button for direct JSON loading
                if st.button("🧪 Load Perfect JSON & Generate Document"):
                    with st.spinner("Loading perfect JSON data..."):
                        perfect_data = test_perfect_json_generation(formatted_date)
                        if perfect_data:
                            st.session_state.extracted_info = perfect_data
                            st.success("✅ Perfect JSON loaded! Ready for document generation.")
                            with st.expander("📋 View Perfect JSON Data"):
                                st.json(perfect_data)
            else:
                st.info("📊 Normal mode: Will exclude existing meeting from context to avoid circular reference.")
                st.session_state.test_mode = False
        else:
            st.session_state.test_mode = False
    
    with col2:
        st.header("Transcription & Résultat")
        input_method = st.radio("Méthode d'entrée :", ("Télécharger Audio", "Entrer la Transcription"))
        
        if input_method == "Télécharger Audio":
            uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["mp3", "wav", "m4a", "flac"])
            whisper_model = st.selectbox("Modèle Whisper", ["tiny", "base", "small", "medium", "large"], index=1)
            
            st.info("💡 **Tip:** Use 'medium' or 'large' models for better French transcription quality!")
            
            if uploaded_file and st.button("Transcrire l'Audio"):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with st.spinner(f"Transcription avec Whisper {whisper_model}..."):
                    transcription = transcribe_audio(uploaded_file, file_extension, whisper_model)
                    if transcription and not transcription.startswith("Error"):
                        st.session_state.transcription = transcription
                        st.success("✅ Transcription completed!")
                        st.text_area("Transcription résultante", transcription, height=200)
                        
                        with st.spinner("Extraction des informations avec contexte historique..."):
                            # Use test mode if enabled
                            allow_circular = getattr(st.session_state, 'test_mode', False)
                            if allow_circular:
                                st.info("🧪 Test mode: Using existing JSON data for perfect generation")
                            
                            extracted_info = extract_info(st.session_state.transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), st.session_state.deepseek_api_key, st.session_state.get("previous_context", ""), allow_circular)
                            if extracted_info:
                                # Apply smart historical filling with test mode
                                extracted_info = smart_historical_data_filling(extracted_info, meeting_date.strftime("%d/%m/%Y"), allow_circular=allow_circular)
                                st.session_state.extracted_info = extracted_info
                                st.success("✅ Information extraction completed!")
                                with st.expander("📋 View Extracted Information"):
                                    st.json(extracted_info)
        else:
            st.info("💡 **Teams Transcript?** Consider uploading the audio file for better quality!")
            transcription_input = st.text_area("Entrez la transcription :", height=200, help="Tip: Add speaker names like 'Grace: ...' for better extraction")
            if st.button("Soumettre la Transcription") and transcription_input:
                st.session_state.transcription = transcription_input
                with st.spinner("Extraction des informations avec contexte historique..."):
                    # Use test mode if enabled
                    allow_circular = getattr(st.session_state, 'test_mode', False)
                    if allow_circular:
                        st.info("🧪 Test mode: Using existing JSON data for perfect generation")
                    
                    extracted_info = extract_info(st.session_state.transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"), st.session_state.deepseek_api_key, st.session_state.get("previous_context", ""), allow_circular)
                    if extracted_info:
                        # Apply smart historical filling with test mode  
                        extracted_info = smart_historical_data_filling(extracted_info, meeting_date.strftime("%d/%m/%Y"), allow_circular=allow_circular)
                        st.session_state.extracted_info = extracted_info
                        st.success("✅ Information extraction completed!")
                        with st.expander("📋 View Extracted Information"):
                            st.json(extracted_info)
        
        if 'extracted_info' in st.session_state and st.button("Générer et Télécharger le Document"):
            with st.spinner("Génération du document avec données complètes..."):
                docx_data = fill_template_and_generate_docx(st.session_state.extracted_info, meeting_title, meeting_date)
                if docx_data:
                    st.success("✅ Document generated successfully!")
                    st.download_button(
                        label="Télécharger le Document",
                        data=docx_data,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()