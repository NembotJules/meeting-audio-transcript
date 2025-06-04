from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import os
from mistralai import Mistral
import requests
from dataclasses import dataclass
import tempfile
import re

@dataclass
class MeetingSchema:
    """Schema definition for meeting data"""
    meeting_metadata: Dict[str, str]  # date, title
    attendance: Dict[str, List[str]]  # present, absent
    agenda_items: List[str]
    activities_review: List[Dict[str, str]]  # actor, dossier, activities, results, perspectives
    resolutions_summary: List[Dict[str, str]]  # date, dossier, resolution, responsible, deadline, status
    key_highlights: List[str]
    miscellaneous: List[str]
    sanctions_summary: List[Dict[str, str]]  # name, reason, amount, date, status

class MeetingProcessor:
    def __init__(self, mistral_api_key: str, deepseek_api_key: str):
        """Initialize with necessary API keys."""
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.deepseek_api_key = deepseek_api_key

    def extract_text_from_document(self, file) -> str:
        """
        Extract text from document using Mistral OCR.
        
        Args:
            file: File object (PDF, PNG, JPG, JPEG)
            
        Returns:
            Extracted text maintaining document structure
        """
        try:
            # Upload the file to Mistral
            uploaded_file = self.mistral_client.files.upload(
                file={
                    "file_name": file.name,
                    "content": file.getvalue(),
                },
                purpose="ocr"
            )
            
            # Get the signed URL
            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Set document type based on file extension
            file_extension = file.name.split('.')[-1].lower()
            document_type = "document_url" if file_extension == 'pdf' else "image_url"
            
            # Extract text using Mistral OCR
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from the document, maintaining the structure and formatting."
                        },
                        {
                            "type": document_type,
                            document_type: signed_url.url
                        }
                    ]
                }
            ]
            
            response = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=messages
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting text from document: {str(e)}")

    def extract_structured_info(self, text: str, meeting_date: str, meeting_title: str) -> Dict:
        """
        Extract structured information from text using LLM.
        
        Args:
            text: Extracted text from document
            meeting_date: Date of the meeting
            meeting_title: Title of the meeting
            
        Returns:
            Structured meeting information as JSON
        """
        prompt = f"""
        You are a precise JSON extractor. Your task is to extract structured information from a meeting note and format it as valid JSON.
        DO NOT include any explanations or additional text in your response, ONLY return the JSON object.
        
        Meeting Date: {meeting_date}
        Meeting Title: {meeting_title}
        
        Extract the following information from the meeting note and structure it exactly as shown:
        
        1. Liste de présence:
           - Present members -> attendance.present[]
           - Absent members -> attendance.absent[]
        
        2. Ordre du jour -> agenda_items[]
        
        3. Revue d'activités (Table):
           For each entry in activities_review[]:
           - Acteur -> actor
           - Dossier -> dossier
           - Activités -> activities
           - Résultats -> results
           - Perspectives -> perspectives
        
        4. Récapitulatif des Résolutions (Table):
           For each entry in resolutions_summary[]:
           - Date -> date (DD/MM/YYYY)
           - Dossier -> dossier
           - Résolution -> resolution
           - Responsable -> responsible
           - Délai d'exécution -> deadline (DD/MM/YYYY)
           - Statut -> status
        
        5. Faits Saillants -> key_highlights[]
        
        6. Divers -> miscellaneous[]
        
        7. Récapitulatif des Sanctions (Table):
           For each entry in sanctions_summary[]:
           - Nom -> name
           - Motif -> reason
           - Montant -> amount (number)
           - Date -> date (DD/MM/YYYY)
           - Statut -> status

        Meeting Note:
        {text}

        IMPORTANT: Your response must be ONLY the JSON object, with no additional text, markdown, or formatting.
        Use this exact structure:
        {{
            "meeting_metadata": {{
                "date": "{meeting_date}",
                "title": "{meeting_title}"
            }},
            "attendance": {{
                "present": [],
                "absent": []
            }},
            "agenda_items": [],
            "activities_review": [
                {{
                    "actor": "",
                    "dossier": "",
                    "activities": "",
                    "results": "",
                    "perspectives": ""
                }}
            ],
            "resolutions_summary": [
                {{
                    "date": "",
                    "dossier": "",
                    "resolution": "",
                    "responsible": "",
                    "deadline": "",
                    "status": ""
                }}
            ],
            "key_highlights": [],
            "miscellaneous": [],
            "sanctions_summary": [
                {{
                    "name": "",
                    "reason": "",
                    "amount": 0,
                    "date": "",
                    "status": ""
                }}
            ]
        }}
        """
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
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
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}")
            
            # Extract JSON from response
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            def clean_json_string(json_str: str) -> str:
                """Helper function to clean and prepare JSON string"""
                try:
                    # Remove any markdown code block markers
                    json_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str)
                    
                    # Find the actual JSON content
                    json_match = re.search(r'({[\s\S]*)', json_str)
                    if not json_match:
                        raise Exception("No valid JSON object found in response")
                    json_str = json_match.group(1)

                    def clean_text_content(text):
                        """Clean text content while preserving special characters"""
                        # First, escape any existing quotes
                        text = text.replace('"', '\\"')
                        
                        # Handle French apostrophes and quotes
                        text = text.replace('"', '\\"').replace('"', '\\"')
                        text = text.replace("'", "'").replace("'", "'")
                        
                        # Replace smart quotes with regular quotes
                        text = text.replace('"', '\\"').replace('"', '\\"')
                        
                        # Handle French contractions - escape the apostrophes
                        text = re.sub(r"(\w)'(\w)", r"\1\\'\2", text)
                        
                        return text

                    def preprocess_json(json_str):
                        """Preprocess JSON string to fix common issues"""
                        # Basic cleanup
                        json_str = json_str.replace('\n', ' ')
                        json_str = ' '.join(json_str.split())
                        
                        # First, protect string content
                        protected_strings = {}
                        
                        def protect_string(match):
                            content = match.group(1)
                            # Clean and protect the string content
                            cleaned = clean_text_content(content)
                            key = f"__STRING_{len(protected_strings)}__"
                            protected_strings[key] = cleaned
                            return f'"{key}"'
                        
                        # Protect string contents
                        json_str = re.sub(r'"([^"]*)"', protect_string, json_str)
                        
                        # Fix structural issues
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                        json_str = re.sub(r':\s*,', ': null,', json_str)    # Fix empty values
                        json_str = re.sub(r':\s*}', ': null}', json_str)    # Fix empty values at end
                        json_str = re.sub(r':\s*]', ': null]', json_str)    # Fix empty values in arrays
                        
                        # Restore protected strings
                        for key, value in protected_strings.items():
                            json_str = json_str.replace(f'"{key}"', f'"{value}"')
                        
                        return json_str

                    def validate_and_fix_json_structure(json_str):
                        """Validate JSON structure and fix unclosed brackets"""
                        stack = []
                        in_string = False
                        escaped = False
                        chars = list(json_str)
                        i = 0
                        
                        while i < len(chars):
                            char = chars[i]
                            
                            if char == '\\':
                                escaped = not escaped
                                i += 1
                                continue
                            
                            if not escaped and char == '"':
                                in_string = not in_string
                            elif not in_string:
                                if char in '{[':
                                    stack.append((char, i))
                                elif char in '}]':
                                    if not stack:
                                        chars.pop(i)
                                        continue
                                    opening, _ = stack.pop()
                                    if (opening == '{' and char != '}') or (opening == '[' and char != ']'):
                                        chars[i] = '}' if opening == '{' else ']'
                            
                            escaped = False
                            i += 1
                        
                        while stack:
                            opening, _ = stack.pop()
                            chars.append('}' if opening == '{' else ']')
                        
                        return ''.join(chars)

                    # Process the JSON in stages
                    json_str = preprocess_json(json_str)
                    json_str = validate_and_fix_json_structure(json_str)
                    
                    try:
                        # Try parsing the preprocessed and fixed JSON
                        data = json.loads(json_str)
                        return json.dumps(data, ensure_ascii=False)
                        
                    except json.JSONDecodeError as e:
                        print("\nJSON Parsing Error Details:")
                        print(f"Error message: {str(e)}")
                        print(f"Error position: {e.pos}")
                        print(f"Line number: {e.lineno}")
                        print(f"Column number: {e.colno}")
                        
                        # Show more context around the error
                        context_start = max(0, e.pos - 200)
                        context_end = min(len(json_str), e.pos + 200)
                        error_context = json_str[context_start:context_end]
                        print("\nContext around error:")
                        print("..." if context_start > 0 else "", end="")
                        print(error_context, end="")
                        print("..." if context_end < len(json_str) else "")
                        print(" " * (min(200, e.pos - context_start)) + "^")  # Point to error location
                        raise

                except Exception as e:
                    print(f"\nJSON Processing Error:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    if isinstance(e, json.JSONDecodeError):
                        print(f"Error position: {e.pos}")
                        print(f"Line number: {e.lineno}")
                        print(f"Column number: {e.colno}")
                    raise Exception(f"Failed to clean JSON: {str(e)}")
            
            try:
                # Process the JSON
                cleaned_json = clean_json_string(content)
                extracted_data = json.loads(cleaned_json)
                
                # Validate the extracted data
                return self.validate_meeting_json(extracted_data)
                
            except Exception as e:
                print(f"Error processing JSON: {str(e)}")
                raise Exception(f"Error extracting structured information: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Error extracting structured information: {str(e)}")

    def validate_meeting_json(self, data: Dict) -> Dict:
        """
        Validate the extracted JSON against our schema.
        
        Args:
            data: Extracted meeting data
            
        Returns:
            Validated and cleaned meeting data
        """
        required_fields = {
            "meeting_metadata": {"date", "title"},
            "attendance": {"present", "absent"},
            "agenda_items": list,
            "activities_review": list,
            "resolutions_summary": list,
            "key_highlights": list,
            "miscellaneous": list,
            "sanctions_summary": list
        }
        
        # Check required fields
        for field, subfields in required_fields.items():
            if field not in data:
                data[field] = {} if isinstance(subfields, set) else []
            if isinstance(subfields, set):
                for subfield in subfields:
                    if subfield not in data[field]:
                        data[field][subfield] = []
        
        # Validate dates
        def validate_date(date_str):
            try:
                if not date_str:
                    return ""
                datetime.strptime(date_str, "%d/%m/%Y")
                return date_str
            except ValueError:
                return ""
        
        data["meeting_metadata"]["date"] = validate_date(data["meeting_metadata"].get("date", ""))
        
        # Validate lists
        for field in ["agenda_items", "key_highlights", "miscellaneous"]:
            if not isinstance(data[field], list):
                data[field] = []
        
        # Validate complex structures
        for resolution in data["resolutions_summary"]:
            resolution["date"] = validate_date(resolution.get("date", ""))
            resolution["deadline"] = validate_date(resolution.get("deadline", ""))
        
        for sanction in data["sanctions_summary"]:
            sanction["date"] = validate_date(sanction.get("date", ""))
            try:
                sanction["amount"] = float(sanction.get("amount", 0))
            except (ValueError, TypeError):
                sanction["amount"] = 0
        
        return data

    def process_historical_meeting(self, file, meeting_date: str, meeting_title: str) -> Dict:
        """
        Process a historical meeting document end-to-end.
        
        Args:
            file: File object (PDF, PNG, JPG, JPEG)
            meeting_date: Date of the meeting (DD/MM/YYYY)
            meeting_title: Title of the meeting
            
        Returns:
            Structured meeting data
        """
        try:
            # Extract text from document
            extracted_text = self.extract_text_from_document(file)
            
            # Extract structured information
            structured_data = self.extract_structured_info(
                extracted_text,
                meeting_date,
                meeting_title
            )
            
            return structured_data
            
        except Exception as e:
            raise Exception(f"Error processing historical meeting: {str(e)}")

def save_meeting_data(data: Dict, output_dir: str) -> str:
    """
    Save meeting data to JSON file.
    
    Args:
        data: Meeting data
        output_dir: Directory to save the file
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from meeting date and title
    date = data["meeting_metadata"]["date"].replace("/", "-")
    title = data["meeting_metadata"]["title"].replace(" ", "_")
    filename = f"meeting_{date}_{title}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath 