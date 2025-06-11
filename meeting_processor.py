"""Module for processing meeting transcripts and extracting structured information."""
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

def clean_json_response(response: str) -> str:
    """Clean and fix common JSON formatting issues in API responses."""
    if not response:
        return None
    
    try:
        # Remove any markdown code block markers and extra whitespace
        cleaned = response.strip()
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```', '', cleaned)
        cleaned = cleaned.strip()
        
        # Try to find JSON content - look for content between { } or [ ]
        json_patterns = [
            r'({.*})',  # Object
            r'(\[.*\])',  # Array
        ]
        
        json_str = None
        for pattern in json_patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                json_str = match.group(1)
                break
        
        if not json_str:
            print(f"No JSON found in response: {response}")
            return None
        
        # Try to parse the JSON as-is first
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {str(e)}")
            print(f"Attempting to fix JSON...")
        
        # Check if JSON appears to be truncated (incomplete)
        if not json_str.rstrip().endswith(('}', ']')):
            print("JSON appears to be truncated!")
            # Try to close incomplete objects/arrays
            json_str = try_fix_truncated_json(json_str)
        
        # Simple fixes for common issues
        fixed_json = json_str
        
        # Fix trailing commas
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
        
        # Try parsing again
        try:
            json.loads(fixed_json)
            return fixed_json
        except json.JSONDecodeError as e:
            print(f"Still cannot parse JSON after fixes: {str(e)}")
            print(f"Problematic JSON (first 500 chars): {fixed_json[:500]}...")
            return None
        
    except Exception as e:
        print(f"Error in clean_json_response: {str(e)}")
        print(f"Original response (first 500 chars): {response[:500]}...")
        return None

def try_fix_truncated_json(json_str: str) -> str:
    """Attempt to fix truncated JSON by adding missing closing brackets."""
    try:
        # Count opening and closing brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Find the last complete object/array and truncate there
        # This is a simple approach - remove incomplete trailing content
        lines = json_str.split('\n')
        
        # Work backwards to find the last complete line with proper structure
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.endswith('}') or line.endswith('},'):
                # Found a complete object, keep up to this line
                truncated_json = '\n'.join(lines[:i+1])
                
                # Remove trailing comma if present
                if truncated_json.rstrip().endswith(','):
                    truncated_json = truncated_json.rstrip()[:-1]
                
                # Add missing closing brackets
                missing_braces = open_braces - close_braces
                missing_brackets = open_brackets - close_brackets
                
                # Only count what's in our truncated version
                trunc_open_braces = truncated_json.count('{')
                trunc_close_braces = truncated_json.count('}')
                trunc_open_brackets = truncated_json.count('[')
                trunc_close_brackets = truncated_json.count(']')
                
                # Add the missing closing brackets
                for _ in range(trunc_open_braces - trunc_close_braces):
                    truncated_json += '\n    }'
                for _ in range(trunc_open_brackets - trunc_close_brackets):
                    truncated_json += '\n]'
                
                print(f"Attempted to fix truncated JSON by closing at line {i+1}")
                return truncated_json
        
        # If we couldn't find a good truncation point, just add closing brackets
        for _ in range(open_braces - close_braces):
            json_str += '}'
        for _ in range(open_brackets - close_brackets):
            json_str += ']'
        
        return json_str
        
    except Exception as e:
        print(f"Error trying to fix truncated JSON: {str(e)}")
        return json_str

class MeetingProcessor:
    def __init__(self, mistral_api_key: str, deepseek_api_key: str, context_dir: str = "processed_meetings"):
        """Initialize with necessary API keys and context directory."""
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.deepseek_api_key = deepseek_api_key
        self.context_dir = context_dir

    def load_historical_context(self, max_meetings: int = 3, exclude_date: str = "") -> str:
        """
        Load the most recent historical meetings as context.
        
        Args:
            max_meetings: Maximum number of meetings to include as context
            exclude_date: Date to exclude from context
            
        Returns:
            Formatted context string for the LLM
        """
        try:
            if not os.path.exists(self.context_dir):
                print(f"Context directory {self.context_dir} not found")
                return ""
            
            # Get all JSON files in the context directory
            json_files = []
            for file in os.listdir(self.context_dir):
                if file.endswith('.json'):
                    filepath = os.path.join(self.context_dir, file)
                    
                    # Skip files that match the exclude_date to avoid circular context
                    if exclude_date:
                        exclude_date_formatted = exclude_date.replace("/", "-")
                        if exclude_date_formatted in file:
                            print(f"Excluding current meeting {file} from historical context to avoid circular reference")
                            continue
                    
                    # Get file modification time for sorting
                    mtime = os.path.getmtime(filepath)
                    json_files.append((mtime, filepath, file))
            
            if not json_files:
                print("No historical meeting files found")
                return ""
            
            # Sort by modification time (most recent first) and take the last max_meetings
            json_files.sort(key=lambda x: x[0], reverse=True)
            recent_files = json_files[:max_meetings]
            
            print(f"Loading {len(recent_files)} historical meetings as context:")
            
            context_parts = []
            for mtime, filepath, filename in recent_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        meeting_data = json.load(f)
                    
                    print(f"- {filename}")
                    
                    # Format this meeting's data for context
                    context_parts.append(self.format_meeting_context(meeting_data))
                    
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue
            
            if context_parts:
                full_context = "=== HISTORICAL MEETING CONTEXT ===\n\n" + "\n\n".join(context_parts)
                print(f"Historical context loaded successfully ({len(full_context)} characters)")
                return full_context
            else:
                return ""
                
        except Exception as e:
            print(f"Error loading historical context: {str(e)}")
            return ""

    def format_meeting_context(self, meeting_data: Dict) -> str:
        """
        Format a single meeting's data for use as context.
        
        Args:
            meeting_data: Meeting data dictionary
            
        Returns:
            Formatted context string
        """
        try:
            context = []
            
            # Meeting metadata
            metadata = meeting_data.get("meeting_metadata", {})
            context.append(f"Meeting Date: {metadata.get('date', 'Unknown')}")
            context.append(f"Meeting Title: {metadata.get('title', 'Unknown')}")
            
            # Ongoing activities (focus on perspectives - what's planned for next)
            activities = meeting_data.get("activities_review", [])
            if activities:
                context.append("\nOngoing Activities:")
                for activity in activities[:10]:  # Limit to first 10 to avoid too much context
                    actor = activity.get("actor", "Unknown")
                    dossier = activity.get("dossier", "Unknown")
                    perspectives = activity.get("perspectives", "")
                    if perspectives and perspectives != "RAS":
                        context.append(f"- {actor} ({dossier}): {perspectives}")
            
            # Pending resolutions
            resolutions = meeting_data.get("resolutions_summary", [])
            if resolutions:
                context.append("\nResolutions from this meeting:")
                for resolution in resolutions:
                    responsible = resolution.get("responsible", "Unknown")
                    deadline = resolution.get("deadline", "")
                    resolution_text = resolution.get("resolution", "")
                    status = resolution.get("status", "")
                    if resolution_text:
                        res_info = f"- {resolution_text} (Responsible: {responsible}"
                        if deadline:
                            res_info += f", Deadline: {deadline}"
                        if status:
                            res_info += f", Status: {status}"
                        res_info += ")"
                        context.append(res_info)
            
            # Previous sanctions (for reference)
            sanctions = meeting_data.get("sanctions_summary", [])
            if sanctions and any(s.get("name", "") != "Aucune" for s in sanctions):
                context.append("\nSanctions from this meeting:")
                for sanction in sanctions:
                    name = sanction.get("name", "")
                    reason = sanction.get("reason", "")
                    amount = sanction.get("amount", "")
                    if name != "Aucune" and name:
                        context.append(f"- {name}: {reason} ({amount} FCFA)")
            
            return "\n".join(context)
            
        except Exception as e:
            print(f"Error formatting meeting context: {str(e)}")
            return f"Meeting Date: {meeting_data.get('meeting_metadata', {}).get('date', 'Unknown')}"

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
        Extract structured information from text using LLM with historical context.
        
        Args:
            text: Extracted text from document
            meeting_date: Date of the meeting
            meeting_title: Title of the meeting
            
        Returns:
            Structured meeting information as JSON
        """
        def safe_parse_json(json_str: str, default_value: Dict) -> Dict:
            """Safely parse JSON with fallback to default value."""
            if not json_str:
                return default_value
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic JSON: {json_str}")
                return default_value

        try:
            # Load historical context (excluding current meeting to avoid circular reference)
            historical_context = self.load_historical_context(max_meetings=3, exclude_date=meeting_date)
            
            # Create enhanced prompts with context and proper French defaults
            context_instruction = ""
            if historical_context:
                context_instruction = f"""
IMPORTANT: Use the following historical context from previous meetings to inform your extraction. 
Pay special attention to:
- Ongoing activities and their progress
- People mentioned in previous meetings
- Pending resolutions and their status
- Previous sanctions for reference

{historical_context}

=====================================
"""

            # Define expected team members
            expected_members = [
                "Grace Divine", "Vladimir SOUA", "Gael KIAMPI", "Emmanuel TEINGA",
                "Francis KAMSU", "Jordan KAMSU-KOM", "Loïc KAMENI", "Christian DJIMELI",
                "Daniel BAYECK", "Brice DZANGUE", "Sherelle KANA", "Jules NEMBOT",
                "Nour MAHAMAT", "Franklin TANDJA", "Marcellin SEUJIP", "Divine NDE",
                "Brian ELLA ELLA", "Amelin EPOH", "Franklin YOUMBI", "Cédric DONFACK",
                "Wilfried DOPGANG", "Ismaël POUNGOUM", "Éric BEIDI", "Boris ON MAKONG",
                "Charlène GHOMSI"
            ]

            # 1. Extract attendance with context
            attendance_prompt = f"""
{context_instruction}
Extract ONLY the attendance information from the meeting note.
Consider people mentioned in the historical context above.
Expected team members: {', '.join(expected_members)}
Return as two lists in this EXACT format (no other text):
{{"present": ["name1", "name2"], "absent": ["name3", "name4"]}}

Meeting Note:
{text}
"""
            attendance_response = self._make_api_call(attendance_prompt)
            attendance_data = safe_parse_json(attendance_response, {"present": [], "absent": []})

            # 2. Extract agenda items with proper French defaults
            agenda_prompt = f"""
{context_instruction}
Extract ONLY the agenda items from the meeting note.
IMPORTANT: Unless explicitly mentioned differently, use these French defaults:
- "I- Relecture du Compte Rendu"
- "II- Récapitulatif des Résolutions et des Sanctions"
- "III- Revue d'activités"
- "IV- Faits Saillants"
- "V- Divers"

Return as a list in this EXACT format (no other text):
{{"agenda_items": ["I- Relecture du Compte Rendu", "II- Récapitulatif des Résolutions et des Sanctions", "III- Revue d'activités", "IV- Faits Saillants", "V- Divers"]}}

Meeting Note:
{text}
"""
            agenda_response = self._make_api_call(agenda_prompt)
            agenda_data = safe_parse_json(agenda_response, {"agenda_items": [
                "I- Relecture du Compte Rendu",
                "II- Récapitulatif des Résolutions et des Sanctions", 
                "III- Revue d'activités",
                "IV- Faits Saillants",
                "V- Divers"
            ]})

            # 3. Extract activities review with context (most important for continuity)
            activities_prompt = f"""
{context_instruction}
Extract ONLY the activities review from the meeting note.
CRITICAL: Create entries for ALL expected team members: {', '.join(expected_members)}
- If a member is not mentioned, use "RAS" for activities, results, and perspectives
- Use the historical context to understand continuing activities

Return as a list of objects in this EXACT format (no other text):
{{"activities_review": [
    {{"actor": "name", "dossier": "text", "activities": "text", "results": "text", "perspectives": "text"}}
]}}

Meeting Note:
{text}
"""
            activities_response = self._make_api_call(activities_prompt)
            activities_data = safe_parse_json(activities_response, {"activities_review": []})

            # Ensure all expected members are included
            mentioned_actors = {activity.get("actor", "") for activity in activities_data.get("activities_review", [])}
            complete_activities = list(activities_data.get("activities_review", []))
            
            for member in expected_members:
                if member not in mentioned_actors:
                    complete_activities.append({
                        "actor": member,
                        "dossier": "Non spécifié",
                        "activities": "RAS",
                        "results": "RAS",
                        "perspectives": "RAS"
                    })
            
            activities_data["activities_review"] = complete_activities

            # 4. Extract resolutions with context
            resolutions_prompt = f"""
{context_instruction}
Extract ONLY the resolutions from the meeting note.
Consider any previous resolutions that might be referenced or updated.
Return as a list of objects in this EXACT format (no other text):
{{"resolutions_summary": [
    {{"date": "DD/MM/YYYY", "dossier": "text", "resolution": "text", "responsible": "name", "deadline": "DD/MM/YYYY", "status": "text"}}
]}}

Meeting Note:
{text}
"""
            resolutions_response = self._make_api_call(resolutions_prompt)
            resolutions_data = safe_parse_json(resolutions_response, {"resolutions_summary": []})

            # Fix resolution dates using historical data
            fixed_resolutions = []
            for resolution in resolutions_data.get("resolutions_summary", []):
                dossier = resolution.get("dossier", "")
                responsible = resolution.get("responsible", "")
                resolution_text = resolution.get("resolution", "")
                
                # Look up historical date for this resolution
                historical_date = get_historical_resolution_date(
                    dossier, responsible, resolution_text, 
                    context_dir=self.context_dir, exclude_date=meeting_date
                )
                
                # Use historical date if found, otherwise keep LLM-suggested date
                if historical_date:
                    resolution["date"] = historical_date
                    print(f"Updated resolution date for {responsible} ({dossier}): {historical_date}")
                
                fixed_resolutions.append(resolution)
            
            resolutions_data["resolutions_summary"] = fixed_resolutions

            # 5. Extract sanctions with context
            sanctions_prompt = f"""
{context_instruction}
Extract ONLY the sanctions from the meeting note.
Consider previous sanctions for reference and context.
Return as a list of objects in this EXACT format (no other text):
{{"sanctions_summary": [
    {{"name": "text", "reason": "text", "amount": "number", "date": "DD/MM/YYYY", "status": "text"}}
]}}

Meeting Note:
{text}
"""
            sanctions_response = self._make_api_call(sanctions_prompt)
            sanctions_data = safe_parse_json(sanctions_response, {"sanctions_summary": []})

            # 6. Extract miscellaneous items with context
            misc_prompt = f"""
{context_instruction}
Extract ONLY miscellaneous items and key highlights from the meeting note.
Return in this EXACT format (no other text):
{{"key_highlights": ["item1", "item2"], "miscellaneous": ["item1", "item2"]}}

Meeting Note:
{text}
"""
            misc_response = self._make_api_call(misc_prompt)
            misc_data = safe_parse_json(misc_response, {"key_highlights": [], "miscellaneous": []})

            # Combine all data with proper error handling
            combined_data = {
                "meeting_metadata": {
                    "date": meeting_date,
                    "title": meeting_title
                },
                "attendance": attendance_data,
                "agenda_items": agenda_data.get("agenda_items", [
                    "I- Relecture du Compte Rendu",
                    "II- Récapitulatif des Résolutions et des Sanctions",
                    "III- Revue d'activités", 
                    "IV- Faits Saillants",
                    "V- Divers"
                ]),
                "activities_review": activities_data.get("activities_review", []),
                "resolutions_summary": resolutions_data.get("resolutions_summary", []),
                "sanctions_summary": sanctions_data.get("sanctions_summary", []),
                "key_highlights": misc_data.get("key_highlights", []),
                "miscellaneous": misc_data.get("miscellaneous", [])
            }

            # Validate the combined data
            return self.validate_meeting_json(combined_data)

        except Exception as e:
            print(f"\nProcessing Error:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise Exception(f"Failed to extract structured information: {str(e)}")

    def _make_api_call(self, prompt: str) -> str:
        """Make API call to Deepseek and return the response content."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 8000  # Increased from 4000 to handle longer responses
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            content = response.json()["choices"][0]["message"]["content"].strip()
            print(f"Raw API response length: {len(content)} characters")
            
            # Check if response was truncated
            finish_reason = response.json()["choices"][0].get("finish_reason", "")
            if finish_reason == "length":
                print("WARNING: Response was truncated due to token limit!")
            
            # Clean the JSON response
            cleaned_json = clean_json_response(content)
            if not cleaned_json:
                raise Exception("Failed to extract valid JSON from response")
            
            print(f"Successfully cleaned JSON")
            return cleaned_json
            
        except Exception as e:
            print(f"\nAPI Call Error:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise Exception(f"API call failed: {str(e)}")

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

def get_historical_resolution_date(dossier, responsible, resolution_text, context_dir="processed_meetings", exclude_date=None):
    """
    Look up the original assignment date for a resolution/dossier from historical meetings.
    
    Args:
        dossier: Dossier name to match
        responsible: Responsible person to match  
        resolution_text: Resolution text to match
        context_dir: Directory containing historical meeting JSONs
        exclude_date: Date to exclude from search (current meeting)
        
    Returns:
        Original date if found, or None if not found
    """
    try:
        if not os.path.exists(context_dir):
            return None
        
        # Get all JSON files in the context directory
        json_files = []
        for file in os.listdir(context_dir):
            if file.endswith('.json'):
                filepath = os.path.join(context_dir, file)
                
                # Skip files that match the exclude_date
                if exclude_date:
                    exclude_date_formatted = exclude_date.replace("/", "-")
                    if exclude_date_formatted in file:
                        continue
                
                # Get file modification time for sorting
                mtime = os.path.getmtime(filepath)
                json_files.append((mtime, filepath, file))
        
        if not json_files:
            return None
        
        # Sort by modification time (most recent first)
        json_files.sort(key=lambda x: x[0], reverse=True)
        
        # Search through historical meetings for matching resolution
        for mtime, filepath, filename in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    meeting_data = json.load(f)
                
                historical_resolutions = meeting_data.get("resolutions_summary", [])
                
                for resolution in historical_resolutions:
                    hist_dossier = resolution.get("dossier", "").strip().lower()
                    hist_responsible = resolution.get("responsible", "").strip().lower()
                    hist_resolution = resolution.get("resolution", "").strip().lower()
                    hist_date = resolution.get("date", "").strip()
                    
                    # Normalize inputs for comparison
                    search_dossier = dossier.strip().lower()
                    search_responsible = responsible.strip().lower()
                    search_resolution = resolution_text.strip().lower()
                    
                    # Try different matching strategies
                    matches = []
                    
                    # 1. Exact dossier + responsible match
                    if hist_dossier == search_dossier and hist_responsible == search_responsible:
                        matches.append(("exact_dossier_responsible", hist_date))
                    
                    # 2. Partial dossier match + responsible match  
                    if (search_dossier in hist_dossier or hist_dossier in search_dossier) and hist_responsible == search_responsible:
                        matches.append(("partial_dossier_responsible", hist_date))
                    
                    # 3. Responsible match + similar resolution text
                    if hist_responsible == search_responsible and len(search_resolution) > 10:
                        # Check if significant words overlap
                        search_words = set(search_resolution.split())
                        hist_words = set(hist_resolution.split())
                        overlap = len(search_words.intersection(hist_words))
                        if overlap >= min(3, len(search_words) // 2):
                            matches.append(("responsible_resolution", hist_date))
                    
                    # 4. Strong dossier match (even if responsible is different)
                    if hist_dossier == search_dossier and hist_date:
                        matches.append(("dossier_only", hist_date))
                    
                    # Return the first (best) match found
                    if matches:
                        match_type, found_date = matches[0]
                        if found_date:  # Only return if date is not empty
                            print(f"Found historical date for '{dossier}' ({responsible}): {found_date} (match: {match_type})")
                            return found_date
                
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                continue
        
        return None
        
    except Exception as e:
        print(f"Error looking up historical resolution date: {str(e)}")
        return None 