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
            # 1. Extract attendance
            attendance_prompt = f"""
            Extract ONLY the attendance information from the meeting note.
            Return as two lists in this EXACT format (no other text):
            {{"present": ["name1", "name2"], "absent": ["name3", "name4"]}}

            Meeting Note:
            {text}
            """
            attendance_response = self._make_api_call(attendance_prompt)
            attendance_data = safe_parse_json(attendance_response, {"present": [], "absent": []})

            # 2. Extract agenda items
            agenda_prompt = f"""
            Extract ONLY the agenda items from the meeting note.
            Return as a list in this EXACT format (no other text):
            {{"agenda_items": ["item1", "item2", "item3"]}}

            Meeting Note:
            {text}
            """
            agenda_response = self._make_api_call(agenda_prompt)
            agenda_data = safe_parse_json(agenda_response, {"agenda_items": []})

            # 3. Extract activities review
            activities_prompt = f"""
            Extract ONLY the activities review from the meeting note.
            Return as a list of objects in this EXACT format (no other text):
            {{"activities_review": [
                {{"actor": "name", "dossier": "text", "activities": "text", "results": "text", "perspectives": "text"}}
            ]}}

            Meeting Note:
            {text}
            """
            activities_response = self._make_api_call(activities_prompt)
            activities_data = safe_parse_json(activities_response, {"activities_review": []})

            # 4. Extract resolutions
            resolutions_prompt = f"""
            Extract ONLY the resolutions from the meeting note.
            Return as a list of objects in this EXACT format (no other text):
            {{"resolutions_summary": [
                {{"date": "DD/MM/YYYY", "dossier": "text", "resolution": "text", "responsible": "name", "deadline": "DD/MM/YYYY", "status": "text"}}
            ]}}

            Meeting Note:
            {text}
            """
            resolutions_response = self._make_api_call(resolutions_prompt)
            resolutions_data = safe_parse_json(resolutions_response, {"resolutions_summary": []})

            # 5. Extract sanctions
            sanctions_prompt = f"""
            Extract ONLY the sanctions from the meeting note.
            Return as a list of objects in this EXACT format (no other text):
            {{"sanctions_summary": [
                {{"name": "text", "reason": "text", "amount": number, "date": "DD/MM/YYYY", "status": "text"}}
            ]}}

            Meeting Note:
            {text}
            """
            sanctions_response = self._make_api_call(sanctions_prompt)
            sanctions_data = safe_parse_json(sanctions_response, {"sanctions_summary": []})

            # 6. Extract miscellaneous items
            misc_prompt = f"""
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
                "agenda_items": agenda_data.get("agenda_items", []),
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