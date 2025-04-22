import streamlit as st
import os
import tempfile
from datetime import datetime
import io
import whisper
import requests
import json
from pydub import AudioSegment


st.set_page_config(page_title="Meeting Transcription Tool", page_icon=":microphone:", layout="wide")



def transcribe_audio(audio_file, file_extension):
    """Transcribe the uploaded audio file to text using OpenAI whisper API"""

    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name


    try: 
        # Loading Whisper model (base model)
        model = whisper.load_model("base")

        #Transcription...
        result = model.transcribe(temp_audio_path)
        text = result["text"]

    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        text = "Error transcribing audio"

    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)

    return text


def format_meeting_notes_with_llm(transcript, meeting_title, date, attendees, template, api_key, action_items = None):
    """Format the transcript into company meeting notes template using Deepseek API"""

    if action_items is None:
        action_items = []

    # Prepare action items as a string if they exist
    action_items_text = ""
    if action_items:
        for idx, item in enumerate(action_items, 1):
            action_items_text += f"{idx}. {item}\n"

    else: 
        action_items_text = "No action items were captured during the meeting."

    # The prompt for the LLM
    prompt = f"""
    You are a professional meeting notes formatter. Format the following meeting transcript according to the provided template.
    
    Meeting Details:
    - Meeting Title: {meeting_title}
    - Date: {date}
    - Attendees: {attendees}
    - Action Items:
    {action_items_text}

    Meeting Transcript: 
    {transcript}

    Meeting Template:
    {template}

   Please format the meeting transcript according to this template, making it professional and well-organized.
"""
    
    try: 
        # Call Deepseek API
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
            headers = headers, 
            json = payload
        )

        if response.status_code == 200:
            result = response.json()
            formatted_notes = result["choices"][0]["message"]["content"].strip()
            return formatted_notes
        else: 
            st.error(f"Error formatting meeting notes: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"Error Formatting Notes: {e}")
        return format_meeting_notes_fallback(transcript, meeting_title, date, attendees, template, action_items)
    

def format_meeting_notes_fallback(transcript, meeting_title, date, attendees, template, action_items):
    """Fallback formatter if the LLM call fails"""

    # Format the template
    meeting_notes = f"""
    # {meeting_title}
    # {date}
    # Attendees: {attendees}
    # Action Items:
    {action_items}
    """

    # Add action items if they exist
    if action_items:
       for idx, item in enumerate(action_items, 1):
           meeting_notes += f"{idx}. {item}\n"

    else: 
        meeting_notes += "No action items were captured during the meeting."

    return meeting_notes


def main():
    st.title("Meeting Audio Transcription Tool")

    # Inititalize state for API key and template

    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    if 'template' not in st.session_state:
        st.session_state.template = """

# [MEETING TITLE]
**Date:** [DATE]
**Attendees:** [ATTENDEES]

## Summary
[BRIEF SUMMARY OF KEY POINTS]

## Discussion Points
[MAIN DISCUSSION POINTS EXTRACTED FROM TRANSCRIPT]

## Decisions
[KEY DECISIONS MADE]

## Action Items
[LIST OF ACTION ITEMS WITH RESPONSIBLE PERSONS]

## Next Steps
[FOLLOW-UP ACTIONS OR NEXT MEETING]
"""

#Sidebar for uploading and options

    with st.sidebar:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])

        # Model options
        st.header("Transcription Options")
        whisper_model = st.selectbox(
            "Whisper Model Size", 
            ["tiny", "base", "small", "medium", "large"], 
            index = 1, 
            help = "Larger models are more accurate but slower"
      )
        
        # API settings

        st.header("Deepseek API Settings")
        api_key = st.text_input("Deepseek API Key", 
                                value = st.session_state.api_key, 
                                type = "password", 
                                help = "Enter your Deepseek API key")
        
        #Save API key to session state
        st.session_state.api_key = api_key

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"File uploaded: {uploaded_file.name}")

            # Button to start transcription
            transcribe_button = st.button("Transcribe Audio")

            # Main content area with two columns
            if uploaded_file is not None: 
                col1, col2 = st.columns(2)

                with col1: 
                    st.header("Meeting Details")
                    meeting_title = st.text_input("Meeting Title")
           
            
    # [MEETING TITLE]
        
        


    
    