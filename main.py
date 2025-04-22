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
    You are a professional meeting notes formatter
