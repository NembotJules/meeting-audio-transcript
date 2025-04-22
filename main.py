import streamlit as st
import os
import tempfile
from datetime import datetime
import whisper
import requests




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
    
    # Initialize state for API key and template
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
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac"])
        
        # Model options
        st.header("Transcription Options")
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower"
        )
        
        # API settings
        st.header("Deepseek API Settings")
        api_key = st.text_input("Deepseek API Key", 
                                value=st.session_state.api_key, 
                                type="password",
                                help="Enter your Deepseek API key")
        # Save API key to session state
        st.session_state.api_key = api_key
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"File uploaded: {uploaded_file.name}")
            
            # Add a button to start transcription
            transcribe_button = st.button("Transcribe Audio")
    
    # Main content area with two columns
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Meeting Details")
            meeting_title = st.text_input("Meeting Title")
            meeting_date = st.date_input("Meeting Date", datetime.now())
            attendees = st.text_area("Attendees (comma separated)")
            
            # Template customization
            st.subheader("Meeting Notes Template")
            template = st.text_area("Customize Template", 
                                   value=st.session_state.template, 
                                   height=250)
            # Save template to session state
            st.session_state.template = template
            
            # Container for dynamic action items
            st.subheader("Action Items")
            action_items_container = st.container()
            
            # Initialize session state for action items if not already
            if 'action_items' not in st.session_state:
                st.session_state.action_items = [""]
                
            # Display all current action items
            with action_items_container:
                new_action_items = []
                
                for i, item in enumerate(st.session_state.action_items):
                    # For each action item, create a row with text input and delete button
                    cols = st.columns([0.9, 0.1])
                    with cols[0]:
                        new_item = st.text_input(f"Item {i+1}", item, key=f"item_{i}")
                    with cols[1]:
                        if st.button("𝗫", key=f"del_{i}"):
                            pass  # We'll handle deletion by not adding to new list
                        else:
                            new_action_items.append(new_item)
                
                # Update session state with filtered list (handles deletions)
                st.session_state.action_items = new_action_items if new_action_items else [""]
                
            # Button to add a new action item
            if st.button("Add Action Item"):
                st.session_state.action_items.append("")
                st.rerun()  # Force refresh to show the new field
        
        with col2:
            st.header("Transcription & Output")
            
            if transcribe_button:
                with st.spinner("Transcribing audio with OpenAI Whisper..."):
                    transcript = transcribe_audio(uploaded_file, file_extension)
                
                if transcript:
                    st.success("Transcription complete!")
                    
                    # Store the transcript in session state
                    st.session_state.transcript = transcript
                    
                    # Display transcription
                    st.subheader("Raw Transcript")
                    st.text_area("Edit if needed:", transcript, height=200, key="edited_transcript")
                    
                    # Format notes button
                    if st.button("Format Meeting Notes"):
                        # Use edited transcript
                        edited_transcript = st.session_state.get("edited_transcript", transcript)
                        
                        # Filter out empty action items
                        action_items = [item for item in st.session_state.action_items if item.strip()]
                        
                        if st.session_state.api_key:
                            with st.spinner("Formatting with Deepseek LLM..."):
                                # Format using the LLM
                                formatted_notes = format_meeting_notes_with_llm(
                                    edited_transcript,
                                    meeting_title,
                                    meeting_date.strftime("%B %d, %Y"),
                                    attendees,
                                    st.session_state.template,
                                    st.session_state.api_key,
                                    action_items
                                )
                        else:
                            st.warning("No Deepseek API key provided. Using fallback formatter.")
                            # Format using the fallback formatter
                            formatted_notes = format_meeting_notes_fallback(
                                edited_transcript,
                                meeting_title,
                                meeting_date.strftime("%B %d, %Y"),
                                attendees,
                                action_items
                            )
                        
                        # Store the formatted notes in session state
                        st.session_state.formatted_notes = formatted_notes
                        
                        # Display the formatted notes
                        st.subheader("Formatted Meeting Notes")
                        st.text_area("Preview:", formatted_notes, height=300)
                        
                        # Create a download button for the formatted notes
                        st.download_button(
                            label="Download Meeting Notes",
                            data=formatted_notes,
                            file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.md",
                            mime="text/markdown"
                        )
            
            # If we have already transcribed, show the results
            elif hasattr(st.session_state, 'transcript'):
                # Display transcription
                st.subheader("Raw Transcript")
                st.text_area("Edit if needed:", st.session_state.transcript, height=200, key="edited_transcript")
                
                # Format notes button
                if st.button("Format Meeting Notes"):
                    # Use edited transcript
                    edited_transcript = st.session_state.get("edited_transcript", st.session_state.transcript)
                    
                    # Filter out empty action items
                    action_items = [item for item in st.session_state.action_items if item.strip()]
                    
                    if st.session_state.api_key:
                        with st.spinner("Formatting with Deepseek LLM..."):
                            # Format using the LLM
                            formatted_notes = format_meeting_notes_with_llm(
                                edited_transcript,
                                meeting_title,
                                meeting_date.strftime("%B %d, %Y"),
                                attendees,
                                st.session_state.template,
                                st.session_state.api_key,
                                action_items
                            )
                    else:
                        st.warning("No Deepseek API key provided. Using fallback formatter.")
                        # Format using the fallback formatter
                        formatted_notes = format_meeting_notes_fallback(
                            edited_transcript,
                            meeting_title,
                            meeting_date.strftime("%B %d, %Y"),
                            attendees,
                            action_items
                        )
                    
                    # Store the formatted notes in session state
                    st.session_state.formatted_notes = formatted_notes
                    
                    # Display the formatted notes
                    st.subheader("Formatted Meeting Notes")
                    st.text_area("Preview:", formatted_notes, height=300)
                    
                    # Create a download button for the formatted notes
                    st.download_button(
                        label="Download Meeting Notes",
                        data=formatted_notes,
                        file_name=f"{meeting_title}_{meeting_date.strftime('%Y-%m-%d')}_notes.md",
                        mime="text/markdown"
                    )
            
            # If we already have formatted notes, show them
            elif hasattr(st.session_state, 'formatted_notes'):
                st.subheader("Formatted Meeting Notes")
                st.text_area("Preview:", st.session_state.formatted_notes, height=300)
                
                # Create a download button for the formatted notes
                st.download_button(
                    label="Download Meeting Notes",
                    data=st.session_state.formatted_notes,
                    file_name=f"meeting_notes_{datetime.now().strftime('%Y-%m-%d')}.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()