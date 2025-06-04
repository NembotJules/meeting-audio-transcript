import streamlit as st
from meeting_processor import MeetingProcessor, save_meeting_data
from datetime import datetime
import json
import pandas as pd

st.set_page_config(page_title="Historical Meeting Processor", layout="wide")

def main():
    st.title("Historical Meeting Notes Processor")
    
    # API Keys input
    with st.sidebar:
        mistral_api_key = st.text_input("Mistral API Key", type="password")
        deepseek_api_key = st.text_input("Deepseek API Key", type="password")
        
        st.markdown("---")
        st.markdown("""
        ### Instructions
        1. Enter your API keys
        2. Upload a meeting note (PDF/Image)
        3. Enter meeting date and title
        4. Process the document
        5. Review and save the extracted data
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload meeting note",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Upload a PDF or image file of the meeting note"
        )
        
        # Meeting metadata
        meeting_date = st.date_input(
            "Meeting Date",
            datetime.now(),
            help="Select the date of the meeting"
        )
        meeting_title = st.text_input(
            "Meeting Title",
            "Réunion DRI",
            help="Enter the title of the meeting"
        )
        
        # Process button
        if st.button("Process Meeting Note"):
            if not uploaded_file:
                st.error("Please upload a file first.")
            elif not mistral_api_key or not deepseek_api_key:
                st.error("Please provide both API keys.")
            else:
                with st.spinner("Processing meeting note..."):
                    try:
                        # Initialize processor
                        processor = MeetingProcessor(mistral_api_key, deepseek_api_key)
                        
                        # Process the document
                        extracted_data = processor.process_historical_meeting(
                            uploaded_file,
                            meeting_date.strftime("%d/%m/%Y"),
                            meeting_title
                        )
                        
                        # Store in session state
                        st.session_state.extracted_data = extracted_data
                        st.success("Meeting note processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing meeting note: {str(e)}")
    
    with col2:
        st.header("Output")
        
        if "extracted_data" in st.session_state:
            # Display extracted data in a formatted way
            data = st.session_state.extracted_data
            
            # Attendance
            st.subheader("Attendance")
            st.write("Present:", ", ".join(data["attendance"]["present"]))
            st.write("Absent:", ", ".join(data["attendance"]["absent"]))
            
            # Agenda
            st.subheader("Agenda Items")
            for item in data["agenda_items"]:
                st.write(f"- {item}")
            
            # Activities Review
            st.subheader("Activities Review")
            if data["activities_review"]:
                df_activities = pd.DataFrame(data["activities_review"])
                st.dataframe(df_activities)
            
            # Resolutions
            st.subheader("Resolutions")
            if data["resolutions_summary"]:
                df_resolutions = pd.DataFrame(data["resolutions_summary"])
                st.dataframe(df_resolutions)
            
            # Sanctions
            st.subheader("Sanctions")
            if data["sanctions_summary"]:
                df_sanctions = pd.DataFrame(data["sanctions_summary"])
                st.dataframe(df_sanctions)
            
            # Key Highlights
            st.subheader("Key Highlights")
            for highlight in data["key_highlights"]:
                st.write(f"- {highlight}")
            
            # Miscellaneous
            st.subheader("Miscellaneous")
            for item in data["miscellaneous"]:
                st.write(f"- {item}")
            
            # Save button
            if st.button("Save Extracted Data"):
                try:
                    filepath = save_meeting_data(data, "processed_meetings")
                    st.success(f"Data saved to {filepath}")
                    
                    # Offer download
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_str = f.read()
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=filepath.split("/")[-1],
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Error saving data: {str(e)}")

if __name__ == "__main__":
    main() 