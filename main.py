import streamlit as st
from docx import Document
import requests

def extract_context_from_report(docx_file):
    """Extract the 'Revue d'activités' section from a Word document."""
    try:
        doc = Document(docx_file)
        context = ""
        in_section = False
        for para in doc.paragraphs:
            text = para.text.strip()
            if "Revue d'activités" in text.lower():
                in_section = True
                continue
            if in_section:
                if text and not text.startswith("◆") and "RÉCAPITULATIF" not in text.upper():
                    context += text + "\n"
                else:
                    break
        return context.strip() if context else "Section 'Revue d'activités' not found."
    except Exception as e:
        st.error(f"Error extracting context: {e}")
        return "Error extracting context."

def answer_question_with_context(question, context, api_key):
    """Answer a question based on the extracted context using Deepseek API."""
    if not context:
        return "No context available."
    prompt = f"""
    As an assistant, answer the following question based on the provided context.

    **Context**:
    {context}

    **Question**:
    {question}

    **Answer**:
    """
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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

def extract_info(transcription, title, date, api_key, previous_context):
    """Extract information from transcription using context (placeholder function)."""
    # This is a simplified placeholder. In a full implementation, it would use the API similarly to answer_question_with_context.
    prompt = f"""
    Extract key meeting details from the transcription below, using the previous context if available.

    **Previous Context**:
    {previous_context if previous_context else "No context available."}

    **Transcription**:
    {transcription}

    **Meeting Title**:
    {title}

    **Date**:
    {date}
    """
    # Add API call here in a full implementation
    return "Extracted info (placeholder)"

def main():
    st.title("Meeting Notes Formatter")

    # Sidebar for API key and previous report
    st.sidebar.header("Configuration")
    st.session_state.api_key = st.sidebar.text_input("Deepseek API Key", type="password")

    st.sidebar.header("Previous Context")
    previous_report = st.sidebar.file_uploader("Upload Previous Report (optional)", type=["docx"])
    if previous_report:
        status_text = st.sidebar.empty()
        status_text.text("Extracting context...")
        context = extract_context_from_report(previous_report)
        status_text.text("Context extracted successfully!")
        st.session_state.previous_context = context
    else:
        st.session_state.previous_context = ""

    # Context testing section
    st.sidebar.subheader("Test the Context")
    question = st.sidebar.text_input("Ask a question about the previous context:")
    if st.sidebar.button("Get Answer") and question and 'previous_context' in st.session_state:
        with st.spinner("Generating answer..."):
            answer = answer_question_with_context(question, st.session_state.previous_context, st.session_state.api_key)
        st.sidebar.write("**Answer:**", answer)

    # Main app content (simplified for example)
    transcription = st.text_area("Enter Meeting Transcription")
    meeting_title = st.text_input("Meeting Title")
    meeting_date = st.date_input("Meeting Date")

    if st.button("Format Meeting Notes"):
        if transcription:
            previous_context = st.session_state.get("previous_context", "")
            with st.spinner("Processing..."):
                extracted_info = extract_info(
                    transcription, meeting_title, meeting_date.strftime("%d/%m/%Y"),
                    st.session_state.api_key, previous_context
                )
            st.write("**Extracted Information:**", extracted_info)
        else:
            st.warning("Please enter a transcription.")

if __name__ == "__main__":
    main()