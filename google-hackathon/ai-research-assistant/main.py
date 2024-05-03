import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
import os
#from dotenv import load_dotenv
import google.generativeai as genai

# Initialize and configure the environment
fetched_api_key = os.getenv("API_KEY")
google_credentials = st.secrets["GOOGLE_CREDENTIALS"]
genai.configure(api_key=fetched_api_key)
model = genai.GenerativeModel("gemini-pro")


def process_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''.join([page.extract_text() or " " for page in pdf.pages])
    except Exception as e:
        st.error(f"An error occurred with pdfplumber: {e}")

    if not text:  # Fallback to PyPDF2 if pdfplumber fails or returns empty text
        try:
            uploaded_file.seek(0)  # Reset file pointer for PyPDF2
            reader = PdfReader(uploaded_file)
            text = ''.join([page.extract_text() or " " for page in reader.pages])
        except Exception as e:
            st.error(f"An error occurred with PyPDF2: {e}")

    if not text:
        st.error("Failed to extract text from the PDF.")
        return None

    return text


def send_message_to_genai(prompt, pdf_text):
    """Sends a message to the Gemini model, including the PDF text only on the first request."""
    full_prompt = prompt  # Default to just the prompt

    if 'pdf_sent' not in st.session_state:
        if pdf_text:  # Make sure there is PDF text to include
            full_prompt = f"{pdf_text}\n{prompt}"
            st.session_state.pdf_text = pdf_text
            st.session_state.pdf_sent = True  # Mark the PDF text as sent AFTER confirming it's included
        else:
            st.error("PDF text is empty. Sending prompt without PDF context.")

    if 'chat' not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])

    response = st.session_state.chat.send_message(full_prompt)
    return response.text


def handle_chat_interaction(prompt, messages_container):
    """
    Handles sending the prompt to the Gemini model and updating the chat interface with both the prompt and the model's response.

    Parameters:
    - prompt (str): The user's question or a direct summary request.
    - messages_container (streamlit.container): The Streamlit container to display chat messages.
    """
    # Ensure there's text to process/send
    if not prompt.strip():
        st.warning("Please enter some text to send.")
        return

    # Update chat history with user's prompt and display it
    st.session_state.chat_history.append({"sender": "user", "text": prompt})

    # Send the prompt to the Gemini model and get the response
    try:
        with st.spinner("Thinking..."):
            pdf_text = st.session_state.get('pdf_text', '')  # Get the processed PDF text if available
            response_text = send_message_to_genai(prompt, pdf_text)
    except Exception as e:
        response_text = f"An error occurred: {str(e)}"
        st.error(response_text)

    # Update chat history with model's response and display it
    st.session_state.chat_history.append({"sender": "assistant", "text": response_text})
    # Display full chat history
    for message in st.session_state.chat_history:
        if message["sender"] == "user":
            messages_container.chat_message("user").write(message['text'])
        else:  # sender == "assistant"
            messages_container.chat_message("assistant").write(message['text'])


def main():
    st.title("AI Chatbot with PDF Question Answering & Summarization")

    # Initialize or reset chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Handle PDF upload and processing
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file:
        pdf_text = process_pdf(uploaded_file)
        st.session_state.pdf_text = pdf_text if pdf_text else ""

    # Chat interface
    if 'pdf_text' in st.session_state and st.session_state.pdf_text:
        user_prompt = None
        messages = st.container()

        if st.button("Get Summary"):
            user_prompt = "Provide a concise summary of the document."

        # Allows for asking questions after or instead of summary
        custom_prompt = st.chat_input("Or ask your question:")
        if custom_prompt:
            user_prompt = custom_prompt

        if user_prompt:
            handle_chat_interaction(user_prompt, messages)


if __name__ == "__main__":
    main()