import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from agents.technical_expert import ResponseAgent
import time
import speech_recognition as sr
from gtts import gTTS
import io
import base64
from PIL import Image
import pandas as pd
from fpdf import FPDF
import tempfile

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Custom CSS for animations and styling
st.markdown("""
<style>
/* Custom container for chat messages */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-out;
}

.user-message {
    background-color: #2e3d49;
    border-left: 5px solid #4CAF50;
}

.bot-message {
    background-color: #1e2a35;
    border-left: 5px solid #2196F3;
}

/* Suggestion buttons */
.stButton button {
    margin: 0.2rem;
    border-radius: 1rem;
    background-color: rgba(33, 150, 243, 0.1);
    border: 1px solid #2196F3;
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: rgba(33, 150, 243, 0.2);
    transform: translateY(-1px);
}

/* Media upload area */
.upload-area {
    border: 2px dashed #2196F3;
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
    margin: 1rem 0;
}

/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = ResponseAgent()
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = False

def convert_to_pdf(content: str) -> bytes:
    """Convert content to PDF format."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content)
    return pdf.output(dest='S').encode('latin-1')

def convert_to_spreadsheet(content: str) -> bytes:
    """Convert content to spreadsheet format."""
    # Simple conversion - could be enhanced based on content structure
    df = pd.DataFrame([content.split('\n')])
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()

def get_download_link(content: str, format: str) -> str:
    """Generate download link for content in specified format."""
    if format == "pdf":
        b64 = base64.b64encode(convert_to_pdf(content)).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="response.pdf">Download PDF</a>'
    elif format == "markdown":
        b64 = base64.b64encode(content.encode()).decode()
        return f'<a href="data:text/markdown;base64,{b64}" download="response.md">Download Markdown</a>'
    elif format == "txt":
        b64 = base64.b64encode(content.encode()).decode()
        return f'<a href="data:text/plain;base64,{b64}" download="response.txt">Download Text</a>'
    elif format == "spreadsheet":
        b64 = base64.b64encode(convert_to_spreadsheet(content)).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="response.xlsx">Download Spreadsheet</a>'

def handle_voice_input() -> str:
    """Handle voice input and convert to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Click 'Stop' when finished.")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except Exception as e:
            st.error(f"Error processing voice input: {str(e)}")
            return ""

def generate_suggestions(content: str) -> list:
    """Generate follow-up suggestions based on content."""
    # This could be enhanced with more sophisticated suggestion generation
    prompt = f"""
    Based on this content, generate 4 follow-up questions or prompts in these categories:
    1. Dig deeper into a specific aspect
    2. Explore related topics
    3. Continue the current discussion
    4. Make an unexpected connection
    
    Content: {content}
    """
    
    response = st.session_state.agent.generate_response(prompt, stream=False)
    suggestions = [s.strip() for s in response.split('\n') if s.strip()]
    return suggestions[:4]

def display_chat_message(message: str, is_user: bool, media_content: dict = None):
    """Display a chat message with animation and optional media content."""
    message_type = "user" if is_user else "bot"
    
    # Display media content if present
    if media_content:
        if media_content.get('image'):
            st.image(media_content['image'])
        if media_content.get('video'):
            st.video(media_content['video'])
    
    # Display message
    st.markdown(
        f'<div class="chat-message {message_type}-message">{message}</div>',
        unsafe_allow_html=True
    )
    
    # Display download options and suggestions for bot messages
    if not is_user:
        # Download options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(get_download_link(message, "pdf"), unsafe_allow_html=True)
        with col2:
            st.markdown(get_download_link(message, "markdown"), unsafe_allow_html=True)
        with col3:
            st.markdown(get_download_link(message, "txt"), unsafe_allow_html=True)
        with col4:
            st.markdown(get_download_link(message, "spreadsheet"), unsafe_allow_html=True)
        
        # Generate and display suggestions
        suggestions = generate_suggestions(message)
        st.write("Follow-up suggestions:")
        cols = st.columns(len(suggestions))
        for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    return suggestion
    
    return None

def stream_callback(chunk: str):
    """Callback function for streaming responses."""
    if 'current_response' not in st.session_state:
        st.session_state.current_response = ''
    
    st.session_state.current_response += chunk
    
    if 'message_placeholder' in st.session_state:
        st.session_state.message_placeholder.markdown(
            f'<div class="chat-message bot-message">{st.session_state.current_response}</div>',
            unsafe_allow_html=True
        )

def main():
    st.title("AI Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar controls
    with st.sidebar:
        st.write("Input Options")
        voice_input = st.toggle("Voice Input", value=st.session_state.voice_input)
        
        st.write("Upload Media")
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "gif"])
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "webm", "ogg"])
    
    # Display chat history
    for message in st.session_state.messages:
        suggestion = display_chat_message(
            message['content'],
            message['is_user'],
            message.get('media_content')
        )
        if suggestion:
            user_input = suggestion
            break
    
    # Handle voice input
    if voice_input and not st.session_state.voice_input:
        user_input = handle_voice_input()
        st.session_state.voice_input = False
    else:
        # Chat input
        user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Prepare media content
        media_content = {}
        if uploaded_image:
            media_content['image'] = Image.open(uploaded_image)
        if uploaded_video:
            media_content['video'] = uploaded_video.read()
        
        # Display user message
        display_chat_message(user_input, True, media_content)
        
        # Add to chat history
        st.session_state.messages.append({
            'content': user_input,
            'is_user': True,
            'media_content': media_content
        })
        
        # Create placeholder for bot response
        st.session_state.message_placeholder = st.empty()
        
        # Reset current response
        st.session_state.current_response = ''
        
        # Generate response
        response = st.session_state.agent.generate_response(
            user_input,
            stream=True,
            stream_callback=stream_callback
        )
        
        # Add response to chat history
        st.session_state.messages.append({
            'content': response,
            'is_user': False
        })

if __name__ == "__main__":
    main() 