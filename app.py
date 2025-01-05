import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from agents.technical_expert import ResponseAgent
import time
import io
import base64
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Optional imports with error handling
OPTIONAL_FEATURES = {
    'voice': False,
    'image': False,
    'spreadsheet': False,
    'pdf': False
}

try:
    import speech_recognition as sr
    from gtts import gTTS
    OPTIONAL_FEATURES['voice'] = True
except ImportError:
    st.sidebar.warning("""
    Voice features are disabled. To enable voice features, install required packages:
    ```bash
    pip install SpeechRecognition gTTS PyAudio
    ```
    """)

try:
    from PIL import Image
    OPTIONAL_FEATURES['image'] = True
except ImportError:
    st.sidebar.warning("""
    Image features are disabled. To enable image features, install:
    ```bash
    pip install Pillow
    ```
    """)

try:
    import pandas as pd
    import openpyxl
    OPTIONAL_FEATURES['spreadsheet'] = True
except ImportError:
    st.sidebar.warning("""
    Spreadsheet export is disabled. To enable, install:
    ```bash
    pip install pandas openpyxl
    ```
    """)

try:
    from fpdf import FPDF
    OPTIONAL_FEATURES['pdf'] = True
except ImportError:
    st.sidebar.warning("""
    PDF export is disabled. To enable, install:
    ```bash
    pip install fpdf
    ```
    """)

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

def convert_to_pdf(content: str) -> Optional[bytes]:
    """Convert content to PDF format."""
    if not OPTIONAL_FEATURES['pdf']:
        return None
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content)
    return pdf.output(dest='S').encode('latin-1')

def convert_to_spreadsheet(content: str) -> Optional[bytes]:
    """Convert content to spreadsheet format."""
    if not OPTIONAL_FEATURES['spreadsheet']:
        return None
    
    df = pd.DataFrame([content.split('\n')])
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()

def get_download_link(content: str, format: str) -> Optional[str]:
    """Generate download link for content in specified format."""
    try:
        if format == "pdf" and OPTIONAL_FEATURES['pdf']:
            pdf_content = convert_to_pdf(content)
            if pdf_content:
                b64 = base64.b64encode(pdf_content).decode()
                return f'<a href="data:application/pdf;base64,{b64}" download="response.pdf">Download PDF</a>'
        elif format == "markdown":
            b64 = base64.b64encode(content.encode()).decode()
            return f'<a href="data:text/markdown;base64,{b64}" download="response.md">Download Markdown</a>'
        elif format == "txt":
            b64 = base64.b64encode(content.encode()).decode()
            return f'<a href="data:text/plain;base64,{b64}" download="response.txt">Download Text</a>'
        elif format == "spreadsheet" and OPTIONAL_FEATURES['spreadsheet']:
            spreadsheet_content = convert_to_spreadsheet(content)
            if spreadsheet_content:
                b64 = base64.b64encode(spreadsheet_content).decode()
                return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="response.xlsx">Download Spreadsheet</a>'
    except Exception as e:
        st.error(f"Error generating {format} download: {str(e)}")
    return None

def handle_voice_input() -> str:
    """Handle voice input and convert to text."""
    if not OPTIONAL_FEATURES['voice']:
        st.error("Voice input is not available. Please install required packages.")
        return ""
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Click 'Stop' when finished.")
            audio = r.listen(source)
            return r.recognize_google(audio)
    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")
        return ""

def generate_suggestions(content: str) -> list:
    """Generate follow-up suggestions based on content."""
    prompt = f"""
    Based on this content, generate 4 follow-up questions or prompts in these categories:
    1. Dig deeper into a specific aspect
    2. Explore related topics
    3. Continue the current discussion
    4. Make an unexpected connection
    
    Content: {content}
    """
    
    try:
        response = st.session_state.agent.generate_response(prompt, stream=False)
        suggestions = [s.strip() for s in response.split('\n') if s.strip()]
        return suggestions[:4]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

def display_chat_message(message: str, is_user: bool, media_content: Optional[Dict[str, Any]] = None):
    """Display a chat message with animation and optional media content."""
    message_type = "user" if is_user else "bot"
    
    # Display media content if present and supported
    if media_content and OPTIONAL_FEATURES['image']:
        if media_content.get('image'):
            try:
                st.image(media_content['image'])
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        if media_content.get('video'):
            try:
                st.video(media_content['video'])
            except Exception as e:
                st.error(f"Error displaying video: {str(e)}")
    
    # Display message
    st.markdown(
        f'<div class="chat-message {message_type}-message">{message}</div>',
        unsafe_allow_html=True
    )
    
    # Display download options and suggestions for bot messages
    if not is_user:
        # Download options
        download_formats = [
            ("pdf", OPTIONAL_FEATURES['pdf']),
            ("markdown", True),
            ("txt", True),
            ("spreadsheet", OPTIONAL_FEATURES['spreadsheet'])
        ]
        
        enabled_formats = [f for f, enabled in download_formats if enabled]
        if enabled_formats:
            cols = st.columns(len(enabled_formats))
            for col, format in zip(cols, enabled_formats):
                with col:
                    link = get_download_link(message, format)
                    if link:
                        st.markdown(link, unsafe_allow_html=True)
        
        # Generate and display suggestions
        suggestions = generate_suggestions(message)
        if suggestions:
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
        voice_input = st.toggle("Voice Input", value=st.session_state.voice_input, disabled=not OPTIONAL_FEATURES['voice'])
        
        if OPTIONAL_FEATURES['image']:
            st.write("Upload Media")
            uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "gif"])
            uploaded_video = st.file_uploader("Upload Video", type=["mp4", "webm", "ogg"])
        else:
            uploaded_image = None
            uploaded_video = None
    
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
    if voice_input and not st.session_state.voice_input and OPTIONAL_FEATURES['voice']:
        user_input = handle_voice_input()
        st.session_state.voice_input = False
    else:
        # Chat input
        user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Prepare media content
        media_content = {}
        if OPTIONAL_FEATURES['image']:
            if uploaded_image:
                try:
                    media_content['image'] = Image.open(uploaded_image)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            if uploaded_video:
                try:
                    media_content['video'] = uploaded_video.read()
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        
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
        
        try:
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
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main() 