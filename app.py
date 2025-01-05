import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import io
import base64
import pyperclip
from typing import Optional, Dict, Any
from PIL import Image
import mimetypes
import traceback
import sys

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
else:
    genai.configure(api_key=api_key)

# Optional imports with error handling
OPTIONAL_FEATURES = {
    'image': False,
    'spreadsheet': False,
    'pdf': False
}

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
/* Dark theme colors */
:root {
    --background-color: #1a1a1a;
    --input-background: #2d2d2d;
    --text-color: #ffffff;
    --border-color: #404040;
    --hover-color: rgba(255, 255, 255, 0.1);
}

/* Main container */
.main {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Chat input styling */
.stChatInput {
    background-color: var(--input-background) !important;
    border-radius: 20px !important;
    border: 1px solid var(--border-color) !important;
    padding: 10px 20px !important;
}

.stChatInput:focus {
    border-color: var(--border-color) !important;
    box-shadow: none !important;
}

/* Upload button styling */
.upload-button {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-color) !important;
    cursor: pointer !important;
    padding: 8px !important;
    transition: background-color 0.2s !important;
}

.upload-button:hover {
    background-color: var(--hover-color) !important;
}

/* Upload menu styling */
.upload-menu {
    background-color: var(--input-background);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    padding: 8px 0;
}

.upload-option {
    color: var(--text-color);
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.upload-option:hover {
    background-color: var(--hover-color);
}

/* Chat message styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    background-color: var(--input-background);
    border-left: 5px solid var(--border-color);
    animation: fadeIn 0.5s ease-out;
}

.user-message {
    border-left-color: #4CAF50;
}

.bot-message {
    border-left-color: #2196F3;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Animations */
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

def copy_to_clipboard(text: str):
    """Copy text to clipboard."""
    try:
        pyperclip.copy(text)
        st.toast("Copied to clipboard!")
    except Exception as e:
        st.error(f"Error copying to clipboard: {str(e)}")

def display_three_dot_menu(message: str):
    """Display three-dot menu with options."""
    with st.expander("⋮", expanded=False):
        if st.button("Copy", key=f"copy_{hash(message)}"):
            copy_to_clipboard(message)
        
        st.write("Download as:")
        download_formats = [
            ("PDF", "pdf", OPTIONAL_FEATURES['pdf']),
            ("Markdown", "markdown", True),
            ("Text", "txt", True),
            ("Spreadsheet", "spreadsheet", OPTIONAL_FEATURES['spreadsheet'])
        ]
        
        for label, format_type, enabled in download_formats:
            if enabled:
                link = get_download_link(message, format_type)
                if link:
                    st.markdown(link, unsafe_allow_html=True)

def process_file_upload(uploaded_file):
    """Process uploaded file according to Gemini's capabilities."""
    if not uploaded_file:
        return None
        
    try:
        # Get file info
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        # Image handling (Gemini supports these formats)
        if file_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']:
            img = Image.open(uploaded_file)
            # Gemini requires RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return {
                'type': 'image',
                'data': img,
                'mime_type': file_type,
                'name': file_name
            }
            
        # Text file handling
        elif file_type in ['text/plain', 'text/markdown', 'text/csv']:
            text_content = uploaded_file.read().decode('utf-8')
            return {
                'type': 'text',
                'data': text_content,
                'mime_type': file_type,
                'name': file_name
            }
            
        # PDF handling (extract text if possible)
        elif file_type == 'application/pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                return {
                    'type': 'text',
                    'data': text_content,
                    'mime_type': file_type,
                    'name': file_name
                }
            except Exception as e:
                st.error(f"Could not process PDF: {str(e)}")
                return None
                
        else:
            st.warning(f"File type {file_type} is not supported by the model.")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def prepare_messages(text_input: str, file_data: dict = None) -> list:
    """Prepare messages for the Gemini model."""
    parts = []
    
    # Add file content if present
    if file_data:
        if file_data['type'] == 'image':
            # For images, we need to pass the PIL Image object directly
            parts.append({
                'mime_type': file_data['mime_type'],
                'data': file_data['data']  # This is the PIL Image object
            })
        elif file_data['type'] == 'text':
            # For text content, add it as a text part
            parts.append({
                'text': f"Content from {file_data['name']}:\n{file_data['data']}"
            })
    
    # Add user's text input
    if text_input:
        parts.append({
            'text': text_input
        })
        
    return parts

def chat_interface():
    """Main chat interface with file upload and input handling."""
    st.title("AI Assistant")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat history display
    for msg in st.session_state.messages:
        with st.container():
            if msg['role'] == 'user':
                st.markdown(f"🧑 **You**: {msg['content']}")
                if msg.get('file_data'):
                    if msg['file_data']['type'] == 'image':
                        st.image(msg['file_data']['data'], caption=msg['file_data']['name'])
                    elif msg['file_data']['type'] == 'text':
                        with st.expander(f"📄 {msg['file_data']['name']}"):
                            st.text(msg['file_data']['data'])
            else:
                st.markdown(f"🤖 **Assistant**: {msg['content']}")
    
    # Input area
    with st.container():
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file (images, text, PDF)",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
            help="The model can process images, text files, and PDFs"
        )
        
        # Process uploaded file
        file_data = None
        if uploaded_file:
            file_data = process_file_upload(uploaded_file)
            if file_data:
                st.success(f"File '{file_data['name']}' processed successfully!")
        
        # Text input and send button
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            user_input = st.text_input(
                "Your message",
                key="user_input",
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("Send", use_container_width=True)
        
        # Handle input submission
        if send_button or (user_input and len(user_input.strip()) > 0):
            if not user_input and not file_data:
                st.warning("Please enter a message or upload a file.")
                return
            
            # Prepare messages for the model
            parts = prepare_messages(user_input, file_data)
            
            # Add to chat history
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input if user_input else "Uploaded file for analysis",
                'file_data': file_data
            })
            
            try:
                # Select appropriate model based on content
                model_name = 'gemini-pro-vision' if file_data and file_data['type'] == 'image' else 'gemini-pro'
                model = genai.GenerativeModel(model_name)
                
                # Generate response
                response = model.generate_content(
                    parts,  # Pass the parts directly
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.95,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                
                # Add response to chat history
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response.text
                })
                
                # Clear input
                st.session_state.user_input = ""
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.error("Full error details:", exc_info=True)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Configure Gemini
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
    else:
        genai.configure(api_key=api_key)
        chat_interface() 