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

# Custom CSS for modern chat interface
st.markdown("""
<style>
/* Modern dark theme */
:root {
    --background-color: #1a1a1a;
    --input-background: #2d2d2d;
    --text-color: #ffffff;
    --border-color: #404040;
}

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Chat container */
.main > div:first-child {
    padding-bottom: 80px;  /* Make room for input container */
}

/* Input container */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--background-color);
    padding: 1rem;
    border-top: 1px solid var(--border-color);
    z-index: 1000;
}

/* Chat messages */
.stChatMessage {
    background-color: var(--input-background);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Chat input */
.stChatInput {
    border-radius: 20px !important;
    border: 1px solid var(--border-color) !important;
    background-color: var(--input-background) !important;
    color: var(--text-color) !important;
    padding-right: 50px !important;
}

/* File uploader */
.stFileUploader {
    position: absolute;
    right: 60px;
    bottom: 10px;
    width: 40px;
    overflow: hidden;
}

.stFileUploader > div > div {
    opacity: 0;
    position: absolute;
}

.stFileUploader::before {
    content: "📎";
    position: absolute;
    right: 10px;
    bottom: 10px;
    font-size: 20px;
    cursor: pointer;
    color: var(--text-color);
    opacity: 0.7;
    transition: opacity 0.2s;
}

.stFileUploader:hover::before {
    opacity: 1;
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
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return {
                'type': 'image',
                'data': img_byte_arr,
                'display_data': img,  # Keep original PIL Image for display
                'mime_type': 'image/jpeg',  # Always use JPEG for consistency
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
            parts.append({
                'inline_data': {
                    'mime_type': file_data['mime_type'],
                    'data': base64.b64encode(file_data['data']).decode('utf-8')
                }
            })
        elif file_data['type'] == 'text':
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
    """Modern chat interface with bottom input."""
    st.title("AI Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    chat_container = st.container()
    
    # Fixed input container at bottom
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Two columns: file upload and chat input
        cols = st.columns([0.1, 0.9])
        
        with cols[0]:
            uploaded_file = st.file_uploader(
                "",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
                label_visibility="collapsed",
                key="chat_file_uploader"
            )
        
        with cols[1]:
            prompt = st.chat_input("Message")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat history in the main container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "file_data" in message:
                    if message["file_data"]["type"] == "image":
                        st.image(message["file_data"]["display_data"])
                    elif message["file_data"]["type"] == "text":
                        with st.expander("📄 View File Content"):
                            st.text(message["file_data"]["data"])
    
    # Handle input
    if prompt:
        # Process any uploaded file
        file_data = None
        if uploaded_file:
            file_data = process_file_upload(uploaded_file)
            if file_data:
                st.toast(f"📎 {file_data['name']} attached")
        
        # Add user message to chat
        st.chat_message("user").write(prompt)
        if file_data:
            if file_data["type"] == "image":
                st.chat_message("user").image(file_data["display_data"])
            else:
                st.chat_message("user").write(f"📎 Attached: {file_data['name']}")
        
        # Add to message history
        user_message = {
            "role": "user",
            "content": prompt
        }
        if file_data:
            user_message["file_data"] = file_data
        st.session_state.messages.append(user_message)
        
        try:
            # Prepare messages for the model
            parts = prepare_messages(prompt, file_data)
            
            # Generate response using Gemini
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(
                parts,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response.text)
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text
            })
            
            # Rerun to clear the input
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if st.checkbox("Show detailed error"):
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