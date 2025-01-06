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

/* Chat messages */
.stChatMessage {
    background: var(--input-background);
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Chat container */
.chat-container {
    margin-bottom: 140px;  /* Space for input area */
}

/* Input container */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--background-color);
    border-top: 1px solid var(--border-color);
    padding: 1rem;
    z-index: 1000;
}

/* Style Streamlit's chat input */
.stChatInput {
    margin-bottom: 0 !important;
}

.stChatInput > div {
    padding: 0 !important;
}

/* Style file uploader */
[data-testid="stFileUploader"] {
    padding: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stFileUploader"] > div {
    padding: 0.5rem !important;
}

/* Ensure content doesn't get hidden */
.main .block-container {
    padding-bottom: 160px;
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
    if 'clear_files' not in st.session_state:
        st.session_state.clear_files = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = "file_uploader_0"
    # Initialize model settings
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.95
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 40
    if 'max_output_tokens' not in st.session_state:
        st.session_state.max_output_tokens = 2048

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

def prepare_messages(text_input: str, files_data: list = None) -> list:
    """Prepare messages for the Gemini model."""
    parts = []
    
    # Add files content if present
    if files_data:
        for file_data in files_data:
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

def model_settings_sidebar():
    """Sidebar for model settings."""
    with st.sidebar:
        st.title("Model Settings")
        
        # Temperature slider
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make the output more creative but less focused"
        )
        
        # Top P slider
        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.05,
            help="Controls diversity via nucleus sampling"
        )
        
        # Top K slider
        st.session_state.top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=st.session_state.top_k,
            step=1,
            help="Controls diversity via top-k sampling"
        )
        
        # Max Output Tokens slider
        st.session_state.max_output_tokens = st.slider(
            "Max Output Length",
            min_value=256,
            max_value=4096,
            value=st.session_state.max_output_tokens,
            step=256,
            help="Maximum length of the response"
        )
        
        # Add a divider
        st.divider()
        
        # Display current settings
        st.write("Current Settings:")
        settings = {
            "Temperature": f"{st.session_state.temperature:.1f}",
            "Top P": f"{st.session_state.top_p:.2f}",
            "Top K": st.session_state.top_k,
            "Max Tokens": st.session_state.max_output_tokens
        }
        for key, value in settings.items():
            st.text(f"{key}: {value}")

def chat_interface():
    """Modern chat interface with minimal design."""
    st.title("AI Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "clear_files" not in st.session_state:
        st.session_state.clear_files = False
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = "file_uploader_0"
    
    # Show model settings sidebar
    model_settings_sidebar()
    
    # Container for chat history
    with st.container():
        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "files_data" in message:
                    for file_data in message["files_data"]:
                        if file_data["type"] == "image":
                            st.image(file_data["display_data"])
                        elif file_data["type"] == "text":
                            with st.expander(f"📄 {file_data['name']}"):
                                st.text(file_data["data"])
    
    # Fixed input container at bottom
    with st.container():
        # File uploader (multiple files)
        uploaded_files = st.file_uploader(
            "📎 Attach files",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
            accept_multiple_files=True,
            key=st.session_state.file_uploader_key
        )
        
        # Chat input
        prompt = st.chat_input("Message")
        
        # Handle input
        if prompt:
            # Process uploaded files
            files_data = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_data = process_file_upload(uploaded_file)
                    if file_data:
                        files_data.append(file_data)
                        st.toast(f"📎 {file_data['name']} attached")
            
            # Display user message
            st.chat_message("user").write(prompt)
            if files_data:
                for file_data in files_data:
                    if file_data["type"] == "image":
                        st.chat_message("user").image(file_data["display_data"])
                    else:
                        st.chat_message("user").write(f"📎 Attached: {file_data['name']}")
            
            # Add to history
            user_message = {
                "role": "user",
                "content": prompt,
                "files_data": files_data
            } if files_data else {
                "role": "user",
                "content": prompt
            }
            st.session_state.messages.append(user_message)
            
            try:
                # Prepare messages with all files
                parts = prepare_messages(prompt, files_data)
                
                # Generate response using current settings
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(
                    parts,
                    generation_config={
                        'temperature': st.session_state.temperature,
                        'top_p': st.session_state.top_p,
                        'top_k': st.session_state.top_k,
                        'max_output_tokens': st.session_state.max_output_tokens,
                    }
                )
                
                # Display response
                with st.chat_message("assistant"):
                    st.write(response.text)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text
                })
                
                # Update file uploader key to clear files
                st.session_state.file_uploader_key = f"file_uploader_{int(time.time())}"
                
                # Clear input
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