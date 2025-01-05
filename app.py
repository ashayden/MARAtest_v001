import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from agents.technical_expert import ResponseAgent
import time
import io
import base64
import pyperclip
from typing import Optional, Dict, Any
from PIL import Image
import mimetypes

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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

def process_uploaded_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process uploaded file and return content in appropriate format for Gemini."""
    if not uploaded_file:
        return None
    
    try:
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        
        # Handle images
        if mime_type and mime_type.startswith('image/'):
            img = Image.open(uploaded_file)
            # Convert to RGB if necessary (Gemini requires RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return {
                'type': 'image',
                'content': img,
                'mime_type': mime_type
            }
        
        # Handle text files
        elif mime_type and mime_type.startswith('text/'):
            content = uploaded_file.read().decode('utf-8')
            return {
                'type': 'text',
                'content': content,
                'mime_type': mime_type
            }
        
        # Handle PDFs and other documents
        else:
            return {
                'type': 'file',
                'content': uploaded_file.read(),
                'mime_type': mime_type or 'application/octet-stream'
            }
    
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

def prepare_gemini_message(text: str, media_content: Optional[Dict[str, Any]] = None) -> list:
    """Prepare message for Gemini model including any media content."""
    message_parts = []
    
    # Add any media content first
    if media_content:
        if media_content.get('type') == 'image':
            message_parts.append({
                'type': 'image',
                'image': media_content['content']
            })
        elif media_content.get('type') == 'text':
            message_parts.append({
                'type': 'text',
                'text': f"Content from uploaded file:\n{media_content['content']}\n\nUser message:"
            })
    
    # Add the text content
    message_parts.append({
        'type': 'text',
        'text': text
    })
    
    return message_parts

def display_upload_options():
    """Display upload options in a dropdown menu."""
    with st.container():
        # Create a custom container for the upload menu using HTML/CSS
        st.markdown("""
        <style>
        .upload-menu {
            position: relative;
            display: inline-block;
        }
        .upload-button {
            background-color: transparent;
            border: none;
            color: #ffffff;
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
        }
        .upload-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .upload-options {
            display: none;
            position: absolute;
            background-color: #2d2d2d;
            min-width: 160px;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
            border-radius: 8px;
            padding: 8px 0;
        }
        .upload-option {
            color: white;
            padding: 12px 16px;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .upload-option:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .upload-icon {
            width: 20px;
            height: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create columns for the + button and upload options
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            if st.button("➕", key="upload_button"):
                st.session_state.show_upload_options = not st.session_state.get('show_upload_options', False)

        if st.session_state.get('show_upload_options', False):
            with st.container():
                options = [
                    ("🖼️", "Image", ["png", "jpg", "jpeg", "gif"]),
                    ("📎", "Files", ["txt", "pdf", "doc", "docx"])
                ]

                for icon, label, file_types in options:
                    if st.button(f"{icon} {label}", key=f"upload_{label.lower()}"):
                        uploaded_file = st.file_uploader(
                            f"Upload {label}",
                            type=file_types,
                            key=f"uploader_{label.lower()}"
                        )
                        if uploaded_file:
                            return process_uploaded_file(uploaded_file)
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

def display_chat_message(message: str, is_user: bool, media_content: Optional[Dict[str, Any]] = None):
    """Display a chat message with animation and optional media content."""
    message_type = "user" if is_user else "bot"
    
    # Create a container for the message and menu
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        # Display media content if present
        if media_content and media_content.get('type') == 'image':
            try:
                st.image(media_content['content'])
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        
        # Display message
        st.markdown(
            f'<div class="chat-message {message_type}-message">{message}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        if not is_user:
            display_three_dot_menu(message)
    
    # Display suggestions for bot messages
    if not is_user:
        suggestions = generate_suggestions(message)
        if suggestions:
            st.write("Follow-up suggestions:")
            cols = st.columns(len(suggestions))
            for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        return suggestion
    
    return None

def main():
    st.title("AI Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Chat container for history
    chat_container = st.container()
    
    # Input container at the bottom
    input_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message['content'],
                message['is_user'],
                message.get('media_content')
            )
    
    with input_container:
        # Create two columns: one narrow for upload, one wide for text input
        col1, col2 = st.columns([0.15, 0.85])
        
        # Handle file uploads in the first column
        with col1:
            uploaded_content = None
            upload_type = st.selectbox(
                "",
                options=["➕", "🖼️ Image", "📎 Files"],
                key="upload_selector",
                label_visibility="collapsed"
            )
            
            if upload_type == "🖼️ Image":
                uploaded_file = st.file_uploader(
                    "Upload Image",
                    type=["png", "jpg", "jpeg", "gif"],
                    key="image_uploader",
                    label_visibility="collapsed"
                )
                if uploaded_file:
                    uploaded_content = process_uploaded_file(uploaded_file)
            
            elif upload_type == "📎 Files":
                uploaded_file = st.file_uploader(
                    "Upload File",
                    type=["txt", "pdf", "doc", "docx"],
                    key="file_uploader",
                    label_visibility="collapsed"
                )
                if uploaded_file:
                    uploaded_content = process_uploaded_file(uploaded_file)
        
        # Handle text input in the second column
        with col2:
            user_input = st.text_input(
                "Message",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Type your message here..."
            )
            
            # Add a send button
            if st.button("Send", key="send_button") or user_input:
                if user_input:  # Only process if there's actual input
                    # Display user message
                    display_chat_message(user_input, True, uploaded_content)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        'content': user_input,
                        'is_user': True,
                        'media_content': uploaded_content
                    })
                    
                    try:
                        # Prepare message for Gemini
                        message_parts = prepare_gemini_message(user_input, uploaded_content)
                        
                        # Generate response using Gemini
                        model = genai.GenerativeModel('gemini-pro-vision' if uploaded_content and uploaded_content.get('type') == 'image' else 'gemini-pro')
                        response = model.generate_content(
                            message_parts,
                            generation_config={
                                'temperature': 0.7,
                                'top_p': 0.95,
                                'top_k': 40,
                                'max_output_tokens': 2048,
                            }
                        )
                        
                        # Get the response text
                        final_response = response.text
                        
                        # Display bot response
                        display_chat_message(final_response, False)
                        
                        # Add response to chat history
                        st.session_state.messages.append({
                            'content': final_response,
                            'is_user': False
                        })
                        
                        # Clear the input
                        st.session_state.user_input = ""
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        
                    # Rerun to clear the input and update the chat
                    st.rerun()

if __name__ == "__main__":
    main() 