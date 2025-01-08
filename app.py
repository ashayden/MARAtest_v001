import streamlit as st

# Configure page before any other Streamlit commands
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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
import streamlit.config as st_config
import re

# Load environment variables
load_dotenv()

# Configure Streamlit for reduced file watching
try:
    st_config.set_option('server.fileWatcherType', 'none')
except Exception:
    # Fallback if setting option fails
    pass

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

# Core CSS for layout and theme
st.markdown("""
<style>
/* Base theme */
[data-testid="stAppViewContainer"] {
    background-color: #1a1a1a;
}

/* Main container */
.main .block-container {
    max-width: 1200px;
    padding-bottom: 100px;
    margin: 0 auto;
}

/* Settings columns */
[data-testid="stHorizontalBlock"] {
    gap: 2rem;
    margin-bottom: 2rem;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 1rem;
}

.streamlit-expanderContent {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 0 0 8px 8px;
    padding: 1rem;
    margin-top: -1px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Chat input container */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    max-width: 100%;
    background-color: #1a1a1a;
    padding: 1rem;
    z-index: 10000;
    border-top: 1px solid #404040;
    box-sizing: border-box;
}

[data-testid="stChatInput"] > div {
    margin: 0 auto;
    max-width: 1200px;
}

[data-testid="stChatInput"] textarea {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
    min-height: 44px !important;
    max-height: 200px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #4CAF50 !important;
    box-shadow: 0 0 0 1px #4CAF50 !important;
}

/* Status indicator */
[data-testid="stStatus"] {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    width: 67%;
    max-width: 800px;
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 8px;
    z-index: 998;
}

/* Buttons */
.stButton > button {
    background-color: transparent;
    border: 1px solid #404040;
    color: white;
    border-radius: 6px;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    border-color: #4CAF50;
    background-color: rgba(76, 175, 80, 0.1);
}

/* Hide default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Ensure content doesn't go behind input */
.main .block-container {
    padding-bottom: 100px !important;
}

/* Sliders */
.stSlider {
    padding: 1rem 0;
}

/* Info boxes */
.stAlert {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    # Initialize all session state variables at startup
    defaults = {
        'messages': [],
        'suggestions': [],
        'clear_files': False,
        'file_uploader_key': "file_uploader_0",
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 40,
        'max_output_tokens': 2048,
        'specialist_responses': {},  # Initialize as empty dictionary
        'current_domains': [],       # Initialize as empty list
        'current_analysis': None     # Store current analysis
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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

def get_download_link(content: str, filename: str = "report.md") -> str:
    """Generate a download link for content."""
    b64 = base64.b64encode(content.encode()).decode()
    return f"""
        <a href="data:text/markdown;base64,{b64}" 
           download="{filename}"
           class="download-link">
           💾 Download as Markdown
        </a>
    """

def copy_to_clipboard(text: str):
    """Copy text to clipboard using JavaScript."""
    js = f"""
        <script>
        function copyToClipboard() {{
            const el = document.createElement('textarea');
            el.value = {repr(text)};
            el.setAttribute('readonly', '');
            el.style.position = 'absolute';
            el.style.left = '-9999px';
            document.body.appendChild(el);
            const selected =
                document.getSelection().rangeCount > 0
                    ? document.getSelection().getRangeAt(0)
                    : false;
            el.select();
            document.execCommand('copy');
            document.body.removeChild(el);
            if (selected) {{
                document.getSelection().removeAllRanges();
                document.getSelection().addRange(selected);
            }}
            window.parent.postMessage({{
                type: 'streamlit:showToast',
                data: {{ message: 'Copied to clipboard!', kind: 'info' }}
            }}, '*');
        }}
        copyToClipboard();
        </script>
    """
    components = st.components.v1.html(js, height=0)
    return components

def generate_suggestions(content: str) -> list:
    """Generate follow-up suggestions based on content."""
    # Get rate limiter instance
    from agents.base_template import RateLimiter
    rate_limiter = RateLimiter.get_instance()
    
    try:
        # Wait for rate limiter with a shorter timeout
        rate_limiter.wait_if_needed(timeout=5)  # 5 second timeout
        
        # Create a more focused prompt to reduce token usage
        prompt = f"""
        Generate 3 follow-up questions based on the key points in this content.
        Format as:
        Q1: [question]
        Q2: [question]
        Q3: [question]
        
        Content: {content[:2000]}  # Limit content length to reduce tokens
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,  # Reduced from 0.9
                'top_p': 0.9,       # Reduced from 0.95
                'top_k': 30,        # Reduced from 40
                'max_output_tokens': 512,  # Reduced from 1024
            }
        )
        
        suggestions = []
        current_question = None
        
        # Simpler parsing logic
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                question = line.split(':', 1)[1].strip()
                # Use first few words as headline
                words = question.split()
                headline = ' '.join(words[:5]) + '...'
                suggestions.append((headline, question))
        
        return suggestions[:3]
        
    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            st.info("Skipping follow-up questions due to rate limits.")
        else:
            st.warning(f"Could not generate follow-up questions: {str(e)}")
        return []

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
                    'text': f"Content from {file_data['name']}:\n{file_data['data']}\n\n"
                })
    
    # Add user's text input
    if text_input:
        parts.append({'text': text_input})
    
    # Ensure there's at least one part with text
    if not parts or not any('text' in part for part in parts):
        parts.append({'text': text_input if text_input else ''})
    
    return parts

def get_orchestrator():
    """Get or create the agent orchestrator."""
    if 'orchestrator' not in st.session_state:
        from agents.orchestrator import AgentOrchestrator
        st.session_state.orchestrator = AgentOrchestrator()
    
    return st.session_state.orchestrator

def get_domain_avatar(domain: str) -> str:
    """Get avatar emoji for each domain/role."""
    domain_avatars = {
        'initial_analysis': '🎯',
        'history': '📚',
        'culture': '🎭',
        'music': '🎵',
        'food': '🍳',
        'architecture': '🏛️',
        'art': '🎨',
        'literature': '📖',
        'geography': '🗺️',
        'economics': '📈',
        'sociology': '👥',
        'politics': '⚖️',
        'science': '🔬',
        'technology': '💻',
        'environment': '🌿',
        'sports': '⚽',
        'religion': '🕊️',
        'philosophy': '🤔',
        'final_synthesis': '📊',
        'suggestions': '💡'
    }
    return domain_avatars.get(domain.lower(), '🔍')

def display_message(message: dict, container=None):
    """Display a chat message with clean formatting and streaming support."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    message_type = message.get('type', '')
    is_streaming = message.get('streaming', False)
    
    # Use provided container or create new one
    if container is None:
        container = st.chat_message("user" if role == 'user' else "assistant")
    
    # Store placeholders in session state to persist across reruns
    container_id = str(id(container))
    content_key = f"content_{container_id}"
    
    if content_key not in st.session_state:
        st.session_state[content_key] = container.empty()
    
    with container:
        if role == 'user':
            st.session_state[content_key].markdown(content)
        
        elif role == 'assistant':
            # Generate creative title based on content
            if not is_streaming and content:
                # Extract first line or section header for title
                first_line = content.split('\n')[0].strip()
                if first_line.startswith('#'):
                    title = first_line.lstrip('#').strip()
                else:
                    # Generate a title from first sentence
                    title = first_line[:50] + ('...' if len(first_line) > 50 else '')
                
                container.markdown(f"### {title}")
                container.markdown("---")
            
            # Clean and format content
            if content:
                # Remove any existing titles/headers
                content = re.sub(r'^#+ .*$', '', content, flags=re.MULTILINE)
                # Remove AI/specialist references
                content = re.sub(r'(?i)(AI|artificial intelligence|specialist|expert).*?analysis', '', content)
                content = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', content)
                
                # Format sections consistently
                content = re.sub(r'^(\d+\.)', r'### \1', content, flags=re.MULTILINE)
                
                st.session_state[content_key].markdown(content)
            
            # Show actions when streaming is complete
            if not is_streaming:
                if message_type == "synthesis":
                    col1, col2 = container.columns([1, 4])
                    with col1:
                        if st.button("📋 Copy", key=f"copy_{message_type}_{hash(content)}_{int(time.time())}"):
                            copy_to_clipboard(content)
                    with col2:
                        report_content = generate_full_report()
                        st.download_button(
                            "💾 Download Report",
                            report_content,
                            file_name="analysis_report.md",
                            mime="text/markdown",
                            key=f"download_{message_type}_{hash(content)}_{int(time.time())}"
                        )
                elif message_type == "suggestions":
                    for idx, (headline, full_question) in enumerate(message.get("suggestions", [])):
                        if st.button(f"💡 {headline}", key=f"suggest_{message_type}_{idx}_{hash(str(headline))}_{int(time.time())}"):
                            st.session_state.next_prompt = full_question
                            st.rerun()
                elif message_type != "suggestions":
                    if st.button("📋 Copy", key=f"copy_{message_type}_{hash(content)}_{int(time.time())}"):
                            copy_to_clipboard(content)
    
    return container

def generate_full_report() -> str:
    """Generate a full report from all messages."""
    report_content = "# Analysis Report\n\n"
    for msg in st.session_state.messages:
        if msg.get("type") in ["initial_analysis", "specialist", "synthesis"]:
            report_content += f"## {msg.get('type', '').replace('_', ' ').title()}\n\n"
            report_content += msg["content"] + "\n\n"
    return report_content

def process_with_orchestrator(orchestrator, prompt: str, files_data: list = None):
    """Process input through the collaborative agent system with streaming support."""
    status_container = st.empty()
    error_container = st.empty()
    
    try:
        # Reset state
        st.session_state.specialist_responses = {}
        st.session_state.current_domains = []
        st.session_state.suggestions = []
        
        # Prepare messages
        parts = prepare_messages(prompt, files_data)
        
        # Process through orchestrator
        current_container = None
        current_message = None
        
        for message in orchestrator.process_input(parts[0]['text']):
            message_type = message.get('type')
            
            if message_type == 'status':
                status_container.write(message['content'])
                continue
                
            if message_type == 'error':
                error_container.error(f"Error: {message['content']}")
                continue
            
            if message_type in ['initial_analysis', 'specialist', 'synthesis']:
                if current_message is None or current_message.get('type') != message_type:
                    # Create new message container
                    current_message = {
                        "role": "assistant",
                        "type": message_type,
                        "content": message['content'],
                        "avatar": message.get('avatar', '🤖'),
                        "streaming": message['streaming']
                    }
                    
                    if message_type == 'specialist':
                        current_message['domain'] = message['domain']
                    
                    current_container = display_message(current_message)
                else:
                    # Update existing message
                    current_message['content'] = message['content']
                    current_message['streaming'] = message['streaming']
                    display_message(current_message, current_container)
                
                # Store completed messages
                if not message['streaming']:
                    st.session_state.messages.append(current_message)
                    if message_type == 'specialist':
                        st.session_state.specialist_responses[message['domain']] = message['content']
                    current_message = None
                    current_container = None
            
            elif message_type == 'suggestions':
                suggestions_message = {
                    "role": "assistant",
                    "type": "suggestions",
                    "suggestions": message['content'],
                    "avatar": "💡"
                }
                st.session_state.messages.append(suggestions_message)
                display_message(suggestions_message)
        
        status_container.empty()
        
    except Exception as e:
        error_container.error(f"Error: {str(e)}")
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

def chat_interface():
    """Modern chat interface following Streamlit best practices."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Set up the layout
        st.title("AI Assistant")
        st.caption("Powered by Google's Gemini API")
        
        # Add settings and documentation to main page
        col1, col2 = st.columns(2)
        
        # Model Settings in first column
        with col1:
            with st.expander("⚙️ Model Settings", expanded=False):
                st.subheader("🔬 Domain Specialist Settings")
                st.info("""
                Adjust these settings to control how domain specialists analyze and respond to queries.
                Other agents (Initializer and Synthesizer) use fixed settings for consistency.
                """)
                
                # Temperature slider
                st.session_state.temperature = st.slider(
                    "Creativity Level",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.temperature,
                    step=0.1,
                    help="Higher values make specialist responses more creative but less focused"
                )
                
                # Top P slider
                st.session_state.top_p = st.slider(
                    "Response Diversity",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.top_p,
                    step=0.05,
                    help="Controls how diverse specialist responses can be"
                )
                
                # Top K slider
                st.session_state.top_k = st.slider(
                    "Choice Range",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.top_k,
                    step=1,
                    help="Controls how many options specialists consider for each word"
                )
                
                # Max Output Tokens slider
                st.session_state.max_output_tokens = st.slider(
                    "Maximum Response Length",
                    min_value=256,
                    max_value=4096,
                    value=st.session_state.max_output_tokens,
                    step=256,
                    help="Maximum length of specialist responses"
                )
        
        # How it Works in second column
        with col2:
            with st.expander("ℹ️ How it Works", expanded=False):
                st.markdown("""
                This AI Assistant uses the Gemini 2.0 Flash Experimental model to provide comprehensive analysis through a coordinated multi-agent system.
                
                ### System Configuration
                - **Model**: Gemini 2.0 Flash Experimental
                - **Rate Limits**: 3 requests per minute (free tier)
                - **Request Interval**: 1.5 seconds minimum between requests
                
                ### Multi-Agent Response Architecture
                
                #### 1. Initial Analysis Agent (Fixed Settings)
                - Temperature: 0.5 (balanced)
                - Purpose: Analyzes input and identifies required expertise
                - Determines which specialists to consult
                - Provides initial context framework
                
                #### 2. Domain Specialists (Adjustable Settings)
                - Created dynamically based on input topic
                - Expertise determined in real-time
                - Controlled by model settings
                - Each specialist provides domain-specific insights
                
                #### 3. Synthesis Agent (Fixed Settings)
                - Temperature: 0.3 (focused)
                - Integrates all specialist responses
                - Creates cohesive final report
                - Maintains consistent academic structure
                
                ### Rate Limit Management
                - Maximum 3 requests per minute
                - 1.5 second pause between requests
                - Manual retry required if limits exceeded
                - Clear error messages when limits are reached
                
                ### Features
                - Real-time streaming responses
                - Dynamic specialist creation
                - Persistent chat history
                - Downloadable reports
                - Copy functionality
                - Follow-up suggestions
                
                ### Model Settings
                Adjust these settings to control specialist behavior:
                - **Creativity Level**: Controls response variety
                - **Response Diversity**: Affects token selection
                - **Choice Range**: Influences word selection
                - **Maximum Length**: Sets response length limit
                
                Note: Initial Analysis and Synthesis agents use fixed settings for consistency.
                """)
        
        # Add horizontal line after settings
        st.markdown("---")
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Display chat messages
        for message in st.session_state.messages:
            display_message(message)
        
        # Create centered input area
        prompt = st.chat_input("Message", key="chat_input")
        
        # Process new input
        if prompt:
            # Add user message
            st.chat_message("user").markdown(prompt)
            
            # Add to message history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Process through orchestrator
            try:
                # Ensure status container is properly initialized
                status_container = st.empty()
                status_container.write("Processing...")
                response = process_with_orchestrator(orchestrator, prompt)
                status_container.write("Complete!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
        
        # Handle suggestion clicks
        if 'next_prompt' in st.session_state:
            prompt = st.session_state.next_prompt
            del st.session_state.next_prompt
            st.rerun()
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

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