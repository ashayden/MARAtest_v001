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
/* Core theme and layout */
.stApp {
    max-width: 100%;
}

/* Main container */
.main .block-container {
    padding-top: 1rem;
    padding-right: 3rem;
    padding-left: 3rem;
    padding-bottom: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    min-width: 18rem;
    max-width: 35rem;
    background-color: #1a1a1a;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 2rem;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem auto;
    max-width: 1200px;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

.streamlit-expanderContent {
    border: none !important;
    padding: 1rem !important;
    background-color: transparent !important;
}

/* Input area */
.stChatInput {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #1a1a1a;
    padding: 1rem 3rem;
    margin: 0;
    z-index: 999;
}

.stChatInput > div {
    max-width: 1200px;
    margin: 0 auto;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #2d2d2d;
    border: 1px dashed #404040;
    border-radius: 8px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background-color: transparent;
    color: white;
    border: 1px solid #404040;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #45a049;
    border-color: #4CAF50;
}

/* Download link */
a.download-link {
    display: inline-block;
    padding: 0.75rem 1rem;
    background-color: #2d2d2d;
    color: white;
    text-decoration: none;
    border: 1px solid #404040;
    border-radius: 6px;
    transition: all 0.2s ease;
}

a.download-link:hover {
    background-color: #45a049;
    border-color: #4CAF50;
}

/* Progress indicator */
.processing-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    background-color: #2d2d2d;
    border-radius: 8px;
    margin: 0.5rem auto;
    max-width: 1200px;
}

/* Add padding at bottom for fixed input */
.main .block-container::after {
    content: "";
    display: block;
    height: 8rem;
}

/* Hide default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
    prompt = f"""
    Based on the following content, generate exactly 3 follow-up questions that are:
    1. A deeper dive into the most interesting specific aspect
    2. An extension into related areas or implications
    3. An unexpected connection to another field
    
    Format each as:
    HEADLINE: [5-7 word summary]
    FULL: [Complete question]
    
    Content: {content}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.9,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
        )
        
        suggestions = []
        current_headline = None
        
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('HEADLINE:'):
                current_headline = line.replace('HEADLINE:', '').strip()
            elif line.startswith('FULL:') and current_headline:
                full_question = line.replace('FULL:', '').strip()
                suggestions.append((current_headline, full_question))
                current_headline = None
        
        return suggestions[:3]  # Ensure we return exactly 3 suggestions
        
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
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

def how_it_works_sidebar():
    """Add How it Works section to sidebar."""
    with st.sidebar:
        with st.expander("ℹ️ How it Works", expanded=False):
            st.markdown("""
            This AI Assistant is unique in its ability to dynamically create specialist agents based on your query's topic. Unlike traditional chatbots that use a one-size-fits-all approach, this system:

            - Creates domain experts in real-time based on your specific topic
            - Combines insights from multiple specialists for comprehensive answers
            - Allows fine-tuning of specialist behavior through model controls
            
            ---
            
            ### Multi-Agent Response Architecture
            
            #### Core Components
            1. **Initializer Agent** (Fixed Settings)
               - Analyzes input to identify required expertise
               - Determines which specialists to consult
               - Provides initial context framework
            
            2. **Domain Specialists** (Adjustable Settings)
               - Created dynamically based on input
               - Expertise determined in real-time
               - Controlled by sidebar model settings
               - Provide domain-specific insights
            
            3. **Synthesis Agent** (Fixed Settings)
               - Integrates all specialist responses
               - Creates cohesive final report
               - Maintains consistent structure
            
            #### Workflow
            1. User submits query/content
            2. Initializer analyzes and identifies needed expertise
            3. Relevant specialists are created/activated
            4. Each specialist provides domain insights
            5. Synthesis agent creates final structured report
            
            #### Dynamic Specialist Control
            - **Creativity Level**: Influences specialist response creativity
            - **Response Diversity**: Controls response variation
            - **Choice Range**: Affects word selection breadth
            - **Maximum Length**: Sets response length limit
            
            #### Features
            - Real-time streaming responses
            - Collapsible specialist insights
            - Multi-modal input support
            - Persistent chat history
            """)

def model_settings_sidebar():
    """Sidebar for model settings."""
    with st.sidebar:
        # Model Settings section
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
        
        # Add How it Works section immediately after Model Settings
        how_it_works_sidebar()

def get_orchestrator():
    """Get or create the agent orchestrator."""
    if 'orchestrator' not in st.session_state:
        from agents.orchestrator import AgentOrchestrator
        from agents.base_template import AgentConfig
        
        config = AgentConfig(
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            top_k=st.session_state.top_k,
            max_output_tokens=st.session_state.max_output_tokens
        )
        
        st.session_state.orchestrator = AgentOrchestrator(config)
    
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

def process_with_orchestrator(orchestrator, prompt: str, files_data: list = None):
    """Process input through the collaborative agent system."""
    try:
        # Create placeholders outside of state reset
        progress_placeholder = st.empty()
        error_container = st.empty()
        
        def update_progress(message):
            progress_placeholder.markdown(f"""
            <div class="processing-message">
                <div class="spinner"></div>
                <span>{message}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Reset state for new request
        st.session_state.specialist_responses = {}
        st.session_state.current_domains = []
        st.session_state.suggestions = []
        
        # Prepare messages
        parts = prepare_messages(prompt, files_data)
        
        # Get initial analysis
        update_progress("Performing initial analysis...")
        
        initial_response = ""
        for chunk in orchestrator.agents['initializer'].generate_response(parts, stream=True):
            if chunk:
                initial_response += chunk
        
        if initial_response:
            # Create initial analysis message
            initial_message = {
                "role": "assistant",
                "type": "initial_analysis",
                "content": initial_response,
                "avatar": "🎯"
            }
            st.session_state.messages.append(initial_message)
            
            # Update display without full rerun
            with st.empty():
                display_message(initial_message)
        
        # Extract text content for specialist identification
        text_content = ""
        for part in parts:
            if 'text' in part:
                text_content += part['text'] + "\n"
        
        # Identify required specialists
        update_progress("Identifying required specialists...")
        domains = orchestrator.identify_required_specialists(text_content.strip())
        st.session_state.current_domains = domains
        
        # Process specialist responses
        synthesis_inputs = [{'text': initial_response}]  # Start with initial analysis
        
        if domains:
            for domain in domains:
                try:
                    update_progress(f"Consulting {domain.title()} specialist...")
                    
                    if domain not in orchestrator.agents:
                        orchestrator.agents[domain] = orchestrator.create_specialist(domain)
                    
                    # Generate specialist response
                    specialist_response = ""
                    for chunk in orchestrator.agents[domain].generate_response(
                        parts,
                        previous_responses=synthesis_inputs.copy(),
                        stream=True
                    ):
                        if chunk:
                            specialist_response += chunk
                    
                    if specialist_response:
                        # Create specialist message
                        specialist_message = {
                            "role": "assistant",
                            "type": "specialist",
                            "domain": domain,
                            "content": specialist_response,
                            "avatar": get_domain_avatar(domain)
                        }
                        st.session_state.messages.append(specialist_message)
                        synthesis_inputs.append({'text': specialist_response})
                        
                        # Update display without full rerun
                        with st.empty():
                            display_message(specialist_message)
                    
                except Exception as e:
                    error_container.error(f"Error with {domain} specialist: {str(e)}")
                    continue
        
        # Generate final synthesis
        update_progress("Synthesizing insights...")
        
        synthesis = ""
        for chunk in orchestrator.agents['reasoner'].generate_response(
            parts,
            previous_responses=synthesis_inputs,
            stream=True
        ):
            if chunk:
                synthesis += chunk
        
        if synthesis:
            # Create synthesis message
            synthesis_message = {
                "role": "assistant",
                "type": "synthesis",
                "content": synthesis,
                "avatar": "📊"
            }
            st.session_state.messages.append(synthesis_message)
            
            # Update display without full rerun
            with st.empty():
                display_message(synthesis_message)
            
            # Generate suggestions
            update_progress("Generating follow-up questions...")
            try:
                suggestions = generate_suggestions(synthesis)
                if suggestions:
                    suggestions_message = {
                        "role": "assistant",
                        "type": "suggestions",
                        "suggestions": suggestions,
                        "avatar": "💡"
                    }
                    st.session_state.messages.append(suggestions_message)
                    
                    # Update display without full rerun
                    with st.empty():
                        display_message(suggestions_message)
            except Exception as e:
                error_container.error(f"Error generating suggestions: {str(e)}")
        
        # Clear progress indicator only after everything is complete
        progress_placeholder.empty()
        return synthesis
        
    except Exception as e:
        error_container.error(f"Error: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.error("Full error details:", exc_info=True)
        return None

def chat_interface():
    """Modern chat interface following Streamlit best practices."""
    try:
        # Initialize state and configuration
        initialize_session_state()
        
        # Set up the layout
        st.title("AI Assistant")
        
        # Configure sidebar
        with st.sidebar:
            model_settings_sidebar()
            how_it_works_sidebar()
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Create main chat container
        chat_container = st.container()
        
        # Create message display area
        with chat_container:
            # Display existing messages
            for message in st.session_state.messages:
                display_message(message)
            
            # Create input area container
            input_container = st.container()
            with input_container:
                # Create columns for file upload and input
                col1, col2 = st.columns([1, 4])
                with col1:
                    uploaded_files = st.file_uploader(
                        "📎 Attach files",
                        type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
                        accept_multiple_files=True,
                        key=st.session_state.file_uploader_key,
                        label_visibility="collapsed"
                    )
                with col2:
                    prompt = st.chat_input("Message", key="chat_input")
        
        # Process new input
        if prompt:
            # Handle file uploads
            files_data = process_uploads(uploaded_files) if uploaded_files else []
            
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "files_data": files_data if files_data else None
            })
            
            # Process through orchestrator
            try:
                response = process_with_orchestrator(orchestrator, prompt, files_data)
                if response:
                    st.session_state.file_uploader_key = f"file_uploader_{int(time.time())}"
            except Exception as e:
                handle_error(e)
        
        # Handle suggestion clicks
        if 'next_prompt' in st.session_state:
            prompt = st.session_state.next_prompt
            del st.session_state.next_prompt
            st.rerun()
            
    except Exception as e:
        handle_error(e)

def process_uploads(uploaded_files):
    """Process multiple file uploads with progress tracking."""
    files_data = []
    if uploaded_files:
        progress_text = "Processing uploads..."
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_data = process_file_upload(uploaded_file)
            if file_data:
                files_data.append(file_data)
                st.toast(f"📎 {file_data['name']} attached")
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
    return files_data

def handle_error(error):
    """Handle errors in a user-friendly way."""
    st.error(f"Error: {str(error)}")
    with st.expander("Show detailed error", expanded=False):
        st.code(traceback.format_exc())

def display_message(message: dict):
    """Display a chat message following Streamlit best practices."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    message_type = message.get('type', '')
    
    if role == 'user':
        with st.chat_message("user"):
            st.markdown(content)
            display_attachments(message.get("files_data", []))
    
    elif role == 'assistant':
        avatar = message.get("avatar", "🤖")
        with st.chat_message("assistant", avatar=avatar):
            if message_type == "initial_analysis":
                display_analysis_message("Initial Analysis", content)
            
            elif message_type == "specialist":
                display_analysis_message(f"{message['domain'].title()} Analysis", content)
            
            elif message_type == "synthesis":
                display_synthesis_message(content)
            
            elif message_type == "suggestions":
                display_suggestions(message.get("suggestions", []))
            
            else:
                st.markdown(content)

def display_attachments(files_data):
    """Display file attachments in chat messages."""
    if not files_data:
        return
        
    for file_data in files_data:
        if file_data["type"] == "image":
            st.image(file_data["display_data"])
        elif file_data["type"] == "text":
            with st.expander(f"📄 {file_data['name']}", expanded=False):
                st.text(file_data["data"])

def display_analysis_message(title: str, content: str):
    """Display an analysis message with copy functionality."""
    with st.expander(title, expanded=False):
        st.markdown(content)
        st.markdown("---")
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("📋", key=f"copy_{hash(content)}", help="Copy to clipboard"):
                copy_to_clipboard(content)

def display_synthesis_message(content: str):
    """Display the synthesis message with enhanced functionality."""
    with st.expander("Final Synthesis", expanded=True):
        st.markdown(content)
        st.markdown("---")
        
        # Create action buttons
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            if st.button("📋 Copy", key=f"copy_synthesis_{hash(content)}"):
                copy_to_clipboard(content)
        
        with col2:
            # Generate full report content
            report_content = generate_full_report()
            st.markdown(
                get_download_link(report_content, "analysis_report.md"),
                unsafe_allow_html=True
            )

def display_suggestions(suggestions):
    """Display follow-up suggestions."""
    st.markdown("### 🤔 Explore Further")
    for idx, (headline, full_question) in enumerate(suggestions):
        with st.container():
            if st.button(
                f"💡 {headline}",
                key=f"suggestion_{idx}_{hash(str(headline))}",
                help=full_question,
                use_container_width=True
            ):
                st.session_state.next_prompt = full_question
                st.experimental_rerun()

def generate_full_report() -> str:
    """Generate a full report from all messages."""
    report_content = "# Analysis Report\n\n"
    for msg in st.session_state.messages:
        if msg.get("type") in ["initial_analysis", "specialist", "synthesis"]:
            report_content += f"## {msg.get('type', '').replace('_', ' ').title()}\n\n"
            report_content += msg["content"] + "\n\n"
    return report_content

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