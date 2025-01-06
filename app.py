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

# Custom CSS for modern chat interface
st.markdown("""
<style>
/* Modern dark theme */
:root {
    --background-color: #1a1a1a;
    --input-background: #2d2d2d;
    --text-color: #ffffff;
    --border-color: #404040;
    --accent-color: #4CAF50;
    --hover-color: #45a049;
}

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: var(--input-background) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-bottom: 0.5rem !important;
}

.streamlit-expanderHeader:hover {
    background-color: #363636 !important;
}

.streamlit-expanderContent {
    background-color: var(--background-color) !important;
    border: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1rem !important;
}

/* Sidebar menu items */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background-color: transparent !important;
    border: none !important;
    padding: 0.5rem !important;
}

section[data-testid="stSidebar"] .streamlit-expanderContent {
    border: none !important;
    background-color: transparent !important;
    padding: 0.5rem !important;
}

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
    width: 100%;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] > section {
    padding: 0.5rem !important;
    border-radius: 8px;
    background: var(--input-background);
    border: 1px dashed var(--border-color);
}

[data-testid="stFileUploader"] > section:hover {
    border-color: var(--accent-color);
}

[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
    display: none;  /* Hide the default label */
}

/* File uploader drop zone */
.uploadedFile {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 4px;
    padding: 0.25rem 0.5rem !important;
    margin: 0.25rem 0;
}

/* File uploader button */
[data-testid="stFileUploader"] button {
    background: var(--accent-color) !important;
    color: white !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    border-radius: 4px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--hover-color) !important;
}

/* Custom file uploader label */
.file-uploader-label {
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.8;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Ensure proper spacing between elements */
.stChatInput {
    margin-top: 1rem !important;
}

/* Add custom file uploader label */
.file-uploader-container::before {
    content: "📎 Attach files";
    display: block;
    margin-bottom: 0.5rem;
    color: var(--text-color);
    opacity: 0.8;
    font-size: 0.9rem;
}

/* Ensure content doesn't get hidden */
.main .block-container {
    padding-bottom: 160px;
}

/* Specialist container styling */
.specialist-container {
    margin: 1rem 0;
    border-left: 3px solid var(--accent-color);
    padding-left: 1rem;
}

/* Analysis sections */
.initial-analysis {
    border-left-color: #2196F3;
}

.final-synthesis {
    border-left-color: #9C27B0;
}

/* Spinner Animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
    margin-right: 10px;
    vertical-align: middle;
}

.processing-message {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin: 10px 0;
}

/* Input container styling */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--background-color);
    border-top: 1px solid var(--border-color);
    padding: 1rem;
    z-index: 1000;
    backdrop-filter: blur(10px);
}

/* Ensure content doesn't get hidden behind input */
.main .block-container {
    padding-bottom: 200px !important;  /* Increased padding to prevent content hiding */
    max-width: 1000px;  /* Limit content width for better readability */
}

/* Style file uploader in fixed container */
.input-container .stFileUploader {
    margin-bottom: 0.5rem;
}

/* Chat input styling */
.stChatInput {
    margin-bottom: 0 !important;
}

.stChatInput > div {
    padding: 0 !important;
}

/* Adjust chat message container */
.chat-message-container {
    margin-bottom: 180px;  /* Space for fixed input area */
}

/* Glass effect for input container */
.input-container {
    background: rgba(26, 26, 26, 0.8);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* Hide default Streamlit padding at the bottom */
.main .block-container {
    padding-bottom: 200px !important;
}

/* Custom styling for the file uploader */
.file-uploader-container {
    background: rgba(45, 45, 45, 0.5);
    border-radius: 8px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}

.file-uploader-container .uploadedFile {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 0.25rem;
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
    Based on the following content, generate exactly 3 follow-up questions that are:
    1. A deeper dive into the most interesting specific aspect of the topic
    2. An extension of the topic into related areas or broader implications
    3. An unexpected or unique connection to another field or concept
    
    For each question, also provide a short headline version (maximum 5-7 words) that captures the key idea.
    Format as:
    HEADLINE: [Short Version]
    FULL: [Complete Question]
    
    Make each question thoughtful, specific, and directly related to the content.
    
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
        
        # Parse headlines and full questions
        headlines = []
        full_questions = []
        current_headline = None
        current_full = None
        
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('HEADLINE:'):
                current_headline = line.replace('HEADLINE:', '').strip()
            elif line.startswith('FULL:'):
                current_full = line.replace('FULL:', '').strip()
                if current_headline and current_full:
                    headlines.append(current_headline)
                    full_questions.append(current_full)
                    current_headline = None
                    current_full = None
        
        # Ensure we have pairs of headlines and full questions
        suggestions = list(zip(headlines[:3], full_questions[:3]))
        return suggestions
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

def display_suggestions():
    """Display suggestion buttons."""
    if st.session_state.suggestions:
        st.markdown("---")  # Add separator
        st.markdown("### 🤔 Explore Further")
        
        # Create three columns with equal width
        cols = st.columns(3)
        
        # Define button styles for each type
        button_styles = [
            "💡",  # For deep dive
            "🔄",  # For related topics
            "🌟"   # For unexpected connections
        ]
        
        # Display each suggestion in its own column
        for idx, ((headline, full_question), style, col) in enumerate(zip(
            st.session_state.suggestions,
            button_styles,
            cols
        )):
            with col:
                # Create a container for the button
                button_container = st.container()
                with button_container:
                    if st.button(
                        f"{style} {headline}",
                        key=f"suggestion_{idx}",
                        help=full_question  # Show full question on hover
                    ):
                        st.session_state.next_prompt = full_question
                        st.rerun()
                    
                    # Show truncated version of full question below button
                    st.caption(full_question[:100] + "..." if len(full_question) > 100 else full_question)

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
        if not parts:
            # If no files, make text input the primary message
            parts = [{'text': text_input}]
        else:
            # If there are files, append text input
            parts.append({'text': text_input})
    elif not parts:
        # Ensure there's at least one message
        parts = [{'text': ''}]
        
    return parts

def how_it_works_sidebar():
    """Add How it Works section to sidebar."""
    with st.sidebar:
        with st.expander("ℹ️ How it Works", expanded=False):
            st.markdown("""
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

def process_with_orchestrator(orchestrator, prompt: str, files_data: list = None):
    """Process input through the collaborative agent system."""
    try:
        # Reset state for new request
        st.session_state.specialist_responses = {}
        st.session_state.current_domains = []
        st.session_state.suggestions = []
        
        # Prepare messages
        parts = prepare_messages(prompt, files_data)
        
        # Progress indicators
        progress = st.empty()
        error_container = st.empty()
        
        # Get initial analysis
        progress.markdown("🔍 Performing initial analysis...")
        initial_response = ""
        for chunk in orchestrator.agents['initializer'].generate_response(parts, stream=True):
            if chunk:
                initial_response += chunk
        
        if initial_response:
            st.session_state.specialist_responses['initial_analysis'] = initial_response
        
        # Identify required specialists
        domains = []
        if isinstance(parts, list) and parts and 'text' in parts[0]:
            domains = orchestrator.identify_required_specialists(parts[0]['text'])
        st.session_state.current_domains = domains
        
        # Process specialist responses
        if domains:
            for domain in domains:
                try:
                    progress.markdown(f"🤔 Consulting {domain.title()} specialist...")
                    
                    if domain not in orchestrator.agents:
                        orchestrator.agents[domain] = orchestrator.create_specialist(domain)
                    
                    # Get previous responses for context
                    previous_responses = []
                    if initial_response:
                        previous_responses.append(initial_response)
                    
                    # Add previous specialist responses
                    for prev_domain in st.session_state.current_domains:
                        if prev_domain != domain and prev_domain in st.session_state.specialist_responses:
                            prev_response = st.session_state.specialist_responses.get(prev_domain)
                            if prev_response:
                                previous_responses.append(prev_response)
                    
                    # Generate specialist response
                    specialist_response = ""
                    for chunk in orchestrator.agents[domain].generate_response(
                        parts,
                        previous_responses=previous_responses,
                        stream=True
                    ):
                        if chunk:
                            specialist_response += chunk
                    
                    if specialist_response:
                        st.session_state.specialist_responses[domain] = specialist_response
                    
                except Exception as e:
                    error_container.error(f"Error with {domain} specialist: {str(e)}")
                    return None
        
        # Generate final synthesis
        progress.markdown("🎯 Synthesizing insights...")
        
        # Collect all responses for synthesis
        synthesis_inputs = []
        if initial_response:
            synthesis_inputs.append(initial_response)
        
        for domain in st.session_state.current_domains:
            if domain in st.session_state.specialist_responses:
                synthesis_inputs.append(st.session_state.specialist_responses[domain])
        
        # Generate synthesis
        synthesis = ""
        for chunk in orchestrator.agents['reasoner'].generate_response(
            parts,
            previous_responses=synthesis_inputs,
            stream=True
        ):
            if chunk:
                synthesis += chunk
        
        if synthesis:
            st.session_state.specialist_responses['final_synthesis'] = synthesis
        
        # Clear progress indicator
        progress.empty()
        
        return synthesis
        
    except Exception as e:
        error_container.error(f"Error: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.error("Full error details:", exc_info=True)
        return None

def chat_interface():
    """Modern chat interface with minimal design."""
    try:
        st.title("AI Assistant")
        initialize_session_state()
        model_settings_sidebar()
        orchestrator = get_orchestrator()
        
        # Input area at top
        input_container = st.container()
        with input_container:
            uploaded_files = st.file_uploader(
                "📎 Attach files",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
                accept_multiple_files=True,
                key=st.session_state.file_uploader_key,
                label_visibility="collapsed"
            )
            prompt = st.chat_input("Message", key="chat_input")
            st.markdown("---")
        
        # Process new input
        if prompt:
            files_data = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_data = process_file_upload(uploaded_file)
                    if file_data:
                        files_data.append(file_data)
                        st.toast(f"📎 {file_data['name']} attached")
            
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "files_data": files_data if files_data else None
            })
            
            try:
                # Process through orchestrator
                response = process_with_orchestrator(orchestrator, prompt, files_data if files_data else None)
                if response:
                    st.session_state.file_uploader_key = f"file_uploader_{int(time.time())}"
                    suggestions = generate_suggestions(response)
                    if suggestions:
                        st.session_state.suggestions = suggestions
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if st.checkbox("Show detailed error"):
                    st.error("Full error details:", exc_info=True)
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    if message.get("files_data"):
                        for file_data in message["files_data"]:
                            if file_data["type"] == "image":
                                st.image(file_data["display_data"])
                            elif file_data["type"] == "text":
                                with st.expander(f"📄 {file_data['name']}", expanded=False):
                                    st.text(file_data["data"])
            
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    # Display responses in order: Initial → Specialists → Synthesis
                    if 'initial_analysis' in st.session_state.specialist_responses:
                        with st.expander("🎯 Initial Analysis", expanded=False):
                            st.markdown(st.session_state.specialist_responses['initial_analysis'])
                    
                    # Display specialist responses in order they were called
                    displayed_domains = set()
                    if 'current_domains' in st.session_state:
                        for domain in st.session_state.current_domains:
                            if domain not in displayed_domains and domain in st.session_state.specialist_responses:
                                with st.expander(f"🔍 {domain.title()} Analysis", expanded=False):
                                    st.markdown(st.session_state.specialist_responses[domain])
                                displayed_domains.add(domain)
                    
                    # Display final synthesis
                    if 'final_synthesis' in st.session_state.specialist_responses:
                        with st.expander("📊 Final Synthesis", expanded=True):
                            st.markdown(st.session_state.specialist_responses['final_synthesis'])
                            
                            # Display suggestions only after synthesis
                            if st.session_state.suggestions:
                                st.markdown("---")
                                st.markdown("### 🤔 Explore Further")
                                cols = st.columns(3)
                                button_styles = ["💡", "🔄", "🌟"]
                                
                                for idx, ((headline, full_question), style, col) in enumerate(zip(
                                    st.session_state.suggestions,
                                    button_styles,
                                    cols
                                )):
                                    with col:
                                        if st.button(
                                            f"{style} {headline}",
                                            key=f"suggestion_{idx}",
                                            help=full_question
                                        ):
                                            st.session_state.next_prompt = full_question
                                            st.rerun()
                                        st.caption(full_question[:100] + "..." if len(full_question) > 100 else full_question)
        
        # Handle suggestion clicks
        if 'next_prompt' in st.session_state:
            prompt = st.session_state.next_prompt
            del st.session_state.next_prompt
            
    except Exception as e:
        st.error(f"Error in chat interface: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.error("Full error details:", exc_info=True)

def create_specialist_container(specialist_name: str, response: str):
    """Create a collapsible container for specialist responses."""
    with st.expander(f"🔍 {specialist_name.replace('specialist_', '').title()} Specialist Analysis"):
        st.markdown(response)

def display_message(message: dict):
    """Display a chat message with appropriate styling."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    
    if role == 'user':
        st.markdown(f"**You:** {content}")
    elif role == 'assistant':
        if 'specialist_responses' in message:
            st.markdown("**AI Assistant Analysis:**")
            # Display initial analysis
            if 'initial_analysis' in message:
                with st.expander("🎯 Initial Analysis", expanded=True):
                    st.markdown(message['initial_analysis'])
            
            # Display specialist responses in collapsible containers
            for specialist, response in message['specialist_responses'].items():
                create_specialist_container(specialist, response)
            
            # Display final synthesis
            if 'final_synthesis' in message:
                with st.expander("📊 Final Synthesis", expanded=True):
                    st.markdown(message['final_synthesis'])
        else:
            st.markdown(f"**AI Assistant:** {content}")

def process_agent_response(user_input: str):
    """Process user input through the agent system and display responses."""
    try:
        # Initialize response structure
        current_response = {
            'role': 'assistant',
            'specialist_responses': {},
            'initial_analysis': '',
            'final_synthesis': ''
        }
        
        # Process through orchestrator
        response_stream = agent_orchestrator.process_input([{'text': user_input}])
        
        # Create placeholder for streaming responses
        response_placeholder = st.empty()
        
        # Track current specialist
        current_specialist = None
        current_content = ""
        
        for chunk in response_stream:
            if '### SPECIALIST:' in chunk:
                # Save previous specialist content if any
                if current_specialist and current_content:
                    current_response['specialist_responses'][current_specialist] = current_content
                
                # Extract new specialist name
                current_specialist = chunk.split('### SPECIALIST:')[1].strip()
                current_content = ""
            elif '### INITIAL_ANALYSIS:' in chunk:
                current_specialist = 'initial_analysis'
                current_content = ""
            elif '### FINAL_SYNTHESIS:' in chunk:
                # Save previous content
                if current_specialist and current_content:
                    current_response['specialist_responses'][current_specialist] = current_content
                current_specialist = 'final_synthesis'
                current_content = ""
            else:
                current_content += chunk
                
                # Update display
                if current_specialist == 'initial_analysis':
                    current_response['initial_analysis'] = current_content
                elif current_specialist == 'final_synthesis':
                    current_response['final_synthesis'] = current_content
                elif current_specialist:
                    current_response['specialist_responses'][current_specialist] = current_content
                
                # Update display
                with response_placeholder.container():
                    display_message(current_response)
        
        # Save final content
        if current_specialist and current_content:
            if current_specialist == 'initial_analysis':
                current_response['initial_analysis'] = current_content
            elif current_specialist == 'final_synthesis':
                current_response['final_synthesis'] = current_content
            else:
                current_response['specialist_responses'][current_specialist] = current_content
        
        # Add to session state
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        st.session_state.messages.append(current_response)
        
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")

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