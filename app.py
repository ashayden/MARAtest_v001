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
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-bottom: 0.5rem !important;
}

.streamlit-expanderHeader:hover {
    background-color: #363636 !important;
}

.streamlit-expanderContent {
    background-color: var(--background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 1rem !important;
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
        'specialist_responses': [],  # Store specialist responses
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
        st.write("Explore Further:")
        
        # Create three columns with different widths for better layout
        col1, col2, col3 = st.columns([1.2, 1.2, 1.2])
        
        # Define button styles for each type
        button_styles = [
            "💡",  # For specific aspect
            "🔄",  # For related topics
            "🌟"   # For unexpected connections
        ]
        
        # Display each suggestion in its own column with styled prefix
        for idx, (col, (headline, full_question), style) in enumerate(zip(
            [col1, col2, col3], 
            st.session_state.suggestions, 
            button_styles
        )):
            with col:
                if st.button(f"{style} {headline}", key=f"suggestion_{idx}"):
                    # Use the full question when clicked
                    st.session_state.next_prompt = full_question
                    st.rerun()

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
        with st.expander("⚙️ Model Settings", expanded=True):
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
        # Prepare messages
        parts = prepare_messages(prompt, files_data)
        specialist_responses = []
        
        # Create containers for dynamic updates
        progress_text = st.empty()
        specialist_containers = st.container()
        
        # Progress indicator with spinner
        progress_text.markdown("""
        <div class="processing-message">
            <div class="spinner"></div>
            <span>Analyzing input...</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Get initial analysis silently
        initial_response = ""
        for chunk in orchestrator.agents['initializer'].generate_response(parts, stream=True):
            if chunk:
                initial_response += chunk
        
        # Identify needed specialists
        if isinstance(parts, list) and parts and 'text' in parts[0]:
            domains = orchestrator.identify_required_specialists(parts[0]['text'])
        else:
            domains = []
        
        # Store current analysis in session state
        st.session_state.current_analysis = {
            'initial_response': initial_response,
            'domains': domains
        }
        
        # Process specialist responses
        if domains:
            # Create placeholders for each specialist
            specialist_placeholders = {}
            with specialist_containers:
                for domain in domains:
                    specialist_placeholders[domain] = st.empty()
            
            for domain in domains:
                # Update progress message for current specialist
                progress_text.markdown(f"""
                <div class="processing-message">
                    <div class="spinner"></div>
                    <span>Consulting {domain.title()} specialist...</span>
                </div>
                """, unsafe_allow_html=True)
                
                if domain not in orchestrator.agents:
                    orchestrator.agents[domain] = orchestrator.create_specialist(domain)
                
                specialist_response = ""
                response_container = specialist_placeholders[domain]
                
                # Initialize the expander
                with response_container:
                    with st.expander(f"🔍 {domain.title()} Analysis", expanded=False):
                        response_text = st.empty()
                
                for chunk in orchestrator.agents[domain].generate_response(
                    parts,
                    previous_responses=[initial_response] + [r['response'] for r in specialist_responses],
                    stream=True
                ):
                    specialist_response += chunk
                    # Update the response text in real-time
                    response_text.markdown(specialist_response)
                
                specialist_responses.append({
                    'domain': domain,
                    'response': specialist_response
                })
            
            # Store specialist responses in session state
            st.session_state.specialist_responses = specialist_responses
        
        # Generate synthesis
        progress_text.markdown("""
        <div class="processing-message">
            <div class="spinner"></div>
            <span>Synthesizing insights...</span>
        </div>
        """, unsafe_allow_html=True)
        
        synthesis = ""
        for chunk in orchestrator.agents['reasoner'].generate_response(
            parts,
            previous_responses=[initial_response] + [r['response'] for r in specialist_responses],
            stream=True
        ):
            if chunk:
                synthesis += chunk
        
        # Clear progress indicator
        progress_text.empty()
        
        return synthesis
        
    except Exception as e:
        st.error(f"Orchestrator error: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.error("Full error details:", exc_info=True)
        return None

def chat_interface():
    """Modern chat interface with minimal design."""
    try:
        st.title("AI Assistant")
        
        # Initialize session state
        initialize_session_state()
        
        # Show model settings sidebar
        model_settings_sidebar()
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Create persistent containers
        specialist_container = st.container()
        chat_container = st.container()
        input_container = st.container()
        
        # Display specialist analysis if available
        with specialist_container:
            if st.session_state.specialist_responses:
                with st.expander("🔍 Domain Expert Analysis", expanded=False):
                    for response in st.session_state.specialist_responses:
                        st.markdown(f"### {response['domain'].title()} Analysis")
                        st.markdown(response['response'])
                        st.divider()
        
        # Chat history
        with chat_container:
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
            
            # Display suggestions after the last message
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                display_suggestions()
        
        # Fixed input container at bottom
        with input_container:
            # File uploader (multiple files)
            uploaded_files = st.file_uploader(
                "📎 Attach files",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md', 'csv', 'pdf'],
                accept_multiple_files=True,
                key=st.session_state.file_uploader_key
            )
            
            # Chat input
            prompt = st.chat_input("Message")
            
            # Check for suggestion click
            if 'next_prompt' in st.session_state:
                prompt = st.session_state.next_prompt
                del st.session_state.next_prompt
            
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
                    # Process through orchestrator
                    response = process_with_orchestrator(orchestrator, prompt, files_data)
                    
                    if response:
                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Generate new suggestions
                        st.session_state.suggestions = generate_suggestions(response)
                        
                        # Update file uploader key to clear files
                        st.session_state.file_uploader_key = f"file_uploader_{int(time.time())}"
                        
                        # Clear input
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if st.checkbox("Show detailed error"):
                        st.error("Full error details:", exc_info=True)

    except OSError as e:
        if "inotify watch limit reached" in str(e):
            st.error("""
            File watch limit reached. Please run these commands on your server:
            ```bash
            sudo sysctl -w fs.inotify.max_user_watches=524288
            echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
            sudo sysctl -p
            ```
            Then restart the Streamlit server.
            """)
        else:
            raise e

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