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
    # Create containers outside the try block
    status_container = st.empty()
    error_container = st.empty()
    error_details_container = st.empty()
    synthesis = None  # Initialize synthesis variable
    
    try:
        def update_progress(message):
            status_container.write(message)
        
        # Reset state for new request
        st.session_state.specialist_responses = {}
        st.session_state.current_domains = []
        st.session_state.suggestions = []
        
        # Prepare messages
        parts = prepare_messages(prompt, files_data)
        
        # Get initial analysis with retry logic
        update_progress("Performing initial analysis...")
        
        # Get rate limiter instance
        from agents.base_template import RateLimiter
        rate_limiter = RateLimiter.get_instance()
        
        initial_response = None
        retry_count = 0
        while retry_count < 3 and not initial_response:
            try:
                # Wait for rate limiter with timeout
                rate_limiter.wait_if_needed(timeout=10)  # 10 second timeout
                
                # Reduce tokens for initial analysis
                orchestrator.agents['initializer'].config.max_output_tokens = 1024
                
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
                
            except Exception as e:
                retry_count += 1
                if retry_count < 3:
                    error_container.warning(f"Retrying initial analysis (attempt {retry_count + 1}/3)...")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    orchestrator.agents['initializer'].config.max_output_tokens = 512  # Reduce tokens for retry
                else:
                    error_container.error("Could not complete initial analysis after 3 attempts.")
                    return None
        
        if not initial_response:
            error_container.error("Failed to generate initial analysis.")
            return None
        
        # Extract specialists directly from initial analysis
        specialists = orchestrator.extract_specialists_from_analysis(initial_response)
        
        # Process specialist responses
        synthesis_inputs = [{'text': initial_response}]  # Start with initial analysis
        
        if specialists:
            for i, specialist in enumerate(specialists):
                try:
                    domain = specialist['domain']
                    update_progress(f"Consulting {domain.title()} specialist...")
                    
                    # Add delay between specialists
                    if i > 0:
                        time.sleep(2)  # 2 second delay between specialists
                    rate_limiter.wait_if_needed(timeout=10)  # Wait for rate limit with timeout
                    
                    # Create specialist with reduced token limit
                    specialist_agent = orchestrator.create_specialist(
                        domain=domain,
                        expertise=specialist['expertise'],
                        focus_areas=specialist['focus']
                    )
                    specialist_agent.config.max_output_tokens = 1024
                    
                    # Generate specialist response with retry
                    retry_count = 0
                    specialist_response = None
                    
                    while retry_count < 3 and not specialist_response:
                        try:
                            response_text = ""
                            for chunk in specialist_agent.generate_response(
                                parts,
                                previous_responses=[initial_response],
                                stream=True
                            ):
                                if chunk:
                                    response_text += chunk
                            
                            if response_text:
                                specialist_response = response_text
                                # Create specialist message
                                specialist_message = {
                                    "role": "assistant",
                                    "type": "specialist",
                                    "domain": domain,
                                    "content": specialist_response,
                                    "avatar": get_domain_avatar(domain)
                                }
                                st.session_state.messages.append(specialist_message)
                                st.session_state.specialist_responses[domain] = specialist_response
                                synthesis_inputs.append({'text': specialist_response})
                                
                                # Update display without full rerun
                                with st.empty():
                                    display_message(specialist_message)
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count < 3:
                                error_container.warning(f"Retrying {domain} specialist (attempt {retry_count + 1}/3)...")
                                time.sleep(2 ** retry_count)
                                specialist_agent.config.max_output_tokens = 512
                            else:
                                error_container.error(f"Could not complete {domain} specialist analysis after 3 attempts.")
                    
                except Exception as e:
                    error_container.error(f"Error with {domain} specialist: {str(e)}")
                    continue
        
        # Only proceed with synthesis if we have specialist responses
        if st.session_state.specialist_responses:
            # Generate final synthesis
            update_progress("Synthesizing insights...")
            
            synthesis = ""
            retry_count = 0
            
            while retry_count < 3 and not synthesis:
                try:
                    rate_limiter.wait_if_needed(timeout=15)  # Wait for rate limit with timeout
                    
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
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < 3:
                        error_container.warning(f"Retrying synthesis (attempt {retry_count + 1}/3)...")
                        time.sleep(2 ** retry_count)
                        orchestrator.agents['reasoner'].config.max_output_tokens = 1024
                    else:
                        error_container.error("Could not complete synthesis after 3 attempts.")
                        synthesis = None
        
        # Update status to complete
        status_container.write("Complete!")
        return synthesis
        
    except Exception as e:
        error_container.error(f"Error: {str(e)}")
        error_details = st.expander("Show error details")
        with error_details:
            st.code(traceback.format_exc())
        return None

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

def display_message(message: dict):
    """Display a chat message with clean formatting."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    message_type = message.get('type', '')
    
    if role == 'user':
        with st.chat_message("user"):
            st.markdown(content)
    
    elif role == 'assistant':
        avatar = message.get("avatar", "🤖")
        
        # Clean title formatting
        if message_type == "specialist":
            # Remove markdown and clean domain name
            domain = message.get("domain", "")
            domain = domain.replace("_", " ").replace("*", "").strip()
            title = f"{domain.title()} Specialist"
        else:
            title_map = {
                "initial_analysis": "Initial Analysis",
                "synthesis": "Final Synthesis",
                "suggestions": "Follow-up Questions"
            }
            title = title_map.get(message_type, "Assistant")
        
        with st.chat_message("assistant", avatar=avatar):
            # Display clean title
            st.markdown(f"### {title}")
            st.markdown("---")
            
            # Add AI content warning for synthesis
            if message_type == "synthesis":
                st.caption("⚠️ This content is AI-generated and should be reviewed for accuracy.")
            
            # Clean and display content
            if content:
                # Remove any markdown artifacts from content
                content = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', content)
                st.markdown(content)
            
            # Handle message actions
            if message_type == "synthesis":
                col1, col2 = st.columns([1, 4])
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
                st.markdown("### Follow-up Questions")
                for idx, (headline, full_question) in enumerate(message.get("suggestions", [])):
                    if st.button(f"💡 {headline}", key=f"suggest_{message_type}_{idx}_{hash(str(headline))}_{int(time.time())}"):
                        st.session_state.next_prompt = full_question
                        st.rerun()
            elif message_type != "suggestions":
                if st.button("📋 Copy", key=f"copy_{message_type}_{hash(content)}_{int(time.time())}"):
                    copy_to_clipboard(content)

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