import google.generativeai as genai
from typing import Optional, Dict, Any, Callable, List
from abc import ABC, abstractmethod
from datetime import datetime
from .config import AgentConfig, AgentMode, ResponseFormat

class AgentTemplate(ABC):
    """Abstract base template for creating specialized AI agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent template with configuration."""
        self.config = config or AgentConfig()
        self.model = genai.GenerativeModel(self.config.model_name)
        
        # Set up generation configuration
        self.generation_config = {
            'temperature': self.config.temperature,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': self.config.max_tokens,
        }
        
        # Initialize memory if enabled
        self.conversation_history: List[Dict[str, Any]] = [] if self.config.enable_memory else None
        
        # Initialize knowledge base if enabled
        if self.config.enable_knowledge_base and self.config.knowledge_base_path:
            self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize knowledge base if configured."""
        # Implement knowledge base initialization
        pass
    
    def format_prompt(self, user_input: str) -> str:
        """Format the prompt with appropriate context and instructions."""
        # Start with base prompt
        formatted_prompt = f"""
        Agent Name: {self.config.name}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S') if self.config.include_timestamps else ''}
        Mode: {self.config.mode.value}
        
        Instructions:
        {self.config.custom_instructions or self.get_default_instructions()}
        
        Previous Context:
        {self.get_conversation_context() if self.config.enable_memory else ''}
        
        Current Query:
        {user_input}
        
        Response Template:
        {self.config.get_template(self.config.mode.value)}
        """
        
        return formatted_prompt
    
    def get_default_instructions(self) -> str:
        """Get default instructions based on mode."""
        mode_instructions = {
            AgentMode.CHAT: """
                Provide clear, conversational responses while maintaining professionalism.
                Focus on direct answers and relevant information.
            """,
            AgentMode.ANALYSIS: """
                Conduct thorough analysis with supporting evidence.
                Present findings in a structured, logical manner.
            """,
            AgentMode.REPORT: """
                Generate comprehensive reports with clear sections.
                Include executive summary, analysis, and recommendations.
            """,
            AgentMode.CALCULATION: """
                Show all calculations clearly with explanations.
                Include units and assumptions where relevant.
            """
        }
        return mode_instructions.get(self.config.mode, mode_instructions[AgentMode.CHAT])
    
    def get_conversation_context(self) -> str:
        """Get relevant conversation history."""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-self.config.memory_window_size:]
        return "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in recent_history
        ])
    
    def update_conversation_history(self, user_input: str, response: str):
        """Update conversation history if memory is enabled."""
        if self.config.enable_memory:
            self.conversation_history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now().isoformat()
            })
    
    def format_response(self, response: str) -> str:
        """Format the response according to configuration."""
        if self.config.response_format == ResponseFormat.HTML:
            # Convert markdown to HTML if needed
            return self.markdown_to_html(response)
        elif self.config.response_format == ResponseFormat.PLAIN:
            # Strip formatting if plain text is requested
            return self.strip_formatting(response)
        return response
    
    @staticmethod
    def markdown_to_html(markdown: str) -> str:
        """Convert markdown to HTML."""
        # Implement markdown to HTML conversion
        return markdown  # Placeholder
    
    @staticmethod
    def strip_formatting(text: str) -> str:
        """Remove formatting from text."""
        # Implement formatting removal
        return text  # Placeholder
    
    def generate_response(
        self, 
        prompt: str,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate a response using the configured model and prompts."""
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(prompt)
            
            if stream and stream_callback and self.config.enable_streaming:
                # Stream response with callback
                response_stream = self.model.generate_content(
                    formatted_prompt,
                    generation_config=self.generation_config,
                    stream=True
                )
                
                final_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        final_response += chunk.text
                        stream_callback(chunk.text)
                
                formatted_response = self.format_response(final_response)
                self.update_conversation_history(prompt, formatted_response)
                return formatted_response
            else:
                # Generate complete response
                response = self.model.generate_content(
                    formatted_prompt,
                    generation_config=self.generation_config
                )
                
                formatted_response = self.format_response(response.text)
                self.update_conversation_history(prompt, formatted_response)
                return formatted_response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if stream_callback:
                stream_callback(error_msg)
            return error_msg 