from .base_template import AgentTemplate
from .config import AgentConfig, AgentMode
from typing import Optional

class ResponseAgent(AgentTemplate):
    """General-purpose agent for structured responses."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the response agent."""
        # Create base configuration
        response_config = config or AgentConfig()
        response_config.update(
            name="AI Assistant",
            description="General-purpose response agent",
            mode=AgentMode.CHAT,
            temperature=0.7,  # Balanced for general responses
            persona={
                'tone': 'clear and informative',
                'style': 'well-structured',
                'expertise_level': 'knowledgeable',
                'communication_style': 'engaging and precise'
            }
        )
        
        # Add response template if not present
        if 'structured_response' not in response_config.output_templates:
            response_config.add_template('structured_response', """
            # Topic Overview
            Clear introduction to the subject
            
            ## Background
            Context and foundational information
            
            ## Key Points
            - Main aspects
            - Important details
            - Notable features
            - Significant elements
            
            ## Analysis
            Deeper examination of the topic
            
            ## Practical Relevance
            Real-world applications or implications
            
            ## Further Information
            Additional resources or references
            """)
        
        super().__init__(response_config)
    
    def format_prompt(self, user_input: str) -> str:
        """Format the prompt with appropriate context."""
        base_prompt = super().format_prompt(user_input)
        
        # Add response-specific context
        response_context = """
        Guidelines:
        1. Information Quality:
           - Present accurate information
           - Include relevant details
           - Support claims with evidence
           - Maintain objectivity
        
        2. Organization:
           - Present logical flow of ideas
           - Use clear categorization
           - Provide smooth transitions
           - Maintain consistent structure
        
        3. Engagement:
           - Use clear language
           - Provide relevant examples
           - Address key aspects
           - Maintain reader interest
        
        4. Presentation:
           - Use appropriate formatting
           - Include section headers
           - Highlight key points
           - Ensure readability
        """
        
        return f"{base_prompt}\n{response_context}" 