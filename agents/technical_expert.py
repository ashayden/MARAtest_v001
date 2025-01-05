from .base_template import AgentTemplate
from typing import Dict, Any

class ResponseAgent(AgentTemplate):
    """General-purpose agent for structured responses."""
    
    def __init__(self):
        """Initialize the response agent."""
        # Define general persona
        persona = {
            'tone': 'clear and informative',
            'style': 'well-structured',
            'expertise_level': 'knowledgeable',
            'communication_style': 'engaging and precise'
        }
        
        # General instructions for structured responses
        custom_instructions = """
        Provide comprehensive responses that:
        
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
        
        Response Format:
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
        """
        
        # Initialize with balanced configuration
        super().__init__(
            temperature=0.7,  # Balanced for general responses
            persona=persona,
            custom_instructions=custom_instructions
        )
    
    def preprocess_input(self, prompt: str) -> str:
        """
        Preprocess the input prompt.
        
        Args:
            prompt: User's query
            
        Returns:
            str: Processed prompt
        """
        return f"""
        Query:
        {prompt}
        
        Required:
        - Comprehensive coverage
        - Clear structure
        - Relevant details
        - Balanced perspective
        """
    
    def postprocess_response(self, response: str) -> str:
        """
        Postprocess the response for consistent formatting.
        
        Args:
            response: Raw response from the model
            
        Returns:
            str: Formatted response
        """
        # Ensure proper markdown formatting
        if not response.startswith('#'):
            response = f"# Response\n\n{response}"
        
        return response 