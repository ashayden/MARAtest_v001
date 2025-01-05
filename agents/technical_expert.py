from .base_template import AgentTemplate
from typing import Dict, Any

class TechnicalExpertAgent(AgentTemplate):
    """Specialized agent for technical documentation and explanations."""
    
    def __init__(self, domain: str = "software development"):
        """
        Initialize the technical expert agent.
        
        Args:
            domain: Specific technical domain of expertise
        """
        # Define technical expert persona
        persona = {
            'tone': 'technical but accessible',
            'style': 'detailed and precise',
            'expertise_level': f'expert in {domain}',
            'communication_style': 'structured and methodical'
        }
        
        # Custom instructions for technical content
        custom_instructions = f"""
        As a technical expert in {domain}, focus on:
        
        1. Technical Accuracy:
           - Use precise terminology
           - Provide accurate technical details
           - Include code examples when relevant
           - Cite best practices and standards
        
        2. Explanation Structure:
           - Start with high-level concepts
           - Break down complex topics
           - Include practical examples
           - Address common pitfalls
        
        3. Implementation Guidance:
           - Step-by-step instructions
           - Configuration details
           - Performance considerations
           - Security implications
        
        4. Documentation Standards:
           - Clear code comments
           - API documentation style
           - Technical diagrams (described in markdown)
           - Version compatibility notes
        
        Response Format:
        # Main Topic
        Brief overview of the technical concept
        
        ## Technical Background
        Essential background information and context
        
        ## Implementation Details
        - Specific steps or components
        - Code examples or configurations
        - Best practices
        
        ## Common Challenges
        Known issues and solutions
        
        ## Best Practices
        Recommended approaches and standards
        
        ## Additional Resources
        Related documentation or references
        """
        
        # Initialize with technical configuration
        super().__init__(
            temperature=0.3,  # Lower temperature for technical precision
            persona=persona,
            custom_instructions=custom_instructions
        )
    
    def preprocess_input(self, prompt: str) -> str:
        """
        Preprocess the input prompt for technical context.
        
        Args:
            prompt: User's technical question
            
        Returns:
            str: Processed prompt with technical context
        """
        # Add technical context markers
        return f"""
        Technical Query:
        {prompt}
        
        Required:
        - Technical accuracy
        - Practical examples
        - Implementation details
        - Best practices
        """
    
    def postprocess_response(self, response: str) -> str:
        """
        Postprocess the response to ensure technical formatting.
        
        Args:
            response: Raw response from the model
            
        Returns:
            str: Formatted technical response
        """
        # Ensure proper markdown formatting
        if not response.startswith('#'):
            response = f"# Technical Response\n\n{response}"
        
        # Add technical disclaimer if not present
        if "Note:" not in response:
            response += "\n\n**Note:** Always refer to official documentation and verify compatibility with your specific environment."
        
        return response 