"""Orchestrator for managing collaborative agent interactions."""
from typing import List, Dict, Generator
from .base_template import BaseAgent, AgentConfig

class AgentOrchestrator:
    """Manages collaboration between specialized agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the orchestrator."""
        self.config = config
        self.agents = {}
        
        # Initialize base agents
        self.initialize_base_agents()
    
    def initialize_base_agents(self):
        """Initialize the core set of specialized agents."""
        # Create configs with fixed settings for base agents
        initializer_config = AgentConfig(
            temperature=0.5,    # Balanced setting for analysis
            top_p=0.9,         # Slightly reduced for more focused analysis
            top_k=40,          # Standard setting
            max_output_tokens=2048  # Standard length
        )
        
        reasoner_config = AgentConfig(
            temperature=0.3,    # Lower temperature for more consistent synthesis
            top_p=0.8,         # More focused on likely tokens
            top_k=30,          # More constrained selection
            max_output_tokens=3072  # Longer limit for comprehensive synthesis
        )
        
        # Initial agent analyzes input and determines needed specialists
        self.agents['initializer'] = BaseAgent(
            initializer_config,
            role="initializer"
        )
        self.agents['initializer'].system_prompt = """You are the initial analysis agent.
        Your role is to:
        1. Analyze user input to identify key topics and required expertise
        2. Determine which specialized agents should be consulted
        3. Structure the initial response framework
        4. Highlight specific aspects for each specialist to address
        5. Maintain coherence across agent contributions
        
        Keep your analysis technical and focused on identifying the required expertise.
        Do not include any first-person pronouns or references to being an AI.
        """
        
        # Reasoning agent synthesizes and refines responses
        self.agents['reasoner'] = BaseAgent(
            reasoner_config,
            role="reasoner"
        )
        self.agents['reasoner'].system_prompt = """You are the synthesis agent responsible for creating the final report.
        
        Report Structure Requirements:
        1. Begin with a creative, topic-specific title
        2. Follow with numbered sections starting with Introduction
        3. Present comprehensive analysis in clear sections
        4. End with conclusion and references if applicable
        5. Include no commentary before or after the report
        
        Critical Guidelines:
        - Never use first-person pronouns or self-references
        - Never include meta-commentary about the analysis process
        - Never discuss revisions or improvements
        - Never add commentary before or after the report
        - Present information directly and professionally
        - Maintain formal academic tone throughout
        
        Content Organization:
        1. Title: Creative and specific to the topic
        2. Introduction: Context and scope
        3. Main Analysis Sections: Clear headings
        4. Conclusion: Key findings and implications
        5. References: When applicable
        
        Formatting Requirements:
        - Use clear section numbering
        - Include descriptive headings
        - Maintain consistent formatting
        - Use markdown for structure
        """
    
    def create_specialist(self, expertise: str) -> BaseAgent:
        """Create a new specialist agent for a specific domain."""
        # Use the main config for specialists (controlled by sidebar)
        specialist = BaseAgent(
            self.config,
            role=f"specialist_{expertise}"
        )
        
        # Dynamic prompt based on expertise
        specialist.system_prompt = f"""You are a specialized expert in {expertise}.
        Your role is to:
        1. Provide deep domain expertise in {expertise}
        2. Analyze aspects relevant to your specialty
        3. Offer unique insights from your field
        4. Connect your knowledge to the broader context
        5. Suggest implications and applications
        
        Focus on:
        - Technical accuracy in {expertise}
        - Practical applications
        - Current best practices
        - Emerging trends
        - Cross-domain implications
        """
        
        return specialist
    
    def identify_required_specialists(self, input_text: str) -> List[str]:
        """Analyze input to determine required specialist expertise."""
        # Ensure input_text is a string
        if isinstance(input_text, dict) and 'text' in input_text:
            input_text = input_text['text']
        elif isinstance(input_text, list) and input_text and isinstance(input_text[0], dict) and 'text' in input_text[0]:
            input_text = input_text[0]['text']
        
        prompt = f"""Analyze the following input and identify the key domains of expertise needed.
        Return only the domain names, separated by commas.
        
        Input: {input_text}
        
        Required expertise:"""
        
        response = self.agents['initializer'].model.generate_content(prompt)
        domains = [d.strip() for d in response.text.split(',')]
        return domains
    
    def process_input(self, user_input: list, stream: bool = True) -> Generator[str, None, None]:
        """Process input through the collaborative agent system."""
        try:
            # Start initial analysis
            yield "### INITIAL_ANALYSIS:\n"
            initial_response = ""
            for chunk in self.agents['initializer'].generate_response(user_input, stream=True):
                initial_response += chunk
                if stream:
                    yield chunk
            
            # Identify needed specialists
            input_text = user_input
            if isinstance(user_input, list) and user_input:
                if isinstance(user_input[0], dict) and 'text' in user_input[0]:
                    input_text = user_input[0]['text']
                elif isinstance(user_input[0], str):
                    input_text = user_input[0]
            elif isinstance(user_input, dict) and 'text' in user_input:
                input_text = user_input['text']
            elif isinstance(user_input, str):
                input_text = user_input
            
            domains = self.identify_required_specialists(input_text)
            
            # Collect specialist responses
            specialist_responses = []
            for domain in domains:
                if domain not in self.agents:
                    self.agents[domain] = self.create_specialist(domain)
                
                # Mark start of specialist response
                yield f"\n### SPECIALIST: {domain}\n"
                
                specialist_response = ""
                for chunk in self.agents[domain].generate_response(
                    user_input,
                    previous_responses=[initial_response] + specialist_responses,
                    stream=True
                ):
                    specialist_response += chunk
                    if stream:
                        yield chunk
                specialist_responses.append(specialist_response)
            
            # Start final synthesis
            yield "\n### FINAL_SYNTHESIS:\n"
            for chunk in self.agents['reasoner'].generate_response(
                user_input,
                previous_responses=[initial_response] + specialist_responses,
                stream=True
            ):
                if stream:
                    yield chunk
        
        except Exception as e:
            yield f"Error in agent collaboration: {str(e)}" 