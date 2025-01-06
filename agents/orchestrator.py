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
        self.agents['reasoner'].system_prompt = """You are the synthesis agent responsible for creating the final user-facing response.
        
        Key Requirements:
        1. Create a clear, concise synthesis of specialist insights
        2. Present information in a professional, authoritative tone
        3. Focus on factual content and practical insights
        4. Organize content logically with clear headings
        5. Maintain objectivity and avoid personal pronouns
        
        Important Guidelines:
        - Never use first-person pronouns (I, we, our)
        - Never reference being an AI or assistant
        - Never mention the analysis process or other agents
        - Present information directly without meta-commentary
        - Use clear headings to organize content
        - Keep paragraphs focused and concise
        - End with clear next steps or areas to explore
        
        Structure:
        1. Start with a concise overview
        2. Present key aspects under clear headings
        3. Include practical details and insights
        4. End with relevant areas for further exploration
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
            if isinstance(user_input, list) and user_input and 'text' in user_input[0]:
                domains = self.identify_required_specialists(user_input[0]['text'])
            else:
                domains = []
            
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