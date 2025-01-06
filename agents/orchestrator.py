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
        # Initial agent analyzes input and determines needed specialists
        self.agents['initializer'] = BaseAgent(
            self.config,
            role="initializer"
        )
        self.agents['initializer'].system_prompt = """You are the initial analysis agent.
        Your role is to:
        1. Analyze user input to identify key topics and required expertise
        2. Determine which specialized agents should be consulted
        3. Structure the initial response framework
        4. Highlight specific aspects for each specialist to address
        5. Maintain coherence across agent contributions
        """
        
        # Reasoning agent synthesizes and refines responses
        self.agents['reasoner'] = BaseAgent(
            self.config,
            role="reasoner"
        )
        self.agents['reasoner'].system_prompt = """You are the reasoning and synthesis agent.
        Your role is to:
        1. Analyze contributions from all agents
        2. Identify connections and patterns
        3. Resolve any conflicts or inconsistencies
        4. Synthesize a coherent final response
        5. Ensure completeness and clarity
        """
    
    def create_specialist(self, expertise: str) -> BaseAgent:
        """Create a new specialist agent for a specific domain."""
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
            # Get initial analysis
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
            
            # Synthesize final response
            for chunk in self.agents['reasoner'].generate_response(
                user_input,
                previous_responses=[initial_response] + specialist_responses,
                stream=True
            ):
                if stream:
                    yield chunk
        
        except Exception as e:
            yield f"Error in agent collaboration: {str(e)}" 