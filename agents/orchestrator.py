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
        4. For each identified domain:
           - Specify the exact domain name
           - Define the required expertise and focus areas
           - Outline specific aspects to analyze
        5. Maintain coherence across agent contributions
        
        Format your analysis with clear sections:
        1. Overview of the topic
        2. Key Topics and Required Expertise:
           For each domain list:
           DOMAIN: [domain_name]
           EXPERTISE: [specific areas of expertise needed]
           FOCUS: [key aspects to analyze]
        3. Analysis Framework
        
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
    
    def create_specialist(self, domain: str, expertise: str, focus_areas: str) -> BaseAgent:
        """Create a specialist agent for a specific domain with defined expertise and focus areas."""
        
        # Create specialized prompt based on domain requirements
        specialist_prompt = f"""You are an expert {domain} specialist with specific expertise in {expertise}.

Your analysis should focus on: {focus_areas}

Analysis Requirements:
1. Be thorough and evidence-based
2. Focus exclusively on your assigned domain and expertise
3. Provide specific examples and data
4. Draw meaningful connections within your domain
5. Maintain academic rigor

Format your response with:
1. Clear section headings
2. Well-organized analysis
3. Specific evidence and citations
4. Domain-specific insights

Avoid:
- Straying from your assigned focus areas
- Making unsupported claims
- Repeating information without insight
- Using first-person pronouns or AI references"""
        
        # Create and return the specialist agent
        specialist = BaseAgent(
            config=self.config,
            role=f"{domain}_specialist"
        )
        specialist.system_prompt = specialist_prompt
        return specialist
    
    def identify_required_specialists(self, input_text: str) -> List[Dict[str, str]]:
        """Analyze input to determine required specialist expertise."""
        try:
            # Ensure input_text is a string
            if isinstance(input_text, dict) and 'text' in input_text:
                input_text = input_text['text']
            elif isinstance(input_text, list) and input_text and isinstance(input_text[0], dict) and 'text' in input_text[0]:
                input_text = input_text[0]['text']
            
            prompt = f"""Analyze the following input and identify the required specialist expertise.
            For each required domain, specify:
            DOMAIN: [domain name in lowercase]
            EXPERTISE: [specific areas of expertise needed]
            FOCUS: [key aspects to analyze]
            
            Separate each specialist with '---'
            
            Input: {input_text}
            
            Required specialists:"""
            
            response = self.agents['initializer'].model.generate_content(prompt)
            if not response.text:
                return []
            
            # Parse specialist definitions
            specialists = []
            for specialist_def in response.text.split('---'):
                if not specialist_def.strip():
                    continue
                    
                specialist = {}
                for line in specialist_def.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key == 'domain':
                            specialist['domain'] = value.lower()
                        elif key == 'expertise':
                            specialist['expertise'] = value
                        elif key == 'focus':
                            specialist['focus'] = value
                
                if 'domain' in specialist and 'expertise' in specialist and 'focus' in specialist:
                    specialists.append(specialist)
            
            return specialists if specialists else [{'domain': 'general', 'expertise': 'general analysis', 'focus': 'overall topic analysis'}]
            
        except Exception as e:
            print(f"Error identifying specialists: {str(e)}")
            return [{'domain': 'general', 'expertise': 'general analysis', 'focus': 'overall topic analysis'}]
    
    def process_input(self, user_input: list, stream: bool = True) -> Generator[str, None, None]:
        """Process input through the collaborative agent system."""
        try:
            # Normalize input format
            if isinstance(user_input, str):
                normalized_input = [{'text': user_input}]
            elif isinstance(user_input, dict):
                normalized_input = [user_input]
            elif isinstance(user_input, list):
                if not user_input:
                    normalized_input = [{'text': ''}]
                elif isinstance(user_input[0], str):
                    normalized_input = [{'text': user_input[0]}]
                elif isinstance(user_input[0], dict) and 'text' in user_input[0]:
                    normalized_input = user_input
                else:
                    normalized_input = [{'text': str(user_input[0])}]
            else:
                normalized_input = [{'text': str(user_input)}]

            # Extract text for specialist identification
            input_text = ''
            if normalized_input and isinstance(normalized_input[0], dict) and 'text' in normalized_input[0]:
                input_text = normalized_input[0]['text']
            
            try:
                # Start initial analysis
                yield "### INITIAL_ANALYSIS:\n"
                initial_response = ""
                for chunk in self.agents['initializer'].generate_response(normalized_input, stream=True):
                    initial_response += chunk
                    if stream:
                        yield chunk
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    yield "Rate limit exceeded. Please wait a moment before trying again."
                    return
                else:
                    yield f"Error in initial analysis: {str(e)}"
                    return
            
            try:
                # Identify needed specialists
                specialists = self.identify_required_specialists(input_text)
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    yield "Rate limit exceeded during specialist identification. Please wait a moment before trying again."
                    return
                else:
                    yield f"Error identifying specialists: {str(e)}"
                    return
            
            # Initialize specialist responses as dictionary
            specialist_responses = {'initial_analysis': initial_response}
            previous_responses = [initial_response]  # Start with initial analysis
            
            for specialist in specialists:
                domain = specialist['domain']
                try:
                    # Create new specialist with specific expertise
                    self.agents[domain] = self.create_specialist(
                        domain=domain,
                        expertise=specialist['expertise'],
                        focus_areas=specialist['focus']
                    )
                    
                    # Mark start of specialist response
                    yield f"\n### SPECIALIST: {domain}\n"
                    
                    # Prepare previous responses for this specialist
                    current_previous_responses = previous_responses.copy()
                    
                    specialist_response = ""
                    for chunk in self.agents[domain].generate_response(
                        normalized_input,
                        previous_responses=current_previous_responses,
                        stream=True
                    ):
                        specialist_response += chunk
                        if stream:
                            yield chunk
                    
                    # Store in dictionary and add to previous responses
                    specialist_responses[domain] = specialist_response
                    previous_responses.append(specialist_response)
                
                except Exception as e:
                    if "RATE_LIMIT_EXCEEDED" in str(e):
                        yield f"\nRate limit exceeded for {domain} specialist. Skipping and continuing with synthesis."
                        continue
                    else:
                        yield f"\nError with {domain} specialist: {str(e)}"
                        continue
            
            try:
                # Start final synthesis
                yield "\n### FINAL_SYNTHESIS:\n"
                synthesis_response = ""
                for chunk in self.agents['reasoner'].generate_response(
                    normalized_input,
                    previous_responses=previous_responses,
                    stream=True
                ):
                    synthesis_response += chunk
                    if stream:
                        yield chunk
                
                # Add synthesis to responses
                specialist_responses['final_synthesis'] = synthesis_response
            
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    yield "\nRate limit exceeded during synthesis. Please wait a moment before trying again."
                    return
                else:
                    yield f"\nError in synthesis: {str(e)}"
                    return
        
        except Exception as e:
            yield f"Error in agent collaboration: {str(e)}" 