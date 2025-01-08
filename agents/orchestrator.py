"""Orchestrator for managing collaborative agent interactions."""
from typing import List, Dict, Generator
from .base_template import BaseAgent, AgentConfig
import re
import time

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
        2. Determine which specialized agents should be consulted (maximum 3 specialists)
        3. Structure the initial response framework
        4. For each identified domain (up to 3):
           DOMAIN: [domain name in lowercase]
           EXPERTISE: [specific areas of expertise needed]
           FOCUS: [key aspects to analyze]
        5. Maintain coherence across agent contributions
        
        Format your analysis with clear sections:
        1. Overview of the topic
        2. Key Topics and Required Expertise (list exactly what is needed for each specialist)
        3. Analysis Framework
        
        Keep your analysis technical and focused.
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
    
    def extract_specialists_from_analysis(self, analysis: str) -> List[Dict[str, str]]:
        """Extract specialist definitions directly from the initial analysis."""
        specialists = []
        
        # Find all domain definitions in the analysis
        pattern = r"DOMAIN:\s*([^\n]+).*?EXPERTISE:\s*([^\n]+).*?FOCUS:\s*([^\n]+)"
        matches = re.finditer(pattern, analysis, re.DOTALL)
        
        for match in matches:
            domain = match.group(1).strip().lower()
            expertise = match.group(2).strip()
            focus = match.group(3).strip()
            
            if domain and expertise and focus:
                specialists.append({
                    'domain': domain,
                    'expertise': expertise,
                    'focus': focus
                })
        
        return specialists[:3]  # Limit to 3 specialists maximum
    
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
            
            # Get rate limiter instance
            from .base_template import RateLimiter
            rate_limiter = RateLimiter.get_instance()
            
            try:
                # Start initial analysis with rate limit handling
                yield "### INITIAL_ANALYSIS:\n"
                initial_response = ""
                
                # Wait for rate limiter with timeout
                rate_limiter.wait_if_needed(timeout=10)  # 10 second timeout
                
                # Reduce token limit for initial analysis
                self.agents['initializer'].config.max_output_tokens = 1024
                
                # Generate initial analysis with retry logic
                retry_count = 0
                while retry_count < 3:
                    try:
                        for chunk in self.agents['initializer'].generate_response(normalized_input, stream=True):
                            if chunk:
                                initial_response += chunk
                                if stream:
                                    yield chunk
                        break  # Success, exit retry loop
                    except Exception as e:
                        retry_count += 1
                        if retry_count < 3:
                            # Wait with exponential backoff
                            time.sleep(2 ** retry_count)
                            # Reduce tokens for retry
                            self.agents['initializer'].config.max_output_tokens = 512
                            continue
                        else:
                            raise e
                
                if not initial_response:
                    yield "Error: Could not generate initial analysis after retries."
                    return
                
                # Extract specialists directly from initial analysis
                specialists = self.extract_specialists_from_analysis(initial_response)
                
                if not specialists:
                    yield "\nNo domain specialists identified. Proceeding with synthesis..."
                    return
                
                # Process specialists in sequence with rate limiting
                specialist_responses = []  # Track all specialist responses
                
                for specialist in specialists[:3]:  # Limit to 3 specialists
                    domain = specialist['domain']
                    try:
                        # Add delay between specialists
                        time.sleep(2)
                        rate_limiter.wait_if_needed(timeout=10)
                        
                        yield f"\n### SPECIALIST: {domain}\n"
                        
                        # Create specialist with reduced token limit
                        self.agents[domain] = self.create_specialist(
                            domain=domain,
                            expertise=specialist['expertise'],
                            focus_areas=specialist['focus']
                        )
                        self.agents[domain].config.max_output_tokens = 1024
                        
                        # Generate specialist response with retry
                        retry_count = 0
                        specialist_response = ""
                        success = False
                        
                        while retry_count < 3 and not success:
                            try:
                                for chunk in self.agents[domain].generate_response(
                                    normalized_input,
                                    previous_responses=[initial_response],
                                    stream=True
                                ):
                                    if chunk:
                                        specialist_response += chunk
                                        if stream:
                                            yield chunk
                                success = True  # Mark as successful if no exception
                            except Exception as e:
                                retry_count += 1
                                if retry_count < 3:
                                    time.sleep(2 ** retry_count)
                                    self.agents[domain].config.max_output_tokens = 512
                                    continue
                                else:
                                    yield f"\nError with {domain} specialist: {str(e)}"
                                    break
                        
                        if success and specialist_response:
                            specialist_responses.append({
                                'domain': domain,
                                'response': specialist_response
                            })
                        
                    except Exception as e:
                        yield f"\nError with {domain} specialist: {str(e)}"
                        continue
                
                # Only proceed with synthesis if we have specialist responses
                if specialist_responses:
                    try:
                        yield "\n### FINAL_SYNTHESIS:\n"
                        
                        # Wait for rate limiter
                        rate_limiter.wait_if_needed(timeout=15)
                        
                        # Prepare synthesis input
                        synthesis_input = [{'text': initial_response}]
                        synthesis_input.extend({'text': resp['response']} for resp in specialist_responses)
                        
                        # Generate synthesis with retry
                        retry_count = 0
                        synthesis_response = ""
                        success = False
                        
                        while retry_count < 3 and not success:
                            try:
                                for chunk in self.agents['reasoner'].generate_response(
                                    normalized_input,
                                    previous_responses=synthesis_input,
                                    stream=True
                                ):
                                    if chunk and stream:
                                        synthesis_response += chunk
                                        yield chunk
                                success = True
                            except Exception as e:
                                retry_count += 1
                                if retry_count < 3:
                                    time.sleep(2 ** retry_count)
                                    self.agents['reasoner'].config.max_output_tokens = 1024
                                    continue
                                else:
                                    yield f"\nError in synthesis: {str(e)}"
                                    break
                        
                    except Exception as e:
                        yield f"\nError in synthesis: {str(e)}"
                else:
                    yield "\nNo specialist responses available for synthesis."
                
            except Exception as e:
                if "RATE_LIMIT_EXCEEDED" in str(e):
                    yield "Rate limit exceeded. Please wait a moment before trying again."
                else:
                    yield f"Error in processing: {str(e)}"
            
        except Exception as e:
            yield f"Error in input processing: {str(e)}" 

    def process_with_specialists(self, prompt: str, initial_analysis: str, specialists: list) -> list:
        """Process input with specialists with better rate limit handling."""
        specialist_responses = []
        rate_limiter = RateLimiter.get_instance()
        
        for i, specialist in enumerate(specialists[:3]):  # Limit to 3 specialists
            try:
                # Calculate expected token count
                expected_tokens = len(initial_analysis.split()) + len(prompt.split())
                
                # Wait with exponential backoff if needed
                wait_time = rate_limiter.wait_if_needed(
                    timeout=15,  # 15 second timeout
                    token_count=expected_tokens,
                    attempt=0
                )
                
                # Create specialist with reduced token limit
                domain = specialist['domain']
                specialist_agent = self.create_specialist(
                    domain=domain,
                    expertise=specialist.get('expertise', ''),
                    focus_areas=specialist.get('focus', [])
                )
                specialist_agent.config.max_output_tokens = 1024  # Limit tokens
                
                # Generate response with retry logic
                response = None
                retry_count = 0
                while response is None and retry_count < 3:
                    try:
                        response = specialist_agent.generate_response(
                            [{'text': prompt}],
                            previous_responses=[initial_analysis],
                            stream=True
                        )
                        specialist_responses.append({
                            'domain': domain,
                            'response': response,
                            'success': True
                        })
                    except Exception as e:
                        retry_count += 1
                        if retry_count < 3:
                            # Reduce tokens and wait before retry
                            specialist_agent.config.max_output_tokens = 512
                            time.sleep(rate_limiter.calculate_backoff_time(retry_count))
                        else:
                            specialist_responses.append({
                                'domain': domain,
                                'response': str(e),
                                'success': False
                            })
                
                # Add delay between specialists
                if i < len(specialists) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                specialist_responses.append({
                    'domain': specialist.get('domain', 'unknown'),
                    'response': str(e),
                    'success': False
                })
        
        return specialist_responses

    def synthesize_responses(self, prompt: str, initial_analysis: str, specialist_responses: list) -> str:
        """Synthesize all responses with rate limit handling."""
        try:
            # Calculate total tokens
            total_tokens = len(prompt.split()) + len(initial_analysis.split())
            for resp in specialist_responses:
                if resp['success']:
                    total_tokens += len(str(resp['response']).split())
            
            # Wait with exponential backoff
            rate_limiter = RateLimiter.get_instance()
            rate_limiter.wait_if_needed(
                timeout=20,  # 20 second timeout for synthesis
                token_count=total_tokens,
                attempt=0
            )
            
            # Prepare synthesis input
            synthesis_input = [
                {'text': initial_analysis}
            ]
            
            # Add successful specialist responses
            for resp in specialist_responses:
                if resp['success']:
                    synthesis_input.append({'text': str(resp['response'])})
            
            # Generate synthesis with retry
            retry_count = 0
            while retry_count < 3:
                try:
                    return self.agents['reasoner'].generate_response(
                        [{'text': prompt}],
                        previous_responses=synthesis_input,
                        stream=True
                    )
                except Exception as e:
                    retry_count += 1
                    if retry_count < 3:
                        time.sleep(rate_limiter.calculate_backoff_time(retry_count))
                    else:
                        raise e
                    
        except Exception as e:
            return f"Error in synthesis: {str(e)}" 

    def process_specialist(self, specialist: dict, input_parts: list, initial_analysis: str) -> Generator[str, None, None]:
        """Process a single specialist with streaming support."""
        try:
            # Get rate limiter instance
            from .base_template import RateLimiter
            rate_limiter = RateLimiter.get_instance()
            
            # Calculate expected token count
            expected_tokens = len(initial_analysis.split()) + len(str(input_parts).split())
            
            # Wait with exponential backoff if needed
            rate_limiter.wait_if_needed(
                timeout=15,  # 15 second timeout
                token_count=expected_tokens,
                attempt=0
            )
            
            # Create specialist with reduced token limit
            domain = specialist['domain']
            specialist_agent = self.create_specialist(
                domain=domain,
                expertise=specialist.get('expertise', ''),
                focus_areas=specialist.get('focus', '')
            )
            specialist_agent.config.max_output_tokens = 1024  # Limit tokens
            
            # Generate response with retry logic
            retry_count = 0
            while retry_count < 3:
                try:
                    # Stream response chunks
                    for chunk in specialist_agent.generate_response(
                        input_parts,
                        previous_responses=[initial_analysis],
                        stream=True
                    ):
                        if chunk:
                            yield chunk
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < 3:
                        # Reduce tokens and wait before retry
                        specialist_agent.config.max_output_tokens = 512
                        time.sleep(2 ** retry_count)
                        continue
                    else:
                        yield f"\nError with {domain} specialist after {retry_count} retries: {str(e)}"
                        break
            
            # Add delay after specialist
            time.sleep(2)
            
        except Exception as e:
            yield f"\nError processing {specialist.get('domain', 'unknown')} specialist: {str(e)}" 