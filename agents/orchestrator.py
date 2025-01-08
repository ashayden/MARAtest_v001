"""Orchestrator for managing collaborative agent interactions."""
from typing import List, Dict, Generator, Optional
import time
import uuid
import re
import streamlit as st
import google.generativeai as genai

from .base import RateLimiter
from utils.state import StateManager

class AgentOrchestrator:
    def __init__(self):
        """Initialize the orchestrator with dependencies."""
        self._rate_limiter = RateLimiter.get_instance()
        self._chunk_buffer_size = 50
        self._max_retries = 3
        self.state_manager = StateManager()
        
        # Initialize model configurations
        self._initializer_config = {
            'temperature': 0.5,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1024
        }
        
        self._specialist_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 1024
        }
        
        self._synthesis_config = {
            'temperature': 0.3,
            'top_p': 0.8,
            'top_k': 30,
            'max_output_tokens': 2048
        }
    
    def process_with_streaming(self, 
                             prompt: str, 
                             message_id: str, 
                             timeout: Optional[float] = None) -> bool:
        """Process input with proper streaming support and error handling."""
        try:
            # Initialize message
            self.state_manager.update_message(
                message_id=message_id,
                content="",
                is_complete=False,
                message_type="processing"
            )
            
            chunk_buffer = []
            retry_count = 0
            
            while retry_count < self._max_retries:
                try:
                    # Wait for rate limit with timeout
                    if not self._rate_limiter.wait_if_needed(timeout):
                        raise TimeoutError("Rate limit timeout exceeded")
                    
                    # Generate initial analysis
                    for chunk in self._generate_initial_analysis(prompt):
                        if chunk:
                            chunk_buffer.append(chunk)
                            
                            # Update state when buffer is full
                            if len(chunk_buffer) >= self._chunk_buffer_size:
                                combined_chunk = "".join(chunk_buffer)
                                self.state_manager.update_message(
                                    message_id=message_id,
                                    content=combined_chunk,
                                    is_complete=False
                                )
                                chunk_buffer = []
                                st.rerun()
                    
                    # Process final buffer
                    if chunk_buffer:
                        final_chunk = "".join(chunk_buffer)
                        initial_analysis = self.state_manager.update_message(
                            message_id=message_id,
                            content=final_chunk,
                            is_complete=True
                        )
                        
                        # Extract and process specialists
                        specialists = self._extract_specialists(initial_analysis.content)
                        for specialist in specialists:
                            specialist_id = str(uuid.uuid4())
                            success = self._process_specialist(
                                specialist=specialist,
                                prompt=prompt,
                                message_id=specialist_id,
                                initial_analysis=initial_analysis.content,
                                timeout=timeout
                            )
                            if not success:
                                return False
                        
                        # Generate synthesis if needed
                        if specialists:
                            synthesis_id = str(uuid.uuid4())
                            success = self._generate_synthesis(
                                prompt=prompt,
                                message_id=synthesis_id,
                                specialists=specialists,
                                timeout=timeout
                            )
                            if not success:
                                return False
                    
                    return True
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self._max_retries:
                        raise
                    time.sleep(2 ** retry_count)  # Exponential backoff
            
        except Exception as e:
            self.state_manager.update_message(
                message_id=message_id,
                content=f"Error: {str(e)}",
                is_complete=True,
                message_type="error"
            )
            return False
    
    def _generate_initial_analysis(self, prompt: str) -> Generator[str, None, None]:
        """Generate initial analysis with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._initializer_config
        )
        
        prompt_template = """Analyze the following topic to identify key areas requiring specialist expertise.
        For each identified domain (maximum 3):
        DOMAIN: [domain name in lowercase]
        EXPERTISE: [specific areas of expertise needed]
        FOCUS: [key aspects to analyze]
        
        Topic: {prompt}
        """
        
        response = model.generate_content(
            prompt_template.format(prompt=prompt),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def _extract_specialists(self, analysis: str) -> List[Dict[str, str]]:
        """Extract required specialists from analysis."""
        specialists = []
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
                    'focus': focus,
                    'avatar': self._get_domain_avatar(domain)
                })
        
        return specialists[:3]  # Limit to 3 specialists
    
    def _get_domain_avatar(self, domain: str) -> str:
        """Get avatar emoji for domain."""
        avatars = {
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
            'philosophy': '🤔'
        }
        return avatars.get(domain.lower(), '🔍')
    
    def _process_specialist(self,
                          specialist: Dict[str, str],
                          prompt: str,
                          message_id: str,
                          initial_analysis: str,
                          timeout: Optional[float] = None) -> bool:
        """Process a single specialist with streaming."""
        try:
            self.state_manager.update_message(
                message_id=message_id,
                content="",
                is_complete=False,
                message_type="specialist",
                domain=specialist.get('domain'),
                avatar=specialist.get('avatar', '🔍')
            )
            
            chunk_buffer = []
            
            # Wait for rate limit
            if not self._rate_limiter.wait_if_needed(timeout):
                raise TimeoutError("Rate limit timeout exceeded")
            
            # Generate specialist response
            for chunk in self._generate_specialist_response(
                specialist=specialist,
                prompt=prompt,
                initial_analysis=initial_analysis
            ):
                if chunk:
                    chunk_buffer.append(chunk)
                    
                    if len(chunk_buffer) >= self._chunk_buffer_size:
                        combined_chunk = "".join(chunk_buffer)
                        self.state_manager.update_message(
                            message_id=message_id,
                            content=combined_chunk,
                            is_complete=False
                        )
                        chunk_buffer = []
                        st.rerun()
            
            # Process final buffer
            if chunk_buffer:
                final_chunk = "".join(chunk_buffer)
                self.state_manager.update_message(
                    message_id=message_id,
                    content=final_chunk,
                    is_complete=True
                )
            
            return True
            
        except Exception as e:
            self.state_manager.update_message(
                message_id=message_id,
                content=f"Error in {specialist.get('domain', 'specialist')}: {str(e)}",
                is_complete=True,
                message_type="error"
            )
            return False
    
    def _generate_specialist_response(self,
                                   specialist: Dict[str, str],
                                   prompt: str,
                                   initial_analysis: str) -> Generator[str, None, None]:
        """Generate specialist response with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._specialist_config
        )
        
        prompt_template = """You are an expert {domain} specialist with expertise in {expertise}.
        
        Focus your analysis on: {focus}
        
        Analyze the following topic from your specialist perspective:
        {prompt}
        
        Consider this initial analysis:
        {initial_analysis}
        
        Provide a detailed specialist analysis focusing on your domain expertise.
        """
        
        response = model.generate_content(
            prompt_template.format(
                domain=specialist['domain'],
                expertise=specialist['expertise'],
                focus=specialist['focus'],
                prompt=prompt,
                initial_analysis=initial_analysis
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def _generate_synthesis(self,
                          prompt: str,
                          message_id: str,
                          specialists: List[Dict[str, str]],
                          timeout: Optional[float] = None) -> bool:
        """Generate synthesis with streaming."""
        try:
            self.state_manager.update_message(
                message_id=message_id,
                content="",
                is_complete=False,
                message_type="synthesis",
                avatar="📊"
            )
            
            chunk_buffer = []
            
            # Wait for rate limit
            if not self._rate_limiter.wait_if_needed(timeout):
                raise TimeoutError("Rate limit timeout exceeded")
            
            # Generate synthesis
            for chunk in self._generate_synthesis_response(
                prompt=prompt,
                specialists=specialists
            ):
                if chunk:
                    chunk_buffer.append(chunk)
                    
                    if len(chunk_buffer) >= self._chunk_buffer_size:
                        combined_chunk = "".join(chunk_buffer)
                        self.state_manager.update_message(
                            message_id=message_id,
                            content=combined_chunk,
                            is_complete=False
                        )
                        chunk_buffer = []
                        st.rerun()
            
            # Process final buffer
            if chunk_buffer:
                final_chunk = "".join(chunk_buffer)
                self.state_manager.update_message(
                    message_id=message_id,
                    content=final_chunk,
                    is_complete=True
                )
            
            return True
            
        except Exception as e:
            self.state_manager.update_message(
                message_id=message_id,
                content=f"Error generating synthesis: {str(e)}",
                is_complete=True,
                message_type="error"
            )
            return False
    
    def _generate_synthesis_response(self,
                                   prompt: str,
                                   specialists: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Generate synthesis response with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._synthesis_config
        )
        
        specialist_info = "\n".join(
            f"- {s['domain'].title()} Specialist: expertise in {s['expertise']}"
            for s in specialists
        )
        
        prompt_template = """Create a comprehensive synthesis report for the following topic:
        {prompt}
        
        Incorporating insights from these specialists:
        {specialist_info}
        
        Requirements:
        1. Begin with a clear title
        2. Organize content in numbered sections
        3. Integrate specialist perspectives
        4. Maintain academic tone
        5. End with key conclusions
        
        Format the report in markdown with clear section headings.
        """
        
        response = model.generate_content(
            prompt_template.format(
                prompt=prompt,
                specialist_info=specialist_info
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text 