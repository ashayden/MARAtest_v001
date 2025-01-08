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
    
    def process_input(self, prompt: str) -> Generator[Dict[str, any], None, None]:
        """Process input and yield messages for display."""
        try:
            # Initial Analysis
            yield {
                "type": "status",
                "content": "Performing initial analysis..."
            }
            
            initial_analysis = yield from self._generate_initial_analysis(prompt)
            
            # Extract and process specialists
            specialists = self._extract_specialists(initial_analysis)
            
            for specialist in specialists:
                yield {
                    "type": "status",
                    "content": f"Consulting {specialist['domain'].title()} specialist..."
                }
                
                specialist_response = yield from self._generate_specialist_response(
                    specialist=specialist,
                    prompt=prompt,
                    initial_analysis=initial_analysis
                )
                
                # Add delay between specialists
                time.sleep(2)
            
            # Generate synthesis
            if specialists:
                yield {
                    "type": "status",
                    "content": "Synthesizing insights..."
                }
                
                synthesis = yield from self._generate_synthesis_response(
                    prompt=prompt,
                    specialists=specialists
                )
                
                yield {
                    "type": "status",
                    "content": "Generating follow-up questions..."
                }
                
                suggestions = self._generate_suggestions(synthesis)
                if suggestions:
                    yield {
                        "type": "suggestions",
                        "content": suggestions
                    }
            
            yield {
                "type": "status",
                "content": "Complete!"
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "content": str(e)
            }
    
    def _generate_initial_analysis(self, prompt: str) -> Generator[str, None, None]:
        """Generate initial analysis with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._initializer_config
        )
        
        prompt_template = """Analyze the following topic and provide a structured analysis with exactly 3 main sections.
        Each section should focus on a key aspect that requires deeper investigation.
        
        Follow these strict markdown formatting rules:
        1. Start with a creative title using a single '#' header
        2. Use '###' for main section headers
        3. Use '####' for subsection headers
        4. Use '**' for bold text (not __ or ***)
        5. Use '*' for italic text (not _)
        6. Use '- ' for bullet points
        7. Use '1. ' style for numbered lists
        8. Add blank lines before and after headers
        9. Use consistent spacing (single blank line between paragraphs)
        10. Do not use horizontal rules or other decorative elements
        
        Format your response as:
        # [Creative Topic-Specific Title]
        
        ### 1. [First Key Aspect]
        [Detailed analysis with proper markdown formatting]
        
        ### 2. [Second Key Aspect]
        [Detailed analysis with proper markdown formatting]
        
        ### 3. [Third Key Aspect]
        [Detailed analysis with proper markdown formatting]
        
        Topic: {prompt}
        """
        
        response = model.generate_content(
            prompt_template.format(prompt=prompt),
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield {
                    "type": "initial_analysis",
                    "content": full_response,
                    "streaming": True
                }
        
        yield {
            "type": "initial_analysis",
            "content": full_response,
            "streaming": False
        }
        
        return full_response
    
    def _extract_specialists(self, analysis: str) -> List[Dict[str, str]]:
        """Extract required specialists from analysis."""
        specialists = []
        sections = re.findall(r'\d+\.\s+([^:]+):', analysis)
        
        for section in sections[:3]:
            topic = section.strip()
            domain = self._map_topic_to_domain(topic.lower())
            
            specialist = {
                'domain': domain,
                'expertise': topic,
                'focus': f"Analyze and provide insights about {topic.lower()}",
                'avatar': self._get_domain_avatar(domain)
            }
            specialists.append(specialist)
        
        return specialists[:3]
    
    def _map_topic_to_domain(self, topic: str) -> str:
        """Map a topic to a specific domain."""
        domain_mappings = {
            ('ecosystem', 'environment', 'nature'): 'ecology',
            ('culture', 'society', 'community'): 'sociology',
            ('history', 'heritage', 'past'): 'history',
            ('geography', 'location', 'terrain'): 'geography',
            ('conservation', 'preservation'): 'conservation',
            ('wildlife', 'animals', 'species'): 'biology'
        }
        
        for keywords, domain in domain_mappings.items():
            if any(word in topic for word in keywords):
                return domain
        
        return 'science'
    
    def _generate_specialist_response(self,
                                   specialist: Dict[str, str],
                                   prompt: str,
                                   initial_analysis: str) -> Generator[Dict[str, any], None, None]:
        """Generate specialist response with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._specialist_config
        )
        
        prompt_template = """Based on your expertise in {expertise}, provide a focused analysis of the following topic.
        
        Follow these strict markdown formatting rules:
        1. Start with a creative title using a single '#' header
        2. Use '###' for main section headers
        3. Use '####' for subsection headers
        4. Use '**' for bold text (not __ or ***)
        5. Use '*' for italic text (not _)
        6. Use '- ' for bullet points
        7. Use '1. ' style for numbered lists
        8. Add blank lines before and after headers
        9. Use consistent spacing (single blank line between paragraphs)
        10. Do not use horizontal rules or other decorative elements
        
        Format your response as:
        # [Creative Topic-Specific Title]
        
        ### [First Main Section]
        [Analysis with proper markdown formatting]
        
        ### [Second Main Section]
        [Analysis with proper markdown formatting]
        
        Topic to analyze: {prompt}
        
        Consider this context:
        {initial_analysis}
        
        Focus your analysis on: {focus}
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
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield {
                    "type": "specialist",
                    "content": full_response,
                    "domain": specialist['domain'],
                    "avatar": specialist['avatar'],
                    "streaming": True
                }
        
        yield {
            "type": "specialist",
            "content": full_response,
            "domain": specialist['domain'],
            "avatar": specialist['avatar'],
            "streaming": False
        }
        
        return full_response
    
    def _generate_synthesis_response(self,
                                   prompt: str,
                                   specialists: List[Dict[str, str]]) -> Generator[Dict[str, any], None, None]:
        """Generate synthesis response with streaming."""
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=self._synthesis_config
        )
        
        specialist_info = "\n".join(
            f"- {s['expertise']}"
            for s in specialists
        )
        
        prompt_template = """Create a comprehensive synthesis for the following topic:
        {prompt}
        
        Drawing from these areas of expertise:
        {specialist_info}
        
        Follow these strict markdown formatting rules:
        1. Start with a creative title using a single '#' header
        2. Use '###' for main section headers
        3. Use '####' for subsection headers
        4. Use '**' for bold text (not __ or ***)
        5. Use '*' for italic text (not _)
        6. Use '- ' for bullet points
        7. Use '1. ' style for numbered lists
        8. Add blank lines before and after headers
        9. Use consistent spacing (single blank line between paragraphs)
        10. Do not use horizontal rules or other decorative elements
        
        Format your response as:
        # [Creative Topic-Specific Title]
        
        ### Key Insights
        [Integrated analysis with proper markdown formatting]
        
        ### Synthesis of Perspectives
        [Combined insights with proper markdown formatting]
        
        ### Conclusions and Implications
        [Final analysis with proper markdown formatting]
        """
        
        response = model.generate_content(
            prompt_template.format(
                prompt=prompt,
                specialist_info=specialist_info
            ),
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield {
                    "type": "synthesis",
                    "content": full_response,
                    "streaming": True
                }
        
        yield {
            "type": "synthesis",
            "content": full_response,
            "streaming": False
        }
        
        return full_response
    
    def _generate_suggestions(self, content: str) -> List[tuple]:
        """Generate follow-up suggestions."""
        try:
            if not self._rate_limiter.wait_if_needed(timeout=5):
                return []
            
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 30,
                    'max_output_tokens': 512
                }
            )
            
            prompt = f"""Generate 3 follow-up questions based on the key points in this content.
            Format as:
            Q1: [question]
            Q2: [question]
            Q3: [question]
            
            Content: {content[:2000]}
            """
            
            response = model.generate_content(prompt)
            suggestions = []
            
            for line in response.text.split('\n'):
                line = line.strip()
                if line.startswith('Q') and ':' in line:
                    question = line.split(':', 1)[1].strip()
                    words = question.split()
                    headline = ' '.join(words[:5]) + '...'
                    suggestions.append((headline, question))
            
            return suggestions[:3]
            
        except Exception:
            return []
    
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