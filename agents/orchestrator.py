"""Orchestrator for managing collaborative agent interactions."""
from typing import List, Dict, Generator, Optional, Any, Union
import time
import uuid
import re
import streamlit as st
import google.generativeai as genai
from dataclasses import dataclass
from .base import RateLimiter
from utils.state import StateManager

@dataclass
class ModelConfig:
    """Configuration for model generation settings."""
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int

class AgentOrchestrator:
    """Orchestrator for managing collaborative agent interactions."""
    
    def __init__(self) -> None:
        """Initialize the orchestrator with dependencies."""
        self._rate_limiter = RateLimiter.get_instance()
        self._chunk_buffer_size: int = 50
        self._max_retries: int = 3
        
        # Initialize model configurations
        self._initializer_config = ModelConfig(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024
        )
        
        self._specialist_config = ModelConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024
        )
        
        self._synthesis_config = ModelConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=30,
            max_output_tokens=2048
        )
    
    def process_input(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """Process input and yield messages for display."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        try:
            # Initial Analysis
            yield {
                "type": "status",
                "content": "Performing initial analysis..."
            }
            
            initial_analysis = ""
            for message in self._generate_initial_analysis(prompt):
                initial_analysis = message["content"]
                yield message
            
            # Extract and process specialists
            specialists = self._extract_specialists(initial_analysis)
            specialist_responses = []
            
            for specialist in specialists:
                yield {
                    "type": "status",
                    "content": f"Consulting {specialist['domain'].title()} specialist..."
                }
                
                specialist_response = ""
                for message in self._generate_specialist_response(
                    specialist=specialist,
                    prompt=prompt,
                    initial_analysis=initial_analysis
                ):
                    specialist_response = message["content"]
                    yield message
                
                specialist_responses.append({
                    'domain': specialist['domain'],
                    'content': specialist_response
                })
                
                # Add delay between specialists
                time.sleep(1)
            
            # Generate synthesis
            if specialists:
                yield {
                    "type": "status",
                    "content": "Synthesizing insights..."
                }
                
                synthesis = ""
                for message in self._generate_synthesis_response(
                    prompt=prompt,
                    specialists=specialist_responses
                ):
                    synthesis = message["content"]
                    yield message
                
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
                "content": f"Error processing input: {str(e)}"
            }
            raise
    
    def _get_model_config(self, config: ModelConfig) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary for API."""
        return {
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'max_output_tokens': config.max_output_tokens
        }
    
    def _generate_model_response(
        self,
        prompt: str,
        config: ModelConfig,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Generate response from model with proper error handling."""
        try:
            model = genai.GenerativeModel(
                'gemini-2.0-flash-exp',
                generation_config=self._get_model_config(config)
            )
            
            response = model.generate_content(prompt, stream=stream)
            
            if stream:
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                yield response.text
                
        except Exception as e:
            error_msg = f"Model generation error: {str(e)}"
            st.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _map_topic_to_domain(self, topic: str) -> str:
        """Map a topic to a specific domain."""
        if not isinstance(topic, str):
            raise ValueError("Topic must be a string")
            
        domain_mappings = {
            ('ecosystem', 'environment', 'nature', 'climate'): 'ecology',
            ('culture', 'society', 'community', 'social'): 'sociology',
            ('history', 'heritage', 'past', 'historical'): 'history',
            ('geography', 'location', 'terrain', 'spatial'): 'geography',
            ('conservation', 'preservation', 'protection'): 'conservation',
            ('wildlife', 'animals', 'species', 'biological'): 'biology',
            ('food', 'cuisine', 'culinary', 'gastronomy'): 'gastronomy',
            ('economics', 'business', 'finance', 'market'): 'economics',
            ('technology', 'innovation', 'digital', 'tech'): 'technology',
            ('art', 'artistic', 'visual', 'creative'): 'art',
            ('music', 'musical', 'sonic', 'audio'): 'music',
            ('literature', 'literary', 'writing', 'text'): 'literature',
            ('politics', 'political', 'governance', 'policy'): 'politics',
            ('religion', 'religious', 'spiritual', 'faith'): 'religion',
            ('philosophy', 'philosophical', 'ethics', 'thought'): 'philosophy'
        }
        
        topic_lower = topic.lower()
        for keywords, domain in domain_mappings.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _get_domain_avatar(self, domain: str) -> str:
        """Get avatar emoji for domain."""
        if not isinstance(domain, str):
            raise ValueError("Domain must be a string")
            
        avatars = {
            'history': '📚',
            'culture': '🎭',
            'music': '🎵',
            'food': '🍳',
            'gastronomy': '🍽️',
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
            'ecology': '🌳',
            'biology': '🧬',
            'conservation': '♻️',
            'sports': '⚽',
            'religion': '🕊️',
            'philosophy': '🤔',
            'general': '🔍'
        }
        return avatars.get(domain.lower(), '🔍') 

    def _extract_specialists(self, analysis: str) -> List[Dict[str, str]]:
        """Extract required specialists from analysis."""
        if not isinstance(analysis, str):
            raise ValueError("Analysis must be a string")
            
        try:
            specialists = []
            sections = re.findall(r'\d+\.\s+([^:]+):', analysis)
            
            for section in sections[:3]:
                topic = section.strip()
                if not topic:
                    continue
                    
                domain = self._map_topic_to_domain(topic.lower())
                
                specialist = {
                    'domain': domain,
                    'expertise': topic,
                    'focus': f"Analyze and provide insights about {topic.lower()}",
                    'avatar': self._get_domain_avatar(domain)
                }
                specialists.append(specialist)
            
            return specialists[:3]
            
        except Exception as e:
            st.error(f"Error extracting specialists: {str(e)}")
            return []

    def _generate_suggestions(self, content: str) -> List[tuple]:
        """Generate follow-up suggestions."""
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
            
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
            
            prompt = """Generate 3 follow-up questions based on the key points.

            Question Requirements:
            1. Focus on unexplored aspects
            2. Encourage deeper investigation
            3. Connect different themes
            4. Maintain academic rigor
            5. Promote critical thinking

            Formatting Rules:
            1. Start each question with "Q: "
            2. Use clear, concise language
            3. One question per line
            4. No numbering or bullets
            5. No additional formatting

            Format exactly as:
            Q: [First follow-up question]
            Q: [Second follow-up question]
            Q: [Third follow-up question]

            Content: {content}
            """
            
            response = model.generate_content(prompt.format(content=content[:2000]))
            suggestions = []
            
            for line in response.text.split('\n'):
                line = line.strip()
                if line.startswith('Q:'):
                    question = line[2:].strip()
                    if not question:
                        continue
                    words = question.split()
                    headline = ' '.join(words[:5]) + '...'
                    suggestions.append((headline, question))
            
            return suggestions[:3]
            
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            return [] 

    def _generate_initial_analysis(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """Generate initial analysis with streaming."""
        try:
            prompt_template = """Analyze the following topic and provide a structured analysis.

            Content Requirements:
            1. Identify key aspects requiring domain expertise
            2. Break down complex topics into focused areas
            3. Highlight interconnections between aspects
            4. Set foundation for specialist analysis

            Strict Formatting Rules:
            1. Title: Single '#' at the top of the document
            2. Main Sections: '##' for top-level sections
            3. Subsections: '###' for subsections
            4. Lists: Use '- ' for all bullet points
            5. Emphasis: Use '*text*' for italics, '**text**' for bold
            6. Spacing: Single blank line between sections
            7. Numbers: Use '1. ' format for numbered lists
            8. Headers: No periods after numbers in headers
            9. Consistency: Same formatting style throughout
            10. Structure: Clear hierarchy (Title → Sections → Subsections)

            Required Document Structure:
            # [Title]

            ## [First Main Section]
            [Content with proper formatting]

            ## [Second Main Section]
            [Content with proper formatting]

            ## [Third Main Section]
            [Content with proper formatting]

            Topic: {prompt}
            """

            full_response = ""
            for chunk in self._generate_model_response(
                prompt=prompt_template.format(prompt=prompt),
                config=self._initializer_config
            ):
                full_response += chunk
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
            
        except Exception as e:
            error_msg = f"Error in initial analysis: {str(e)}"
            st.error(error_msg)
            yield {
                "type": "error",
                "content": error_msg
            }

    def _generate_specialist_response(
        self,
        specialist: Dict[str, str],
        prompt: str,
        initial_analysis: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate specialist response with streaming."""
        try:
            specialist_template = """As a domain expert in {expertise}, analyze the following topic.

            Expert Analysis Requirements:
            1. Apply deep domain knowledge
            2. Provide evidence-based insights
            3. Identify patterns and implications
            4. Connect to broader context
            5. Highlight critical factors

            Strict Formatting Rules:
            1. Title: Single '#' at the top of the document
            2. Main Sections: '##' for top-level sections
            3. Subsections: '###' for subsections
            4. Lists: Use '- ' for all bullet points
            5. Emphasis: Use '*text*' for italics, '**text**' for bold
            6. Spacing: Single blank line between sections
            7. Numbers: Use '1. ' format for numbered lists
            8. Headers: No periods after numbers in headers
            9. Consistency: Same formatting style throughout
            10. Structure: Clear hierarchy (Title → Sections → Subsections)

            Required Document Structure:
            # [Title]

            ## Analysis
            [Main analysis with proper formatting]

            ## Implications
            [Implications with proper formatting]

            ## Recommendations
            [Recommendations with proper formatting]

            Topic: {prompt}
            Context: {initial_analysis}
            Focus: {focus}
            """

            full_response = ""
            for chunk in self._generate_model_response(
                prompt=specialist_template.format(
                    expertise=specialist['expertise'],
                    prompt=prompt,
                    initial_analysis=initial_analysis,
                    focus=specialist['focus']
                ),
                config=self._specialist_config
            ):
                full_response += chunk
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
            
        except Exception as e:
            error_msg = f"Error in specialist response: {str(e)}"
            st.error(error_msg)
            yield {
                "type": "error",
                "content": error_msg
            }

    def _generate_synthesis_response(
        self,
        prompt: str,
        specialists: List[Dict[str, str]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate synthesis response with streaming."""
        try:
            # Format specialist insights for context
            specialist_info = "\n\n".join(
                f"### {s['domain'].title()} Insights:\n{s['content']}"
                for s in specialists
            )
            
            synthesis_template = """Create a comprehensive synthesis of expert analyses.

            Synthesis Requirements:
            1. Integrate key patterns across analyses
            2. Highlight interconnections
            3. Draw meaningful conclusions
            4. Present cohesive narrative
            5. Identify implications

            Strict Formatting Rules:
            1. Title: Single '#' at the top of the document
            2. Main Sections: '##' for top-level sections
            3. Subsections: '###' for subsections
            4. Lists: Use '- ' for all bullet points
            5. Emphasis: Use '*text*' for italics, '**text**' for bold
            6. Spacing: Single blank line between sections
            7. Numbers: Use '1. ' format for numbered lists
            8. Headers: No periods after numbers in headers
            9. Consistency: Same formatting style throughout
            10. Structure: Clear hierarchy (Title → Sections → Subsections)

            Required Document Structure:
            # [Title]

            ## Key Findings
            [Integrated analysis with proper formatting]

            ## Synthesis
            [Combined insights with proper formatting]

            ## Recommendations
            [Action items with proper formatting]

            Topic: {prompt}

            Expert Insights:
            {specialist_info}
            """

            full_response = ""
            for chunk in self._generate_model_response(
                prompt=synthesis_template.format(
                    prompt=prompt,
                    specialist_info=specialist_info
                ),
                config=self._synthesis_config
            ):
                full_response += chunk
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
            
        except Exception as e:
            error_msg = f"Error in synthesis: {str(e)}"
            st.error(error_msg)
            yield {
                "type": "error",
                "content": error_msg
            } 