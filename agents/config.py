from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

class ResponseFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"

class AgentMode(Enum):
    CHAT = "chat"
    ANALYSIS = "analysis"
    REPORT = "report"
    CALCULATION = "calculation"
    VOICE = "voice"

class SuggestionType(Enum):
    DEEP_DIVE = "deep_dive"
    RELATED_TOPICS = "related_topics"
    CONTINUE_CURRENT = "continue_current"
    SURPRISING_CONNECTIONS = "surprising_connections"

@dataclass
class MediaConfig:
    """Configuration for media handling."""
    allowed_image_types: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "gif"])
    allowed_video_types: List[str] = field(default_factory=lambda: ["mp4", "webm", "ogg"])
    max_file_size_mb: int = 10
    enable_image_analysis: bool = True
    enable_video_analysis: bool = True

@dataclass
class VoiceConfig:
    """Configuration for voice interactions."""
    enable_voice_input: bool = True
    enable_voice_output: bool = True
    voice_language: str = "en-US"
    voice_gender: str = "neutral"
    speech_rate: float = 1.0
    pitch: float = 1.0

@dataclass
class SuggestionsConfig:
    """Configuration for follow-up suggestions."""
    enable_suggestions: bool = True
    max_suggestions: int = 4
    suggestion_types: List[SuggestionType] = field(default_factory=lambda: [
        SuggestionType.DEEP_DIVE,
        SuggestionType.RELATED_TOPICS,
        SuggestionType.CONTINUE_CURRENT,
        SuggestionType.SURPRISING_CONNECTIONS
    ])
    min_confidence_score: float = 0.7

@dataclass
class AgentConfig:
    """Configuration for agent behavior and features."""
    
    # Core Settings
    name: str = "AI Assistant"
    description: str = "General-purpose AI assistant"
    version: str = "1.0.0"
    
    # Model Configuration
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Feature Flags
    enable_streaming: bool = True
    enable_memory: bool = True
    enable_knowledge_base: bool = False
    enable_file_attachments: bool = True
    enable_calculations: bool = False
    enable_citations: bool = False
    enable_web_search: bool = False
    enable_data_visualization: bool = False
    enable_voice_chat: bool = True
    enable_suggestions: bool = True
    
    # Export Configuration
    available_formats: List[ResponseFormat] = field(default_factory=lambda: [
        ResponseFormat.MARKDOWN,
        ResponseFormat.PDF,
        ResponseFormat.PLAIN,
        ResponseFormat.SPREADSHEET
    ])
    
    # Media Configuration
    media_config: MediaConfig = field(default_factory=MediaConfig)
    
    # Voice Configuration
    voice_config: VoiceConfig = field(default_factory=VoiceConfig)
    
    # Suggestions Configuration
    suggestions_config: SuggestionsConfig = field(default_factory=SuggestionsConfig)
    
    # Response Configuration
    response_format: ResponseFormat = ResponseFormat.MARKDOWN
    max_response_length: int = 4000
    include_confidence_scores: bool = False
    include_sources: bool = False
    include_timestamps: bool = True
    
    # Memory Configuration
    memory_window_size: int = 10
    memory_type: str = "conversation"  # conversation, semantic, episodic
    
    # Knowledge Base Configuration
    knowledge_base_path: Optional[str] = None
    kb_embedding_model: Optional[str] = None
    kb_chunk_size: int = 500
    
    # Persona Configuration
    persona: Dict[str, str] = field(default_factory=lambda: {
        'tone': 'professional',
        'style': 'clear and direct',
        'expertise_level': 'expert',
        'communication_style': 'structured'
    })
    
    # Mode Configuration
    mode: AgentMode = AgentMode.CHAT
    mode_specific_config: Dict[str, Any] = field(default_factory=dict)
    
    # Custom Instructions
    custom_instructions: Optional[str] = None
    custom_templates: Dict[str, str] = field(default_factory=dict)
    
    # Output Formatting
    output_templates: Dict[str, str] = field(default_factory=lambda: {
        'report': """
        # {title}
        
        ## Executive Summary
        {summary}
        
        ## Analysis
        {analysis}
        
        ## Key Findings
        {findings}
        
        ## Recommendations
        {recommendations}
        
        ## Supporting Details
        {details}
        """,
        'analysis': """
        # Analysis: {topic}
        
        ## Overview
        {overview}
        
        ## Key Points
        {points}
        
        ## Detailed Analysis
        {analysis}
        
        ## Implications
        {implications}
        """,
        'calculation': """
        # {calculation_type}
        
        ## Input Parameters
        {parameters}
        
        ## Calculations
        {calculations}
        
        ## Results
        {results}
        
        ## Notes
        {notes}
        """
    })
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def enable_feature(self, feature_name: str):
        """Enable a specific feature."""
        feature_flag = f"enable_{feature_name}"
        if hasattr(self, feature_flag):
            setattr(self, feature_flag, True)
    
    def disable_feature(self, feature_name: str):
        """Disable a specific feature."""
        feature_flag = f"enable_{feature_name}"
        if hasattr(self, feature_flag):
            setattr(self, feature_flag, False)
    
    def set_mode(self, mode: AgentMode, **mode_config):
        """Set agent mode with specific configuration."""
        self.mode = mode
        self.mode_specific_config = mode_config
    
    def add_template(self, name: str, template: str):
        """Add a custom output template."""
        self.output_templates[name] = template
    
    def get_template(self, name: str) -> str:
        """Get an output template by name."""
        return self.output_templates.get(name, "") 