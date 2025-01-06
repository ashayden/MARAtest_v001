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

@dataclass
class MediaConfig:
    """Configuration for media handling."""
    allowed_image_types: List[str] = field(default_factory=lambda: ["png", "jpg", "jpeg", "gif"])
    max_file_size_mb: int = 10
    enable_image_analysis: bool = True

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
    enable_file_attachments: bool = True
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
    
    # Response Configuration
    response_format: ResponseFormat = ResponseFormat.MARKDOWN
    max_response_length: int = 4000
    include_timestamps: bool = True
    
    # Memory Configuration
    memory_window_size: int = 10
    memory_type: str = "conversation"
    
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