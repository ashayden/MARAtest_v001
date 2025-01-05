from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ResponseFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"

class AgentMode(Enum):
    CHAT = "chat"
    ANALYSIS = "analysis"
    REPORT = "report"
    CALCULATION = "calculation"

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
    enable_file_attachments: bool = False
    enable_calculations: bool = False
    enable_citations: bool = False
    enable_web_search: bool = False
    enable_data_visualization: bool = False
    
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