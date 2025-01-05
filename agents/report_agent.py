from .base_template import AgentTemplate
from .config import AgentConfig, AgentMode
from typing import Optional

class ReportAgent(AgentTemplate):
    """Specialized agent for generating structured reports."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the report agent with custom configuration."""
        # Create report-specific configuration
        report_config = config or AgentConfig()
        report_config.update(
            name="Report Generator",
            description="Specialized agent for generating detailed reports",
            mode=AgentMode.REPORT,
            temperature=0.3,  # Lower temperature for more focused reports
            enable_citations=True,
            enable_knowledge_base=True,
            include_sources=True
        )
        
        # Add custom report template if not present
        if 'detailed_report' not in report_config.output_templates:
            report_config.add_template('detailed_report', """
            # {title}
            
            ## Executive Summary
            {executive_summary}
            
            ## Introduction
            {introduction}
            
            ## Methodology
            {methodology}
            
            ## Findings and Analysis
            {findings}
            
            ## Key Insights
            {insights}
            
            ## Recommendations
            {recommendations}
            
            ## Implementation Plan
            {implementation}
            
            ## Risk Analysis
            {risks}
            
            ## Conclusion
            {conclusion}
            
            ## References and Sources
            {references}
            
            ## Appendices
            {appendices}
            """)
        
        super().__init__(report_config)
    
    def format_prompt(self, user_input: str) -> str:
        """Enhance the prompt formatting for report generation."""
        base_prompt = super().format_prompt(user_input)
        
        # Add report-specific context
        report_context = f"""
        Additional Report Guidelines:
        1. Structure:
           - Begin with a clear executive summary
           - Present methodology and approach
           - Support findings with evidence
           - Provide actionable recommendations
        
        2. Content Requirements:
           - Use data-driven insights
           - Include relevant statistics
           - Cite authoritative sources
           - Present balanced viewpoints
        
        3. Professional Standards:
           - Maintain formal tone
           - Use industry-standard terminology
           - Follow style guidelines
           - Ensure factual accuracy
        
        4. Visualization Notes:
           - Describe any charts or graphs
           - Include data tables
           - Reference visual elements
           - Explain key metrics
        """
        
        return f"{base_prompt}\n{report_context}"
    
    def format_response(self, response: str) -> str:
        """Enhanced response formatting for reports."""
        formatted_response = super().format_response(response)
        
        # Add report metadata if configured
        if self.config.include_timestamps:
            formatted_response = f"""
            Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            {formatted_response}
            """
        
        if self.config.include_sources:
            formatted_response += "\n\n---\nSources and References:\n"
            # Add source citations here
        
        return formatted_response 