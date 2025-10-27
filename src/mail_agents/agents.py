"""CrewAI agents for email automation.

This module provides a backwards-compatible interface to the new YAML-based
CrewAI configuration system while maintaining the original API.
"""

import os
from pathlib import Path
from typing import Optional

from .crew import CrewManager
from .config import settings
from .model import SpamClassifier


class EmailAgents:
    """Manages CrewAI agents for email automation.
    
    This class provides backwards compatibility with the original interface
    while using the new YAML-based configuration system internally.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize email agents.

        Args:
            config_dir: Directory containing configuration files (deprecated, kept for compatibility)
        """
        # Initialize the new crew manager
        self.crew_manager = CrewManager()
        
        # Backwards compatibility - expose the classifier
        self.classifier = self.crew_manager.classifier
        
        # Print LLM configuration info
        if not os.getenv('OPENAI_API_KEY'):
            print(f"🤖 Using Ollama model: {settings.ollama_model}")
        else:
            print("🤖 Using OpenAI GPT model")
        
        # Print model status
        if self.classifier.model is None:
            print("Warning: Trained model not found. Classification features will be limited.")

    # Backwards compatibility methods - delegate to CrewManager
    def classify_email(self, email_text: str) -> str:
        """Classify an email using ML model and CrewAI agent.

        Args:
            email_text: The email text to classify

        Returns:
            Classification result as string
        """
        return self.crew_manager.classify_email(email_text)

    def extract_information(self, email_text: str) -> str:
        """Extract information from an email.

        Args:
            email_text: The email text to analyze

        Returns:
            Extracted information as string
        """
        return self.crew_manager.extract_information(email_text)

    def draft_response(self, email_text: str, context: str = "") -> str:
        """Draft a response to an email.

        Args:
            email_text: The email to respond to
            context: Additional context for drafting

        Returns:
            Drafted response as string
        """
        return self.crew_manager.draft_response(email_text, context)

    def process_pipeline(self, email_text: str) -> str:
        """Process email through complete pipeline.

        Args:
            email_text: The email to process

        Returns:
            Complete pipeline result as string
        """
        return self.crew_manager.process_pipeline(email_text)

    # New methods that expose the enhanced functionality
    def get_crew_info(self):
        """Get information about the configured crews and capabilities."""
        return self.crew_manager.get_crew_info()
    
    def validate_configuration(self):
        """Validate the current configuration setup."""
        return self.crew_manager.validate_configuration()
    
    def reload_configs(self):
        """Reload all configuration files."""
        return self.crew_manager.reload_configs()

    # Deprecated methods (kept for backwards compatibility)
    def create_spam_classifier_agent(self):
        """Create the spam classifier agent (deprecated - use YAML config)."""
        print("⚠️  Warning: create_spam_classifier_agent() is deprecated. Use YAML configuration instead.")
        return self.crew_manager.agent_factory.create_spam_classifier()

    def create_email_extractor_agent(self):
        """Create the email extractor agent (deprecated - use YAML config)."""
        print("⚠️  Warning: create_email_extractor_agent() is deprecated. Use YAML configuration instead.")
        return self.crew_manager.agent_factory.create_email_extractor()

    def create_email_drafter_agent(self):
        """Create the email drafter agent (deprecated - use YAML config)."""
        print("⚠️  Warning: create_email_drafter_agent() is deprecated. Use YAML configuration instead.")
        return self.crew_manager.agent_factory.create_email_drafter()

    def create_pipeline_coordinator_agent(self):
        """Create the pipeline coordinator agent (deprecated - use YAML config)."""
        print("⚠️  Warning: create_pipeline_coordinator_agent() is deprecated. Use YAML configuration instead.")
        return self.crew_manager.agent_factory.create_pipeline_coordinator()


# For backwards compatibility, also expose CrewManager directly
__all__ = ['EmailAgents', 'CrewManager']
