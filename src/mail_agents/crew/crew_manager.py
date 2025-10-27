"""Crew manager for orchestrating CrewAI email processing workflows."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from crewai import Crew
from .agents import AgentFactory
from .tasks import TaskFactory
from ..model import SpamClassifier
from ..config import settings


class CrewManager:
    """Main manager class for CrewAI email processing workflows."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the crew manager.
        
        Args:
            config_path: Path to the crew configuration YAML file
        """
        self.config_path = config_path or Path(__file__).parent / "configs" / "crew.yaml"
        self.crew_config = self._load_config()
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory(agent_factory=self.agent_factory)
        self.classifier = SpamClassifier(settings.model_path, settings.vectorizer_path)
        
        # Try to load the trained model
        try:
            self.classifier.load()
            print("✅ ML model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Warning: Trained model not found. ML predictions will be limited.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load crew configuration from YAML file.
        
        Returns:
            Dictionary containing crew configurations
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Crew configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def classify_email(self, email_text: str) -> str:
        """Classify an email using ML model and CrewAI agent.
        
        Args:
            email_text: The email text to classify
            
        Returns:
            Classification result as string
        """
        # Get ML model prediction
        ml_prediction = None
        if self.classifier.model is not None:
            predictions = self.classifier.predict([email_text])
            ml_prediction = predictions[0]
        
        # Create agent and task
        agent = self.agent_factory.create_spam_classifier()
        task = self.task_factory.create_classify_email_task(email_text, ml_prediction, agent)
        
        # Execute with minimal crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=self.crew_config.get('email_processing_crew', {}).get('verbose', True)
        )
        
        result = crew.kickoff()
        return str(result)
    
    def extract_information(self, email_text: str) -> str:
        """Extract information from an email.
        
        Args:
            email_text: The email text to analyze
            
        Returns:
            Extracted information as string
        """
        agent = self.agent_factory.create_email_extractor()
        task = self.task_factory.create_extract_information_task(email_text, agent)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=self.crew_config.get('email_processing_crew', {}).get('verbose', True)
        )
        
        result = crew.kickoff()
        return str(result)
    
    def draft_response(self, email_text: str, context: str = "") -> str:
        """Draft a response to an email.
        
        Args:
            email_text: The email to respond to
            context: Additional context for drafting
            
        Returns:
            Drafted response as string
        """
        agent = self.agent_factory.create_email_drafter()
        task = self.task_factory.create_draft_response_task(email_text, context, agent)
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=self.crew_config.get('email_processing_crew', {}).get('verbose', True)
        )
        
        result = crew.kickoff()
        return str(result)
    
    def process_pipeline(self, email_text: str) -> str:
        """Process email through complete pipeline.
        
        Args:
            email_text: The email to process
            
        Returns:
            Complete pipeline result as string
        """
        # Get ML prediction
        ml_prediction = None
        if self.classifier.model is not None:
            predictions = self.classifier.predict([email_text])
            ml_prediction = predictions[0]
        
        # Create all agents for the pipeline
        agents = self.agent_factory.create_all_agents()
        
        # Create pipeline task
        coordinator = agents['pipeline_coordinator']
        task = self.task_factory.create_pipeline_task(email_text, ml_prediction, coordinator)
        
        # Execute with full crew
        crew_config = self.crew_config.get('email_processing_crew', {})
        crew = Crew(
            agents=list(agents.values()),
            tasks=[task],
            verbose=crew_config.get('verbose', True),
            memory=crew_config.get('memory', True),
            cache=crew_config.get('cache', True),
            max_rpm=crew_config.get('max_rpm', 10)
        )
        
        result = crew.kickoff()
        return str(result)
    
    def create_specialized_crew(self, crew_type: str) -> Crew:
        """Create a specialized crew for specific use cases.
        
        Args:
            crew_type: Type of specialized crew to create
            
        Returns:
            Configured specialized crew
        """
        specialized_config = self.crew_config.get('specialized_crews', {})
        
        if crew_type not in specialized_config:
            available_types = list(specialized_config.keys())
            raise KeyError(f"Specialized crew '{crew_type}' not found. Available types: {available_types}")
        
        config = specialized_config[crew_type]
        
        # Create agents for the specialized crew
        agents = []
        for agent_name in config['agents']:
            agents.append(self.agent_factory.create_agent(agent_name))
        
        return Crew(
            agents=agents,
            tasks=[],  # Tasks will be added when needed
            verbose=config.get('verbose', True),
            process=config.get('process', 'sequential')
        )
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get information about the configured crews.
        
        Returns:
            Dictionary with crew configuration details
        """
        return {
            'main_crew': self.crew_config.get('email_processing_crew', {}),
            'specialized_crews': list(self.crew_config.get('specialized_crews', {}).keys()),
            'available_agents': self.agent_factory.list_available_agents(),
            'available_tasks': self.task_factory.list_available_tasks(),
            'available_templates': self.task_factory.list_available_templates(),
            'llm_settings': self.crew_config.get('llm_settings', {}),
            'execution_policies': self.crew_config.get('execution_policies', {})
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration setup.
        
        Returns:
            Validation results with any issues found
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Test agent creation
            agents = self.agent_factory.create_all_agents()
            validation_results['agents_created'] = len(agents)
            
            # Test task creation
            tasks = self.task_factory.list_available_tasks()
            validation_results['tasks_available'] = len(tasks)
            
            # Check ML model
            if self.classifier.model is None:
                validation_results['warnings'].append("ML model not loaded - predictions will be limited")
            
            # Check LLM configuration
            if self.agent_factory.llm is None:
                validation_results['warnings'].append("Using default LLM (requires API keys)")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Configuration error: {str(e)}")
        
        return validation_results
    
    def reload_configs(self) -> None:
        """Reload all configuration files."""
        self.crew_config = self._load_config()
        self.agent_factory.reload_config()
        self.task_factory.reload_config()
        print("✅ All configurations reloaded")