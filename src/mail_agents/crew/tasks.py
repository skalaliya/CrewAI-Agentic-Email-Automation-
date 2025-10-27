"""Task factory for creating CrewAI tasks from YAML configuration."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from crewai import Task, Agent
from .agents import AgentFactory


class TaskFactory:
    """Factory class for creating CrewAI tasks from YAML configuration."""
    
    def __init__(self, config_path: Optional[Path] = None, agent_factory: Optional[AgentFactory] = None):
        """Initialize the task factory.
        
        Args:
            config_path: Path to the tasks configuration YAML file
            agent_factory: AgentFactory instance for creating agents
        """
        self.config_path = config_path or Path(__file__).parent / "configs" / "tasks.yaml"
        self.tasks_config = self._load_config()
        self.agent_factory = agent_factory or AgentFactory()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load task configuration from YAML file.
        
        Returns:
            Dictionary containing task configurations
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Task configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_task(self, task_name: str, agent: Optional[Agent] = None, **kwargs) -> Task:
        """Create a specific task by name.
        
        Args:
            task_name: Name of the task to create
            agent: Specific agent to assign to the task (optional)
            **kwargs: Additional parameters to override task configuration
            
        Returns:
            Configured CrewAI Task instance
            
        Raises:
            KeyError: If task name not found in configuration
        """
        if task_name not in self.tasks_config:
            available_tasks = [name for name in self.tasks_config.keys() 
                             if name not in ['templates', 'default_settings']]
            raise KeyError(f"Task '{task_name}' not found. Available tasks: {available_tasks}")
        
        config = self.tasks_config[task_name].copy()
        
        # Create agent if not provided
        if agent is None:
            agent_name = config.get('agent')
            if agent_name:
                agent = self.agent_factory.create_agent(agent_name)
            else:
                raise ValueError(f"No agent specified for task '{task_name}' and no agent provided")
        
        # Build task parameters
        task_params = {
            'description': config['description'],
            'expected_output': config['expected_output'],
            'agent': agent,
        }
        
        # Add optional parameters
        optional_params = [
            'context', 'tools', 'async_execution', 'human_input', 'max_execution_time'
        ]
        for param in optional_params:
            if param in config:
                task_params[param] = config[param]
        
        # Apply any overrides from kwargs
        task_params.update(kwargs)
        
        return Task(**task_params)
    
    def create_classify_email_task(self, email_text: str, ml_prediction: Optional[Dict] = None, 
                                 agent: Optional[Agent] = None) -> Task:
        """Create a spam classification task with email content.
        
        Args:
            email_text: The email text to classify
            ml_prediction: Optional ML model prediction to include
            agent: Specific agent to use (optional)
            
        Returns:
            Configured classification task
        """
        # Build description with email content
        base_config = self.tasks_config['classify_email']
        
        ml_context = ""
        if ml_prediction:
            ml_context = f"\n\nML Model Prediction: {ml_prediction['prediction']} (confidence: {ml_prediction['confidence']:.2f})"
        
        description = f"{base_config['description']}\n\nEmail to classify:\n{email_text}{ml_context}"
        
        return self.create_task('classify_email', agent=agent, description=description)
    
    def create_extract_information_task(self, email_text: str, agent: Optional[Agent] = None) -> Task:
        """Create an information extraction task with email content.
        
        Args:
            email_text: The email text to analyze
            agent: Specific agent to use (optional)
            
        Returns:
            Configured extraction task
        """
        base_config = self.tasks_config['extract_information']
        description = f"{base_config['description']}\n\nEmail to analyze:\n{email_text}"
        
        return self.create_task('extract_information', agent=agent, description=description)
    
    def create_draft_response_task(self, email_text: str, context: str = "", 
                                 agent: Optional[Agent] = None) -> Task:
        """Create a response drafting task with email content.
        
        Args:
            email_text: The original email to respond to
            context: Additional context for drafting
            agent: Specific agent to use (optional)
            
        Returns:
            Configured drafting task
        """
        base_config = self.tasks_config['draft_response']
        context_text = f"\n\nAdditional Context: {context}" if context else ""
        description = f"{base_config['description']}\n\nEmail to respond to:\n{email_text}{context_text}"
        
        return self.create_task('draft_response', agent=agent, description=description)
    
    def create_pipeline_task(self, email_text: str, ml_prediction: Optional[Dict] = None,
                           agent: Optional[Agent] = None) -> Task:
        """Create a complete pipeline processing task.
        
        Args:
            email_text: The email text to process
            ml_prediction: Optional ML model prediction
            agent: Specific agent to use (optional)
            
        Returns:
            Configured pipeline task
        """
        base_config = self.tasks_config['process_pipeline']
        
        ml_context = ""
        if ml_prediction:
            ml_context = f"\n\nML Model Prediction: {ml_prediction['prediction']} (confidence: {ml_prediction['confidence']:.2f})"
        
        description = f"{base_config['description']}\n\nEmail to process:\n{email_text}{ml_context}"
        
        return self.create_task('process_pipeline', agent=agent, description=description)
    
    def create_tasks_from_template(self, template_name: str, **kwargs) -> Task:
        """Create a task from a template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Parameters to customize the template
            
        Returns:
            Task created from template
        """
        if 'templates' not in self.tasks_config or template_name not in self.tasks_config['templates']:
            available_templates = list(self.tasks_config.get('templates', {}).keys())
            raise KeyError(f"Template '{template_name}' not found. Available templates: {available_templates}")
        
        template = self.tasks_config['templates'][template_name]
        
        # Create task from template
        task_params = {
            'description': template['description'],
            'expected_output': template['expected_output'],
        }
        
        # Apply template overrides
        task_params.update(kwargs)
        
        # Create agent if not provided
        if 'agent' not in task_params:
            # Default to email drafter for response templates
            agent = self.agent_factory.create_email_drafter()
            task_params['agent'] = agent
        
        return Task(**task_params)
    
    def list_available_tasks(self) -> List[str]:
        """Get list of available task names.
        
        Returns:
            List of available task names
        """
        return [name for name in self.tasks_config.keys() 
                if name not in ['templates', 'default_settings']]
    
    def list_available_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of available template names
        """
        return list(self.tasks_config.get('templates', {}).keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task configuration dictionary
        """
        if task_name not in self.tasks_config:
            raise KeyError(f"Task '{task_name}' not found")
        return self.tasks_config[task_name].copy()
    
    def reload_config(self) -> None:
        """Reload the task configuration from file."""
        self.tasks_config = self._load_config()