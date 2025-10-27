"""Agent factory for creating CrewAI agents from YAML configuration."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from crewai import Agent, LLM
from ..config import settings


class AgentFactory:
    """Factory class for creating CrewAI agents from YAML configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the agent factory.
        
        Args:
            config_path: Path to the agents configuration YAML file
        """
        self.config_path = config_path or Path(__file__).parent / "configs" / "agents.yaml"
        self.agents_config = self._load_config()
        self.llm = self._setup_llm()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file.
        
        Returns:
            Dictionary containing agent configurations
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Agent configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_llm(self) -> Optional[LLM]:
        """Setup the LLM based on configuration.
        
        Returns:
            Configured LLM instance or None for default
        """
        if not os.getenv('OPENAI_API_KEY'):
            # Use Ollama when OpenAI is not available
            return LLM(
                model=f"ollama/{settings.ollama_model}",
                base_url=settings.ollama_base_url
            )
        return None  # Use CrewAI default (OpenAI)
    
    def create_agent(self, agent_name: str) -> Agent:
        """Create a specific agent by name.
        
        Args:
            agent_name: Name of the agent to create
            
        Returns:
            Configured CrewAI Agent instance
            
        Raises:
            KeyError: If agent name not found in configuration
        """
        if agent_name not in self.agents_config:
            available_agents = list(self.agents_config.keys())
            raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
        
        config = self.agents_config[agent_name]
        
        # Build agent parameters
        agent_params = {
            'role': config['role'],
            'goal': config['goal'], 
            'backstory': config['backstory'],
            'verbose': config.get('verbose', True),
            'allow_delegation': config.get('allow_delegation', False),
            'max_iter': config.get('max_iter', 3),
            'max_execution_time': config.get('max_execution_time', 30),
        }
        
        # Add LLM if configured
        if self.llm:
            agent_params['llm'] = self.llm # Assigns Ollama LLM to agent
        
        # Add optional parameters if they exist in config
        optional_params = [
            'step_callback', 'system_template', 'prompt_template', 'response_template'
        ]
        for param in optional_params:
            if param in config and config[param] is not None:
                agent_params[param] = config[param]
        
        return Agent(**agent_params)   # Agent now uses local Ollama
    
    def create_spam_classifier(self) -> Agent:
        """Create the spam classifier agent.
        
        Returns:
            Configured spam classifier agent
        """
        return self.create_agent('spam_classifier')
    
    def create_email_extractor(self) -> Agent:
        """Create the email extractor agent.
        
        Returns:
            Configured email extractor agent
        """
        return self.create_agent('email_extractor')
    
    def create_email_drafter(self) -> Agent:
        """Create the email drafter agent.
        
        Returns:
            Configured email drafter agent
        """
        return self.create_agent('email_drafter')
    
    def create_pipeline_coordinator(self) -> Agent:
        """Create the pipeline coordinator agent.
        
        Returns:
            Configured pipeline coordinator agent
        """
        return self.create_agent('pipeline_coordinator')
    
    def create_all_agents(self) -> Dict[str, Agent]:
        """Create all configured agents.
        
        Returns:
            Dictionary mapping agent names to Agent instances
        """
        agents = {}
        for agent_name in self.agents_config.keys():
            if agent_name != 'default_settings':  # Skip default settings
                agents[agent_name] = self.create_agent(agent_name)
        return agents
    
    def list_available_agents(self) -> list[str]:
        """Get list of available agent names.
        
        Returns:
            List of available agent names
        """
        return [name for name in self.agents_config.keys() if name != 'default_settings']
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration dictionary
        """
        if agent_name not in self.agents_config:
            raise KeyError(f"Agent '{agent_name}' not found")
        return self.agents_config[agent_name].copy()
    
    def reload_config(self) -> None:
        """Reload the agent configuration from file."""
        self.agents_config = self._load_config()
        self.llm = self._setup_llm()