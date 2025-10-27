"""CrewAI module for email processing automation.

This module provides a clean, YAML-configured approach to CrewAI agent management
for email processing, classification, and response generation.
"""

from .crew_manager import CrewManager
from .agents import AgentFactory
from .tasks import TaskFactory

__all__ = ['CrewManager', 'AgentFactory', 'TaskFactory']