"""CrewAI agents for email automation."""

from pathlib import Path
from typing import Optional

import yaml
from crewai import Agent, Crew, Task

from .config import settings
from .model import SpamClassifier


class EmailAgents:
    """Manages CrewAI agents for email automation."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize email agents.

        Args:
            config_dir: Directory containing agents.yaml and tasks.yaml
        """
        self.config_dir = config_dir or settings.config_dir
        self.agents_config = self._load_config("agents.yaml")
        self.tasks_config = self._load_config("tasks.yaml")
        self.classifier = SpamClassifier(settings.model_path, settings.vectorizer_path)

        # Try to load the trained model
        try:
            self.classifier.load()
        except FileNotFoundError:
            print("Warning: Trained model not found. Classification features will be limited.")

    def _load_config(self, filename: str) -> dict:
        """Load YAML configuration file.

        Args:
            filename: Name of the config file

        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def create_spam_classifier_agent(self) -> Agent:
        """Create the spam classifier agent."""
        config = self.agents_config["spam_classifier"]
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=True,
            allow_delegation=False,
        )

    def create_email_extractor_agent(self) -> Agent:
        """Create the email extractor agent."""
        config = self.agents_config["email_extractor"]
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=True,
            allow_delegation=False,
        )

    def create_email_drafter_agent(self) -> Agent:
        """Create the email drafter agent."""
        config = self.agents_config["email_drafter"]
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=True,
            allow_delegation=False,
        )

    def create_pipeline_coordinator_agent(self) -> Agent:
        """Create the pipeline coordinator agent."""
        config = self.agents_config["pipeline_coordinator"]
        return Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            verbose=True,
            allow_delegation=True,
        )

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
        agent = self.create_spam_classifier_agent()
        task_config = self.tasks_config["classify_email"]

        ml_context = ""
        if ml_prediction:
            ml_context = f"\n\nML Model Prediction: {ml_prediction['prediction']} (confidence: {ml_prediction['confidence']:.2f})"

        task = Task(
            description=f"{task_config['description']}\n\nEmail to classify:\n{email_text}{ml_context}",
            expected_output=task_config["expected_output"],
            agent=agent,
        )

        # Execute
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        return str(result)

    def extract_information(self, email_text: str) -> str:
        """Extract information from an email.

        Args:
            email_text: The email text to analyze

        Returns:
            Extracted information as string
        """
        agent = self.create_email_extractor_agent()
        task_config = self.tasks_config["extract_information"]

        task = Task(
            description=f"{task_config['description']}\n\nEmail to analyze:\n{email_text}",
            expected_output=task_config["expected_output"],
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
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
        agent = self.create_email_drafter_agent()
        task_config = self.tasks_config["draft_response"]

        context_text = f"\n\nContext: {context}" if context else ""

        task = Task(
            description=f"{task_config['description']}\n\nEmail to respond to:\n{email_text}{context_text}",
            expected_output=task_config["expected_output"],
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        return str(result)

    def process_pipeline(self, email_text: str) -> str:
        """Process email through complete pipeline.

        Args:
            email_text: The email to process

        Returns:
            Complete pipeline result as string
        """
        # Create all agents
        classifier_agent = self.create_spam_classifier_agent()
        extractor_agent = self.create_email_extractor_agent()
        drafter_agent = self.create_email_drafter_agent()
        coordinator_agent = self.create_pipeline_coordinator_agent()

        # Get ML prediction
        ml_prediction = None
        if self.classifier.model is not None:
            predictions = self.classifier.predict([email_text])
            ml_prediction = predictions[0]

        ml_context = ""
        if ml_prediction:
            ml_context = f"\n\nML Model Prediction: {ml_prediction['prediction']} (confidence: {ml_prediction['confidence']:.2f})"

        # Create tasks
        task_config = self.tasks_config["process_pipeline"]
        pipeline_task = Task(
            description=f"{task_config['description']}\n\nEmail to process:\n{email_text}{ml_context}",
            expected_output=task_config["expected_output"],
            agent=coordinator_agent,
        )

        # Execute pipeline
        crew = Crew(
            agents=[coordinator_agent, classifier_agent, extractor_agent, drafter_agent],
            tasks=[pipeline_task],
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)
