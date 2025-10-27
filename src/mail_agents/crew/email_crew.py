"""Email processing crew following CrewAI recommended patterns."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ScrapeWebsiteTool, 
    FileReadTool,
    FileWriterTool
)
from .tools import EmailParserTool, SpamIndicatorTool
from ..config import settings
from ..model import SpamClassifier
from crewai import LLM
import os


@CrewBase
class EmailProcessingCrew:
    """Email processing crew using recommended CrewAI patterns."""
    
    agents_config = "configs/agents.yaml"
    tasks_config = "configs/tasks.yaml"
    
    def __init__(self):
        """Initialize the email processing crew."""
        self.classifier = SpamClassifier(settings.model_path, settings.vectorizer_path)
        
        # Try to load the trained model
        try:
            self.classifier.load()
            print("✅ ML model loaded successfully")
        except FileNotFoundError:
            print("⚠️  Warning: Trained model not found. ML predictions will be limited.")
    
    def _get_llm(self):
        """Get the appropriate LLM configuration."""
        if not os.getenv('OPENAI_API_KEY'):
            # Use Ollama when OpenAI is not available
            return LLM(
                model=f"ollama/{settings.ollama_model}",
                base_url=settings.ollama_base_url
            )
        return None  # Use CrewAI default (OpenAI)

    @agent
    def spam_classifier(self) -> Agent:
        """Create spam classifier agent from YAML config."""
        return Agent(
            config=self.agents_config['spam_classifier'],
            verbose=True,
            llm=self._get_llm(),
            tools=[
                SpamIndicatorTool(),  # Custom spam detection tool - working perfectly!
            ]
        )

    @agent
    def email_extractor(self) -> Agent:
        """Create email extractor agent from YAML config."""
        return Agent(
            config=self.agents_config['email_extractor'],
            verbose=True,
            llm=self._get_llm(),
            tools=[
                EmailParserTool(),    # Custom email parsing tool
                FileWriterTool()      # For saving extracted data
            ]
        )

    @agent
    def email_drafter(self) -> Agent:
        """Create email drafter agent from YAML config."""
        return Agent(
            config=self.agents_config['email_drafter'],
            verbose=True,
            llm=self._get_llm(),
            tools=[
                FileReadTool(),       # For reading email templates
                FileWriterTool()      # For saving draft responses
            ]
        )

    @agent
    def pipeline_coordinator(self) -> Agent:
        """Create pipeline coordinator agent from YAML config."""
        return Agent(
            config=self.agents_config['pipeline_coordinator'],
            verbose=True,
            llm=self._get_llm(),
            tools=[
                FileReadTool(),        # For reading configurations
                FileWriterTool()       # For logging and reporting
            ]
        )

    @task
    def classify_email_task(self) -> Task:
        """Create email classification task from YAML config."""
        return Task(
            config=self.tasks_config['classify_email'],
            agent=self.spam_classifier()
        )

    @task
    def extract_information_task(self) -> Task:
        """Create information extraction task from YAML config."""
        return Task(
            config=self.tasks_config['extract_information'],
            agent=self.email_extractor()
        )

    @task
    def draft_response_task(self) -> Task:
        """Create response drafting task from YAML config."""
        return Task(
            config=self.tasks_config['draft_response'],
            agent=self.email_drafter()
        )

    @task
    def process_pipeline_task(self) -> Task:
        """Create pipeline processing task from YAML config."""
        return Task(
            config=self.tasks_config['process_pipeline'],
            agent=self.pipeline_coordinator()
        )

    @crew
    def email_processing_crew(self) -> Crew:
        """Create the main email processing crew."""
        return Crew(
            agents=[
                self.spam_classifier(),
                self.email_extractor(),
                self.email_drafter(),
                self.pipeline_coordinator()
            ],
            tasks=[
                self.classify_email_task(),
                self.extract_information_task(),
                self.draft_response_task(),
                self.process_pipeline_task()
            ],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def spam_detection_crew(self) -> Crew:
        """Create a specialized spam detection crew."""
        return Crew(
            agents=[self.spam_classifier()],
            tasks=[self.classify_email_task()],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def information_extraction_crew(self) -> Crew:
        """Create a specialized information extraction crew."""
        return Crew(
            agents=[self.email_extractor()],
            tasks=[self.extract_information_task()],
            process=Process.sequential,
            verbose=True
        )

    @crew
    def response_drafting_crew(self) -> Crew:
        """Create a specialized response drafting crew."""
        return Crew(
            agents=[self.email_drafter()],
            tasks=[self.draft_response_task()],
            process=Process.sequential,
            verbose=True
        )

    # Convenience methods for easy access
    def classify_email(self, email_text: str, ml_prediction: dict = None) -> str:
        """Classify an email using the spam detection crew."""
        # Add email content to task description
        task = Task(
            description=f"""
            {self.tasks_config['classify_email']['description']}
            
            Email to classify:
            {email_text}
            
            ML Model Prediction: {ml_prediction if ml_prediction else 'Not available'}
            """,
            expected_output=self.tasks_config['classify_email']['expected_output'],
            agent=self.spam_classifier()
        )
        
        crew = Crew(
            agents=[self.spam_classifier()],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return result.raw

    def extract_information(self, email_text: str) -> str:
        """Extract information from an email."""
        task = Task(
            description=f"""
            {self.tasks_config['extract_information']['description']}
            
            Email to analyze:
            {email_text}
            """,
            expected_output=self.tasks_config['extract_information']['expected_output'],
            agent=self.email_extractor()
        )
        
        crew = Crew(
            agents=[self.email_extractor()],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return result.raw

    def draft_response(self, email_text: str, context: str = "") -> str:
        """Draft a response to an email."""
        context_text = f"\n\nAdditional Context: {context}" if context else ""
        
        task = Task(
            description=f"""
            {self.tasks_config['draft_response']['description']}
            
            Email to respond to:
            {email_text}{context_text}
            """,
            expected_output=self.tasks_config['draft_response']['expected_output'],
            agent=self.email_drafter()
        )
        
        crew = Crew(
            agents=[self.email_drafter()],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return result.raw