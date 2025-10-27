"""FastAPI server for mail agents."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agents import EmailAgents
from .config import settings
from .model import SpamClassifier

app = FastAPI(
    title="Mail Agents API",
    description="Agentic email automation with CrewAI",
    version="0.1.0",
)

# Initialize components
email_agents = EmailAgents()
classifier = SpamClassifier(settings.model_path, settings.vectorizer_path)

# Load model if available
try:
    classifier.load()
except FileNotFoundError:
    print("Warning: Model not loaded. Train the model first using 'mailai train'")


class EmailInput(BaseModel):
    """Input model for email text."""

    text: str
    context: Optional[str] = None


class ClassificationResponse(BaseModel):
    """Response model for email classification."""

    prediction: str
    is_spam: bool
    confidence: float
    agent_analysis: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response model for information extraction."""

    extracted_info: str


class DraftResponse(BaseModel):
    """Response model for email draft."""

    draft: str


class PipelineResponse(BaseModel):
    """Response model for complete pipeline."""

    result: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = classifier.model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "version": "0.1.0",
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(email: EmailInput):
    """Classify an email as spam or ham.

    Args:
        email: Email text to classify

    Returns:
        Classification result with confidence score
    """
    if classifier.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first using 'mailai train'",
        )

    try:
        # Get ML prediction
        predictions = classifier.predict([email.text])
        prediction = predictions[0]

        # Optionally enhance with agent analysis
        agent_analysis = None
        try:
            agent_analysis = email_agents.classify_email(email.text)
        except Exception as e:
            print(f"Agent analysis failed: {e}")

        return ClassificationResponse(
            prediction=prediction["prediction"],
            is_spam=prediction["is_spam"],
            confidence=prediction["confidence"],
            agent_analysis=agent_analysis,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=ExtractionResponse)
async def extract_information(email: EmailInput):
    """Extract key information from an email.

    Args:
        email: Email text to analyze

    Returns:
        Extracted information
    """
    try:
        extracted_info = email_agents.extract_information(email.text)
        return ExtractionResponse(extracted_info=extracted_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/draft", response_model=DraftResponse)
async def draft_response(email: EmailInput):
    """Draft a response to an email.

    Args:
        email: Email text to respond to, with optional context

    Returns:
        Drafted email response
    """
    try:
        draft = email_agents.draft_response(email.text, email.context or "")
        return DraftResponse(draft=draft)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline", response_model=PipelineResponse)
async def process_pipeline(email: EmailInput):
    """Process email through complete pipeline.

    Args:
        email: Email text to process

    Returns:
        Complete pipeline results
    """
    try:
        result = email_agents.process_pipeline(email.text)
        return PipelineResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
