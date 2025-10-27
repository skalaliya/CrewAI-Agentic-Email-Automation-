# API Usage Examples

This document provides examples of how to use the Mail Agents API.

## Starting the Server

```bash
mailai run-api
```

Or with custom host and port:

```bash
mailai run-api --host 127.0.0.1 --port 9000
```

For development with auto-reload:

```bash
mailai run-api --reload
```

## API Endpoints

### Health Check

Check if the API is running and if the ML model is loaded.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

### Classify Email

Classify an email as spam or ham (legitimate).

**Request:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Congratulations! You have won a free iPhone! Click here to claim your prize now!"
  }'
```

**Response:**
```json
{
  "prediction": "spam",
  "is_spam": true,
  "confidence": 0.95,
  "agent_analysis": "This email exhibits classic spam characteristics..."
}
```

### Extract Information

Extract key information from an email.

**Request:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hi John, Can we schedule a meeting with Sarah on Monday at 3pm in Conference Room A to discuss the Q4 budget?"
  }'
```

**Response:**
```json
{
  "extracted_info": "Sender: Unknown\nRecipients: John, Sarah\nDate: Monday at 3pm\nLocation: Conference Room A\nTopic: Q4 budget\nAction Items: Schedule meeting"
}
```

### Draft Response

Draft a professional response to an email.

**Request:**
```bash
curl -X POST http://localhost:8000/draft \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Can we schedule a meeting to discuss the project timeline?",
    "context": "Meeting request from team member"
  }'
```

**Response:**
```json
{
  "draft": "Subject: Re: Meeting to discuss project timeline\n\nDear [Name],\n\nThank you for reaching out. I would be happy to schedule a meeting to discuss the project timeline.\n\nWould next Tuesday at 2pm work for you? We can meet in Conference Room B or via video call if you prefer.\n\nPlease let me know what works best for your schedule.\n\nBest regards,\n[Your Name]"
}
```

### Process Pipeline

Process an email through the complete pipeline (classify, extract, draft).

**Request:**
```bash
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hi, I received your invoice for $500. However, I believe there may be a discrepancy. Can we review this together?"
  }'
```

**Response:**
```json
{
  "result": "Classification: Ham (legitimate email)\nConfidence: 0.98\n\nExtracted Information:\n- Amount: $500\n- Topic: Invoice discrepancy\n- Action Required: Review invoice\n\nDrafted Response:\nSubject: Re: Invoice Review\n\nDear [Name],\n\nThank you for bringing this to my attention. I'd be happy to review the invoice with you to resolve any discrepancies.\n\nCould you please specify which line items you'd like to discuss? This will help me prepare the relevant documentation.\n\nI'm available [times] to go through this together.\n\nBest regards,\n[Your Name]"
}
```

## Python Client Example

```python
import requests

API_BASE = "http://localhost:8000"

# Classify an email
response = requests.post(
    f"{API_BASE}/classify",
    json={"text": "Get rich quick! Click here!"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Extract information
response = requests.post(
    f"{API_BASE}/extract",
    json={"text": "Meeting with John on Monday at 3pm"}
)
result = response.json()
print(f"Extracted: {result['extracted_info']}")

# Draft response
response = requests.post(
    f"{API_BASE}/draft",
    json={
        "text": "Can we schedule a meeting?",
        "context": "Initial meeting request"
    }
)
result = response.json()
print(f"Draft: {result['draft']}")
```

## JavaScript/Node.js Client Example

First, install the axios dependency:

```bash
npm install axios
```

Then use it in your code:

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:8000';

// Classify an email
async function classifyEmail(text) {
  const response = await axios.post(`${API_BASE}/classify`, { text });
  console.log('Prediction:', response.data.prediction);
  console.log('Confidence:', response.data.confidence);
}

// Extract information
async function extractInfo(text) {
  const response = await axios.post(`${API_BASE}/extract`, { text });
  console.log('Extracted:', response.data.extracted_info);
}

// Draft response
async function draftResponse(text, context) {
  const response = await axios.post(`${API_BASE}/draft`, { text, context });
  console.log('Draft:', response.data.draft);
}

// Usage
classifyEmail('Get rich quick! Click here!');
extractInfo('Meeting with John on Monday at 3pm');
draftResponse('Can we schedule a meeting?', 'Initial meeting request');
```

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test the API directly from your browser.

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `422`: Validation error (invalid request format)
- `500`: Internal server error
- `503`: Service unavailable (e.g., model not loaded)

Example error response:

```json
{
  "detail": "Model not loaded. Train the model first using 'mailai train'"
}
```
