
# Health Diagnostic System with LangGraph

## Overview
A scalable healthcare diagnostic system built with FastAPI, LangGraph, and integrated with the Gemini LLM for intelligent diagnosis and treatment planning. This project leverages a PostgreSQL database for patient records and utilizes embeddings for symptom-based matching.

## Features
- Patient Management: Add patient details and generate unique patient IDs.
- Diagnosis: Match symptoms against a database or predict conditions using Gemini LLM.
- Treatment Planning: Retrieve or generate treatment plans with medications, instructions, and warnings.
- Drug Interaction Checking: Identify potential interactions between prescribed medications.
- Follow-up Scheduling: Schedule and record follow-up appointments with notes.
- Concise Responses: Optimized prompts ensure short, focused descriptions and advice.

## Requirements
- Python 3.10+
- PostgreSQL (with pgvector extension for embeddings)
- Google Gemini API Key (set as GOOGLE_API_KEY in .env)
- Dependencies:
  - fastapi
  - uvicorn
  - langgraph
  - langchain-core
  - google-generativeai
  - python-dotenv
  - psycopg2-binary
  - torch (for embeddings)

## Installation
### 1. Clone the Repository
```
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment
- Create a .env file in the project root:
  ```
  GOOGLE_API_KEY=your_gemini_api_key
  DATABASE_URL=postgresql://user:password@localhost:5432/dbname
  ```
- Update DATABASE_URL with your PostgreSQL connection string.

### 5. Set Up Database
- Install PostgreSQL and enable the pgvector extension.
- Create the database and tables by running the SQL scripts in repository.py or a separate .sql file (e.g., schema.sql if provided).

### 6. Run the Application
```
uvicorn routes:app --reload
```

## Usage
### API Endpoints
#### GET /
- Response: {"message": "Connected to Health Diagnostic System with LangGraph!"}
- Purpose: Health check.

#### POST /healthcare
- Request Body:
  ```
  {
    "name": "Aqsa",
    "age": 30,
    "gender": "Female",
    "symptoms": ["fever", "cough", "flu"],
    "medical_history": ["High Blood Pressure"]
  }
  ```
- Response: JSON with patient, patient_id, condition, medications, instructions, warnings, etc.
- Purpose: Diagnose and plan treatment.

#### POST /followup
- Request Body:
  ```
  {
    "patient_id": "49224e31-3438-40db-840a-2439163283f5",
    "date": "2025-07-31T10:00:00Z",
    "notes": "Follow-up for persistent cough"
  }
  ```
- Response: {"followup_id": "..."}
- Purpose: Schedule a follow-up.

### Example Request
```
curl -X 'POST' \
  'http://127.0.0.1:8000/healthcare' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Aqsa",
    "age": 30,
    "gender": "Female",
    "symptoms": ["fever", "cough", "flu"],
    "medical_history": ["High Blood Pressure"]
  }'
```

## Project Structure
- routes.py: Defines FastAPI endpoints.
- agents.py: Contains LangGraph nodes for workflow logic.
- schemas.py: Pydantic models for request/response validation.
- repository.py: Database interaction functions.
- service.py: Gemini API integration (e.g., gemini_generate_content).
- .env: Environment variables (e.g., API keys, database URL).

## Configuration
- Logging: Configured to log at INFO level to stdout.
- Database: Uses PostgreSQL with pgvector for symptom embeddings.
- API Key: Store your Gemini API key in .env.

## Contributing
### 1. Fork the Repository
### 2. Create a Feature Branch
```
git checkout -b feature-name
```
### 3. Commit Changes
```
git commit -m "Add feature"
```
### 4. Push to the Branch
```
git push origin feature-name
```
## Acknowledgments
- [xAI] for inspiration and tools.
- Google for the Gemini API.
- LangChain for workflow management.

