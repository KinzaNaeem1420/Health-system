from pydantic import BaseModel
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class FollowupInput(BaseModel):
    patient_id: str
    date: str
    notes: str

class TreatmentRequest(BaseModel):
    condition: str