from pydantic import BaseModel
from typing import List, Optional, Dict

class PatientInput(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: List[str]
    medical_history: Optional[List[str]] = None 

    class Config:
        extra = "forbid"

class QueryRequest(BaseModel):
    symptoms: List[str]

class TreatmentRequest(BaseModel):
    condition: str

class FollowupInput(BaseModel):
    patient_id: str
    date: str
    notes: str

class QueryResponse(BaseModel):
    answer: str
    diagnoses: List[Dict]
    sources: List[Dict]

class TreatmentResponse(BaseModel):
    treatment: Dict
    drug_interactions: List[Dict]

class FollowupResponse(BaseModel):
    followup_id: int