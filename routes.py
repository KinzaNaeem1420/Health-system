from fastapi import APIRouter, HTTPException
from schemas import PatientInput, FollowupInput, FollowupResponse
from agents import graph_app
from pydantic import BaseModel
import json
from datetime import datetime

router = APIRouter()

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@router.get("/")
def root():
    return {"message": "Connected to Health Diagnostic System with LangGraph!"}

@router.post("/healthcare")
async def healthcare_endpoint(patient: PatientInput) -> dict:
    try:
        result = graph_app.invoke({
            "action": "add_and_diagnose",
            "patient_input": patient.dict(),
            "patient_id": None,
            "symptoms": patient.symptoms,
            "diagnoses": [],
            "treatment": None,
            "drug_interactions": None,
            "followup": None,
            "response": None,
            "sources": []
        })
        if "error" in result["response"]:
            raise HTTPException(status_code=500, detail=result["response"]["error"])
        return json.loads(json.dumps(result["response"], cls=DateTimeEncoder))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing healthcare request: {str(e)}")

@router.post("/followup", response_model=FollowupResponse)
async def followup_endpoint(followup: FollowupInput):
    try:
        result = graph_app.invoke({
            "action": "followup",
            "patient_input": None,
            "patient_id": followup.patient_id,
            "symptoms": None,
            "diagnoses": None,
            "treatment": None,
            "drug_interactions": None,
            "followup": followup.dict(),
            "response": None,
            "sources": None
        })
        if "error" in result["response"]:
            raise HTTPException(status_code=500, detail=result["response"]["error"])
        return json.loads(json.dumps(result["response"], cls=DateTimeEncoder))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling followup: {str(e)}")