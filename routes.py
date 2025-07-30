from fastapi import APIRouter, HTTPException
from schemas import PatientInput, FollowupInput, FollowupResponse
from agents import graph_app
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
def root():
    return {"message": "Connected to Health Diagnostic System with LangGraph!"}

@router.post("/healthcare")
async def healthcare_endpoint(patient: PatientInput) -> dict:
    try:
        logger.debug(f"Received patient input: {patient.dict()}")
        if not patient.dict().get("symptoms"):
            raise HTTPException(status_code=400, detail="Symptoms field is required")
        state = {
            "action": "add_and_diagnose",
            "patient_input": patient.dict()
        }
        result = graph_app.invoke(state)
        logger.debug(f"Final state after invoke: {result}")  # Debug final state
        response = result.get("response", {})
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        # Construct detailed response if not already set
        if not response or not all(key in response for key in ["condition", "medications", "instructions", "warnings"]):
            treatment = result.get("treatment", {})
            response = {
                "patient": patient.dict(),
                "patient_id": result.get("patient_id"),
                "diagnoses": result.get("diagnoses", []),
                "instructions": treatment.get("instructions", "Consult a healthcare provider."),
                "drug_interactions": result.get("drug_interactions", []),
                 "warnings": treatment.get("warnings", []),     
                "status": "completed"
            }
        return response
    except Exception as e:
        logger.error(f"Healthcare endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing healthcare request: {str(e)}")

@router.post("/followup", response_model=FollowupResponse)
async def followup_endpoint(followup: FollowupInput):
    try:
        state = {
            "action": "followup",
            "followup": followup.dict()
        }
        result = graph_app.invoke(state)
        if "error" in result.get("response", {}):
            raise HTTPException(status_code=500, detail=result["response"]["error"])
        return result["response"]
    except Exception as e:
        logger.error(f"Followup endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scheduling followup: {str(e)}")