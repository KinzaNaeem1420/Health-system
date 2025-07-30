from torch import embedding
from repository import get_db_connection, search_documents, fetch_diagnoses, fetch_treatment, fetch_drug_interactions, insert_patient, query_patients, insert_followup
import uuid
import json
from typing import List, Dict
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def gemini_generate_content(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text  # Ensure text is returned as a string
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return json.dumps({"diagnoses": "API error", "treatment": "No treatment available"})

def add_patient(patient_data: dict) -> dict:
    patient_id = str(uuid.uuid4())
    symptom_embedding = embedding.encode(json.dumps(patient_data["symptoms"])).tolist()
    with get_db_connection() as conn:
        insert_patient(conn, patient_id, patient_data["name"], patient_data["age"], patient_data["gender"], patient_data["symptoms"], patient_data.get("medical_history"), symptom_embedding)
    return {"patient_id": patient_id}

def diagnose_conditions(symptoms: List[str]) -> dict:
    with get_db_connection() as conn:
        diagnoses = fetch_diagnoses(conn)
        results = []
        symptom_set = set(symptoms)
        has_match = False
        
        # Check for database matches
        for condition, db_symptoms, description in diagnoses:
            db_symptom_set = set(json.loads(db_symptoms))
            overlap = len(symptom_set.intersection(db_symptom_set)) / len(db_symptom_set)
            if overlap > 0.5:
                has_match = True
                # Enhance description with Gemini LLM
                prompt = f"Provide additional insights for the medical condition '{condition}' based on the symptoms {', '.join(symptoms)}."
                response = model.generate_content(prompt)
                enhanced_description = f"{description} [Enhanced]: {response.text}"
                results.append({"condition": condition, "description": enhanced_description, "match_score": overlap})

        # If no match or insufficient matches, use RAG and LLM to predict
        if not has_match or not results:
            query = ", ".join(symptoms)
            docs = search_documents(query)
            context = "\n".join([doc["content"] for doc in docs]) if docs else "No additional context available."
            prompt = f"Based on the symptoms {query} and the following context: {context}, predict a possible medical diagnosis. Include a brief general advice a. "
            response = model.generate_content(prompt)
            if response.text:
                results.append({"condition": response.text.split('\n')[0].replace('**', ''), "description": '\n'.join(response.text.split('\n')[1:]), "match_score": 0.0})

        return {"diagnoses": sorted(results, key=lambda x: x["match_score"], reverse=True)}

def get_patients(symptoms: str = None) -> list:
    with get_db_connection() as conn:
        results = query_patients(conn, symptoms)
        return [{"id": row[0], "name": row[1], "symptoms": json.loads(row[2]), "distance": row[3] if symptoms else None} for row in results]

def get_treatment(condition: str) -> dict:
    with get_db_connection() as conn:
        result = fetch_treatment(conn, condition)
        if not result:
            raise ValueError("Treatment not found")
        medications = json.loads(result[0])
        treatment_data = {
            "condition": condition,
            "medications": medications,
            "instructions": result[1],
            "warnings": json.loads(result[2])
        }
        interactions = []
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                pair = ",".join(sorted([medications[i], medications[j]]))
                interaction = fetch_drug_interactions(conn, pair)
                if interaction:
                    interactions.append({"medication_pair": pair, "severity": interaction[0], "description": interaction[1]})
        # Enhance treatment instructions with Gemini LLM
        prompt = f"Provide additional advice for treating '{condition}' with medications {', '.join(medications)}"
        response = model.generate_content(prompt)
        treatment_data["instructions"] = f"{treatment_data['instructions']} [Enhanced]: {response.text}"
        return {"treatment": treatment_data, "drug_interactions": interactions}

def schedule_followup(patient_id: str, date: str, notes: str) -> dict:
    with get_db_connection() as conn:
        followup_id = insert_followup(conn, patient_id, date, notes)
        return {"followup_id": followup_id}