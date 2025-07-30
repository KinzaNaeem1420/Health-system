from typing import TypedDict, List, Dict, Optional
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from repository import get_db_connection, fetch_diagnoses, fetch_treatment, fetch_drug_interactions, insert_patient, insert_followup, insert_diagnosis, insert_treatment, insert_drug_interaction, embeddings
from service import gemini_generate_content  # Assuming this imports the function
import json
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Define shared state
class HealthDiagnosticState(TypedDict):
    action: Optional[str]
    patient_input: Optional[Dict]
    patient_id: Optional[str]
    symptoms: Optional[List[str]]
    diagnoses: Optional[List[Dict]]
    treatment: Optional[Dict]
    drug_interactions: Optional[List[Dict]]
    followup: Optional[Dict]
    response: Optional[Dict]
    sources: Optional[List[Dict]]

# Symptom Collector Agent
def symptom_collector_node(state: HealthDiagnosticState) -> HealthDiagnosticState:
    try:
        logger.info(f"Received patient_input: {state.get('patient_input')}")
        if state["action"] in ["add_patient", "add_and_diagnose"]:
            patient_input = state["patient_input"]
            if not patient_input or not patient_input.get("symptoms"):
                raise ValueError("Patient input or symptoms missing")
            state["patient_id"] = str(uuid.uuid4())
            state["symptoms"] = patient_input["symptoms"]
            with get_db_connection() as conn:
                try:
                    symptom_embedding = embeddings.encode(json.dumps(patient_input["symptoms"])).tolist()
                except AttributeError as e:
                    raise ValueError(f"Embedding generation failed: {str(e)}")
                insert_patient(
                    conn,
                    state["patient_id"],
                    patient_input["name"],
                    patient_input["age"],
                    patient_input["gender"],
                    patient_input["symptoms"],
                    patient_input.get("medical_history"),
                    symptom_embedding
                )
                conn.commit()
                logger.info(f"Inserted patient with patient_id: {state['patient_id']}")
            if state["action"] == "add_patient":
                state["response"] = {"patient_id": state["patient_id"]}
        elif state["action"] == "followup":
            if not state["followup"] or not state["followup"].get("patient_id"):
                raise ValueError("Patient ID missing for follow-up")
        return state
    except Exception as e:
        logger.error(f"Symptom Collector failed: {str(e)}")
        state["response"] = {"error": f"Symptom Collector failed: {str(e)}"}
        return state

# Differential Diagnostician Agent 
def differential_diagnostician_node(state: HealthDiagnosticState) -> HealthDiagnosticState:
    try:
        if state["action"] == "add_and_diagnose":
            with get_db_connection() as conn:
                diagnoses = fetch_diagnoses(conn)
                state["diagnoses"] = []
                symptom_set = set(state["symptoms"])
                has_match = False
                
                # Check for database matches
                for condition, db_symptoms, description in diagnoses:
                    logger.debug(f"Checking condition: {condition}, db_symptoms: {db_symptoms}, type: {type(db_symptoms)}")
                    try:
                        db_symptom_set = set(json.loads(db_symptoms)) if isinstance(db_symptoms, str) else set(db_symptoms)
                        overlap = len(symptom_set.intersection(db_symptom_set)) / max(len(db_symptom_set), 1)  # Avoid division by zero
                        if overlap > 0.5:
                            has_match = True
                            prompt = PromptTemplate.from_template(
                                "Provide a description of '{condition}' based on symptoms {symptoms}. "
                                "Include Advice and When to Seek Help in bullet points. "
                                "Focus only on the symptoms {symptoms}, avoiding references to specific patients."
                            )
                            response = model.generate_content(
                                prompt.format(
                                    condition=condition,
                                    symptoms=", ".join(state["symptoms"])
                                )
                            )
                            enhanced_description = f"{description} [Enhanced]: {response.text}"
                            state["diagnoses"].append({
                                "condition": condition.replace("Possible ", ""),
                                "description": enhanced_description,
                                "match_score": overlap
                            })
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse db_symptoms for condition {condition}: {str(e)}, db_symptoms: {db_symptoms}")

                # If no matches, use Gemini to predict
                if not has_match:
                    query = ", ".join(state["symptoms"])
                    prompt = PromptTemplate.from_template(
                        """
                        Based on symptoms {query}, predict a medical diagnosis.
                        Return the response in JSON format with fields: condition (string), symptoms (list of strings), description (string).
                        The description should include Advice and When to Seek Help in bullet points.
                        Focus only on the symptoms {symptoms}, avoiding references to specific patients.
                        If no diagnosis can be determined, return a condition of 'Undetermined' with a description explaining the limitation.
                        """
                    )
                    response = model.generate_content(
                        prompt.format(
                            query=query,
                            symptoms=", ".join(state["symptoms"])
                        )
                    )
                    logger.info(f"Raw Gemini response for prediction: {response.text}")
                    try:
                        response_text = response.text.strip()
                        if response_text.startswith("```json") and response_text.endswith("```"):
                            response_text = response_text[7:-3].strip()
                        generated_diagnosis = json.loads(response_text)
                        if not generated_diagnosis.get("condition"):
                            logger.warning("No condition returned by Gemini, forcing Undetermined")
                            condition = "Undetermined"
                            description = "No valid diagnosis generated by model."
                        else:
                            condition = generated_diagnosis.get("condition", "Undetermined").replace("Possible ", "")
                            description = generated_diagnosis.get("description", "No description provided")
                        symptoms = generated_diagnosis.get("symptoms", state["symptoms"])
                        insert_diagnosis(conn, condition, json.dumps(symptoms), description)
                        state["diagnoses"].append({
                            "condition": condition,
                            "description": description,
                            "match_score": 0.0
                        })
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Gemini response: {str(e)}, Raw response: {response.text}")
                        state["diagnoses"].append({
                            "condition": "Undetermined",
                            "description": f"Failed to parse Gemini response: {str(e)} - {response.text}",
                            "match_score": 0.0
                        })

                # Ensure treatment is set
                if state["diagnoses"]:
                    top_diagnosis = sorted(state["diagnoses"], key=lambda x: x["match_score"], reverse=True)[0]
                    state["treatment"] = {"condition": top_diagnosis["condition"]}
                else:
                    state["treatment"] = {"condition": "Undetermined"}
                    state["diagnoses"].append({
                        "condition": "Undetermined",
                        "description": "No matching or valid diagnosis found.",
                        "match_score": 0.0
                    })
                logger.info(f"Set treatment condition: {state['treatment']['condition']}")
        return state
    except Exception as e:
        logger.error(f"Differential Diagnostician failed: {str(e)}")
        state["response"] = {"error": f"Differential Diagnostician failed: {str(e)}"}
        return state



# Treatment Planner Agent 
def treatment_planner_node(state: HealthDiagnosticState) -> HealthDiagnosticState:
    try:
        if state["action"] == "add_and_diagnose":
            if "treatment" not in state or state["treatment"] is None:
                state["treatment"] = {}
            condition = state["treatment"].get("condition")
            if not condition:
                raise ValueError("No condition provided for treatment planning")
            with get_db_connection() as conn:
                result = fetch_treatment(conn, condition)
                logger.debug(f"Fetch treatment result for {condition}: {result}")  # Debug the raw result
                if result:
                    # Handle medications as list or JSON string
                    medications = result[0]
                    if isinstance(medications, str):
                        medications = json.loads(medications)
                    elif not isinstance(medications, list):
                        medications = []
                    medications = [med.split()[0] for med in medications]  # Remove doses
                    state["treatment"] = {
                        "condition": condition,
                        "medications": medications,
                        "instructions": result[1],
                        "warnings": json.loads(result[2]) if result[2] and isinstance(result[2], str) else []
                    }
                    prompt = PromptTemplate.from_template(
                        "Provide additional advice for treating '{condition}' with medications {medications}. "
                        "Include Lifestyle Tips, Side Effects, and Recovery in bullet points. "
                        "Focus only on the condition and medications, avoiding references to specific patients."
                    )
                    response = model.generate_content(
                        prompt.format(
                            condition=condition,
                            medications=", ".join(medications) if medications else "none"
                        )
                    )
                    state["treatment"]["instructions"] = f"{state['treatment']['instructions']} [Enhanced]: {response.text}"
                else:
                    prompt = PromptTemplate.from_template(
                        """
                        For the condition '{condition}', provide a treatment plan in JSON format with fields:
                        - medications (list of strings, realistic medication names without doses)
                        - instructions (string, detailed treatment instructions)
                        - warnings (list of strings, specific warnings)
                        Ensure the treatment is appropriate for the condition.
                        """
                    )
                    response = model.generate_content(
                        prompt.format(
                            condition=condition
                        )
                    )
                    try:
                        response_text = response.text.strip()
                        if response_text.startswith("```json") and response_text.endswith("```"):
                            response_text = response_text[7:-3].strip()
                        generated_treatment = json.loads(response_text)
                        medications = [med.split()[0] for med in generated_treatment.get("medications", [])]
                        instructions = generated_treatment.get("instructions", "Follow standard care protocols.")
                        warnings = generated_treatment.get("warnings", [])
                        insert_treatment(conn, condition, medications, instructions, json.dumps(warnings) if warnings else "[]")
                        state["treatment"] = {
                            "condition": condition,
                            "medications": medications,
                            "instructions": instructions,
                            "warnings": warnings
                        }
                    except json.JSONDecodeError as e:
                        state["treatment"] = {
                            "condition": condition,
                            "medications": [],
                            "instructions": f"Failed to generate treatment: {str(e)}",
                            "warnings": ["Consult a healthcare provider"]
                        }
                        logger.error(f"Failed to parse Gemini treatment response: {str(e)}, Raw response: {response.text}")
                logger.info(f"Planned treatment for condition: {condition}")
        return state
    except Exception as e:
        logger.error(f"Treatment Planner failed: {str(e)}")
        state["response"] = {"error": f"Treatment Planner failed: {str(e)}"}
        return state

# Drug Interaction Checker Agent
def drug_interaction_checker_node(state: HealthDiagnosticState) -> HealthDiagnosticState:
    try:
        if state["action"] == "add_and_diagnose":
            treatment = state.get("treatment", {})
            if isinstance(treatment, str):
                try:
                    treatment = json.loads(treatment) if treatment else {}
                except json.JSONDecodeError:
                    treatment = {}
            medications = treatment.get("medications", [])
            state["drug_interactions"] = []
            with get_db_connection() as conn:
                for diagnosis in state["diagnoses"]:
                    condition = diagnosis["condition"]
                    if medications:
                        if len(medications) >= 2:
                            pair = ",".join(sorted([medications[0], medications[1]]))
                        else:
                            pair = ",".join(sorted(medications + ["Placebo"]))
                    else:
                        prompt = PromptTemplate.from_template(
                            """
                            For the condition '{condition}', suggest a hypothetical medication pair that might be used.
                            Return the response in JSON format with a single field: medication_pair (string, format: "drug1,drug2").
                            """
                        )
                        response = model.generate_content(prompt.format(condition=condition))
                        try:
                            response_text = response.text.strip()
                            if response_text.startswith("```json") and response_text.endswith("```"):
                                response_text = response_text[7:-3].strip()
                            hypothetical_pair = json.loads(response_text).get("medication_pair", "DrugA,DrugB")
                            pair = hypothetical_pair
                        except json.JSONDecodeError:
                            pair = "DrugA,DrugB"

                    interaction = fetch_drug_interactions(conn, pair)
                    if interaction:
                        state["drug_interactions"].append({
                            "medication_pair": pair,
                            "severity": interaction[0],
                            "description": interaction[1],
                            "condition": condition
                        })
                    else:
                        interaction_prompt = PromptTemplate.from_template(
                            """
                            For the medication pair '{medication_pair}' used for '{condition}', predict potential drug interactions.
                            Return the response in JSON format with fields:
                            - medication_pair (string, format: "drug1,drug2")
                            - severity (string, one of: "minor", "moderate", "severe")
                            - description (string, details of the interaction)
                            If no significant interaction exists, set severity to "minor" and provide a description.
                            Avoid references to specific patients.
                            """
                        )
                        interaction_response = model.generate_content(
                            interaction_prompt.format(medication_pair=pair, condition=condition)
                        )
                        try:
                            response_text = interaction_response.text.strip()
                            if response_text.startswith("```json") and response_text.endswith("```"):
                                response_text = response_text[7:-3].strip()
                            interaction = json.loads(response_text)
                            insert_drug_interaction(
                                conn,
                                interaction.get("medication_pair", pair),
                                interaction.get("severity", "minor"),
                                interaction.get("description", "No significant interaction")
                            )
                            state["drug_interactions"].append({
                                "medication_pair": interaction.get("medication_pair", pair),
                                "severity": interaction.get("severity", "minor"),
                                "description": interaction.get("description", "No significant interaction"),
                                "condition": condition
                            })
                        except json.JSONDecodeError:
                            insert_drug_interaction(
                                conn,
                                pair,
                                "minor",
                                f"Generated interaction: {interaction_response.text}"
                            )
                            state["drug_interactions"].append({
                                "medication_pair": pair,
                                "severity": "minor",
                                "description": f"Generated interaction: {interaction_response.text}",
                                "condition": condition
                            })
            state["response"] = {
                "patient": {
                    "patient_id": state["patient_id"],
                    "name": state["patient_input"]["name"],
                    "age": state["patient_input"]["age"],
                    "gender": state["patient_input"]["gender"],
                    "symptoms": state["patient_input"]["symptoms"],
                    "medical_history": state["patient_input"].get("medical_history")
                },
                "answer": f"Possible conditions: {', '.join([d['condition'] for d in state['diagnoses']])}" if state["diagnoses"] else "No matching conditions found",
                "diagnoses": sorted(state["diagnoses"], key=lambda x: x["match_score"], reverse=True),
                "treatment": state["treatment"],
                "drug_interactions": state["drug_interactions"],
                "sources": state["sources"] or []
            }
            logger.info(f"Completed drug interaction check for patient_id: {state['patient_id']}")
        return state
    except Exception as e:
        logger.error(f"Drug Interaction Checker failed: {str(e)}")
        state["response"] = {"error": f"Drug Interaction Checker failed: {str(e)}"}
        return state

# Follow-up Coordinator Agent
def followup_coordinator_node(state: HealthDiagnosticState) -> HealthDiagnosticState:
    try:
        if state["action"] == "followup":
            followup = state["followup"]
            with get_db_connection() as conn:
                date = followup["date"]
                if isinstance(date, str):
                    date = datetime.fromisoformat(date.replace("Z", "+00:00"))
                elif not isinstance(date, datetime):
                    raise ValueError("Date must be a valid datetime object or ISO string")
                followup_id = insert_followup(conn, followup["patient_id"], date, followup["notes"])
            state["response"] = {"followup_id": followup_id}
            logger.info(f"Scheduled followup for patient_id: {followup['patient_id']}")
            print(f"State after followup: {json.dumps(state, default=str)}")  # Debug state
        return state
    except Exception as e:
        logger.error(f"Follow-up Coordinator failed: {str(e)}")
        state["response"] = {"error": f"Follow-up Coordinator failed: {str(e)}"}
        return state

# Define LangGraph workflow
def create_workflow():
    workflow = StateGraph(HealthDiagnosticState)
    
    workflow.add_node("symptom_collector", symptom_collector_node)
    workflow.add_node("differential_diagnostician", differential_diagnostician_node)
    workflow.add_node("treatment_planner", treatment_planner_node)
    workflow.add_node("drug_interaction_checker", drug_interaction_checker_node)
    workflow.add_node("followup_coordinator", followup_coordinator_node)
    
    workflow.set_entry_point("symptom_collector")
    workflow.add_conditional_edges(
        "symptom_collector",
        lambda state: state["action"],
        {
            "add_patient": END,
            "add_and_diagnose": "differential_diagnostician",
            "followup": "followup_coordinator",
            "error": END
        }
    )
    workflow.add_edge("differential_diagnostician", "treatment_planner")
    workflow.add_edge("treatment_planner", "drug_interaction_checker")
    workflow.add_edge("drug_interaction_checker", END)
    workflow.add_edge("followup_coordinator", END)
    
    return workflow.compile()

graph_app = create_workflow()