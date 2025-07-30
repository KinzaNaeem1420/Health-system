from asyncio.log import logger
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# Initialize embedding model
embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST")
        )
        register_vector(conn)
        return conn
    except Exception as e:
        raise Exception(f"Database connection failed: {str(e)}")

def execute_query(conn, query, params=None):
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or ())
            return cur
    except Exception as e:
        raise Exception(f"Query execution failed: {str(e)}")

from repository import get_db_connection

def search_documents(query):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Search across diagnosis, treatment, followup_notes, and name
        cursor.execute("""
            SELECT id, patient_id, name, diagnosis, treatment, followup_notes, created_at, updated_at
            FROM document
            WHERE COALESCE(diagnosis, '') ILIKE %s
               OR COALESCE(treatment, '') ILIKE %s
               OR COALESCE(followup_notes, '') ILIKE %s
               OR COALESCE(name, '') ILIKE %s
            LIMIT 10
        """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
        results = [
            {
                "id": row[0],
                "patient_id": row[1],
                "name": row[2],
                "diagnosis": row[3],
                "treatment": row[4],
                "followup_notes": row[5],
                "created_at": row[6],
                "updated_at": row[7]
            } for row in cursor.fetchall()
        ]
    except Exception as e:
        print(f"Error querying documents: {e}")
        results = []
    finally:
        cursor.close()
        conn.close()
    return results
# In repository.py
def condition_exists(conn, condition, table):
    cursor = conn.cursor()
    if table == "diagnoses":
        cursor.execute("SELECT COUNT(*) FROM diagnoses WHERE condition = ?", (condition,))
    elif table == "treatments":
        cursor.execute("SELECT COUNT(*) FROM treatments WHERE condition = ?", (condition,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0

def store_document(content: str, metadata: dict) -> int:
    try:
        embedding = embeddings.encode(content).tolist()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (content, embedding, metadata)
                    VALUES (%s, %s::vector, %s)
                    RETURNING id
                    """,
                    (content, embedding, json.dumps(metadata))
                )
                doc_id = cur.fetchone()[0]
                conn.commit()
                return doc_id
    except Exception as e:
        raise Exception(f"Document storage failed: {str(e)}")

def fetch_diagnoses(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT condition, symptoms, description FROM diagnoses")
            return cur.fetchall()
    except Exception as e:
        raise Exception(f"Fetching diagnoses failed: {str(e)}")

def fetch_diagnosis(conn, condition):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT condition, symptoms, description FROM diagnoses WHERE condition = %s", (condition,))
            return cur.fetchone()
    except Exception as e:
        raise Exception(f"Fetching diagnosis failed: {str(e)}")

def insert_diagnosis(conn, condition, symptoms, description):
    try:
        with conn.cursor() as cur:
            # Check if condition already exists to avoid duplicates
            cur.execute("SELECT 1 FROM diagnoses WHERE condition = %s", (condition,))
            if cur.fetchone():
                return  # Condition already exists, skip insertion
            cur.execute(
                """
                INSERT INTO diagnoses (condition, symptoms, description)
                VALUES (%s, %s, %s)
                """,
                (condition, json.dumps(symptoms), description)
            )
            conn.commit()
    except Exception as e:
        raise Exception(f"Diagnosis insertion failed: {str(e)}")

def fetch_treatment(conn, condition):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT medications, instructions, warnings FROM treatments WHERE condition = %s", (condition,))
            return cur.fetchone()
    except Exception as e:
        raise Exception(f"Fetching treatment failed: {str(e)}")

def insert_treatment(conn, condition, medications, instructions, warnings):
    try:
        with conn.cursor() as cur:
            # Check if condition already exists to avoid duplicates
            cur.execute("SELECT 1 FROM treatments WHERE condition = %s", (condition,))
            if cur.fetchone():
                return  # Condition already exists, skip insertion
            cur.execute(
                """
                INSERT INTO treatments (condition, medications, instructions, warnings)
                VALUES (%s, %s, %s, %s)
                """,
                (condition, json.dumps(medications), instructions, json.dumps(warnings))
            )
            conn.commit()
    except Exception as e:
        raise Exception(f"Treatment insertion failed: {str(e)}")

def fetch_drug_interactions(conn, medication_pair):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT severity, description FROM drug_interactions WHERE medication_pair = %s", (medication_pair,))
            return cur.fetchone()
    except Exception as e:
        raise Exception(f"Fetching drug interactions failed: {str(e)}")

def insert_drug_interaction(conn, medication_pair, severity, description):
    try:
        with conn.cursor() as cur:
            # Check if medication_pair already exists to avoid duplicates
            cur.execute("SELECT 1 FROM drug_interactions WHERE medication_pair = %s", (medication_pair,))
            if cur.fetchone():
                return  # Interaction already exists, skip insertion
            cur.execute(
                """
                INSERT INTO drug_interactions (medication_pair, severity, description)
                VALUES (%s, %s, %s)
                """,
                (medication_pair, severity, description)
            )
            conn.commit()
    except Exception as e:
        raise Exception(f"Drug interaction insertion failed: {str(e)}")

def insert_patient(conn, patient_id, name, age, gender, symptoms, medical_history, symptom_embedding):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients (patient_id, name, age, gender, symptoms, medical_history, symptom_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (patient_id, name, age, gender, json.dumps(symptoms), json.dumps(medical_history), symptom_embedding)
            )
    except Exception as e:
        logger.error(f"Patient insertion failed: {str(e)}")
        raise

def query_patients(conn, symptoms=None):
    try:
        with conn.cursor() as cur:
            if symptoms:
                symptom_embedding = embeddings.encode(symptoms).tolist()
                cur.execute(
                    """
                    SELECT id, name, symptoms, symptom_embedding <=> CAST(%s AS vector) AS distance
                    FROM patients
                    ORDER BY symptom_embedding <=> CAST(%s AS vector)
                    LIMIT 3
                    """,
                    (symptom_embedding, symptom_embedding)
                )
            else:
                cur.execute("SELECT id, name, symptoms FROM patients")
            return cur.fetchall()
    except Exception as e:
        raise Exception(f"Patient query failed: {str(e)}")

def insert_followup(conn, patient_id, date, notes):
    try:
        if not isinstance(date, datetime):
            raise ValueError("Date must be a valid datetime object")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO followups (patient_id, date, notes, completed)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (patient_id, date, notes, False)
            )
            followup_id = cur.fetchone()[0]
            conn.commit()
            return followup_id
    except Exception as e:
        raise Exception(f"Follow-up insertion failed: {str(e)}")
        