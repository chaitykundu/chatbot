import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import re
from unidecode import unidecode
import google.generativeai as genai

# ---------------- CONFIG ----------------
PINECONE_INDEX_NAME = "exercise-embeddings"
PINECONE_DIMENSION = 1024
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 3

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Pinecone, embedding model, and Gemini
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# FSM States
class State:
    SMALL_TALK = "SMALL_TALK"
    PERSONAL_FOLLOW_UP = "PERSONAL_FOLLOW_UP"
    EXERCISE_SELECTION = "EXERCISE_SELECTION"
    EXERCISE_INTERACTION = "EXERCISE_INTERACTION"
    EXIT = "EXIT"

# Session context
session_context = {
    "state": State.SMALL_TALK,
    "user_preferences": {
        "topic": "linear_equations",  # Default to linear equations
        "difficulty": "medium",
        "grade": None
    },
    "current_exercise": None,
    "current_chunk_id": None
}

# Function to sanitize string for ASCII-only
def sanitize_ascii(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]', '_', str(text))

# Function to call Gemini-1.5-Flash
def call_llm(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 150,
                "temperature": 0.7
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, something went wrong. Let's continue!"

# Function to query Pinecone
def query_pinecone(query_text: str, filters: Dict = None, top_k: int = TOP_K) -> List[Dict]:
    query_embedding = model.encode([query_text])[0].tolist()
    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }
    if filters:
        if "mathematical_concept" in filters and filters["mathematical_concept"] is None:
            filters["mathematical_concept"] = "linear_equations"
        query_params["filter"] = filters
    print(f"Query filters: {filters}")  # Debug
    try:
        result = index.query(**query_params)
        print(f"Matches found: {len(result['matches'])}")  # Debug
        return result["matches"]
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

# Function to upsert exercises to Pinecone
def upsert_exercises(exercises: List[Dict]):
    for exercise in exercises:
        exercise_id = f"exercise_{exercise['exercise_metadata']['exercise_number']}"
        text = exercise["exercise_content"]["main_data"]["text"]
        for section in exercise["exercise_content"]["sections"]:
            section_id = section["section_id"]
            question_text = section["question"]["text"]
            combined_text = f"{text} {question_text}"
            embedding = model.encode([combined_text])[0].tolist()
            metadata = {
                "chunk_type": "question",
                "exercise_number": exercise["exercise_metadata"]["exercise_number"],
                "section_id": section_id,
                "topic": exercise["exercise_metadata"]["topic"],
                "mathematical_concept": "linear_equations",  # Explicitly set
                "difficulty": exercise["exercise_metadata"]["exercise_type"],
                "grade": exercise["exercise_metadata"]["class"],
                "text": question_text,
                "rows_data": json.dumps(section["question"].get("table", {}).get("rows_data", [])),
                "svg_exists": section["question"]["svg"] is not None
            }
            index.upsert([
                {
                    "id": f"{exercise_id}_section_{section_id}",
                    "values": embedding,
                    "metadata": metadata
                }
            ])
            # Upsert hints and solutions
            if "hint" in section:
                hint_text = section["hint"]["text"]
                embedding = model.encode([hint_text])[0].tolist()
                metadata["chunk_type"] = "hint"
                metadata["text"] = hint_text
                index.upsert([
                    {
                        "id": f"{exercise_id}_section_{section_id}_hint",
                        "values": embedding,
                        "metadata": metadata
                    }
                ])
            if "solution" in section:
                solution_text = section["solution"]["text"]
                embedding = model.encode([solution_text])[0].tolist()
                metadata["chunk_type"] = "solution"
                metadata["text"] = solution_text
                metadata["rows_data"] = json.dumps(section["solution"].get("table", {}).get("rows_data", []))
                index.upsert([
                    {
                        "id": f"{exercise_id}_section_{section_id}_solution",
                        "values": embedding,
                        "metadata": metadata
                    }
                ])

# Prompt template
def build_prompt(state: str, user_input: str, chunks: List[Dict] = None) -> str:
    if state == State.SMALL_TALK:
        return f"""
        You're a friendly math tutor. Engage in casual small talk based on the user's input: '{user_input}'.
        Keep the tone warm and encouraging, like: "Hey, great to hear from you! How’s your day going?"
        Transition to asking about their math preferences, e.g., "What math topics do you enjoy?"
        Keep the response concise and in English.
        """

    elif state == State.PERSONAL_FOLLOW_UP:
        return f"""
        You're a friendly math tutor. The user said: '{user_input}'. Acknowledge their input and ask about their math preferences (e.g., topic or difficulty).
        Example: "Cool, you like math! Do you prefer topics like linear equations or something else?"
        Keep the tone encouraging and concise, in English.
        """

    elif state == State.EXERCISE_SELECTION:
        if not chunks:
            return """
            You're a friendly math tutor. No exercises were found for the user's preferences.
            Apologize and suggest trying a different topic or difficulty.
            Example: "Sorry, I couldn’t find an exercise for that. Want to try a different topic like algebra?"
            Keep the response concise and in English.
            """
        context = "\n".join([
            f"Exercise {i+1}: {chunk['metadata']['text']} (Difficulty: {chunk['metadata']['difficulty']}, Topic: {chunk['metadata']['mathematical_concept']})"
            f"{' Table: ' + chunk['metadata']['rows_data'] if chunk['metadata']['rows_data'] != '[]' else ''}"
            for i, chunk in enumerate(chunks)
        ])
        return f"""
        You're a friendly math tutor. Present the following exercises to the user, keeping the tone engaging and clear.
        Use LaTeX for math expressions (e.g., $y = 2x + 3$). If the exercise involves a diagram (svg_exists: true), describe minor SVG modifications (e.g., scale the graph by 1.2x).
        Maintain solvability but feel free to make minor numeric changes (e.g., change $y = 2x + 3$ to $y = 3x + 5$).
        Example interaction:
        - Tutor: "Here’s a fun exercise: Solve $y = 2x + 3$ for $x = 1$. Need a hint?"
        - User: "Hint."
        - Tutor: "Try substituting $x = 1$ into the equation."
        Exercises:
        {context}
        Instructions:
        - Select one exercise to present to the user.
        - Format any math expressions in LaTeX.
        - If svg_exists is true, suggest a minor SVG adjustment (e.g., "Let’s scale the graph slightly").
        - Ask if the user wants to try it or needs a hint.
        Keep the response concise and in English.
        """

    elif state == State.EXERCISE_INTERACTION:
        current_chunk = session_context["current_exercise"]
        return f"""
        You're a friendly math tutor. The user is working on: '{current_chunk['text']}' (Topic: {current_chunk['mathematical_concept']}).
        User input: '{user_input}'.
        If they ask for a hint, provide one based on the topic (or retrieve a hint chunk).
        If they ask for a solution, provide it (or retrieve a solution chunk).
        If they provide an answer, evaluate it and give feedback.
        Use LaTeX for math expressions (e.g., $y = 2x + 3$).
        If svg_exists is true ({current_chunk['svg_exists']}), suggest a minor SVG adjustment (e.g., "Try zooming the graph by 1.1x").
        Maintain solvability with minor numeric changes if needed.
        Example:
        - User: "Solve $y = 2x + 3$ for $x = 1$."
        - Tutor: "Great try! Let’s check: $y = 2(1) + 3 = 5$. Correct! Want another?"
        Instructions:
        - Respond to the user’s input.
        - Use LaTeX for math.
        - Keep the tone encouraging and concise, in English.
        """

    return "Error: Invalid state."

# FSM Transition and Action Logic
def process_user_input(user_input: str) -> str:
    current_state = session_context["state"]
    user_input_lower = user_input.lower()  # Define at the start to avoid UnboundLocalError
    response = ""

    if current_state == State.SMALL_TALK:
        prompt = build_prompt(State.SMALL_TALK, user_input)
        response = call_llm(prompt)
        session_context["state"] = State.PERSONAL_FOLLOW_UP

    elif current_state == State.PERSONAL_FOLLOW_UP:
        if "linear" in user_input_lower or "equations" in user_input_lower:
            session_context["user_preferences"]["topic"] = "linear_equations"
        elif "quadratic" in user_input_lower:
            session_context["user_preferences"]["topic"] = "quadratic_equations"
        else:
            session_context["user_preferences"]["topic"] = "linear_equations"  # Default
        if "easy" in user_input_lower:
            session_context["user_preferences"]["difficulty"] = "easy"
        elif "hard" in user_input_lower:
            session_context["user_preferences"]["difficulty"] = "hard"
        else:
            session_context["user_preferences"]["difficulty"] = "medium"
        if "grade" in user_input_lower or "8th" in user_input_lower:
            session_context["user_preferences"]["grade"] = "H"

        prompt = build_prompt(State.PERSONAL_FOLLOW_UP, user_input)
        response = call_llm(prompt)
        session_context["state"] = State.EXERCISE_SELECTION
        if not response.endswith("?"):
            response += " Ready to try a math exercise?"

    elif current_state == State.EXERCISE_SELECTION:
        filters = {
            "chunk_type": "question",
            "mathematical_concept": session_context["user_preferences"].get("topic", "linear_equations"),
            "difficulty": session_context["user_preferences"].get("difficulty", "medium")
        }
        if session_context["user_preferences"]["grade"]:
            filters["grade"] = session_context["user_preferences"]["grade"]

        matches = query_pinecone(user_input, filters=filters, top_k=TOP_K)
        if matches:
            session_context["current_exercise"] = matches[0]["metadata"]
            session_context["current_chunk_id"] = matches[0]["id"]
            prompt = build_prompt(State.EXERCISE_SELECTION, user_input, matches)
            response = call_llm(prompt)
            session_context["state"] = State.EXERCISE_INTERACTION
        else:
            prompt = build_prompt(State.EXERCISE_SELECTION, user_input)
            response = call_llm(prompt)
            session_context["state"] = State.PERSONAL_FOLLOW_UP

    elif current_state == State.EXERCISE_INTERACTION:
        if "hint" in user_input_lower:
            filters = {
                "chunk_type": "hint",
                "exercise_number": session_context["current_exercise"]["exercise_number"],
                "section_id": session_context["current_exercise"].get("section_id", "main")
            }
            matches = query_pinecone("", filters=filters, top_k=1)
            if matches:
                response = f"Hint: {matches[0]['metadata']['text']}"
            else:
                response = "No hint available. Want to see the solution?"
        elif "solution" in user_input_lower:
            filters = {
                "chunk_type": {"$in": ["solution", "full_solution"]},
                "exercise_number": session_context["current_exercise"]["exercise_number"],
                "section_id": session_context["current_exercise"].get("section_id", "main")
            }
            matches = query_pinecone("", filters=filters, top_k=1)
            if matches:
                response = f"Solution: {matches[0]['metadata']['text']}"
                if matches[0]["metadata"]["rows_data"] != "[]":
                    response += f" Table: {matches[0]['metadata']['rows_data']}"
            else:
                response = "No solution available. Want another exercise?"
                session_context["state"] = State.EXERCISE_SELECTION
        else:
            prompt = build_prompt(State.EXERCISE_INTERACTION, user_input)
            response = call_llm(prompt)
            response += " Want the solution or another exercise?"
            session_context["state"] = State.EXERCISE_SELECTION

    elif current_state == State.EXIT:
        response = "Goodbye! Come back for more math fun!"
        session_context["state"] = State.SMALL_TALK

    if "exit" in user_input_lower or "quit" in user_input_lower:
        session_context["state"] = State.EXIT
        response = "Goodbye! Come back for more math fun!"

    return response

# ---------------- MAIN LOOP ----------------
def main():
    print("Welcome to the Math Chatbot! Type 'exit' to quit.")
    while session_context["state"] != State.EXIT:
        user_input = input("You: ")
        response = process_user_input(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()