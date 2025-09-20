import json
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from tutor_states import TutorStateMachine, TutorState

# ---------------- CONFIG ----------------
INPUT_FILE = Path("Files/merged_output.json")
MODEL_NAME = "intfloat/multilingual-e5-large"
PINECONE_INDEX_NAME = "exercise-embeddings"

# Load .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Init Pinecone + model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

# ---------------- DATA HELPERS ----------------
def load_exercises():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    exercises = []
    if isinstance(data, dict):
        exercises = [data]
    elif isinstance(data, list):
        for ex in data:
            if isinstance(ex, dict):
                exercises.append(ex)
            elif isinstance(ex, list):  # flatten nested lists
                exercises.extend([e for e in ex if isinstance(e, dict)])
    return exercises

all_exercises = load_exercises()

def get_classes():
    return sorted(set(ex["exercise_metadata"]["class"] for ex in all_exercises))

def get_topics(chosen_class):
    return sorted(set(
        ex["exercise_metadata"]["topic"]
        for ex in all_exercises
        if ex["exercise_metadata"]["class"] == chosen_class
    ))

def get_exercises(chosen_class, chosen_topic):
    return [
        ex for ex in all_exercises
        if ex["exercise_metadata"]["class"] == chosen_class
        and ex["exercise_metadata"]["topic"] == chosen_topic
    ]

def get_exercises_by_topic(topic: str):
    """Get exercises matching the topic (fuzzy search)"""
    matching_exercises = []
    topic_lower = topic.lower()
    
    for ex in all_exercises:
        ex_topic = ex["exercise_metadata"]["topic"].lower()
        ex_class = ex["exercise_metadata"]["class"].lower()
        
        # Check if topic appears in either the topic or class field
        if topic_lower in ex_topic or topic_lower in ex_class:
            matching_exercises.append(ex)
    
    # If no direct matches, try partial matching
    if not matching_exercises:
        for ex in all_exercises:
            ex_topic = ex["exercise_metadata"]["topic"].lower()
            ex_class = ex["exercise_metadata"]["class"].lower()
            
            # Check if any word from the topic appears
            topic_words = topic_lower.split()
            for word in topic_words:
                if word in ex_topic or word in ex_class:
                    matching_exercises.append(ex)
                    break
    
    return matching_exercises[:5]  # Limit to first 5 exercises

def save_svg_to_file(svg_content, filename):
    """Save SVG content to a file for rendering."""
    output_dir = Path("svg_outputs")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"SVG saved to {file_path}. Open it in a browser to view.")
    except Exception as e:
        print(f"Error saving SVG: {e}")

# ---------------- RETRIEVAL CHATBOT (BACKUP) ----------------
def retrieve_answer(query, top_k=3):
    emb = model.encode([query])[0].tolist()
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)

    answers = []
    for match in results["matches"]:
        meta = match["metadata"]
        answers.append({
            "score": match["score"],
            "text": meta.get("text", ""),
            "chunk_type": meta.get("chunk_type"),
            "parent_id": meta.get("parent_id"),
            "exercise_number": meta.get("exercise_number"),
        })
    return answers

# ---------------- MAIN TUTORING SYSTEM ----------------
def main():
    print("Welcome to the AI Math Tutor! 🎓")
    print("I'll guide you through a personalized learning experience.")
    print("Type 'quit' anytime to exit.\n")
    
    # Initialize the state machine
    tutor = TutorStateMachine()
    exercises = []
    
    # Start the conversation
    bot_message = tutor.process_input("")
    print(f"Bot: {bot_message}")
    
    while tutor.context.state != TutorState.COMPLETE:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Thanks for learning with me today! Goodbye! 👋")
            break
        
        # Special handling when we transition to learning state
        if tutor.context.state == TutorState.DIAGNOSTIC and tutor.context.diagnostic_step == 4:
            # About to transition to learning - get exercises based on focus topic
            bot_message = tutor.process_input(user_input)
            print(f"Bot: {bot_message}")
            
            # Now get relevant exercises
            if tutor.context.focus_topic:
                exercises = get_exercises_by_topic(tutor.context.focus_topic)
                if not exercises:
                    print("Bot: I couldn't find exercises for that specific topic. Let me get some general exercises for you.")
                    exercises = all_exercises[:5]  # Use first 5 exercises as fallback
                else:
                    print(f"Bot: I found {len(exercises)} exercises related to '{tutor.context.focus_topic}'. Let's start!")
        
        elif tutor.context.state == TutorState.LEARNING:
            # Handle SVG content if present in current exercise
            if tutor.context.current_exercise:
                exercise = tutor.context.current_exercise
                content = exercise["exercise_content"]
                
                # Save main SVG if present
                if "svg" in content["main_data"] and content["main_data"]["svg"]:
                    save_svg_to_file(
                        content["main_data"]["svg"],
                        f"exercise_{exercise['exercise_metadata']['exercise_number']}_main.svg"
                    )
                
                # Save section SVG if present
                if content["sections"]:
                    for sec in content["sections"]:
                        question = sec.get("question", {})
                        if "svg" in question and question["svg"]:
                            save_svg_to_file(
                                question["svg"],
                                f"exercise_{exercise['exercise_metadata']['exercise_number']}_section_{sec['section_number']}.svg"
                            )
            
            bot_message = tutor.process_input(user_input, exercises)
            print(f"Bot: {bot_message}")
        
        else:
            # Normal processing for opening and diagnostic states
            bot_message = tutor.process_input(user_input)
            print(f"Bot: {bot_message}")
    
    print("\n🎉 Session completed! Great work today!")

if __name__ == "__main__":
    main()