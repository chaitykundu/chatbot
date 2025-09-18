import json
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

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

# ---------------- HELPERS ----------------
def load_exercises():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

        # Ensure we always get a flat list of dict exercises
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

# ---------------- HELPERS ----------------
# ... (other helper functions remain unchanged)

# ---------------- HELPERS ----------------
# ---------------- HELPERS ----------------
def save_svg_to_file(svg_content, filename):
    """Save SVG content to a file for rendering."""
    output_dir = Path("svg_outputs")
    output_dir.mkdir(exist_ok=True)  # Create svg_outputs folder if it doesn't exist
    file_path = output_dir / filename  # Construct path to save in svg_outputs
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"SVG saved to {file_path}. Open it in a browser to view.")
    except Exception as e:
        print(f"Error saving SVG: {e}")

def run_exercise(exercise):
    meta = exercise["exercise_metadata"]
    content = exercise["exercise_content"]

    print(f"\n📘 Exercise {meta['exercise_number']} ({meta['exercise_type']})")
    print("Main text:", content["main_data"]["text"])

    # Check for SVG in main_data
    if "svg" in content["main_data"] and content["main_data"]["svg"]:
        save_svg_to_file(
            content["main_data"]["svg"],
            f"exercise_{meta['exercise_number']}_main.svg"
        )

    for sec in content["sections"]:
        q = sec.get("question", {}).get("text")
        if q:
            print(f"\n❓ Section {sec['section_number']} - {q}")
            # Check for SVG in the question
            question = sec.get("question", {})
            if "svg" in question and question["svg"]:
                save_svg_to_file(
                    question["svg"],
                    f"exercise_{meta['exercise_number']}_section_{sec['section_number']}.svg"
                )
            input("Press Enter to continue to the next question...")

# ... (rest of the code remains unchanged)

# ---------------- RETRIEVAL CHATBOT ----------------
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

# ---------------- CHATBOT MAIN LOOP ----------------
def main():
    print("Welcome to the Math Chatbot! 🎓")
    print("Choose mode: (1) Guided Exercises, (2) Free Q&A")
    mode = input("> ")

    if mode == "1":
        # Guided pipeline
        classes = get_classes()
        print(f"Available classes: {classes}")
        chosen_class = input("Pick a class: ")

        topics = get_topics(chosen_class)
        print(f"Available topics: {topics}")
        chosen_topic = input("Pick a topic: ")

        exercises = get_exercises(chosen_class, chosen_topic)
        if not exercises:
            print("No exercises found for this selection.")
            return

        # Run the first exercise for now
        exercise = exercises[0]
        run_exercise(exercise)

    else:
        # Retrieval chatbot
        print("Ask me anything about math exercises! (type 'quit' to exit)")
        while True:
            query = input("You: ")
            if query.lower() in ["quit", "exit"]:
                break
            results = retrieve_answer(query)
            if results:
                for ans in results:
                    print(f"\n[{ans['chunk_type']} | Ex {ans['exercise_number']} | parent {ans['parent_id']}]")
                    print(ans['text'])
            else:
                print("Sorry, I couldn’t find anything.")

if __name__ == "__main__":
    main()