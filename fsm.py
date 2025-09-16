import threading
from transitions import Machine
from langdetect import detect
import json
import time
import re
from pathlib import Path

# Load knowledge base from JSON file
INPUT_FILE = Path("Files/merged_lessons.json")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Build knowledge base from JSON data
KNOWLEDGE_BASE = {
    "topics": [],
    "difficulties": [],
    "examples": {}
}

# Extract unique topics and difficulties from exercise metadata
for exercise in data if isinstance(data, list) else [data]:
    meta = exercise.get("exercise_metadata", {})
    topic = meta.get("topic", "unknown")
    difficulty = meta.get("difficulty", "medium")
    if topic not in KNOWLEDGE_BASE["topics"]:
        KNOWLEDGE_BASE["topics"].append(topic)
    if difficulty not in KNOWLEDGE_BASE["difficulties"]:
        KNOWLEDGE_BASE["difficulties"].append(difficulty)

    # Extract example text from main_data or sections
    content = exercise.get("exercise_content", {})
    main_data = content.get("main_data", {})
    if main_data.get("text"):
        KNOWLEDGE_BASE["examples"][topic] = main_data["text"][:50] + "..."  # Truncate for brevity
    sections = content.get("sections", [])
    for sec in sections:
        section_data = sec.get("section_data", {})
        if section_data.get("text") and topic in KNOWLEDGE_BASE["examples"]:
            KNOWLEDGE_BASE["examples"][topic] = section_data["text"][:50] + "..."
        elif section_data.get("text"):
            KNOWLEDGE_BASE["examples"][topic] = section_data["text"][:50] + "..."

# Ensure minimum examples if none found
for topic in KNOWLEDGE_BASE["topics"]:
    if topic not in KNOWLEDGE_BASE["examples"]:
        KNOWLEDGE_BASE["examples"][topic] = f"Solve a {topic} problem..."

# Define the states
states = [
    'small_talk',
    'academic_transition',
    'exercise_selection',
    'guidance',
    'inactive'
]

# Define the class to manage the FSM
class DialogueFSM:
    def __init__(self):
        self.state = 'small_talk'
        self.conversation_history = []
        self.context = {"topic": None, "difficulty": None}
        self.inactivity_timer = None
        self.timeout_duration = 30

        self.machine = Machine(
            model=self,
            states=states,
            initial='small_talk',
            transitions=[
                {'trigger': 'to_academic_transition', 'source': 'small_talk', 'dest': 'academic_transition'},
                {'trigger': 'to_exercise_selection', 'source': 'academic_transition', 'dest': 'exercise_selection'},
                {'trigger': 'to_guidance', 'source': '*', 'dest': 'guidance'},
                {'trigger': 'to_inactive', 'source': '*', 'dest': 'inactive'}
            ]
        )
        self.on_enter_small_talk()

    def on_enter_small_talk(self):
        greeting = f"Good morning! It's 09:27 AM +06 on Monday, September 08, 2025. Hope you're doing well. How's it going?"
        self.send_message(greeting)

    def on_enter_academic_transition(self):
        last_response = self.conversation_history[-1][1] if self.conversation_history else ""
        question = f"Glad to hear that! Based on '{last_response}', what did you learn recently? Or when is your next exam?"
        self.send_message(question)

    def on_enter_exercise_selection(self):
        context_str = f"Based on your interest in {self.context['topic']} at {self.context['difficulty']} level, " if self.context["topic"] or self.context["difficulty"] else "Based on our chat, "
        question = f"{context_str}let’s pick an exercise. What topic (e.g., {', '.join(KNOWLEDGE_BASE['topics'])}) or difficulty (e.g., {', '.join(KNOWLEDGE_BASE['difficulties'])}) would you like?"
        self.send_message(question)

    def on_enter_guidance(self):
        example = KNOWLEDGE_BASE["examples"].get(self.context["topic"], "your problem")
        self.send_message(f"Need help? Let’s start with: What do you think is the next step for {example}?")

    def on_enter_inactive(self):
        self.send_message("Are you still there? It's 09:27 AM +06 on Monday, September 08, 2025.")
        self.reset_inactivity_timer()

    def process_user_input(self, user_message):
        self.reset_inactivity_timer()
        lang = detect(user_message)
        direction = "rtl" if lang == "he" else "ltr"
        self.conversation_history.append(("user", user_message, direction))
        self.update_context(user_message)

        if self.state == 'small_talk' and any(word in user_message.lower() for word in ['good', 'fine', 'great', 'ok']):
            self.to_academic_transition()
        elif self.state == 'academic_transition' and any(word in user_message.lower() for word in ['learn', 'exam', 'math', 'science']):
            self.to_exercise_selection()
        elif self.state in ['exercise_selection', 'guidance'] and any(word in user_message.lower() for word in ['help', 'stuck', 'solve']):
            self.to_guidance()

    def update_context(self, user_message):
        user_message = user_message.lower()
        for topic in KNOWLEDGE_BASE["topics"]:
            if topic in user_message:
                self.context["topic"] = topic
                break
        for difficulty in KNOWLEDGE_BASE["difficulties"]:
            if difficulty in user_message:
                self.context["difficulty"] = difficulty
                break

    def send_message(self, message):
        try:
            lang = detect(message) if self.state != 'inactive' else detect(self.conversation_history[-1][1])
            direction = "rtl" if lang == "he" else "ltr"
        except:
            direction = "ltr"
        self.conversation_history.append(("bot", message, direction))
        print(f"[{direction}] Bot: {message}")

    def start_inactivity_timer(self):
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        self.inactivity_timer = threading.Timer(self.timeout_duration, self.to_inactive)
        self.inactivity_timer.start()

    def reset_inactivity_timer(self):
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
        self.start_inactivity_timer()

    def provide_guidance(self):
        if self.state == 'guidance':
            history = [msg for role, msg, *rest in self.conversation_history if isinstance(msg, str)]
            guidance_count = len([h for h in history if h.startswith("Need help?") or h.startswith("Another guiding")])
            if guidance_count < 4:
                example = KNOWLEDGE_BASE["examples"].get(self.context["topic"], "your problem")
                self.send_message(f"Another guiding question: Can you try the next step for {example}?")
            else:
                self.send_message(f"Here’s the full solution with explanation for {self.context['topic'] or 'your problem'}: [LaTeX math here].")

# Interactive loop
def main():
    fsm = DialogueFSM()
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            fsm.process_user_input(user_input)
            if fsm.state == 'guidance':
                fsm.provide_guidance()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()