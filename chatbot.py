# chatbot.py (Enhanced AI-Based Math Tutor with Progressive Guidance and Improved SVG Handling)
import os
import re
import json
import random
import time
import threading
from pathlib import Path
from enum import Enum, auto
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer
import logging
import uuid

# Load environment variables
load_dotenv(dotenv_path=Path(".env"))

# Set up logging

logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = Path("Files/merged_output.json")
SVG_OUTPUT_DIR = Path("svg_outputs")
SVG_OUTPUT_DIR.mkdir(exist_ok=True)

# Pinecone Config
INDEX_NAME = "exercise-embeddings"
EMBED_DIM = 1024
TOP_K_RETRIEVAL = 20

# Embedding Model Config
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# Enhanced Inactivity Settings
INACTIVITY_TIMEOUT = 60  # Increased from 30 seconds
TYPING_DETECTION_THRESHOLD = 10  # Seconds to wait for complete input

# Progressive Guidance Settings
MIN_ATTEMPTS_BEFORE_HINT = 1
MIN_ATTEMPTS_BEFORE_SOLUTION = 2
MAX_GUIDANCE_LEVELS = 3  # 0=encouragement, 1=guiding_question, 2=hint, 3=solution

# -----------------------------
# GenAI Chat Client (using LangChain)
# -----------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt template for the RAG chatbot (bilingual support)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Math AI tutor. 
    
    Language Rules:
    - If the question is in Hebrew → respond in Hebrew
    - If the question is in English → respond in English
    - Always match the user's language preference
    - For Hebrew responses, use Right-to-Left (RTL) formatting for conversational text.
    - Ensure all mathematical expressions and scientific notation remain Left-to-Right (LTR), even within Hebrew sentences.
    
    Teaching Guidelines:
    - Never give direct answers immediately
    - Use a gradual assistance approach: encouragement → guiding questions → hints → solution
    - Ask guiding questions to help students think through problems
    - Build understanding step by step
    - Use the provided context to give accurate information
    - If context lacks crucial information, state what's missing
    - When providing hints, use EXACT TEXT from context when available
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Context: {context}\n\nQuestion: {input}")
])
rag_chain = rag_prompt | llm

# Small talk prompt (bilingual)
small_talk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Math AI tutor starting a conversation.
    
    Language Rules:
    - Detect the user's language and respond in the same language
    - If Hebrew is detected, respond in Hebrew
    - If English is detected, respond in English
    - Default to English if language is unclear
    
    Personality:
    - Warm, encouraging, and approachable
    - Enjoys chatting about hobbies, school, and daily life
    - Enthusiastic about helping with math
    - Keep responses short and conversational (1-2 sentences max) and conversational
    - Understand the user's intent even with spelling mistakes or unclear input
    - Examples: "Hey! How are you doing today?", "Hi there! What's up?", "How's everything going?"
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
small_talk_chain = small_talk_prompt | llm

# Personal follow-up prompt (bilingual)
personal_followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are continuing a casual conversation with a student.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - Keep the same language throughout the conversation
    
    Guidelines:
    - Acknowledge their response warmly
    - Keep it brief and natural (1-2 sentences)
    - Show genuine interest in personal topics like work, sports, daily life
    - Gradually transition toward academic readiness
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
personal_followup_chain = personal_followup_prompt | llm

# Diagnostic prompt (bilingual)
diagnostic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a math tutor conducting a diagnostic assessment to understand the student's needs.
    
    Language Rules:
    - Match the user's language (Hebrew or English)
    - For Hebrew: Use proper RTL formatting for general text, keep math expressions LTR
    
    Diagnostic Guidelines:
    - Ask one diagnostic question at a time based on the sequence: test, last class, focus
    - Acknowledge the previous response briefly if relevant
    - Examples: "Sounds interesting! Do you have a test coming up?" or "Got it. What did you cover in your last class?"
    - Keep it conversational and encouraging
    - Responses should be short (1-2 sentences max)
    - Transition naturally to the next question or to academic topics after all questions
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Previous diagnostic response: {previous_response}\nCurrent question to ask: {current_question}\nGenerate response:"),
])
diagnostic_chain = diagnostic_prompt | llm

academic_transition_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are transitioning from personal chat to academic topics.
        
    Language Rules:
    - Match the user's language (Hebrew or English)
    - For Hebrew: Use proper RTL formatting for general text, keep math expressions LTR

    Academic Transition Guidelines:
    - Ask about recent learning or upcoming academic events, but AVOID directly asking for grade/class—focus on subjects or interests.
    - Examples: "What did you learn recently?", "When is your next exam?", "How's school going?", "What subjects are you studying?"
    - Bridge from personal to academic naturally
    - Keep it friendly but start showing academic interest
    - Keep responses short (1 sentence)
    - Make the transition feel natural
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
academic_transition_chain = academic_transition_prompt | llm

# New prompt for generating question and SVG
question_svg_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Math tutor creating a new question and SVG visualization.
    
    Language: Generate the question in {language} ({'Hebrew' if language == 'he' else 'English'}). Additionally, do not mismatch the language; suppose you start with english, continue in English.
     and if you start with Hebrew, continue in Hebrew. Until the user changes the language.

    Task:
    - Generate a NEW math question for the given topic and grade.
    - Include a corresponding SVG visualization (single <svg> element).
    - For slope questions: Use two distinct points (e.g., (1,1) and (4,7)), calculate slope (e.g., 2).
    - Structure output as JSON with fields: question (str), solution (str), svg (str).
    - SVG Guidelines:
      - Use viewBox="0 0 400 300".
      - Include axes: x (0-10, bottom), y (0-10, left, inverted).
      - Draw a bold blue line for the slope, add light grid, label points and axes.
      - Ensure valid, self-contained SVG XML (no interactions).
    - Question Guidelines:
      - Simple, solvable, grade-appropriate (e.g., grade 7/8).
      - Include specific points (e.g., "Find the slope of the line through (1,1) and (4,7)").
      - Solution should show steps (e.g., "Slope = (y2-y1)/(x2-x1) = (7-1)/(4-1) = 6/3 = 2").
    - Return ONLY the JSON object, no explanations or markdown."""),
    ("user", "Topic: {topic}\nGrade: {grade}")
])
question_svg_generation_chain = question_svg_generation_prompt | llm


# -----------------------------
# Localization (Bilingual Support)
# -----------------------------
I18N = {
    "en": {
        "choose_language": "Choose language:\n1) English (default)",
        "ask_grade": "Nice! Before we start, what grade are you in? (e.g., 7, 8)",
        "ask_topic": "Which topic would you like to practice? (e.g., {topics})",
        "ready_for_question": "Awesome! Let's start with this exercise:",
        "hint_prefix": "💡 Hint: ",
        "solution_prefix": "✅ Solution: ",
        "wrong_answer": "Not quite right. Let me help you think through this...",
        "guiding_question": "🤔 Let me ask you this: ",
        "encouragement": "You're making progress — give it try first!",
        "try_again": "Can you try again? Think about your approach.",
        "need_more_attempts": "Give it another try first - I believe you can work through this! {guiding_prompt}",
        "no_exercises": "No exercises found for grade {grade} and topic {topic}.",
        "no_more_hints": "No more hints available. Would you like to see the solution?",
        "no_relevant_exercises": "I couldn't find any relevant exercises for your query.",
        "ask_for_solution": "Would you like me to show you the solution?",
        "irrelevant_msg": "I can only help with math exercises and related questions.",
        "inactivity_check": "Are you still there? I'm here to help whenever you're ready!",
        "session_timeout": "It looks like you stepped away. Feel free to continue whenever you're ready!",
        "ask_for_doubts": "Great work! You've completed several exercises on {topic}. Do you have any questions or doubts about this topic?",
        "no_doubts_response": "Perfect! It looks like you understand {topic} well. Great job today!",
        "doubt_clearing_intro": "Good question! Let me address your question about {topic}:",
        "ask_more_doubts": "Do you have any other questions about {topic}?",
        "small_talk_hobbies": "What hobbies do you have?",
        "diagnostic_test": "Okay! Do you have a test coming up?",
        "diagnostic_last_class": "What did you cover in your last class?",
        "diagnostic_focus": "What would you like to work on today?",
        "doubt_answer_complete": "I hope that helps clarify things about {topic} for you!",
        "lesson_closing": "Great, that was an awesome lesson! I’ll send you similar exercises for practice and see you in the next session. If you have questions, feel free to message me. And if you get stuck – just remember, you’re a genius. Bye!"
    },
    "he": {
        "choose_language": "בחר שפה:\n1) אנגלית (ברירת מחדל)",
        "ask_grade": "נחמד! לפני שנתחיל, באיזו כיתה אתה? (למשל, ז, ח)",
        "ask_topic": "מצוין! כיתה {grade}. באיזה נושא תרצה להתרגל? (לדוגמה: {topics})",
        "ready_for_question": "מעולה! בואו נתחיל עם התרגיל הזה:",
        "hint_prefix": "💡 רמז: ",
        "solution_prefix": "✅ פתרון: ",
        "wrong_answer": "לא בדיוק נכון. בוא אעזור לך לחשוב על זה...",
        "guiding_question": "🤔 תן לי לשאול אותך את זה: ",
        "encouragement": "אתה מתקדם - תנסה קודם!",
        "try_again": "תוכל לנסות שוב? חשוב על הגישה שלך.",
        "need_more_attempts": "תן לזה עוד ניסיון - אני מאמין שאתה יכול לעבוד על זה!",
        "no_exercises": "לא נמצאו תרגילים עבור כיתה {grade} ונושא {topic}.",
        "no_more_hints": "אין עוד רמזים זמינים. האם תרצה לראות את הפתרון?",
        "no_relevant_exercises": "לא הצלחתי למצוא תרגילים רלוונטיים לשאלתך.",
        "ask_for_solution": "האם תרצה שאראה לך את הפתרון?",
        "irrelevant_msg": "אני יכול לעזור רק עם תרגילי מתמטיקה ושאלות קשורות.",
        "inactivity_check": "אתה עדיין כאן? אני כאן לעזור בכל עת שתהיה מוכן!",
        "session_timeout": "נראה שיצאת לרגע. הרגש בנוח להמשיך בכל עת שתהיה מוכן!",
        "ask_for_doubts": "עבודה מעולה! השלמת מספר תרגילים על {topic}. יש לך שאלות או ספקות על הנושא הזה?",
        "no_doubts_response": "מושלם! נראה שאתה מבין את {topic} היטב. עבודה נהדרת היום!",
        "doubt_clearing_intro": "אני כאן לעזור! תן לי לענות על השאלה שלך על {topic}:",
        "ask_more_doubts": "יש לך שאלות נוספות על {topic}?",
        "small_talk_hobbies": "אילו תחביבים יש לך?",
        "diagnostic_test": "יש לך מבחן בקרוב?",
        "diagnostic_last_class": "מה סקרתם בשיעור האחרון שלך?",
        "diagnostic_focus": "על מה תרצה לעבוד היום?",
        "doubt_answer_complete": "אני מקווה שזה עוזר להבהיר דברים על {topic} עבורך!",
        "lesson_closing": "נהדר, זה היה שיעור מדהים! אשלח לך תרגילים דומים לתרגול וניפגש בשיעור הבא. אם יש לך שאלות, אל תהסס לפנות אליי. ואם תיתקע – זכור, אתה גאון. להתראות!"
    }
}

# -----------------------------
# FSM STATES
# -----------------------------
class State(Enum):
    START = auto()
    SMALL_TALK = auto()
    PERSONAL_FOLLOWUP = auto()
    DIAGNOSTIC = auto()
    ACADEMIC_TRANSITION = auto()
    ASK_GRADE = auto()
    EXERCISE_SELECTION = auto()
    QUESTION_ANSWER = auto()
    GUIDING_QUESTION = auto()
    PROVIDING_HINT = auto()
    ASK_FOR_DOUBTS = auto()
    DOUBT_CLEARING = auto()

# -----------------------------
# Helper Functions
# -----------------------------
def detect_language(text: str) -> str:
    """Detect if text is Hebrew or English."""
    if any('\u0590' <= char <= '\u05FF' for char in text):
        return "he"
    return "en"

# -----------------------------
# Helper Functions-Cleantext
# -----------------------------
def clean_math_text(text: str) -> str:
    if not text: return text
    # Remove $$...$$ and $...$
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
    # (Do NOT remove backslash commands globally)
    text = re.sub(r'\s+', ' ', text).replace('$', '')
    return text.strip()

def translate_text_to_english(text: str) -> str:
    """Translate text (likely Hebrew) to English using GenAI."""
    if not text or not text.strip():
        return text
    try:
        translation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise translator. Translate the following text to English. If it's already in English, return it as is. Provide ONLY the translation."),
            ("user", "{input}"),
        ])
        translation_chain = translation_prompt | llm
        response = translation_chain.invoke({"input": text.strip()})
        translated = response.content.strip()
        translated = clean_math_text(translated)
        if is_likely_hebrew(text) and not is_likely_hebrew(translated):
             return translated
        elif not is_likely_hebrew(text):
             return text
        else:
             logger.warning(f"Potential translation issue. Input: {text}, Output: {translated}")
             return translated
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"[Translation Error: {text}]"

def translate_text_to_conversation_language(text: str, target_language: str) -> str:
    """Translate text to match the conversation language (en or he)."""
    if not text or not text.strip():
        return text
    
    # If target language is English
    if target_language == "en":
        # Only translate if the text is actually in Hebrew
        if is_likely_hebrew(text):
            return translate_text_to_english(text)
        else:
            # Text is already in English (or other Latin script), return as is
            return text
    
    # If target language is Hebrew
    elif target_language == "he":
        # Only translate if the text is in English/Latin script and not already Hebrew
        if not is_likely_hebrew(text) and text.strip():
            try:
                translation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a precise translator. Translate the following text to Hebrew. If it's already in Hebrew, return it as is. Provide ONLY the translation."),
                    ("user", "{input}"),
                ])
                translation_chain = translation_prompt | llm
                response = translation_chain.invoke({"input": text.strip()})
                translated = response.content.strip()
                translated = clean_math_text(translated)
                return translated
            except Exception as e:
                logger.error(f"Error translating text to Hebrew: {str(e)}")
                return text
        else:
            # Text is already in Hebrew, return as is
            return text
    
    # Fallback: return original text
    return text

def is_likely_hebrew(text: str) -> bool:
    """Simple heuristic to check if text contains Hebrew characters."""
    return any('\u0590' <= char <= '\u05FF' for char in text)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_exercises(input_file: Path = None):
    """Load exercises with flattening logic for nested structures."""
    if input_file is None:
        input_file = INPUT_FILE  # Use default if not specified
    
    with open(input_file, "r", encoding="utf-8") as f:
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

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedding_model = None

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise EnvironmentError("PINECONE_API_KEY not found in .env")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(INDEX_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using SentenceTransformer."""
    if embedding_model is None:
        logger.error("Embedding model not loaded.")
        return []
    try:
        return embedding_model.encode([text], show_progress_bar=False)[0].tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

def retrieve_relevant_chunks(query: str, pc_index: Any, grade: Optional[str] = None, topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from Pinecone based on a query."""
    query = clean_math_text(query)
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return []

    filter_dict = {}
    if grade:
        filter_dict["grade"] = {"$eq": grade}
    if topic:
        filter_dict["topic"] = {"$eq": topic}

    try:
        response = pc_index.query(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVAL,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        return [match.metadata for match in response.matches]
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {str(e)}", exc_info=True)
        return []

def describe_svg_content(svg_content: str) -> str:
    """Describe SVG content using GenAI."""
    try:
        svg_description_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Provide a CONCISE English description of the main mathematical elements in the SVG (e.g., axes, points, lines, shapes). Do not include the raw SVG code. Provide only a brief description."),
            ("user", "Describe the following SVG content:\n```svg\n{svg_input}\n```"),
        ])
        svg_description_chain = svg_description_prompt | llm
        response = svg_description_chain.invoke({"svg_input": svg_content})
        return clean_math_text(response.content)
    except Exception as e:
        logger.error(f"Error describing SVG content: {str(e)}")
        return "An error occurred while describing the image."

# -----------------------------
# Enhanced Inactivity Timer with Typing Detection
# -----------------------------
class EnhancedInactivityTimer:
    def __init__(self, callback, timeout=INACTIVITY_TIMEOUT):
        self.callback = callback
        self.timeout = timeout
        self.timer = None
        self.last_activity_time = time.time()
        self.typing_detected = False
        
    def start(self):
        self.stop()
        self.timer = threading.Timer(self.timeout, self._check_inactivity)
        self.timer.daemon = True
        self.timer.start()
        
    def stop(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
            
    def reset(self):
        self.last_activity_time = time.time()
        self.typing_detected = False
        self.start()
    
    def mark_typing(self):
        """Mark that user is typing - prevents premature timeout."""
        self.typing_detected = True
        self.last_activity_time = time.time()
    
    def _check_inactivity(self):
        """Check if user is truly inactive."""
        current_time = time.time()
        time_since_activity = current_time - self.last_activity_time
        
        if self.typing_detected and time_since_activity < TYPING_DETECTION_THRESHOLD:
            self.start()
        elif time_since_activity >= self.timeout:
            self.callback()
        else:
            remaining_time = self.timeout - time_since_activity
            self.timer = threading.Timer(remaining_time, self._check_inactivity)
            self.timer.daemon = True
            self.timer.start()

# -----------------------------
# Enhanced Attempt Tracking
# -----------------------------
class AttemptTracker:
    def __init__(self):
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
        
    def reset(self):
        """Reset for new question."""
        self.total_attempts = 0
        self.incorrect_attempts = 0
        self.guidance_level = 0
        self.has_requested_hint = False
        self.has_requested_solution = False
    
    def record_attempt(self, is_correct: bool):
        """Record an attempt and return if guidance should be offered."""
        self.total_attempts += 1
        if not is_correct:
            self.incorrect_attempts += 1
        
        return not is_correct and self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT
    
    def can_provide_hint(self) -> bool:
        """Check if hint can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_HINT or 
                self.has_requested_hint)
    
    def can_provide_solution(self) -> bool:
        """Check if solution can be provided based on attempts."""
        return (self.incorrect_attempts >= MIN_ATTEMPTS_BEFORE_SOLUTION or
                self.has_requested_solution or
                (self.has_requested_hint and self.incorrect_attempts >= 1))
    
    def should_encourage_more_attempts(self, is_hint_request: bool = False, is_solution_request: bool = False) -> bool:
        """Determine if we should encourage more attempts instead of giving help."""
        if is_solution_request and not self.can_provide_solution():
            return True
        if is_hint_request and not self.can_provide_hint():
            return True
        return False
    
    def generate_exercise_with_llm(topic: str, grade: str) -> Optional[Dict[str, Any]]:
        """Generates a new exercise and SVG using a generative LLM."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = question_svg_generation_prompt.format(topic=topic, grade=grade)
            response = model.generate_content(prompt)
            
            # Extract and parse the JSON from the response
            generated_json = response.text.strip('` \n').strip('json\n')
            exercise = json.loads(generated_json)
            return exercise
        except Exception as e:
            logger.error(f"❌ Error generating exercise with LLM: {e}")
            return None

# -----------------------------
# Enhanced Dialogue FSM
# -----------------------------
class DialogueFSM:
    def __init__(self, exercises_data, pinecone_index):
        self.state = State.START
        self.grade = None
        self.hebrew_grade = None
        self.topic = None
        self.exercises_data = exercises_data
        self.current_exercise = None
        self.current_hint_index = 0
        self.current_question_index = 0
        self.pinecone_index = pinecone_index
        self.chat_history = []
        self.current_svg_description = None
        self.recently_asked_exercise_ids = []
        self.RECENTLY_ASKED_LIMIT = 5
        self.small_talk_turns = 0
        self.user_language = "en"
        self.topic_exercises_count = 0   # Track number of completed exercises per topic
        self.MAX_EXERCISES = 2  # Strictly 2 exercises before doubt checking
        
        self.small_talk_chain = small_talk_chain
        self.personal_followup_chain = personal_followup_chain
        self.diagnostic_chain = diagnostic_chain  
        self.academic_transition_chain = academic_transition_chain

        self.small_talk_question_index = 0
        self.small_talk_responses = []  # Store user responses for small talk
        self.diagnostic_question_index = 0
        self.diagnostic_responses = []  # Store user responses for test, last class, and focus

        # Enhanced attempt tracking
        self.attempt_tracker = AttemptTracker()
        
        # Doubt clearing functionality
        self.doubt_questions_count = 0
        self.MAX_DOUBT_QUESTIONS = 2
        # Enhanced inactivity timer
        self.inactivity_timer = EnhancedInactivityTimer(self._handle_inactivity)
        self._start_inactivity_timer()
        
        # SVG handling attributes
        self.current_svg_file_path = None  # Track the current SVG file
        self.svg_generated_for_question = False  # Track if SVG was generated for current question

    def _start_inactivity_timer(self):
        """Start or reset the inactivity timer."""
        self.inactivity_timer.reset()

    def _handle_inactivity(self):
        """Handle inactivity timeout - only triggers when truly inactive."""
        lang_dict = I18N[self.user_language]
        
        if self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            self._send_inactivity_message(lang_dict["inactivity_check"])
        else:
            self._send_inactivity_message(lang_dict["session_timeout"])
    
    def _send_inactivity_message(self, message):
        logger.info(f"[INACTIVITY TIMEOUT] {message}")

    @staticmethod
    def _translate_grade_to_hebrew(grade_num: str) -> str:
        grade_map = {"7": "ז", "8": "ח"}
        return grade_map.get(grade_num, grade_num)

    def _get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text based on current user language."""
        lang_dict = I18N[self.user_language]
        text = lang_dict.get(key, I18N["en"][key])
        return text.format(**kwargs) if kwargs else text
    
    def _extract_grade_from_input(self, text: str) -> Optional[str]:
        """Extract grade number (7/8 or Hebrew equivalent) from user input using regex."""
        import re
        
        # English grades: 7 or 8 (case-insensitive, whole word)
        english_match = re.search(r'\b(7|8)\b', text, re.IGNORECASE)
        if english_match:
            return english_match.group(1)
        
        # Hebrew grades: ז (7), ח (8) - using the existing map
        hebrew_map = {"ז": "7", "ח": "8"}  # Matches _translate_grade_to_hebrew
        hebrew_match = re.search(r'(ז|ח)', text)
        if hebrew_match:
            hebrew_char = hebrew_match.group(1)
            return hebrew_map.get(hebrew_char, None)
        
        return None

    def _generate_grade_acknowledgment(self, user_input: str, grade: str) -> str:
        """Generate AI-based acknowledgment for grade selection."""
        try:
            acknowledgment_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly Math AI tutor acknowledging a student's grade selection.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Acknowledge the user's input warmly and briefly (1 sentence, 5-10 words)
                - Mention the grade they selected
                - Be encouraging and conversational
                - Examples: 'Great, grade 8 sounds awesome!', 'Nice choice, grade 7!'
                - Return ONLY the acknowledgment sentence
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "User input: {input}\nSelected grade: {grade}")
            ])
            acknowledgment_chain = acknowledgment_prompt | llm
            response = acknowledgment_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input,
                "grade": grade
            })
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating grade acknowledgment: {e}")
            return f"Got it, grade {grade} is perfect!" if self.user_language == "en" else f"מצוין, כיתה {self._translate_grade_to_hebrew(grade)} נהדרת!"

    def _generate_ai_small_talk(self, user_input: str = "") -> str:
        """Generate AI-based small talk response."""
        try:
            response = self.small_talk_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())  # Clean small talk response
        except Exception as e:
            logger.error(f"Error generating AI small talk: {e}")
            return "Hey! How's it going today?"

    def _generate_ai_personal_followup(self, user_input: str = "") -> str:
        """Generate AI-based personal follow-up response."""
        try:
            response = self.personal_followup_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())  # Clean personal talk response
        except Exception as e:
            logger.error(f"Error generating AI personal follow-up: {e}")
            return "That's interesting! How was your day yesterday?"
        
    def _get_small_talk_question(self) -> str:
        """Get the small talk question (only hobbies)."""
        return self._get_localized_text("small_talk_hobbies")
        
    def _get_diagnostic_question(self) -> str:
        """Get the current diagnostic question based on index."""
        diagnostic_questions = [
            self._get_localized_text("diagnostic_test"),
            self._get_localized_text("diagnostic_last_class"),
            self._get_localized_text("diagnostic_focus")
        ]
        return diagnostic_questions[self.diagnostic_question_index]

    def _generate_academic_transition(self, user_input: str = "") -> str:
        """Generate AI-based academic transition response."""
        try:
            response = self.academic_transition_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "input": user_input or ""
            })
            return clean_math_text(response.content.strip())  # Clean small talk response
        except Exception as e:
            logger.error(f"Error generating academic transition: {e}")
            return "By the way, what have you been learning lately?"

    def _generate_guiding_question(self, user_answer: str, question: str, context: str) -> str:
        """Generate a guiding question to help student think through the problem."""
        try:
            guiding_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a Math AI tutor. Generate a guiding question to help the student think through the problem step by step.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Ask a question that guides them toward the solution
                - Don't give away the answer directly
                - Focus on the mathematical concept or method
                - Be encouraging and supportive
                - Keep it concise (1-2 sentences)
                
                Example guiding questions:
                - "What operation should we use first?"
                - "Can you identify what type of equation this is?"
                - "What do you think the first step should be?"
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Problem: {question}\nStudent's Answer: {answer}\nContext: {context}\n\nGenerate a helpful guiding question:")
            ])
            
            guiding_chain = guiding_prompt | llm
            response = guiding_chain.invoke({
                "chat_history": self.chat_history[-3:],
                "question": question,
                "answer": user_answer,
                "context": context
            })
            
            return clean_math_text(response.content.strip())  # Clean guiding question
        except Exception as e:
            logger.error(f"Error generating guiding question: {e}")
            guiding_text = self._get_localized_text("guiding_question")
            return f"{guiding_text}What do you think the first step should be?"
        
    def _generate_question_and_svg_with_llm(self, topic: str, grade: str) -> Optional[Dict[str, Any]]:
            """Generate a new question and SVG using LLM."""
            try:
                response = question_svg_generation_chain.invoke({
                    "topic": topic,
                    "grade": grade,
                    "language": self.user_language
                })
                result = json.loads(response.content.strip())
                if not all(key in result for key in ["question", "solution", "svg"]):
                    logger.error("Incomplete JSON from LLM: missing question, solution, or svg")
                    return None
                if not result["svg"].startswith('<svg') or not result["svg"].endswith('</svg>'):
                    logger.error("Invalid SVG generated by LLM")
                    return None
                return {
                    "canonical_exercise_id": f"generated_{uuid.uuid4().hex}",
                    "grade": grade,
                    "topic": topic,
                    "text": {
                        "question": [result["question"]],
                        "solution": [result["solution"]],
                        "hint": []  # No hints for generated questions
                    },
                    "svg": [result["svg"]]
                }
            except Exception as e:
                logger.error(f"Error generating question and SVG with LLM: {e}")
                return None

    def _generate_progressive_hint(self, hint_level: int = 0) -> Optional[str]:
        """Generate progressive hints based on level."""
        if not self.current_exercise:
            return None
        
        # Check if it's a local exercise (merged_output.json format)
        if "exercise_content" in self.current_exercise:
            exercise_content = self.current_exercise["exercise_content"]
            sections = exercise_content.get("sections", [])
            
            if not sections or not (0 <= self.current_question_index < len(sections)):
                return None
            
            current_section = sections[self.current_question_index]
            hint_text = current_section.get("hint", {}).get("text", "")
            
            if hint_text:
                hint_text = clean_math_text(hint_text)
                return translate_text_to_conversation_language(hint_text, self.user_language)
            return None
        
        # Fallback for original RAG exercise format
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("hint") and
            isinstance(self.current_exercise["text"]["hint"], list)):
            
            hints = self.current_exercise["text"]["hint"]
            if hint_level < len(hints):
                hint_text = hints[hint_level]  # Remove commas and $ for consistency
                hint_text = clean_math_text(hint_text)  # ← remove $
                return translate_text_to_conversation_language(hint_text, self.user_language)
        return None

    def _evaluate_answer_with_guidance(self, user_input: str, question: str, solution: str, context: str) -> Dict[str, Any]:
        """Enhanced answer evaluation with progressive guidance system."""
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Math AI tutor evaluating a student's answer.
            
            Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
            
            Evaluation Guidelines:
            1. Determine if the answer is CORRECT or INCORRECT
            2. If INCORRECT, identify the specific mistake or misconception
            3. Provide encouragement regardless of correctness
            4. DO NOT reveal the correct answer
            5. Be supportive and educational
            
            Response Format:
            CORRECT: [brief encouraging comment]
            OR
            SORRY: [brief explanation of what went wrong without giving the answer]
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question}\nStudent Answer: {answer}\nContext: {context}\n\nEvaluate the answer:")
        ])
        
        evaluation_chain = evaluation_prompt | llm
        try:
            eval_response = evaluation_chain.invoke({
                "chat_history": self.chat_history[-4:],
                "question": question,
                "answer": user_input,
                "context": context
            })
            
            evaluation_result = clean_math_text(eval_response.content.strip())  # Clean evaluation response
            is_correct = evaluation_result.lower().startswith("correct:")
            
            return {
                "is_correct": is_correct,
                "feedback": evaluation_result,
                "needs_guidance": not is_correct
            }
        except Exception as e:
            logger.error(f"Error in answer evaluation: {e}")
            return {
                "is_correct": False,
                "feedback": "I couldn't evaluate your answer right now.",
                "needs_guidance": True
            }

    def _provide_progressive_guidance(self, user_input: str, question: str, context: str, is_forced: bool = False) -> str:
        """Provide progressive guidance based on attempts and current guidance level."""
        lang_dict = I18N[self.user_language]
        
        if self.attempt_tracker.guidance_level == 0:  # Encouragement
            
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
    
            return f"{guiding_prefix}{guiding_q}"
            
        elif self.attempt_tracker.guidance_level == 1:  # Second Guiding Question
            self.attempt_tracker.guidance_level = 2
            self.state = State.GUIDING_QUESTION
            # Generate a different guiding question, possibly using more context or rephrasing
            guiding_q = self._generate_guiding_question(user_input, question, context)
            guiding_prefix = lang_dict["guiding_question"]
            return f"{guiding_prefix}{guiding_q}"

        elif self.attempt_tracker.guidance_level == 2:  # Hint
            if not is_forced and not self.attempt_tracker.can_provide_hint():
                return f"{lang_dict['encouragement']}{lang_dict['try_again']}"
            self.attempt_tracker.guidance_level = 3
            self.state = State.PROVIDING_HINT
            hint = self._generate_progressive_hint(0)
            if hint:
                hint_prefix = lang_dict["hint_prefix"]
                return f"{hint_prefix}{hint}"
            else:
                self.attempt_tracker.guidance_level = 4
                return self._get_current_solution()
                
        else:  # guidance_level >= 3, provide solution
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            return f"{solution_prefix}{solution}\n\n{self._move_to_next_exercise_or_question()}"

    def _handle_hint_request(self, user_input: str) -> str:
        """Handle explicit hint requests by providing guiding questions."""
        lang_dict = I18N[self.user_language]
        
        # Get current question context
        current_question = self._get_current_question()
        retrieved_context = retrieve_relevant_chunks(
            f"Question: {current_question} User's Answer: {user_input}",
            self.pinecone_index,
            grade=self.hebrew_grade,
            topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
        )
        context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
        context_str = clean_math_text(context_str)  # Clean context
        
        # Instead of checking attempts, provide progressive guidance
        return self._provide_progressive_guidance(user_input, current_question, context_str, is_forced=True)

    def _handle_solution_request(self, user_input: str) -> str:
        """Handle explicit solution requests by providing guiding questions first."""
        lang_dict = I18N[self.user_language]

        # Ensure we are working with the CURRENT question only
        current_question = self._get_current_question()
        current_solution = self._get_current_solution()
        
        if self.attempt_tracker.guidance_level < 3:  # Haven't completed guiding questions + hint
            current_question = self._get_current_question()
            retrieved_context = retrieve_relevant_chunks(
                f"Question: {current_question} User's Answer: {user_input}",
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            context_str = clean_math_text(context_str)  # Clean context
            
            # Provide guiding question instead of direct solution
            encouragement = lang_dict["encouragement"]
            guiding_q = self._generate_guiding_question(user_input, current_question, context_str)
            guiding_prefix = lang_dict["guiding_question"]
            self.attempt_tracker.guidance_level = 1
            self.state = State.GUIDING_QUESTION
            return f"{encouragement}{guiding_prefix}{guiding_q}"
        
        # If they've been through guidance, provide solution
        else:
            self.attempt_tracker.has_requested_solution = True
            solution_prefix = lang_dict["solution_prefix"]
            solution = self._get_current_solution()
            
            # Generate detailed explanation using AI
            try:
                explanation_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are a Math AI tutor providing a detailed solution explanation.
                    
                    Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                    
                    Guidelines:
                    - Always explain in steps: Step 1, Step 2, Step 3...
                    - First: state the key formula or rule used.
                    - Second: substitute values from the problem.
                    - Third: simplify step by step.
                    - Fourth: conclude with the final answer.
                    - End with a short "check your answer" verification if possible.
                    - Be clear, concise, and educational.
                    - Never show raw $ signs or LaTeX markup.
                    """),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "Question: {question}\nSolution: {solution}\n\nProvide a step-by-step explanation:")
                ])
                
                explanation_chain = explanation_prompt | llm
                current_question = self._get_current_question()
                
                response = explanation_chain.invoke({
                    "chat_history": self.chat_history[-3:],
                    "question": current_question,
                    "solution": solution
                })
                
                explanation = clean_math_text(response.content.strip())  # Clean explanation
                
                # Generate NEW SVG for solution explanation
                svg_reference = ""
                if self.current_exercise and self.current_exercise.get("svg"):
                    svg_reference = self._generate_and_save_svg(for_solution_explanation=True)
                
                result = f"{solution_prefix}{solution}\n\n{explanation}"
                if svg_reference:
                    result += f"\n{svg_reference}"
                result += self._move_to_next_exercise_or_question()
                return result
                
            except Exception as e:
                logger.error(f"Error generating solution explanation: {e}")
                result = f"{solution_prefix}{solution}"
                result += self._move_to_next_exercise_or_question()
                return result

    def _reset_attempt_tracking(self):
        """Reset attempt tracking for new exercise."""
        self.attempt_tracker.reset()
        # Reset SVG tracking for new question/exercise
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

    def _get_exercise_by_id(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        # Try to find by canonical_exercise_id first (for generated exercises)
        for ex in self.exercises_data:
            if isinstance(ex, dict) and ex.get("canonical_exercise_id") == exercise_id:
                return ex
        
        # For loaded exercises, create an ID from metadata
        for ex in self.exercises_data:
            if isinstance(ex, dict) and "exercise_metadata" in ex:
                metadata = ex["exercise_metadata"]
                generated_id = f"{metadata.get('class', '')}_{metadata.get('lesson_number', '')}_{metadata.get('exercise_number', '')}"
                if generated_id == exercise_id:
                    return ex
        
        return None

    def _find_matching_topic(self, user_topic: str, grade: str) -> Optional[str]:
        """Find a matching Hebrew topic for user input."""
        # Get available topics for the grade
        available_topics = []
        try:
            for ex in self.exercises_data:
                if isinstance(ex, dict) and ex.get("exercise_metadata", {}).get("class") == grade:
                    topic = ex.get("exercise_metadata", {}).get("topic", "")
                    if topic and topic not in available_topics:
                        available_topics.append(topic)
        except Exception as e:
            logger.error(f"Error getting available topics: {e}")
            return None
        
        if not available_topics:
            return None
        
        # Create English-to-Hebrew mappings for user input matching
        topic_mappings = {
            # English terms that users might type -> Hebrew topics
            "coordinate system": ["מערכת צירים", "מערכת צירים ומשוואות (חזרה)"],
            "coordinate": ["מערכת צירים", "מערכת צירים ומשוואות (חזרה)"],
            "equation of a line": ["משוואת ישר", "משוואת ישר ושיפוע"],
            "equation": ["משוואת ישר", "משוואת ישר ושיפוע"],
            "line": ["משוואת ישר", "משוואת ישר ושיפוע", "הצגה של ישר "],
            "slope": ["משוואת ישר ושיפוע", "מציאת שיפוע ונקודת חיתוך עם ציר Y", "פונקציות - שיפוע ומשוואת ישר"],
            "linear functions": ["פונקציות לינאריות"],
            "functions": ["פונקציות לינאריות", "פונקציות - משוואת ישר מנקודות", "פונקציות - שיפוע ומשוואת ישר"],
            "line representation": ["הצגה של ישר "],
            "representation": ["הצגה של ישר "],
            "parallelogram": ["מקבילית"],
            "y-intercept": ["מציאת שיפוע ונקודת חיתוך עם ציר Y"],
            "intercept": ["מציאת שיפוע ונקודת חיתוך עם ציר Y"]
        }
        
        user_lower = user_topic.lower().strip()
        
        # Direct mapping check
        for english_key, hebrew_topics in topic_mappings.items():
            if english_key in user_lower:
                for hebrew_topic in hebrew_topics:
                    if hebrew_topic in available_topics:
                        return hebrew_topic
        
        # Fuzzy matching for Hebrew input
        if any(ord(char) > 127 for char in user_topic):  # Contains non-ASCII (Hebrew)
            for topic in available_topics:
                if user_topic.strip() in topic or topic in user_topic.strip():
                    return topic
        
        # Return first available topic if no match found
        return available_topics[0] if available_topics else None

    def _get_english_topic_mapping(self, hebrew_topic: str) -> str:
        """Map Hebrew topics to their English equivalents using predefined mappings."""
        topic_mappings = {
            # Grade 7 topics
            "מערכת צירים": "Coordinate system",
            "מקבילית": "Parallelogram",
            
            # Grade 8 topics (in order as they appear in JSON)
            "מערכת צירים ומשוואות (חזרה)": "Coordinate system and equations (review)",
            "הצגה של ישר ": "Line representation",  # Note: includes trailing space
            "משוואת ישר ושיפוע": "Equation of a line and slope", 
            "פונקציות - משוואת ישר מנקודות": "Functions - Equation of a line from points",
            "פונקציות - שיפוע ומשוואת ישר": "Functions - Slope and equation of a line",
            "משוואת ישר": "Equation of a line",
            "פונקציות לינאריות": "Linear functions",
            "מציאת שיפוע ונקודת חיתוך עם ציר Y": "Finding slope and Y-intercept"
        }
        
        return topic_mappings.get(hebrew_topic, hebrew_topic)

    def _pick_new_exercise_rag(self, query: str, grade: str = None, topic: str = None):
        """Retrieve relevant exercises using RAG, with local fallback."""
        
        # Try RAG first only if Pinecone is available
        if self.pinecone_index:
            try:
                relevant_chunks = retrieve_relevant_chunks(query, self.pinecone_index, grade=grade, topic=topic)
                
                if relevant_chunks:
                    all_exercise_ids = list(set(chunk["exercise_id"] for chunk in relevant_chunks))
                    if all_exercise_ids:
                        # Filter out recently asked exercises
                        available_exercise_ids = [ex_id for ex_id in all_exercise_ids if ex_id not in self.recently_asked_exercise_ids]

                        if not available_exercise_ids:
                            logger.info("All retrieved exercises were recently asked. Clearing history.")
                            self.recently_asked_exercise_ids.clear()
                            available_exercise_ids = all_exercise_ids

                        if available_exercise_ids:
                            chosen_exercise_id = random.choice(available_exercise_ids)
                            self.current_exercise = self._get_exercise_by_id(chosen_exercise_id)
                            
                            if self.current_exercise:
                                logger.info(f"Selected exercise via RAG: {chosen_exercise_id}")
                                self._setup_selected_exercise(chosen_exercise_id)
                                return
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}, falling back to local exercises")
        
        # Fallback to local exercises from merged_output.json
        logger.info("Using local exercises as primary source")
        self._pick_local_exercise(grade=grade, topic=topic)

    def _pick_local_exercise(self, grade: str = None, topic: str = None):
        """Pick exercise from local JSON files (merged_output.json)."""
        if not self.exercises_data:
            logger.error("No local exercises data available")
            self.current_exercise = None
            return
        
        # Filter exercises by grade and topic
        filtered_exercises = []
        for ex in self.exercises_data:
            if not isinstance(ex, dict) or "exercise_metadata" not in ex:
                continue
                
            metadata = ex["exercise_metadata"]
            
            # Check grade match
            if grade and metadata.get("class") != grade:
                continue
                
            # Check topic match (if topic is specified)
            if topic:
                exercise_topic = metadata.get("topic", "")
                # Use fuzzy matching for topic
                if not (topic.lower() in exercise_topic.lower() or exercise_topic.lower() in topic.lower()):
                    continue
            
            filtered_exercises.append(ex)
        
        if not filtered_exercises:
            logger.warning(f"No local exercises found for grade={grade}, topic={topic}")
            self.current_exercise = None
            return
        
        # Select random exercise from filtered ones
        self.current_exercise = random.choice(filtered_exercises)
        logger.info(f"Selected local exercise: grade {grade}, topic {topic}, total options: {len(filtered_exercises)}")
        
        # Setup the exercise (reset counters, handle SVG, etc.)
        self._setup_selected_exercise()

    def _setup_selected_exercise(self, exercise_id: str = None):
        """Setup selected exercise with counters and SVG handling."""
        if not self.current_exercise:
            return
            
        # Reset exercise-specific counters
        self.current_hint_index = 0
        self.current_question_index = 0
        self.current_svg_description = None
        self._reset_attempt_tracking()
        
        # Reset SVG tracking for new exercise
        self.svg_generated_for_question = False
        self.current_svg_file_path = None

        # Add to recently asked list (for RAG exercises with IDs)
        if exercise_id:
            self.recently_asked_exercise_ids.append(exercise_id)
            if len(self.recently_asked_exercise_ids) > self.RECENTLY_ASKED_LIMIT:
                self.recently_asked_exercise_ids.pop(0)

    def _generate_and_save_svg(self, for_solution_explanation: bool = False) -> str:
        """Generate SVG dynamically using AI based on the current question. Returns the file reference text."""
        if not self.current_exercise:
            return ""
            
        try:
            # Get the current question text
            current_question = self._get_current_question()
            if not current_question:
                return ""
            
            # Generate SVG using AI
            svg_content = self._generate_svg_with_ai(current_question, for_solution_explanation)
            if not svg_content:
                return ""
                
            # Generate unique filename
            suffix = "_solution" if for_solution_explanation else ""
            svg_filename = f"dynamic_exercise_q{self.current_question_index}{suffix}_{uuid.uuid4().hex[:8]}.svg"
            svg_filepath = SVG_OUTPUT_DIR / svg_filename
            
            # Save SVG file
            with open(svg_filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)
            
            # Store file path for future reference
            if not for_solution_explanation:
                self.current_svg_file_path = str(svg_filepath)
            
            # Return file reference
            if self.user_language == "he":
                return f"📊 תרשים זמין: {svg_filepath.name}"
            else:
                return f"📊 Diagram available: {svg_filepath.name}"
                
        except Exception as e:
            logger.error(f"Error generating and saving dynamic SVG: {e}")
            return ""

    def _generate_svg_with_ai(self, question_text: str, for_solution: bool = False) -> str:
        """Generate SVG content using AI based on the question text."""
        try:
            context = "solution explanation" if for_solution else "problem visualization"
            
            svg_generation_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are an expert math visualization generator. Create a clean, educational SVG for the given math question.

                Context: Creating {context}

                SVG Requirements:
                - Use viewBox="0 0 400 300" for consistent sizing
                - Include coordinate axes if the problem involves coordinates, graphs, or geometry
                - Add a subtle grid if appropriate for the math concept
                - Use clear colors: black for axes, blue for main elements, red for points
                - Include proper labels and numbers
                - Make it educational and clear
                - Ensure valid SVG XML structure
                - NO interactive elements - static visualization only

                Question Types & Visualizations:
                - Coordinate geometry: Show coordinate plane with labeled points
                - Line equations: Show the line with equation label
                - Slope: Show two points with rise/run visualization
                - Triangles: Show triangle with labeled vertices
                - Functions: Show function graph with key points

                Return ONLY the SVG code, no explanations."""),
                ("user", f"Generate an SVG visualization for this math question: {question_text}")
            ])
            
            svg_chain = svg_generation_prompt | llm
            response = svg_chain.invoke({"question_text": question_text})
            
            svg_content = response.content.strip()
            
            # Clean up the response - remove any markdown formatting
            svg_content = re.sub(r'```svg\s*', '', svg_content)
            svg_content = re.sub(r'```\s*$', '', svg_content)
            svg_content = svg_content.strip()
            
            # Validate SVG structure
            if svg_content.startswith('<svg') and svg_content.endswith('</svg>'):
                return svg_content
            else:
                logger.warning(f"Invalid SVG structure generated: {svg_content[:100]}...")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating SVG with AI: {e}")
            return ""

    def _should_generate_svg(self, question_text: str) -> bool:
        """Determine if the question would benefit from an SVG visualization."""
        # Convert to lowercase for keyword matching
        text_lower = question_text.lower()
        
        # Keywords that suggest visual content would be helpful
        visual_keywords = [
            # Coordinate geometry
            "coordinate", "נקודה", "נקודות", "point", "points", "axes", "axis", "צירים",
            # Geometry
            "triangle", "משולש", "parallelogram", "מקבילית", "line", "ישר", "קו",
            # Graphing and functions
            "graph", "גרף", "plot", "function", "פונקציה", "slope", "שיפוע",
            # Visual math terms
            "draw", "תרשים", "diagram", "מערכת", "system", "equation", "משוואה"
        ]
        
        # Check if any visual keywords are present
        for keyword in visual_keywords:
            if keyword in text_lower:
                return True
        
        # Check for coordinate patterns like (x,y) or specific numbers
        coordinate_patterns = [
            r'\([+-]?\d+\s*,\s*[+-]?\d+\)',  # (x,y) coordinates
            r'\([+-]?\d+\s*,\s*__\)',        # (x,__) incomplete coordinates
            r'\(__\s*,\s*[+-]?\d+\)',        # (__,y) incomplete coordinates
        ]
        
        for pattern in coordinate_patterns:
            if re.search(pattern, question_text):
                return True
        
        return False

    def _get_current_question(self) -> str:
        """Retrieves the current question text for the exercise."""
        if not self.current_exercise:
            return "No question available."
        
        # Check if it's a local exercise (merged_output.json format)
        if "exercise_content" in self.current_exercise:
            exercise_content = self.current_exercise["exercise_content"]
            main_instruction = exercise_content.get("main_data", {}).get("text", "")
            sections = exercise_content.get("sections", [])
            
            if not sections:
                return main_instruction if main_instruction else "No question available."
            
            # Get current section
            if not (0 <= self.current_question_index < len(sections)):
                logger.warning(f"Invalid question index: {self.current_question_index}")
                return main_instruction if main_instruction else "No question available."
            
            current_section = sections[self.current_question_index]
            question_text = current_section.get("question", {}).get("text", "")
            section_number = current_section.get("section_number", "")
            
            # Combine main instruction with specific question
            if main_instruction and question_text:
                full_question = f"{main_instruction}\n\n{section_number}) {question_text}"
            elif question_text:
                full_question = f"{section_number}) {question_text}" if section_number else question_text
            else:
                full_question = main_instruction if main_instruction else "No question available."
            
            # Translate to match conversation language and clean
            clean_question = clean_math_text(full_question)
            return translate_text_to_conversation_language(clean_question, self.user_language)
        
        # Fallback for original RAG exercise format
        if not (self.current_exercise and 
                self.current_exercise.get("text", {}).get("question") and
                isinstance(self.current_exercise["text"]["question"], list)):
            logger.warning("Invalid exercise data structure for question retrieval.")
            return "No question available."

        questions = self.current_exercise["text"]["question"]
        if not (0 <= self.current_question_index < len(questions)):
            logger.warning(f"Invalid question index: {self.current_question_index}")
            return "No question available."

        q_text = questions[self.current_question_index].replace(',', '')
        q_text = clean_math_text(q_text)   # Remove $

        # Generate SVG dynamically for math questions that could benefit from visualization
        if not self.svg_generated_for_question and self._should_generate_svg(q_text):
            svg_reference = self._generate_and_save_svg(for_solution_explanation=False)
            if svg_reference and isinstance(svg_reference, str):
                q_text += f"\n{svg_reference}"
            self.svg_generated_for_question = True
        elif self.current_svg_file_path and self.svg_generated_for_question:
            # Reuse existing SVG file reference for hints
            q_text += f"\n\n[Image File: {self.current_svg_file_path.as_posix()}]"

        # Translate to match conversation language
        return translate_text_to_conversation_language(q_text, self.user_language)

    def _get_current_solution(self) -> str:
        if not self.current_exercise:
            return "No solution available."
        
        # Check if it's a local exercise (merged_output.json format)
        if "exercise_content" in self.current_exercise:
            exercise_content = self.current_exercise["exercise_content"]
            sections = exercise_content.get("sections", [])
            
            if not sections or not (0 <= self.current_question_index < len(sections)):
                return "No solution available."
            
            current_section = sections[self.current_question_index]
            solution_text = current_section.get("solution", {}).get("text", "")
            
            if solution_text:
                sol_text = clean_math_text(solution_text)
                # Translate to match conversation language
                return translate_text_to_conversation_language(sol_text, self.user_language)
            return "No solution available."
        
        # Fallback for original RAG exercise format
        if (self.current_exercise and
            self.current_exercise.get("text", {}).get("solution") and
            isinstance(self.current_exercise["text"]["solution"], list) and
            self.current_question_index < len(self.current_exercise["text"]["solution"])):
            sol_text = self.current_exercise["text"]["solution"][self.current_question_index]
            sol_text = clean_math_text(sol_text)   # ← remove $
            
            # Translate to match conversation language
            return translate_text_to_conversation_language(sol_text, self.user_language)
        return "No solution available."
    
    def _generate_lesson_summary(self) -> str:
        closing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly math tutor closing a short session.
            Write a 2–3 sentence positive summary of the lesson.
            End with a light humorous encouragement."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Summarize the lesson we just completed.")
        ])
        summary_chain = closing_prompt | llm
        try:
            response = summary_chain.invoke({"chat_history": self.chat_history[-10:]})
            return clean_math_text(response.content.strip())
        except Exception as e:
            logger.error(f"Error generating lesson summary: {e}")
            return "Great job today! You tackled some tough problems with confidence."

    def _has_more_questions(self) -> bool:
        """Check if the current exercise has more questions."""
        if not self.current_exercise:
            return False
        
        # Check for local exercise format
        if "exercise_content" in self.current_exercise:
            sections = self.current_exercise["exercise_content"].get("sections", [])
            return self.current_question_index < len(sections) - 1
        
        # Check for RAG exercise format
        if ("text" in self.current_exercise and
            "question" in self.current_exercise["text"] and
            isinstance(self.current_exercise["text"]["question"], list)):
            return self.current_question_index < len(self.current_exercise["text"]["question"]) - 1
        
        return False

    def _advance_to_next_question(self) -> str:
        """Advance to next question and return the formatted question text."""
        self.current_question_index += 1
        self.current_hint_index = 0
        self._reset_attempt_tracking()  # This will reset SVG tracking too
        return f"\n\nNext question:\n{self._get_current_question()}"

    def _move_to_next_exercise_or_question(self) -> str:
        """Enhanced version that strictly provides 2 exercises before doubt checking."""
        
        # Check if there are more questions in the current exercise
        if self._has_more_questions():
            # Move to next question within the same exercise
            return self._advance_to_next_question()
        else:
            # Current exercise is finished - increment counter
            self.topic_exercises_count += 1

            # Strictly check for exactly 2 exercises
            if self.topic_exercises_count >= self.MAX_EXERCISES:
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"

            # Construct query for a new exercise
            query = f"Next exercise for grade {self.hebrew_grade}"
            topic_for_query = None
            if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"]:
                query += f" on topic {self.topic}"
                topic_for_query = self.topic

            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_query)

            if not self.current_exercise:
                self.state = State.ASK_FOR_DOUBTS
                topic_name = self.topic or "this topic"
                return f"\n\n{self._get_localized_text('ask_for_doubts', topic=topic_name)}"

            return f"\n\nNext exercise:\n{self._get_current_question()}"

    def _generate_doubt_clearing_response(self, user_question: str) -> str:
        """Generate response to clear student's doubts using RAG."""
        try:
            # Only translate to English for retrieval if the question is in Hebrew
            # This helps with Pinecone search but doesn't affect the response language
            search_question = translate_text_to_english(user_question) if is_likely_hebrew(user_question) else user_question
            
            # Retrieve relevant context for the doubt
            retrieved_context = retrieve_relevant_chunks(
                search_question,
                self.pinecone_index,
                grade=self.hebrew_grade,
                topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
            )
            
            context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
            context_str = clean_math_text(context_str)  # Clean context
            
            # Create a doubt-clearing prompt
            doubt_clearing_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a helpful Math AI tutor addressing a student's doubt or question.
                
                Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                
                Guidelines:
                - Provide clear, detailed explanations
                - Use the context to give accurate information
                - Be patient and encouraging
                - Break down complex concepts into simple steps
                - If context doesn't contain relevant information, acknowledge this
                - Start with encouragement
                - Give step-by-step explanation
                - Use examples from context when available
                - End with confirmation question
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "Student's Question: {question}\n\nRelevant Context: {context}")
            ])
            
            doubt_clearing_chain = doubt_clearing_prompt | llm
            response = doubt_clearing_chain.invoke({
                "chat_history": self.chat_history[-5:],
                "question": user_question,  # Use original question in user's language
                "context": context_str
            })
            
            topic_name = self.topic or "this topic"
            intro = self._get_localized_text("doubt_clearing_intro", topic=topic_name)
            return f"{intro}\n\n{clean_math_text(response.content.strip())}"
            
        except Exception as e:
            logger.error(f"Error generating doubt clearing response: {e}")
            return "I'd be happy to help with your question, but I'm having trouble processing it right now. Could you try asking it in a different way?"

    def transition(self, user_input: str) -> str:
        """Enhanced FSM transition with progressive guidance and improved attempt tracking."""
        
        # Mark user activity (prevents premature inactivity timeout)
        if user_input.strip():
            self.inactivity_timer.reset()
            
        text_lower = (user_input or "").strip().lower()

        # Detect user language from input
        if user_input:
            detected_lang = detect_language(user_input)
            if detected_lang != self.user_language and detected_lang in ["he", "en"]:
                self.user_language = detected_lang
            
        # Add user input to chat history
        if user_input:
            self.chat_history.append(HumanMessage(content=clean_math_text(user_input)))

        # --- State Transitions ---
        if self.state == State.START:
            self.state = State.SMALL_TALK
            self.small_talk_question_index = 0
            self.small_talk_responses = []
            self.small_talk_turns = 1
            simple_greetings = ["Hey! How are you?", "Hi there!", "What's up?", "How's it going?"]
            ai_response = random.choice(simple_greetings)
            self.chat_history.append(AIMessage(content=ai_response))
            return ai_response

        elif self.state == State.SMALL_TALK:
            if self.small_talk_question_index == 0:
                # Generate contextual response to greeting + hobbies question
                try:
                    response = self.small_talk_chain.invoke({
                        "chat_history": self.chat_history[-3:],
                        "input": user_input or ""
                    })
                    response_text = clean_math_text(response.content.strip())
                    # Ensure hobbies question is included
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_text = f"{response_text} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                except Exception as e:
                    logger.error(f"Error generating contextual small talk: {e}")
                    # Fallback: static response + hobbies
                    fallback_response = "I'm doing great, thanks for asking!" if self.user_language == "en" else "אני בסדר, תודה ששאלת!"
                    hobbies_q = self._get_localized_text("small_talk_hobbies")
                    response_text = f"{fallback_response} {hobbies_q}"
                    self.small_talk_question_index += 1
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
            else:
                # User responded to hobbies, move to personal followup
                self.small_talk_responses.append(user_input)
                self.state = State.PERSONAL_FOLLOWUP
                ai_response = self._generate_ai_personal_followup(user_input)
                self.chat_history.append(AIMessage(content=ai_response))
                return ai_response

        elif self.state == State.PERSONAL_FOLLOWUP:
            self.state = State.DIAGNOSTIC
            self.diagnostic_question_index = 0
            self.diagnostic_responses = []  # Reset responses
            response_text = self._get_diagnostic_question()
            self.chat_history.append(AIMessage(content=response_text))
            return response_text
        
        elif self.state == State.DIAGNOSTIC:
            # Store the user's response
            self.diagnostic_responses.append(user_input)
            self.diagnostic_question_index += 1

            # Check if there are more diagnostic questions
            if self.diagnostic_question_index < 3:  # 3 questions total
                # Get next diagnostic question
                next_question = self._get_diagnostic_question()
                # Generate short contextual acknowledgment
                try:
                    contextual_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""You are a friendly math tutor.
                        Language: Respond in {self.user_language} ({'Hebrew' if self.user_language == 'he' else 'English'})
                        Guidelines:
                        - Acknowledge the user's previous response in ONE short sentence (5-10 words).
                        - Be warm, conversational, and encouraging.
                        - Examples: 'Wow, great to hear!', 'That's awesome!', 'Cool, love that!'
                        - Return ONLY the acknowledgment sentence."""),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("user", "{input}")
                    ])
                    contextual_chain = contextual_prompt | llm
                    ack_response = contextual_chain.invoke({
                        "chat_history": self.chat_history[-3:],
                        "input": user_input
                    })
                    acknowledgment = clean_math_text(ack_response.content.strip())
                    if not acknowledgment:  # Ensure non-empty
                        acknowledgment = "That's awesome!"
                except Exception as e:
                    logger.error(f"Error generating contextual acknowledgment: {e}")
                    acknowledgment = "That's awesome!"  # Fallback

                # Combine acknowledgment with diagnostic question
                response_text = f"{acknowledgment} So, {next_question}"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            else:
                # All questions done - move to academic transition
                self.state = State.ACADEMIC_TRANSITION
                ai_response = self._generate_academic_transition(user_input)
                self.chat_history.append(AIMessage(content=ai_response))
                return ai_response
            
        elif self.state == State.ACADEMIC_TRANSITION:
            # Use existing academic transition
            ai_response = self._generate_academic_transition(user_input)
            self.state = State.ASK_GRADE
            grade_ask = self._get_localized_text("ask_grade")
            response_text = f"{ai_response}\n\n{grade_ask}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.ASK_GRADE:
            # Extract grade number from user input using regex and AI
            extracted_grade = self._extract_grade_from_input(user_input.strip())
            
            if not extracted_grade:
                # Ask again if no valid grade found - localized with examples
                examples = "7 or 8" if self.user_language == "en" else "ז or ח"
                retry_message = f"{self._get_localized_text('ask_grade')} ({self._get_localized_text('examples', examples=examples)})"
                self.chat_history.append(AIMessage(content=retry_message))
                return retry_message
            
            # Success: Set grades
            self.grade = extracted_grade
            self.hebrew_grade = self._translate_grade_to_hebrew(self.grade)
            self.state = State.EXERCISE_SELECTION
            
            # Generate contextual acknowledgment using AI
            grade_acknowledgment = self._generate_grade_acknowledgment(user_input, self.grade)
            
            # Get available topics with error handling - preserve order from JSON
            available_topics_hebrew = []
            try:
                # Get topics in the order they appear in the JSON, without duplicates
                seen_topics = set()
                for ex in self.exercises_data:
                    if isinstance(ex, dict) and ex.get("exercise_metadata", {}).get("class") == self.hebrew_grade:
                        topic = ex.get("exercise_metadata", {}).get("topic", "")
                        if topic and topic != "Unknown" and topic not in seen_topics:
                            available_topics_hebrew.append(topic)
                            seen_topics.add(topic)
            except Exception as e:
                logger.error(f"Error extracting topics: {e}")
                available_topics_hebrew = []
            
            if available_topics_hebrew:
                if self.user_language == "en":
                    # Use predefined English mappings instead of AI translation
                    english_topics = []
                    for topic in available_topics_hebrew:  # Show ALL topics, not just [:3]
                        english_topic = self._get_english_topic_mapping(topic)
                        english_topics.append(english_topic)
                    topics_str = ", ".join(english_topics)
                else:
                    topics_str = ", ".join(available_topics_hebrew)  # Show ALL topics
            else:
                topics_str = "Any topic"
                
            topic_question = self._get_localized_text("ask_topic", grade=self.grade, topics=topics_str)
            response_text = f"{grade_acknowledgment} {topic_question}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state == State.EXERCISE_SELECTION:
            self.topic = clean_math_text(user_input.strip()) # Clean topic input
            # Reset counters when new topic is selected
            self.topic_exercises_count = 0  # Reset for new topic
            self.doubt_questions_count = 0
            
            # Try to find a matching Hebrew topic
            matched_topic = self._find_matching_topic(self.topic, self.hebrew_grade)
            
            # Use the improved exercise picking method (tries RAG first, then local)
            query = f"Find an exercise for grade {self.hebrew_grade} on topic {self.topic}"
            topic_for_picking = matched_topic if matched_topic else (self.topic if self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None)
            self._pick_new_exercise_rag(query=query, grade=self.hebrew_grade, topic=topic_for_picking)

            # Final fallback: try any exercise for the grade if still no match
            if not self.current_exercise:
                logger.info(f"No exercises found for grade {self.hebrew_grade} and topic {self.topic}. Trying any exercise for the grade.")
                self._pick_new_exercise_rag(query=f"Find an exercise for grade {self.hebrew_grade}", grade=self.hebrew_grade, topic=None)

            if not self.current_exercise:
                self.state = State.EXERCISE_SELECTION
                no_exercises = self._get_localized_text("no_exercises", grade=self.grade, topic=self.topic)
                no_relevant = self._get_localized_text("no_relevant_exercises")
                response_text = f"{no_exercises}\n{no_relevant}\n\nPlease choose another topic:"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            self.state = State.QUESTION_ANSWER
            ready_text = self._get_localized_text("ready_for_question")
            response_text = f"{ready_text}\n{self._get_current_question()}"
            self.chat_history.append(AIMessage(content=response_text))
            return response_text

        elif self.state in [State.QUESTION_ANSWER, State.GUIDING_QUESTION, State.PROVIDING_HINT]:
            # Handle irrelevant questions
            irrelevant_keywords = [
                "recipe", "cake", "story", "joke", "weather", "song", "news", "football",
                "music", "movie", "politics", "food", "travel", "holiday"
            ]
            if any(word in text_lower for word in irrelevant_keywords):
                irrelevant_msg = self._get_localized_text("irrelevant_msg")
                response_text = irrelevant_msg + "\n\nLet's focus on the current exercise."
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle explicit hint requests with attempt checking
            hint_keywords = ["hint", "help", "clue", "tip", "stuck", "don't know", "not sure", "confused", "רמז", "עזרה"]
            if (text_lower == "hint" or 
                any(keyword in text_lower for keyword in hint_keywords) or
                ("give" in text_lower and any(keyword in text_lower for keyword in ["hint", "help", "clue"])) or
                ("can you" in text_lower and any(keyword in text_lower for keyword in ["hint", "help"]))):
                
                response_text = self._handle_hint_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle solution requests with attempt checking
            solution_keywords = ["solution", "answer", "pass", "skip", "give up", "show me the solution", "פתרון", "תשובה"]
            if (text_lower in {"solution", "pass"} or
                any(keyword in text_lower for keyword in solution_keywords) or
                ("give me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"])) or
                ("show me" in text_lower and any(keyword in text_lower for keyword in ["solution", "answer"]))):
                
                response_text = self._handle_solution_request(user_input)
                self.chat_history.append(AIMessage(content=response_text))
                return response_text

            # Handle answer evaluation with enhanced attempt tracking
            else:
                current_question = self._get_current_question()
                current_solution = self._get_current_solution()

                # Get context for evaluation
                retrieved_context = retrieve_relevant_chunks(
                    f"Question: {current_question} User's Answer: {user_input}",
                    self.pinecone_index,
                    grade=self.hebrew_grade,
                    topic=self.topic if self.topic and self.topic.lower() not in ["anyone", "any", "anything", "random", "whatever", "any topic"] else None
                )
                context_str = "\n".join([c.get("text", "") for c in retrieved_context if c.get("text")])
                if self.current_svg_description:
                    context_str += f"\n\nImage Description: {self.current_svg_description}"

                # Evaluate answer
                evaluation_result = self._evaluate_answer_with_guidance(user_input, current_question, current_solution, context_str)
                
                # Record the attempt
                should_offer_guidance = self.attempt_tracker.record_attempt(evaluation_result["is_correct"])
                
                if evaluation_result["is_correct"]:
                    response_text = "✅ Correct!"
                    response_text += self._move_to_next_exercise_or_question()
                    self.state = State.QUESTION_ANSWER
                    self._reset_attempt_tracking()
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text
                
                else:
                    # Incorrect answer - handle based on attempt count
                    feedback_lines = evaluation_result.get("feedback", self._get_localized_text("wrong_answer")).split('\n')
                    main_feedback = feedback_lines[0] if feedback_lines else self._get_localized_text("wrong_answer")
                    
                    if should_offer_guidance:
                        # Provide progressive guidance
                        guidance = self._provide_progressive_guidance(user_input, current_question, context_str)
                        response_text = f"{main_feedback}\n\n{guidance}"
                    else:
                        # Encourage another attempt
                        encouragement = self._get_localized_text("encouragement")
                        try_again = self._get_localized_text("try_again")
                        response_text = f"{main_feedback}\n\n{encouragement}{try_again}"
                    
                    self.chat_history.append(AIMessage(content=response_text))
                    return response_text

        elif self.state == State.ASK_FOR_DOUBTS:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no doubts", "no questions", "לא", "אין", "בסדר"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators):
                # Generate lesson summary and reset for new topic
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.EXERCISE_SELECTION
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.current_exercise = None
                response_text = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
            
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                
                if "?" in user_input:
                    doubt_response = self._generate_doubt_clearing_response(user_input)
                else:
                    doubt_response = f"I'm ready to help! What would you like me to explain or clarify about {topic_name}?"
                
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            
            else:
                # Treat as a doubt/question
                self.state = State.DOUBT_CLEARING
                self.doubt_questions_count = 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        elif self.state == State.DOUBT_CLEARING:
            no_doubt_indicators = ["no", "nope", "nothing", "i'm good", "all clear", "no more", "that's all", "thanks", "לא", "אין", "תודה", "זה הכל"]
            doubt_indicators = ["yes", "yeah", "yep", "i have", "question", "doubt", "confused", "don't understand", "כן", "יש לי", "שאלה"]
            topic_name = self.topic or "this topic"
            
            if any(indicator in text_lower for indicator in no_doubt_indicators) or self.doubt_questions_count >= self.MAX_DOUBT_QUESTIONS:
                # Generate lesson summary and reset for new topic
                summary = self._generate_lesson_summary()
                closing_message = self._get_localized_text("lesson_closing")
                self.state = State.EXERCISE_SELECTION
                self.topic_exercises_count = 0
                self.doubt_questions_count = 0
                self.current_exercise = None
                response_text = f"{summary}\n\n{closing_message}\n\nWould you like to continue with more exercises on this topic or choose a new topic?"
                self.chat_history.append(AIMessage(content=response_text))
                return response_text
                
            elif any(indicator in text_lower for indicator in doubt_indicators) or "?" in user_input:
                self.doubt_questions_count += 1
                doubt_response = self._generate_doubt_clearing_response(user_input)
                if self.doubt_questions_count < self.MAX_DOUBT_QUESTIONS:
                    doubt_response += f"\n\n{self._get_localized_text('ask_more_doubts', topic=topic_name)}"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response
            else:
                # Unclear response
                doubt_response = f"Could you clarify your question about {topic_name} or say 'no' if you're ready to move on?"
                self.chat_history.append(AIMessage(content=doubt_response))
                return doubt_response

        # Default fallback
        response_text = "I'm not sure how to proceed. Type 'exit' to quit."
        self.chat_history.append(AIMessage(content=response_text))
        return response_text

# -----------------------------
# BOT.PY INTEGRATION - GUIDED EXERCISES AND RETRIEVAL SYSTEM
# -----------------------------

# Additional configuration for bot.py functionality
BOT_INPUT_FILE = Path("parsed_outputs/merged_output.json")  # Fixed path
BOT_MODEL_NAME = "intfloat/multilingual-e5-large"
BOT_PINECONE_INDEX_NAME = "mathtutor-rag-e5-large"  # Use same index as main chatbot

# Initialize bot.py components
try:
    bot_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    bot_index = bot_pc.Index(BOT_PINECONE_INDEX_NAME)
    bot_model = SentenceTransformer(BOT_MODEL_NAME)
    bot_all_exercises = []
    
    def load_bot_exercises():
        """Load exercises for bot.py functionality."""
        global bot_all_exercises
        try:
            with open(BOT_INPUT_FILE, "r", encoding="utf-8") as f:
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
            
            bot_all_exercises = exercises
            logger.info(f"Loaded {len(bot_all_exercises)} exercises for bot functionality")
            return exercises
        except Exception as e:
            logger.error(f"Error loading bot exercises: {e}")
            return []
    
    # Load exercises on startup
    load_bot_exercises()
    
except Exception as e:
    logger.error(f"Error initializing bot.py components: {e}")
    bot_pc = None
    bot_index = None
    bot_model = None
    bot_all_exercises = []

def get_bot_classes():
    """Get available classes from bot exercises."""
    if not bot_all_exercises:
        return []
    return sorted(set(ex["exercise_metadata"]["class"] for ex in bot_all_exercises if "exercise_metadata" in ex))

def get_bot_topics(chosen_class):
    """Get available topics for a specific class."""
    if not bot_all_exercises:
        return []
    return sorted(set(
        ex["exercise_metadata"]["topic"]
        for ex in bot_all_exercises
        if "exercise_metadata" in ex and ex["exercise_metadata"]["class"] == chosen_class
    ))

def get_bot_exercises(chosen_class, chosen_topic):
    """Get exercises for specific class and topic."""
    if not bot_all_exercises:
        return []
    return [
        ex for ex in bot_all_exercises
        if "exercise_metadata" in ex 
        and ex["exercise_metadata"]["class"] == chosen_class
        and ex["exercise_metadata"]["topic"] == chosen_topic
    ]

def run_bot_exercise(exercise):
    """Run a guided exercise from bot.py functionality."""
    try:
        meta = exercise["exercise_metadata"]
        content = exercise["exercise_content"]

        print(f"\n📘 Exercise {meta['exercise_number']} ({meta['exercise_type']})")
        print("Main text:", content["main_data"]["text"])

        student_answers = []
        for sec in content["sections"]:
            q = sec.get("question", {}).get("text")
            if q:
                print(f"\n❓ Section {sec['section_number']} - {q}")
                answer = input("Your answer: ")
                student_answers.append({
                    "section": sec['section_number'],
                    "question": q,
                    "answer": answer
                })
                print(f"Answer recorded: {answer}")
                input("Press Enter to continue to the next question...")

        print("\nYour answers:")
        for ans in student_answers:
            print(f"Section {ans['section']}: {ans['answer']}")
            
        return student_answers
    except Exception as e:
        logger.error(f"Error running bot exercise: {e}")
        print("Sorry, there was an error running this exercise.")
        return []

def retrieve_bot_answer(query, top_k=3):
    """Retrieve answers using bot.py retrieval system."""
    if not bot_model or not bot_index:
        return []
    
    try:
        emb = bot_model.encode([query])[0].tolist()
        results = bot_index.query(vector=emb, top_k=top_k, include_metadata=True)

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
    except Exception as e:
        logger.error(f"Error in bot retrieval: {e}")
        return []

def run_bot_mode():
    """Run the bot.py main functionality as an alternative mode."""
    print("Welcome to the Math Chatbot! 🎓")
    print("Choose mode: (1) Guided Exercises, (2) Free Q&A")
    mode = input("> ")

    if mode == "1":
        # Guided pipeline
        classes = get_bot_classes()
        if not classes:
            print("No classes available in bot exercises.")
            return
            
        print(f"Available classes: {classes}")
        chosen_class = input("Pick a class: ")

        topics = get_bot_topics(chosen_class)
        if not topics:
            print(f"No topics available for class {chosen_class}.")
            return
            
        print(f"Available topics: {topics}")
        chosen_topic = input("Pick a topic: ")

        exercises = get_bot_exercises(chosen_class, chosen_topic)
        if not exercises:
            print("No exercises found for this selection.")
            return

        # Run the first exercise for now
        exercise = exercises[0]
        results = run_bot_exercise(exercise)
        
        if len(exercises) > 1:
            print(f"\nThere are {len(exercises)} exercises available for this topic.")
            continue_choice = input("Would you like to continue with more exercises? (y/n): ")
            if continue_choice.lower() in ['y', 'yes']:
                for i, ex in enumerate(exercises[1:], 2):
                    print(f"\n=== Exercise {i} ===")
                    run_bot_exercise(ex)

    else:
        # Retrieval chatbot
        print("Ask me anything about math exercises! (type 'quit' to exit)")
        while True:
            query = input("You: ")
            if query.lower() in ["quit", "exit"]:
                break
            results = retrieve_bot_answer(query)
            if results:
                print("\n🔍 Found these relevant answers:")
                for i, ans in enumerate(results, 1):
                    print(f"\n[{i}] [{ans['chunk_type']} | Ex {ans['exercise_number']} | parent {ans['parent_id']} | Score: {ans['score']:.3f}]")
                    print(ans['text'])
            else:
                print("Sorry, I couldn't find anything relevant.")

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not INPUT_FILE.exists():
        logger.error("❌ Missing JSON file.")
        return

    try:
        exercises = load_exercises()
        pinecone_index = get_pinecone_index()
    except Exception as e:
        logger.error(f"❌ Error loading data or connecting to Pinecone: {e}")
        return

    fsm = DialogueFSM(exercises, pinecone_index)

    # Initial transition to start the conversation
    initial_response = fsm.transition("")
    print(f"A_GUY: {initial_response}")

    while True:
        try:
            # Enhanced input handling with typing detection
            fsm.inactivity_timer.mark_typing()
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            fsm.inactivity_timer.stop()
            break
        if user_input.lower() in {"exit", "quit", "done"}:
            print("👋 Bye!")
            fsm.inactivity_timer.stop()
            break

        response = fsm.transition(user_input)

        print(f"A_GUY: {response}")

if __name__ == "__main__":
    main()