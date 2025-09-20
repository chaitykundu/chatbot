import json
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# ---------------- STATE MANAGEMENT ----------------
class TutorState(Enum):
    OPENING = 1
    DIAGNOSTIC = 2
    LEARNING = 3
    COMPLETE = 4

@dataclass
class ConversationContext:
    state: TutorState = TutorState.OPENING
    user_hobbies: Optional[str] = None
    upcoming_test: Optional[str] = None
    last_class_topic: Optional[str] = None
    focus_topic: Optional[str] = None
    exercise_counter: int = 0
    current_exercise: Optional[Dict[str, Any]] = None
    misunderstanding_step: int = 0  # 0=first attempt, 1=reread, 2=guide1, 3=guide2, 4=hint, 5=solution
    opening_step: int = 1  # Track which opening question we're on (1-5)
    diagnostic_step: int = 1  # Track which diagnostic question we're on (1-3)

# ---------------- LLM HELPERS ----------------
def generate_personalized_followup(hobby: str) -> str:
    """Generate personalized follow-up question based on hobby"""
    prompt = f"""
    The student mentioned their hobby is: {hobby}
    
    Generate a short, friendly follow-up question about their hobby. 
    Keep it conversational and show genuine interest.
    Examples:
    - If hobby is basketball: "Did you get to play today?"
    - If hobby is reading: "What book are you reading right now?"
    - If hobby is gaming: "What game are you into lately?"
    
    Just return the question, nothing else.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating followup: {e}")
        return "That sounds interesting! Tell me more about it."

def generate_humorous_reaction(hobby: str, followup_answer: str) -> str:
    """Generate short humorous reaction"""
    prompt = f"""
    The student's hobby is {hobby} and they just answered: "{followup_answer}"
    
    Generate a short, light humorous reaction that transitions to studying.
    Examples:
    - "Great! Let's see if your brain is as fit as your legs."
    - "Nice! Now let's exercise your math muscles too."
    - "Awesome! Time to level up your math skills."
    
    Keep it positive, encouraging, and brief. Just return the reaction, nothing else.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating reaction: {e}")
        return "Great! Now let's get those math skills warmed up."

    def generate_contextual_response(self, user_input: str, response_type: str) -> str:
        """Generate contextual AI responses based on user input"""
        return generate_contextual_response(self, user_input, response_type)
    """Generate contextual AI responses based on user input"""
    if response_type == "day_followup":
        prompt = f"""
        The student just responded to "Hey hey, how are you?" with: "{user_input}"
        
        Generate a natural, contextual follow-up question about their day that acknowledges their response. 
        Be friendly, conversational, and show you're listening to what they said.
        
        Examples:
        - If they said "good": "That's great to hear! How was your day? Anything interesting happen?"
        - If they said "tired": "Oh, sounds like you've had a long day! What kept you busy?"
        - If they said "stressed": "Sorry to hear that! Rough day? What's been going on?"
        
        Just return the contextual response, nothing else.
        """
    
    elif response_type == "hobby_question":
        prompt = f"""
        We just asked about their day and they responded: "{user_input}"
        
        Generate a natural transition to asking about their hobbies that acknowledges their day response.
        Be conversational and connecting.
        
        Examples:
        - If they had a good day: "Awesome! So when you're not having good days like today, what do you like to do to unwind? Any hobbies?"
        - If they had a tough day: "Well hopefully we can make it better! What do you usually do to relax? Any hobbies you're into?"
        - If neutral: "Fair enough! So what do you like to do in your free time? Any hobbies?"
        
        Just return the contextual question, nothing else.
        """
    
    elif response_type == "diagnostic_followup":
        prompt = f"""
        We're in the diagnostic phase and the student just answered: "{user_input}"
        
        Generate a natural, contextual response that acknowledges what they said and flows into the next diagnostic question.
        Be supportive and show you're building on their previous answer.
        
        Just return the contextual response, nothing else.
        """
    
    else:
        return "I'm not sure how to respond to that."
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating contextual response: {e}")
        # Fallback responses
        if response_type == "day_followup":
            return "That's nice! How was your day? Anything interesting happen?"
        elif response_type == "hobby_question":
            return "Cool! So what do you like to do in your free time? Any hobbies?"
        else:
            return "Thanks for sharing that! Let me ask you something else..."
    """Generate missing guiding questions, hints, or solutions"""
    exercise_text = exercise["exercise_content"]["main_data"]["text"]
    
    if content_type == "guiding_question_1":
        prompt = f"""
        Exercise: {exercise_text}
        Current question: {question_text}
        
        The student is struggling with this question. Generate the FIRST guiding question to help them think about the problem step by step. This should be a gentle nudge in the right direction, not giving away the answer.
        
        Just return the guiding question, nothing else.
        """
    elif content_type == "guiding_question_2":
        prompt = f"""
        Exercise: {exercise_text}
        Current question: {question_text}
        
        The student still needs help after the first guiding question. Generate a SECOND guiding question that provides more specific direction while still encouraging them to think.
        
        Just return the guiding question, nothing else.
        """
    elif content_type == "hint":
        prompt = f"""
        Exercise: {exercise_text}
        Current question: {question_text}
        
        The student needs a hint. Provide a helpful hint that points toward the solution method or key insight needed to solve this problem.
        
        Just return the hint, nothing else.
        """
    elif content_type == "solution":
        prompt = f"""
        Exercise: {exercise_text}
        Current question: {question_text}
        
        Provide a complete, step-by-step solution to this problem. Explain each step clearly so the student can understand the reasoning.
        
        Just return the solution, nothing else.
        """
    else:
        return "I'm not sure how to help with that."
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating {content_type}: {e}")
        return f"I'm having trouble generating the {content_type}. Let me try to help in a different way."

# ---------------- STATE HANDLERS ----------------
class TutorStateMachine:
    def __init__(self):
        self.context = ConversationContext()
    
    def handle_opening_state(self, user_input: str = "") -> str:
        """Handle STATE_1 - Opening"""
        if self.context.opening_step == 1:
            self.context.opening_step = 2
            return "Hey hey, how are you?"
        
        elif self.context.opening_step == 2:
            # Generate contextual response to their mood/state
            contextual_response = self.generate_contextual_response(user_input, "day_followup")
            self.context.opening_step = 3
            return contextual_response
        
        elif self.context.opening_step == 3:
            # Generate contextual hobby question based on their day response
            hobby_question = self.generate_contextual_response(user_input, "hobby_question")
            self.context.opening_step = 4
            return hobby_question
        
        elif self.context.opening_step == 4:
            # Store hobbies and generate personalized follow-up
            self.context.user_hobbies = user_input
            self.context.opening_step = 5
            return generate_personalized_followup(user_input)
        
        elif self.context.opening_step == 5:
            # Generate humorous reaction and transition
            reaction = generate_humorous_reaction(self.context.user_hobbies, user_input)
            self.context.state = TutorState.DIAGNOSTIC
            self.context.opening_step = 1  # Reset for next time
            return reaction
        
        return "Let's move on to understanding what you need to work on!"
    
    def handle_diagnostic_state(self, user_input: str = "") -> str:
        """Handle STATE_2 - Diagnostic"""
        if self.context.diagnostic_step == 1:
            self.context.diagnostic_step = 2
            return "Do you have a test coming up?"
        
        elif self.context.diagnostic_step == 2:
            # Store test info and generate contextual response
            self.context.upcoming_test = user_input
            contextual_response = generate_contextual_response(self, user_input, "diagnostic_followup")
            self.context.diagnostic_step = 3
            # Add the actual question after the contextual response
            return f"{contextual_response} What did you cover in the last class?"
        
        elif self.context.diagnostic_step == 3:
            # Store last class topic and generate contextual response
            self.context.last_class_topic = user_input
            contextual_response = generate_contextual_response(self, user_input, "diagnostic_followup")
            self.context.diagnostic_step = 4
            return f"{contextual_response} What would you like to work on today?"
        
        elif self.context.diagnostic_step == 4:
            # Store focus topic and transition to learning
            self.context.focus_topic = user_input
            self.context.state = TutorState.LEARNING
            self.context.diagnostic_step = 1  # Reset for next time
            return f"Perfect! I can see you want to focus on {user_input}. That's a great choice! Let's start working on some exercises that will help you with that."
        
        return "Great! Let's begin with some exercises."
    
    def handle_learning_state(self, user_input: str, exercises: List[Dict[str, Any]]) -> str:
        """Handle STATE_3 - Learning Stage"""
        # Check if we've completed 2 exercises
        if self.context.exercise_counter >= 2:
            self.context.state = TutorState.COMPLETE
            return "Great job! You've completed both exercises. Keep up the good work!"
        
        # If no current exercise, start a new one
        if self.context.current_exercise is None:
            if self.context.exercise_counter < len(exercises):
                self.context.current_exercise = exercises[self.context.exercise_counter]
                self.context.misunderstanding_step = 0
                return self.present_exercise(self.context.current_exercise)
            else:
                self.context.state = TutorState.COMPLETE
                return "We've worked through all available exercises! Great job!"
        
        # Handle student response based on misunderstanding step
        return self.handle_student_response(user_input)
    
    def present_exercise(self, exercise: Dict[str, Any]) -> str:
        """Present the exercise to the student"""
        meta = exercise["exercise_metadata"]
        content = exercise["exercise_content"]
        
        exercise_text = f"\n📘 Exercise {meta['exercise_number']} ({meta['exercise_type']})\n"
        exercise_text += f"Main text: {content['main_data']['text']}\n"
        
        # Present the first section question
        if content["sections"]:
            first_section = content["sections"][0]
            question = first_section.get("question", {}).get("text", "")
            if question:
                exercise_text += f"\n❓ Question: {question}\n"
        
        return exercise_text
    
    def handle_student_response(self, user_input: str) -> str:
        """Handle student response with misunderstanding mechanism"""
        if self.context.misunderstanding_step == 0:
            # First response - assume incorrect, guide them
            self.context.misunderstanding_step = 1
            return "Try reading the question again carefully..."
        
        elif self.context.misunderstanding_step == 1:
            # Provide first guiding question
            self.context.misunderstanding_step = 2
            current_section = self.context.current_exercise["exercise_content"]["sections"][0]
            question_text = current_section.get("question", {}).get("text", "")
            return generate_missing_content(self.context.current_exercise, "guiding_question_1", question_text)
        
        elif self.context.misunderstanding_step == 2:
            # Provide second guiding question
            self.context.misunderstanding_step = 3
            current_section = self.context.current_exercise["exercise_content"]["sections"][0]
            question_text = current_section.get("question", {}).get("text", "")
            return generate_missing_content(self.context.current_exercise, "guiding_question_2", question_text)
        
        elif self.context.misunderstanding_step == 3:
            # Provide hint
            self.context.misunderstanding_step = 4
            current_section = self.context.current_exercise["exercise_content"]["sections"][0]
            question_text = current_section.get("question", {}).get("text", "")
            return generate_missing_content(self.context.current_exercise, "hint", question_text)
        
        elif self.context.misunderstanding_step == 4:
            # Provide full solution and move to next exercise
            current_section = self.context.current_exercise["exercise_content"]["sections"][0]
            question_text = current_section.get("question", {}).get("text", "")
            solution = generate_missing_content(self.context.current_exercise, "solution", question_text)
            
            # Move to next exercise
            self.context.exercise_counter += 1
            self.context.current_exercise = None
            self.context.misunderstanding_step = 0
            
            return f"{solution}\n\nNow let's move on to the next exercise!"
        
        return "Let's continue with the next exercise!"
    
    def process_input(self, user_input: str, exercises: List[Dict[str, Any]] = None) -> str:
        """Main method to process user input based on current state"""
        if self.context.state == TutorState.OPENING:
            return self.handle_opening_state(user_input)
        
        elif self.context.state == TutorState.DIAGNOSTIC:
            return self.handle_diagnostic_state(user_input)
        
        elif self.context.state == TutorState.LEARNING:
            if exercises is None:
                exercises = []
            return self.handle_learning_state(user_input, exercises)
        
        elif self.context.state == TutorState.COMPLETE:
            return "Thanks for learning with me today! Feel free to start a new session anytime."
        
        return "I'm not sure how to help with that right now."