📘 RAG Math Tutor (IvriTutor Demo)
This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to assist students with 8th-grade math exercises.
It uses a knowledge base of parsed exercises, Pinecone for vector search, and Google Gemini (Qwen) for natural language understanding.

🚀 Features
Interactive Math Tutor Chatbot with FSM-based flow
Supports retrieval from JSON exercise bank
Provides hints, solutions, and answer evaluation
Uses Pinecone for efficient vector search
Powered by Google Gemini/Qwen for natural language understanding
📋 Prerequisites
Python: Version 3.8 or higher
API Keys:
Pinecone → Get API Key
Google Gemini (Qwen) → Google AI Studio or Google Cloud Console
(Optional) Virtual Environment for dependency isolation
⚙️ Setup
1. Clone the Repository
git clone https://github.com/your-username/IvriTutor_Demo.git
cd IvriTutor_Demo

2. Create a Virtual Environment (Recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

## 4. Configure Environment Variables

Create a .env file in the project root:

PINECONE_API_KEY=your_actual_pinecone_api_key_here
GEMINI_API_KEY=your_actual_gemini_api_key_here

📚 Preparing the Knowledge Base

Place parsed exercise JSON files in:

IvriTutor_Demo/parsed_outputs/


Generate embeddings (first run or after data changes):

python IvriTutor_Demo/parsed_outputs/embedding.py


Index embeddings into Pinecone:

python IvriTutor_Demo/parsed_outputs/index_embedding.py


⚠️ Make sure the index name in index_embedding.py matches the one used in chatbot.py (default: mathtutor-e5-large).

💬 Running the Chatbot

Make sure your virtual environment is active, then run:

## python chatbot.py

Interaction Flow:

The chatbot greets you

You select grade and topic

It presents math exercises

You can:

Provide an answer → chatbot evaluates

Ask for a hint

Request the solution


## Project Instructions
1. Chat Flow and User Engagement
Lead the Conversation: The chat should initiate the conversation with a friendly greeting like "Hey!", "How's it going?". "What's new?", or "How are you doing?"
Personalization: Following the initial greeting, the chat should ask about something related to the student's personal life. This could be their job search, how yesterday's game was, etc.
Academics: After the personal conversation, the chat(teacher) should transition to academic topics by asking questions like "What did you learn recently?" or "When is your next exam?".
2. Guiding the Student
Gradual Assistance: If the student indicates they don't know the answer/asking for the full solution,
the system should offer them a chance to reconsider before providing help. The support should follow this specific sequence:
Guiding question -> Another guiding question -> A hint -> The full solution with an explanation
3. Support for Multiple Languages
The system needs to support both Hebrew (RTL) and English (LTR). It's crucial that the text direction is correct for each language.
This applies to all conversational elements and any mathematical expressions or scientific notation, which should remain LTR.
4. Handling the Full Solution
If the student explicitly requests the full solution, they should only receive it after they have been guided through the steps
outlined in point 2 (guiding question, guiding question, hint). The full solution should not be given immediately.
5. Inactivity Timeout
If the student doesn't respond for 30 seconds, the chat should check in to make sure they are still there.


