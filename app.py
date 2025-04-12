# app.py
import os
import re
import json
import random
import datetime
import time
import tempfile # For handling uploaded audio files
from dotenv import load_dotenv

# Langchain & LLM Imports
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
# operator and RunnablePassthrough/Lambda might not be needed if invoking directly
# from operator import itemgetter
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
# Removed RunnableSequence

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Text-to-Speech Imports
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

# Speech-to-Text Imports
import speech_recognition as sr

# Flask Web Framework
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

# --- Flask App Initialization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR, static_folder=os.path.join(BASE_DIR, "static"))

# --- Environment Variables & Configuration ---
load_dotenv(dotenv_path="pass.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Critical Error: GROQ_API_KEY is not set.")
    # exit(1)

# Directories and Files
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "responses_output")
ERROR_LOG_FILE = os.path.join(BASE_DIR, "error_log.txt")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "user_uploads")
# --- LTM File ---
LTM_FILE_PATH = os.path.join(BASE_DIR, "mindpal_ltm.json") # Long-Term Memory file

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Global Variables ---
response_counter = 1
chat_memory = None # Session memory (Langchain)
groq_chat = None # LLM instance
tts_model = None
tts_tokenizer = None
vader_analyzer = None
speech_recognizer = None
long_term_memory = {} # Persistent memory (Loaded from file)

# MindPal Specific Data (Activity Suggestions, Crisis Keywords) - Same as before
activity_suggestions = {
    "Negative": [
        "Sometimes just taking a few slow, deep breaths can make a difference. Would you like to try?",
        "Perhaps listening to some calming music could soothe your mind right now.",
        "Writing down your thoughts, like in a journal, can often help clarify feelings. Is that something you might consider?",
        "If possible, a gentle walk, even just for a few minutes, might help shift your perspective.",
        "Remember to be kind to yourself during tough moments."
    ],
    "Neutral": [
        "This might be a good moment for a brief mindfulness exercise, just noticing your surroundings.",
        "How about taking a moment to think of one small thing you're grateful for today?",
        "Sometimes a little planning helps. Could you think of one small, enjoyable thing to do later?",
        "A quiet moment to simply 'be' can be quite restorative."
    ],
    "Positive": [
        "That's wonderful to hear! Take a moment to really soak in this positive feeling.",
        "Maybe you could channel this positive energy into something creative or enjoyable?",
        "Sharing positive feelings can amplify them. Is there someone you could share this with?",
        "Acknowledging positive moments helps build resilience. Well done."
    ]
}
CRISIS_KEYWORDS = ["suicide", "kill myself", "hopeless", "self-harm", "want to die", "can't go on", "no reason to live"]

# --- Utility Functions ---
# log_error, load_chat_history, save_chat_history (Same as before)
def log_error(error_msg, exc_info=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {error_msg}\n"
    if exc_info:
        import traceback
        log_entry += traceback.format_exc() + "\n"
    print(f"ERROR: {log_entry.strip()}")
    try:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"🚨 Critical Error: Could not write to log file {ERROR_LOG_FILE}: {e}")

def load_chat_history():
    history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
                for msg_dict in raw_history:
                    msg_type = msg_dict.get('type')
                    content = msg_dict.get('content', '')
                    if msg_type == 'human':
                        history.append(HumanMessage(content=content))
                    elif msg_type == 'ai':
                        history.append(AIMessage(content=content))
            print(f"Loaded {len(history)} messages from chat history.")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {CHAT_HISTORY_FILE}. Starting fresh history.")
            log_error(f"JSONDecodeError loading chat history from {CHAT_HISTORY_FILE}")
        except Exception as e:
            print(f"⚠️ Warning: Error loading chat history: {e}. Starting fresh history.")
            log_error(f"Error loading chat history: {e}", exc_info=True)
    return history

def save_chat_history(messages):
    try:
        history_to_save = []
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(msg, HumanMessage):
                history_to_save.append({"type": "human", "content": content})
            elif isinstance(msg, AIMessage):
                history_to_save.append({"type": "ai", "content": content})
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
        print(f"Saved {len(history_to_save)} messages to chat history.")
    except Exception as e:
        log_error(f"Error saving chat history: {e}", exc_info=True)

# --- Long-Term Memory (LTM) Functions --- ADDED ---
def load_ltm():
    """Loads long-term memory from the JSON file."""
    global long_term_memory
    if os.path.exists(LTM_FILE_PATH):
        try:
            with open(LTM_FILE_PATH, 'r', encoding='utf-8') as f:
                long_term_memory = json.load(f)
                print(f"✅ Long-Term Memory loaded ({len(long_term_memory)} items).")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not decode {LTM_FILE_PATH}. Starting empty LTM.")
            log_error(f"JSONDecodeError loading LTM from {LTM_FILE_PATH}")
            long_term_memory = {}
        except Exception as e:
            print(f"⚠️ Warning: Error loading LTM: {e}. Starting empty LTM.")
            log_error(f"Error loading LTM: {e}", exc_info=True)
            long_term_memory = {}
    else:
        print("No LTM file found. Starting empty LTM.")
        long_term_memory = {}
    # Ensure it's a dictionary
    if not isinstance(long_term_memory, dict):
        print(f"⚠️ Warning: LTM data loaded was not a dictionary ({type(long_term_memory)}). Resetting to empty LTM.")
        long_term_memory = {}

def save_ltm():
    """Saves the current long-term memory dictionary to the JSON file."""
    global long_term_memory
    if not isinstance(long_term_memory, dict):
        log_error(f"Attempted to save LTM, but it was not a dictionary ({type(long_term_memory)}). Skipping save.")
        return
    try:
        with open(LTM_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(long_term_memory, f, indent=4)
        print(f"✅ Long-Term Memory saved ({len(long_term_memory)} items).")
    except Exception as e:
        log_error(f"Error saving LTM: {e}", exc_info=True)

# --- Sentiment & Voice Analysis ---
# analyze_text_sentiment, analyze_voice_sentiment_placeholder (Same as before)
def analyze_text_sentiment(text):
    global vader_analyzer
    if not vader_analyzer: return "Neutral"
    if not text: return "Neutral"
    try:
        sentiment_score = vader_analyzer.polarity_scores(text)
        compound = sentiment_score['compound']
        if compound >= 0.05: return "Positive"
        elif compound <= -0.05: return "Negative"
        else: return "Neutral"
    except Exception as e:
        log_error(f"Error analyzing text sentiment: {e}")
        return "Neutral"

def analyze_voice_sentiment_placeholder(audio_file_path):
    print(f"Placeholder: Would analyze voice sentiment for {audio_file_path}")
    return None

# --- Speech-to-Text ---
# transcribe_audio (Same as before)
def transcribe_audio(audio_file_path):
    global speech_recognizer
    if not speech_recognizer:
        log_error("Speech recognizer not initialized.")
        return None, "Error: Speech recognition service not available."
    if not os.path.exists(audio_file_path):
        log_error(f"Audio file not found for transcription: {audio_file_path}")
        return None, "Error: Audio file missing."
    try:
        with sr.AudioFile(audio_file_path) as source:
            print(f"Listening to audio file: {audio_file_path}")
            audio_data = speech_recognizer.record(source)
            print("Transcribing audio...")
            text = speech_recognizer.recognize_google(audio_data)
            print(f"Transcription result: '{text}'")
            return text, None
    except sr.UnknownValueError:
        error_msg = "Could not understand audio. Please speak clearly."
        log_error(f"SpeechRecognition UnknownValueError for file: {audio_file_path}")
        return None, error_msg
    except sr.RequestError as e:
        error_msg = f"Speech recognition service error: {e}."
        log_error(f"SpeechRecognition RequestError: {e}")
        return None, error_msg
    except Exception as e:
        error_msg = "An unexpected error occurred during transcription."
        log_error(f"Unexpected error in transcribe_audio: {e}", exc_info=True)
        return None, error_msg

# --- Text-to-Speech ---
# save_response_to_wav (Same as before)
def save_response_to_wav(response_text, file_name_base):
    global tts_model, tts_tokenizer, response_counter
    if not tts_model or not tts_tokenizer:
        print("⚠️ TTS is unavailable.")
        return None, None
    if not response_text:
        print("⚠️ Cannot generate audio for empty text.")
        return None, None
    try:
        safe_base = re.sub(r'[\\/*?:"<>|]', "", file_name_base)
        unique_file_name = f"{safe_base}_{response_counter}.wav"
        output_path = os.path.join(AUDIO_DIR, unique_file_name)
        output_url = url_for('static', filename=f'responses_output/{unique_file_name}', _external=False)

        print(f"Generating TTS for '{response_text[:50]}...'")
        inputs = tts_tokenizer(response_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
            if not isinstance(output, torch.Tensor):
                 raise TypeError("Expected TTS output waveform to be a Tensor")
            audio = output.squeeze().cpu().numpy()
        if audio.ndim > 1:
            if audio.shape[0] > 1 and audio.ndim == 2:
                audio = audio[0, :]
            else:
                audio = audio.flatten()

        sampling_rate = tts_model.config.sampling_rate
        sf.write(output_path, audio, samplerate=sampling_rate)
        print(f"🎙️ Generated audio: {output_path} (URL: {output_url})")
        response_counter += 1
        return output_path, output_url
    except Exception as e:
        log_error(f"Error during TTS generation for text '{response_text[:50]}...': {e}", exc_info=True)
        return None, None

# --- Dynamic Prompt Creation --- CORRECTED (Alternative) ---
def create_mindpal_prompt(text_sentiment_tag, voice_sentiment_tag, is_vent_mode, ltm_data):
    """Creates the dynamic system prompt content for MindPal, incorporating sentiment and LTM."""
    # ... (Serialization logic for ltm_facts_str remains the same) ...
    ltm_facts_str = "No specific long-term memories noted yet." # Default value
    if ltm_data and isinstance(ltm_data, dict) and ltm_data:
        try:
            ltm_facts_str = json.dumps(ltm_data, indent=None, ensure_ascii=False)
            if len(ltm_facts_str) > 1500:
                 truncated_ltm = dict(list(ltm_data.items())[:15])
                 ltm_facts_str = json.dumps({"summary": "Key facts remembered (details summarized).", **truncated_ltm }, ensure_ascii=False)
                 print(f"⚠️ LTM data too large for prompt ({len(ltm_facts_str)} chars), sending truncated summary.")
            elif ltm_facts_str == '{}':
                 ltm_facts_str = "No specific long-term memories noted yet."
        except Exception as e:
            log_error(f"Error serializing LTM for prompt: {e}", exc_info=True)
            ltm_facts_str = "Error retrieving remembered facts."
    if not ltm_facts_str:
        ltm_facts_str = "No specific long-term memories noted yet."

    # Base prompt definition remains the same...
    base_prompt = """You are MindPal, a supportive mental wellness companion AI. 
    Your primary goal is to listen empathetically, help users reflect on their mood, and offer gentle, evidence-informed suggestions for mindfulness or well-being activities appropriate for their current state. 
    You should remember key facts the user tells you to remember across sessions.
    \n\n**Remembered Facts (Long-Term Memory):**\n{ltm_facts_str}\n\n**Your Core Principles:**\n- **Be Kind & Empathetic:** 
    Respond with warmth, patience, and non-judgment. Acknowledge and validate the user's feelings. 
    Use remembered facts naturally when relevant.\n- **Be Supportive, Not Therapeutic:** 
    You are a wellness tool, **NOT** a replacement for professional therapy or diagnosis. Avoid giving medical advice.\n- 
    **Focus on the Present & History:** Keep the conversation grounded in the current check-in and recent history ({chat_history}). 
    Refer back to LTM when appropriate.\n- **Suggest Gently:** If offering activities, tailor them based on the user's likely mood ({text_sentiment_tag}). Do not pressure the user.\n- **Consider Tone (If Possible):** Pay attention to the user's language. If their tone sounds particularly flat or agitated (based on external analysis - currently placeholder), adjust your empathy accordingly. [{voice_sentiment_info}]\n- **Maintain Safety:** If the user expresses thoughts of self-harm, suicide, or being in immediate crisis, **STOP** your standard response flow. Provide crisis support resources and state clearly that you cannot offer therapeutic help for such situations. [Crisis handling is managed by external logic, but be aware].
    \n- **Keep it Concise:** Aim for clear, calming, and relatively brief responses suitable for a chat interface.\n
"""
    # voice_info definition remains the same...
    voice_info = f"Voice analysis suggests: {voice_sentiment_tag}" if voice_sentiment_tag else "Voice tone analysis not available or inconclusive."

    # mode_instruction definition remains the same...
    if is_vent_mode:
        mode_instruction = """
**Current Mode: Vent Mode**
... (rest of vent mode instruction) ...
"""
    else: # Standard Mode
        mode_instruction = f"""
**Current Mode: Standard Chat**
... (rest of standard mode instruction) ...
"""

    # --- FIXED CONCATENATION using Parentheses ---
    final_prompt_content = (
        base_prompt.replace("{ltm_facts_str}", ltm_facts_str)
                   .replace("{text_sentiment_tag}", text_sentiment_tag or "Unknown")
                   .replace("{voice_sentiment_info}", voice_info)
                   .replace("{chat_history}", "{chat_history}") # Keep placeholder
        + mode_instruction
        + "\n**Current Task:** Respond to the human input now."
    ) # No backslashes needed inside parentheses

    # Final check for {chat_history} placeholder remains the same...
    if "{chat_history}" not in final_prompt_content:
         log_error("Placeholder {chat_history} was unexpectedly removed during prompt creation.")

    return final_prompt_content

# --- Initialization Function --- MODIFIED ---
def initialize_assistant():
    """Loads models, memory (session & LTM), and sets up the MindPal state."""
    global chat_memory, groq_chat, tts_model, tts_tokenizer, vader_analyzer, speech_recognizer, response_counter, long_term_memory
    print("🚀 Initializing MindPal Backend...")

    # --- Load LTM First --- ADDED ---
    load_ltm() # Load persistent memory into the global dict

    # Initialize VADER
    try:
        vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER Sentiment Analyzer initialized.")
    except Exception as e:
        log_error(f"Failed to initialize VADER: {e}", exc_info=True); vader_analyzer = None

    # Initialize Speech Recognition
    try:
        speech_recognizer = sr.Recognizer()
        print("✅ Speech Recognition engine initialized.")
    except Exception as e:
        log_error(f"Failed to initialize SpeechRecognition: {e}", exc_info=True); speech_recognizer = None

    # Initialize Text-to-Speech (TTS)
    try:
        model_id = "facebook/mms-tts-eng"
        print(f"Loading TTS model: {model_id}...")
        tts_model = VitsModel.from_pretrained(model_id)
        tts_tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✅ TTS Model loaded successfully.")
    except Exception as e:
        print(f"❌ Warning: Failed to load TTS model ({model_id}): {e}")
        log_error(f"Failed to load TTS model: {e}", exc_info=True); tts_model = None; tts_tokenizer = None

    # Initialize LLM and Conversation Memory
    if not GROQ_API_KEY:
        print("❌ Critical Error: GROQ_API_KEY missing. LLM functionality disabled.")
        groq_chat = None; chat_memory = None
    else:
        model_name = "llama3-8b-8192"
        print(f"Initializing Groq Chat LLM: {model_name}...")
        try:
            groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=0.75)
            print(f"✅ Groq Chat LLM ({model_name}) initialized.")

            # Memory Setup (Session Memory)
            conversational_memory_length = 10
            chat_memory = ConversationBufferWindowMemory(
                k=conversational_memory_length, memory_key="chat_history", return_messages=True
            )
            initial_history = load_chat_history()
            for msg in initial_history:
                chat_memory.chat_memory.add_message(msg)
            print(f"✅ Langchain Session memory initialized with {len(initial_history)} previous messages.")

        except Exception as e:
            print(f"❌ Error initializing Groq Chat LLM or Memory: {e}")
            log_error(f"Error initializing Groq/Langchain: {e}", exc_info=True); groq_chat = None; chat_memory = None

    response_counter = 1
    print("👍 MindPal Backend Initialization Complete.")
    print("-" * 30)


# --- Flask Routes ---
# /, /static, /welcome_pa (Same as before)
@app.route('/')
def index():
    print("Serving frontend.html")
    return render_template('frontend.html')

@app.route('/static/<path:path>')
def send_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/welcome_pa', methods=['GET'])
def welcome():
    print("Processing /welcome_pa request")
    welcome_message = "Hello there. I'm MindPal. How are you feeling today? Feel free to talk or type. I can also remember things you tell me to."
    file_path, audio_url = save_response_to_wav(welcome_message, "mindpal_welcome")
    return jsonify({ "response": welcome_message, "audio_url": audio_url })


# --- Chat Endpoint --- MODIFIED ---
@app.route('/chat_pa', methods=['POST'])
def chat():
    """Handles user chat messages (text or audio) for MindPal, including LTM commands."""
    global chat_memory, groq_chat, response_counter, long_term_memory # Ensure LTM is global

    start_time = time.time()
    print("\n--- New MindPal Chat Request ---")

    # Check core components
    if not groq_chat or not chat_memory or not vader_analyzer or not speech_recognizer:
        log_error("Chat endpoint called but core components not ready.")
        return jsonify({"error": "MindPal core services are initializing or unavailable."}), 503

    user_input_text = None
    audio_file_path = None
    stt_error = None
    final_audio_url = None
    suggestion_added = False
    is_crisis = False
    response_text = ""
    intent_handled = False # Flag for specific commands like LTM

    try:
        # --- Input Processing (Audio or Text) ---
        # (Same as before - handles audio upload, STT, or gets text input)
        if 'audio_file' in request.files:
            # ... (rest of audio handling and STT - same as previous version) ...
            audio_file = request.files['audio_file']
            if audio_file.filename != '':
                # Use a temporary directory to avoid clutter and potential name collisions
                with tempfile.TemporaryDirectory(dir=UPLOAD_DIR) as temp_dir:
                    # Ensure filename is safe (though tempfile uses unique dirs)
                    safe_filename = f"upload_{int(time.time())}.wav" # Assuming WAV for STT
                    audio_file_path = os.path.join(temp_dir, safe_filename)
                    try:
                        audio_file.save(audio_file_path)
                        print(f"Received audio file saved temporarily to: {audio_file_path}")
                        user_input_text, stt_error = transcribe_audio(audio_file_path)
                    except Exception as save_err:
                        log_error(f"Error saving uploaded audio file: {save_err}", exc_info=True)
                        stt_error = "Error processing uploaded audio."
                    # audio_file_path will be automatically cleaned up when 'with' block exits
            else: stt_error = "Received empty audio file."
        else: # Process text input
            data = request.get_json()
            if data and "user_input" in data:
                user_input_text = data.get("user_input", "").strip()
                print(f"Received Text Input: '{user_input_text}'")
            else: # Neither audio nor text provided
                log_error("Request has no 'audio_file' and no valid 'user_input' text.")
                return jsonify({"error": "Invalid request: No text or audio input found."}), 400


        # Handle STT errors right away
        if stt_error:
            response_text = stt_error
            _, final_audio_url = save_response_to_wav(response_text, "stt_error")
            return jsonify({"response": response_text, "audio_url": final_audio_url})

        # Handle empty input after potential STT or direct text input
        if not user_input_text:
            response_text = "I didn't catch that. Could you please repeat or type your message?"
            _, final_audio_url = save_response_to_wav(response_text, "stt_empty")
            return jsonify({"response": response_text, "audio_url": final_audio_url})

        # Get vent mode flag (Handle both form-data and json)
        is_vent_mode = False
        try:
            if request.content_type.startswith('multipart/form-data'):
                 # For FormData, data comes from request.form, not request.get_json()
                is_vent_mode = request.form.get("vent_mode", "false").lower() == "true"
            elif request.is_json:
                 data = request.get_json()
                 is_vent_mode = data.get("vent_mode", False) if data else False
        except Exception as e:
             log_error(f"Error reading vent_mode flag: {e}")
             is_vent_mode = False # Default to false on error
        print(f"Vent Mode: {is_vent_mode}")


        # --- Safety Check: Crisis Detection ---
        user_input_lower = user_input_text.lower()
        if any(keyword in user_input_lower for keyword in CRISIS_KEYWORDS):
            # ... (Crisis handling - same as before) ...
            print("🚨 Crisis Keyword Detected!")
            crisis_message = ( # Make sure this message is complete and appropriate
                "I hear that you're in a lot of pain right now, and I want you to know you're not alone. "
                "As an AI, I'm not equipped to provide the help you need in a crisis. "
                "Please reach out to trained professionals who can support you immediately. \n"
                "You can contact:\n"
                "- National Crisis and Suicide Lifeline: Call or text 988 (US & Canada)\n"
                "- Crisis Text Line: Text HOME to 741741 (US & Canada), 85258 (UK)\n"
                "- Befrienders Worldwide: https://www.befrienders.org/ (International Directory)\n"
                "Please reach out now. Help is available."
            )
            log_error(f"Crisis keyword detected in input: '{user_input_text}'")
            _, final_audio_url = save_response_to_wav(crisis_message, "crisis_response")
            is_crisis = True
            response_text = crisis_message
            intent_handled = True # Prevent further processing

        # --- LTM Intent Handling --- ADDED ---
        if not intent_handled:
            # Explicit LTM Store Command
            # Regex improved slightly to handle optional "that" and possessives like "my" more cleanly
            ltm_store_match = re.match(r"remember\s+(?:that\s+)?(?:my\s+)?(.+?)\s+(?:is|are)\s+(.*)", user_input_text, re.IGNORECASE)
            if ltm_store_match:
                print("Intent: Store LTM Item")
                key = ltm_store_match.group(1).strip().replace(" ", "_") # Normalize key: lowercase optional, replace space mandatory
                value = ltm_store_match.group(2).strip()
                if key and value:
                    # Optional: Add simple validation or sanitization for key/value if needed
                    long_term_memory[key.lower()] = value # Store key as lowercase for consistent lookup
                    save_ltm() # Save immediately
                    response_text = f"Okay, I'll remember that your {key.replace('_',' ')} is {value}."
                    intent_handled = True
                else:
                    response_text = "I seem to be missing the piece of information or what to call it."
                    intent_handled = True # Still handle it as an attempt

            # Explicit LTM Recall Command
            # Regex improved slightly to handle optional "my" and optional question mark
            ltm_recall_match = re.match(r"what(?:'s| is)\s+(?:my\s+)?(.+?)\??$", user_input_text, re.IGNORECASE)
            if ltm_recall_match and not intent_handled:
                 print("Intent: Recall LTM Item")
                 key = ltm_recall_match.group(1).strip().replace(" ", "_") # Normalize key
                 # Lookup using lowercase key
                 if key.lower() in long_term_memory:
                     value = long_term_memory[key.lower()]
                     response_text = f"I remember that your {key.replace('_',' ')} is {value}."
                 else:
                     response_text = f"Sorry, I don't have anything stored for '{key.replace('_',' ')}'."
                 intent_handled = True

            # Command to list LTM (for debugging/user clarity)
            if not intent_handled and user_input_lower == "what do you remember":
                 print("Intent: List LTM Items")
                 if long_term_memory:
                     # Format keys nicely for display
                     remembered_items = [f"- {k.replace('_',' ').capitalize()}: {v}" for k, v in long_term_memory.items()]
                     response_text = "Here's what I remember:\n" + "\n".join(remembered_items)
                 else:
                     response_text = "I haven't specifically remembered any key facts yet. Feel free to tell me using 'remember my [thing] is [value]'."
                 intent_handled = True

        # --- Main LLM Conversation (If no intent handled) ---
        if not intent_handled:
            # --- Process Input (Sentiment, Placeholder Voice) ---
            text_sentiment_tag = analyze_text_sentiment(user_input_text)
            print(f"Text Sentiment: {text_sentiment_tag}")
            voice_sentiment_tag = None
            # Placeholder voice analysis only makes sense if audio was provided originally
            # We don't have the audio path here anymore if using tempfile 'with' block earlier
            # If needed, pass audio_file_path down or do analysis within the 'with' block.
            # For now, voice_sentiment_tag remains None.
            # if audio_file_path:
            #     voice_sentiment_tag = analyze_voice_sentiment_placeholder(audio_file_path)

            # --- Generate Dynamic System Prompt (Includes LTM) ---
            memory_variables = chat_memory.load_memory_variables({})
            current_chat_history = memory_variables.get('chat_history', [])
            # Pass the current state of global long_term_memory
            dynamic_system_content = create_mindpal_prompt(text_sentiment_tag, voice_sentiment_tag, is_vent_mode, long_term_memory)

            # Construct messages for LLM
            messages_for_llm = [SystemMessage(content=dynamic_system_content)] + current_chat_history + [HumanMessage(content=user_input_text)]

            # --- LLM Call ---
            try:
                print("MindPal is thinking (invoking LLM)...")
                llm_response_obj = groq_chat.invoke(messages_for_llm)
                response_text = llm_response_obj.content if llm_response_obj else "..."
                print(f"LLM Raw Response: '{response_text}'")
            except Exception as e:
                print(f"❌ Error during LLM conversation: {e}")
                log_error(f"Error invoking LLM: {e}", exc_info=True)
                response_text = "I'm having a little trouble processing that right now. Could you perhaps rephrase?"

            # --- Activity Suggestion ---
            final_response = response_text
            # Suggest only if not in vent mode, response exists, and mood maybe not Positive
            if not is_vent_mode and response_text and text_sentiment_tag != "Positive":
                 if random.random() < 0.4: # Suggest sometimes
                     suggestions = activity_suggestions.get(text_sentiment_tag, [])
                     if suggestions:
                         chosen_suggestion = random.choice(suggestions)
                         # Append suggestion gently
                         final_response += f"\n\n{chosen_suggestion}"
                         suggestion_added = True
                         print(f"Added suggestion: '{chosen_suggestion}'")
            else:
                 final_response = response_text # Ensure final_response is set

        else: # An intent WAS handled (LTM command or crisis)
             final_response = response_text # Use the response set by the intent handler

        # --- Post-Processing ---
        if not final_response: # Catchall if something went wrong
            log_error(f"No final response generated for input: '{user_input_text}'")
            final_response = "I'm not quite sure how to respond to that. Could you tell me more?"

        # Save context to SESSION memory (use transcribed text and final response)
        if not is_crisis and isinstance(final_response, str): # Don't save crisis interactions or non-strings
            try:
                chat_memory.save_context({"human_input": user_input_text}, {"output": final_response})
                print("Context saved to Session memory.")
            except Exception as mem_save_err:
                 log_error(f"Error saving context to session memory: {mem_save_err}", exc_info=True)


        # Generate TTS for the final response
        print(f"Generating TTS for: '{final_response[:100]}...'")
        _, final_audio_url = save_response_to_wav(final_response, "mindpal_response")

        # Temporary audio file cleanup is handled by the 'with tempfile.TemporaryDirectory' block exiting

        end_time = time.time()
        print(f"Request processed in {end_time - start_time:.2f} seconds.")
        print(f"Returning Response: '{final_response}'")
        print(f"Audio URL: {final_audio_url}")
        print("--- End MindPal Chat Request ---")

        # Return JSON Response
        response_payload = {
            "response": final_response,
            "audio_url": final_audio_url,
            "suggestion_added": suggestion_added,
            "is_crisis": is_crisis
        }
        return jsonify(response_payload)

    # --- Global Exception Handler for the Route ---
    except Exception as e:
        log_error(f"Unexpected error in /chat_pa route: {e}", exc_info=True)
        # Temporary audio file cleanup might have already happened or failed,
        # but we don't have the path here anymore if 'with' block was used and exited.
        error_response = "Apologies, I encountered an unexpected issue. Please try again."
        _, audio_url = save_response_to_wav(error_response, "internal_error")
        # Return a generic error response
        return jsonify({"response": error_response, "audio_url": audio_url, "error": "Internal server error"}), 500


# --- Main Execution --- MODIFIED ---
if __name__ == "__main__":
    initialize_assistant()
    if not groq_chat or not chat_memory or not vader_analyzer or not speech_recognizer:
         print("🚨 MindPal cannot start due to critical initialization errors. Please check logs.")
    else:
        print("Starting Flask development server for MindPal on http://127.0.0.1:5050...")
        print("LTM will be loaded/saved from:", LTM_FILE_PATH)
        print("Press Ctrl+C to stop.")
        try:
             # Run with debug=False for proper shutdown hook execution
            app.run(host='127.0.0.1', port=5050, debug=False)
        finally: # This block executes when the server stops (e.g., Ctrl+C)
            # --- Shutdown Hook ---
            print("\nServer shutting down...")
            if chat_memory:
                try:
                    print("Saving final chat history...")
                    messages_to_save = chat_memory.chat_memory.messages
                    save_chat_history(messages_to_save)
                except Exception as e:
                     log_error(f"Error saving chat history on shutdown: {e}", exc_info=True)
            # --- Save LTM on Shutdown --- ADDED ---
            try:
                print("Saving final Long-Term Memory...")
                save_ltm()
            except Exception as e:
                log_error(f"Error saving LTM on shutdown: {e}", exc_info=True)
            print("MindPal application finished.")