# 🧠 MindPal – Your AI Mental Wellness Companion

MindPal is an AI-powered, voice-enabled mental wellness companion designed to support students and professionals struggling with stress, burnout, and emotional isolation. It engages users in meaningful conversation, detects mood, remembers key facts, and offers gentle, personalized support—without the cost or stigma of traditional therapy.

---

## 🚨 Problem Statement

There’s a growing mental health crisis, especially among students and working professionals. Many suffer in silence due to:

- Stress, burnout, and isolation.
- Limited access to mental health professionals.
- The high cost and stigma around therapy.
- Lack of consistent, personalized emotional support.

---

## 💡 Our Solution: MindPal

MindPal provides a safe, judgment-free AI companion that:

- Listens and responds with empathy via **text or voice**.
- Detects **emotional state** using sentiment analysis.
- Suggests calming or uplifting **wellness activities**.
- Remembers important facts for **personalized support**.
- Flags critical messages and redirects to **crisis resources**.

---

## 🎯 Use Cases

- Daily mood check-ins and emotional tracking
- Support during academic or workplace burnout
- Safe space for users to vent without judgment
- Voice-based hands-free conversations
- Early detection of emotional distress or crisis signals

---

## ⚙️ Tech Stack

| Layer           | Technology                             |
|----------------|-----------------------------------------|
| **LLM & AI**    | Groq (LLaMA 3) + LangChain              |
| **Sentiment**   | VADER (text sentiment analysis)         |
| **Voice I/O**   | Google STT, HuggingFace TTS (VITS)      |
| **Memory**      | LangChain session memory + JSON-based LTM |
| **Backend**     | Flask (Python)                          |
| **Frontend**    | HTML, CSS, JS (Voice UI + Chat UI)      |

---

## 🧩 Features

- **Conversational AI**: Human-like, empathetic responses.
- **Long-Term Memory**: Remembers personal facts across sessions.
- **Mood Detection**: Understands how you feel using language cues.
- **Vent Mode**: Just listens—no advice, no suggestions.
- **Crisis Detection**: Flags sensitive keywords, offers real help.
- **Voice Support**: Speech-to-text and text-to-speech built-in.

---

## 🚧 Dependencies & Showstoppers

- Requires **Groq API key** for LLM functionality.
- STT depends on **internet access** for Google API.
- No database yet—**LTM is stored in a local JSON file**.
- Voice tone sentiment is currently a placeholder.
- **Not a medical tool**—does not replace professional care.

---

## 🛠️ Installation & Setup

1. **Clone the Repo**
   ```bash
   git clone https://github.com/Celest14l/MindPal.git
   cd MindPal
