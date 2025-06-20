conda create -n lang10 python=3.11 -y

conda activate lang10

pip install -r requirements.txt


<!-- langchain==0.3.13
langchain-core==0.3.28
langchain-community==0.3.13
langchain-openai==0.2.14 -->


streamlit run app.py 


# 🤖 Customer AI Chatbot (Groq + FAISS + Streamlit)

This AI chatbot enables interactive Q&A with customer-specific documents using Groq's LLaMA 3, FAISS vector search, and Streamlit UI. It supports both text and voice-based queries with multilingual support and natural-sounding audio responses.

---

## 🔄 Project Workflow

1. **User Input**  
   Users submit a query via text or voice. Voice input is transcribed to text using Whisper.

2. **Language Detection & Translation**  
   Non-English queries are translated using Deep Translator.

3. **Document Search via FAISS**  
   FAISS retrieves relevant document chunks from uploaded PDFs.

4. **LLM Processing via Groq**  
   The LLaMA 3 model (accessed via Groq API + LangChain) generates a response based on the query and document context.

5. **Response Output**  
   - Text responses are displayed on the UI.
   - Optional: The response is converted to audio via ElevenLabs.

---

## 🧱 Project Architecture

```text
📂 customer-chatbot/
│
├── app.py                  # Streamlit app UI
├── agent.py                # LLM response handler using Groq
├── vectorstore/            # FAISS index and embeddings
├── utils.py                # Helper functions for audio, PDF parsing, translation
├── requirements.txt        # Project dependencies
├── .env                    # API keys
└── build.sh                # Render deployment script



---

## ⚙️ Tools and Technologies

| Component          | Technology Used              |
|--------------------|------------------------------|
| 🧠 LLM             | Groq (LLaMA 3 via LangChain) |
| 🗃️ Vector DB      | FAISS                        |
| 🧬 Embeddings      | HuggingFace Transformers     |
| 🎙️ Speech-to-Text | Whisper (local)              |
| 🔊 Text-to-Speech  | ElevenLabs                   |
| 🌐 Translation     | Deep Translator (Google API) |
| 📄 PDF Parsing     | PyMuPDF (fitz)               |
| 🎛️ Audio Input    | Sounddevice + SciPy          |
| 🌐 UI Framework    | Streamlit                    |
| 🔐 Env Management  | python-dotenv                |

---

## 🧪 Features

✅ Answer user queries based on preloaded company documents  
✅ Accept text-based or voice-based inputs  
✅ Convert voice to text (Whisper) and text to audio (ElevenLabs)  
✅ Translate queries and responses into multiple languages  
✅ Secure .env-based API key management  
✅ Easy local and Render deployment  

---

## 🛠️ Build From Scratch

### 🧬 Clone the Repository

```bash
git clone https://github.com/yourusername/customer-chatbot.git
cd customer-chatbot


## Set Up Environment
conda create -n lang10 python=3.11 -y

conda activate lang10


### Add API Keys
## Create a .env file:
GROQ_API_KEY=your_groq_key
ELEVENLABS_API_KEY=your_elevenlabs_key


## Run the App

streamlit run app.py

