# RAG-Based LLM Chatbot
RAG-Based LLM Chatbot using LLama-3.2 model, running locally using Docker.

### Getting started:

#### 1. Clone the Respository 
```bash
git clone https://github.com/sunil-thapa99/RAG_Chatbot.git
cd RAG_Chatbot
```

#### 2. Create a virtual environment either using anaconda or venv
```bash
conda create --name rag_chatbot python=3.10
conda activate rag_chatbot
```

#### 3. Download and install Ollama
[Ollama](https://ollama.com/): An open-source repository for models including LLM. 
```bash
ollama run llama3.2
```

#### 4. Setup docker and run QDrant
```bash
docker run -p 6333:6333 -d qdrant/qdrant
```

#### 5. Install all the dependencies
```bash
pip install -r requirements.txt
```

#### 6. Run streamlit application
```bash
streamlit run app.py
```

### Files Information:
- `app.py`: Streamlit application for the chatbot.
- `embeddings.py`: Document ingestion, and creates embedding using BAAI/bge-small-en.
- `chatbot.py`: Retrieves relevant content according to query.
- `requirements.txt`: List of all the dependencies required to run the application.