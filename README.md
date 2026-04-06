# 🏦 Insurance Policy Intelligence System

AI-powered tool that lets financial advisors instantly 
find answers from insurance policy documents using 
natural language — powered by RAG architecture.

## 🚀 Live Demo
👉 [Try it here](https://insurance-policy-intelligence-di7ezpgfuduexnwxpkxi89.streamlit.app/)

## The Problem
Financial advisors spend 15-20 minutes per customer 
query manually searching policy documents. This system 
reduces that to seconds.

## How It Works
1. Upload insurance PDFs through the web interface
2. Documents indexed into Pinecone vector database
3. Ask any natural language question
4. System finds relevant policy sections semantically
5. LLaMA 3.3 70B answers based on actual policy text
6. Source citations show exactly which document and page

## Tech Stack
- **Pinecone** — cloud vector database (production grade)
- **LangChain** — AI framework connecting all components
- **Groq + LLaMA 3.3 70B** — AI response generation
- **SentenceTransformers** — text embeddings
- **Streamlit** — web interface with file upload
- **PyPDF** — PDF processing

## Architecture
```
PDFs → PyPDFLoader → chunks → embeddings → Pinecone (cloud)
Question → embeddings → Pinecone search → relevant chunks → LLaMA → answer
```

## Setup
```bash
pip install -r requirements.txt
```

Add to .env:
```
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
```

Run indexer first (once):
```bash
python indexer.py
```

Launch app:
```bash
streamlit run app.py
```

## ⚠️ Current Limitations
- No image/chart support — only text extracted
- Math formulas may not extract correctly
- Encrypted PDFs cannot be processed
- Scanned PDFs return 0 chunks (OCR needed)
- Table structure lost during extraction

## 🔮 Future Improvements
- Add OCR for scanned PDFs (pytesseract)
- Add vision AI for image understanding
- Upgrade to pdfplumber for better tables
- Deploy with Azure OpenAI for HIPAA compliance
- Add conversation memory across sessions

## Real World Application
This architecture mirrors what companies like 
First Citizens Bank and IU Health use for internal 
document intelligence systems — processing thousands 
of policy documents with sub-second retrieval.

## What I Learned
- Production RAG with persistent Pinecone vector storage
- LangChain document loaders and text splitters
- Cosine similarity vs L2 distance for text search
- Source citation and metadata tracking
- Streamlit session state and file upload handling
- HIPAA compliance considerations for healthcare AI
