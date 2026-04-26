# DocQ&A Bot — RAG-based Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language 
questions over a collection of documents and receive accurate, grounded answers with source 
citations. The system retrieves only relevant document chunks before generating an answer, 
ensuring responses are grounded in the provided documents rather than the model's training data.

## Tech Stack

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.11 |
| RAG Framework | LangChain | 0.2.16 |
| LangChain Community | langchain-community | 0.2.16 |
| LangChain Groq | langchain-groq | 0.1.9 |
| LLM | Groq API — llama-3.3-70b-versatile | — |
| Embeddings | sentence-transformers (local) | 3.0.1 |
| Embedding Model | all-MiniLM-L6-v2 | — |
| Vector Database | FAISS (CPU) | 1.8.0 |
| PDF Loader | PyPDF | 4.3.1 |
| DOCX Loader | docx2txt | 0.8 |
| Environment Variables | python-dotenv | 1.0.1 |

## Architecture Overview

```
Documents (PDF / TXT / DOCX)
           │
           ▼
    ┌─────────────┐
    │  ingest.py  │
    └─────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
  Load Docs   Extract Text
     │
     ▼
  Chunk Text
  (RecursiveCharacterTextSplitter)
  chunk_size=800, overlap=150
     │
     ▼
  Generate Embeddings
  (all-MiniLM-L6-v2, runs locally)
     │
     ▼
  Store in FAISS Index
  (persisted to disk → /vectorstore)
  
  
User Query
     │
     ▼
    ┌─────────────┐
    │  query.py   │
    └─────────────┘
     │
     ▼
  Embed Query
  (same all-MiniLM-L6-v2 model)
     │
     ▼
  Similarity Search
  (FAISS — top-4 chunks retrieved)
     │
     ▼
  Build Prompt
  (retrieved chunks + user question)
     │
     ▼
  Groq LLaMA 3.3-70b
  (generates grounded answer)
     │
     ▼
  Answer + Source Citations
```

## Chunking Strategy

**Strategy used: RecursiveCharacterTextSplitter**
- `chunk_size = 800`
- `chunk_overlap = 150`

**Why recursive over fixed-size splitting:**
Fixed-size splitting cuts text at arbitrary character positions, often breaking mid-sentence 
or mid-thought. RecursiveCharacterTextSplitter tries to split at `\n\n` (paragraphs) first, 
then `\n` (newlines), then `. ` (sentences), and only falls back to character-level splitting 
as a last resort. This keeps each chunk semantically coherent.

**Why chunk_size=800:**
I tested chunk_size=500 — answers felt incomplete as important context was cut off. 
chunk_size=1000 pulled in too much irrelevant text during retrieval. 800 gave the best 
balance between focused retrieval and complete answers.

**Why overlap=150:**
Without overlap, context at chunk boundaries is lost entirely. 150 characters of overlap 
ensures that sentences split across two chunks are still fully represented in at least one 
of them.

## Embedding Model and Vector Database

**Embedding Model — `all-MiniLM-L6-v2`**

Chosen because it runs entirely locally via the `sentence-transformers` library — no API 
call, no cost, and no latency from an external service. Despite being lightweight (384 
dimensions), it performs well on semantic similarity tasks for general and technical text. 
It is one of the most widely benchmarked open-source embedding models for retrieval tasks.

**Vector Database — FAISS (CPU)**

FAISS was chosen over ChromaDB or Qdrant because it requires no external server or process 
to run. It persists the index to disk and loads it back in milliseconds. For a 5-document 
knowledge base with ~700 chunks, FAISS is more than sufficient. The clear separation between 
`ingest.py` (indexing) and `query.py` (retrieval) ensures the index is only rebuilt when 
documents change, not on every query.

**LLM — LLaMA 3.3-70b via Groq API**

LLaMA 3 is Meta's open-source model, satisfying the open-source model requirement. Groq 
was chosen over running Ollama locally because it requires no local GPU, has a generous 
free tier, and provides significantly faster inference. This makes the system accessible 
without any special hardware requirements.

## Setup Instructions

```bash
# 1. Clone the repository
https://github.com/alekhya-1-24/Doc-QA-Bot.git
cd Doc-Q&A-Bot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and paste your Groq API key (see Environment Variables section below)

# 5. Verify your documents are in the /data folder
ls data/
# Should show your PDF, TXT, and DOCX files

# 6. Index the documents (run once, or whenever documents change)
python ingest.py

# 7. Start the bot
python main.py          # CLI mode
```

## Environment Variables

Create a `.env` file in the project root (never commit this file — it is in `.gitignore`):

```
GROQ_API_KEY=your_groq_api_key_here
```

**How to get your Groq API key:**
1. Go to https://console.groq.com
2. Sign up for a free account (no credit card required)
3. Navigate to API Keys → Create API Key
4. Copy the key and paste it into your `.env` file

The `.env.example` file in this repo shows the required format with a placeholder value. 
Never paste your actual key into `.env.example` or any file that gets committed to GitHub.

## Example Queries

| # | Query | Expected Answer Theme |
|---|---|---|
| 1 | What problem does Bitcoin solve? | Double spending, peer-to-peer electronic cash, removing trusted third parties |
| 2 | What does Sun Tzu say about knowing your enemy? | Intelligence, spies, understanding enemy position and strategy |
| 3 | What does the declaration say about the right to education? | Free elementary education, equal access, parental choice |
| 4 | How does Bitcoin prevent double spending? | Proof-of-work, blockchain, longest chain rule |
| 5 | What does Sun Tzu say about using spies? | Five types of spies, importance of secret information, divine manipulation |
| 6 | What is the latest iPhone model? | Bot responds: "I couldn't find this in the provided documents" |

Query 6 is intentional — it demonstrates the bot's hallucination prevention. The system 
will not answer from general knowledge if the information is not present in the indexed documents.

## Known Limitations

**No conversation memory**
Each query is processed independently. If you ask a follow-up question like "Can you 
elaborate on that?", the bot has no memory of the previous answer. Implementing 
conversational memory would require passing chat history into the retrieval step 
using LangChain's ConversationalRetrievalChain.

**Tables and images not parsed**
PDF tables and embedded images are not extracted. The loaders only process prose text. 
Documents that are primarily tables or scanned images will not index correctly.

**TXT files have no page metadata**
Unlike PDFs, plain text files do not have page numbers. Source citations for TXT files 
will show "page N/A". The answer text itself may reference section numbers if they exist 
in the document.

**DOCX loaded as single document**
The Docx2txtLoader treats an entire DOCX file as one document object regardless of length. 
Page metadata is unavailable for DOCX files. The content is still chunked and indexed 
correctly — only the citation detail is affected.

**Context window limit**
With TOP_K=4 chunks passed to the LLM, very broad questions may not get enough context 
to produce a complete answer. Increasing TOP_K improves recall but also increases the 
chance of irrelevant chunks being included.

**Language**
The system works best on English documents. Mixed-language or heavily formatted technical 
PDFs may produce degraded results due to encoding issues during text extraction.
