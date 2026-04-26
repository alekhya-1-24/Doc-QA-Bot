import os
from dotenv import load_dotenv

load_dotenv()  # must be before anything else

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not found in the context below, respond with:
"I couldn't find information about this in the provided documents."
Do NOT use your own training knowledge to answer. Only use the context given.

Context:
{context}

Question: {question}

Give a clear, well-structured answer. At the end, mention which document(s) the information came from.
"""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def build_qa_chain(vectorstore):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure your .env file exists "
            "in the project root and contains: GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def ask(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources


def format_sources(sources):
    seen = set()
    lines = []
    for doc in sources:
        meta = doc.metadata
        filename = os.path.basename(meta.get("source", "Unknown"))
        page = meta.get("page", "N/A")
        if isinstance(page, int):
            page = page + 1
        key = f"{filename}_p{page}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  - {filename}  (page {page})")
    return "\n".join(lines) if lines else "  - Source metadata unavailable"