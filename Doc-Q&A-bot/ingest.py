import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = "data"
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents(data_dir):
    docs = []
    supported = {".pdf", ".txt", ".docx"}

    print("=" * 55)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 55)

    for filename in os.listdir(data_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            print(f"  Skipping unsupported file: {filename}")
            continue

        filepath = os.path.join(data_dir, filename)
        print(f"\nLoading: {filename}")

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(filepath)

            loaded = loader.load()
            docs.extend(loaded)
            print(f"  -> {len(loaded)} page(s) loaded")
            print(f"  -> Preview: {loaded[0].page_content[:80].strip()}...")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    print(f"\nTotal pages loaded across all documents: {len(docs)}")
    return docs


def chunk_documents(docs):
    print("\n" + "=" * 55)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("=" * 55)
    print("Strategy : RecursiveCharacterTextSplitter")
    print("Chunk size: 800 characters")
    print("Overlap   : 150 characters")
    print("Reason    : Preserves semantic meaning by splitting")
    print("            at paragraphs first, then sentences.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"\nSample chunk preview:")
    print("-" * 40)
    print(chunks[0].page_content[:200])
    print("-" * 40)
    return chunks


def build_vectorstore(chunks):
    print("\n" + "=" * 55)
    print("STEP 3: EMBEDDING AND STORING IN FAISS")
    print("=" * 55)
    print("Embedding model : all-MiniLM-L6-v2 (runs locally)")
    print("Vector database : FAISS (persisted to disk)")
    print("\nGenerating embeddings for all chunks...")
    print("(This may take a minute...)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"\nTotal chunks embedded and stored: {len(chunks)}")
    print(f"Vector store saved to: '{VECTORSTORE_PATH}/'")
    print("\n" + "=" * 55)
    print("INDEXING COMPLETE")
    print("Run 'python main.py' to start querying.")
    print("=" * 55)


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}' folder not found. Create it and add your documents.")
        exit(1)

    docs = load_documents(DATA_DIR)

    if not docs:
        print("No documents loaded. Add PDF, TXT, or DOCX files to the /data folder.")
        exit(1)

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)