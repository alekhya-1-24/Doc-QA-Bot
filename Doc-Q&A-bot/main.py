import os
from query import load_vectorstore, build_qa_chain, ask, format_sources

VECTORSTORE_PATH = "vectorstore"

def main():
    print("=" * 55)
    print("       Document Q&A Bot — RAG Pipeline")
    print("  Type 'exit' or 'quit' to stop")
    print("=" * 55)

    if not os.path.exists(VECTORSTORE_PATH):
        print("\nVector store not found.")
        print("Run 'python ingest.py' first to index your documents.")
        return

    print("\nLoading vector store and model...")
    try:
        vectorstore = load_vectorstore()
        qa_chain = build_qa_chain(vectorstore)
        print("Ready! Ask me anything about your documents.\n")
    except Exception as e:
        print(f"Startup error: {e}")
        return

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\nSearching documents...\n")

        try:
            answer, sources = ask(qa_chain, question)
            
            # Show retrieved chunks
            print("=" * 55)
            print("RETRIEVED CHUNKS:")
            print("=" * 55)
            for i, doc in enumerate(sources, 1):
                filename = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                if isinstance(page, int):
                    page = page + 1
                print(f"\nChunk {i} — {filename} (page {page}):")
                print("-" * 40)
                print(doc.page_content[:300] + "...")  # show first 300 chars
            
            print("\n" + "=" * 55)
            print("ANSWER:")
            print("=" * 55)
            print(f"{answer}")
            print(f"\nSources:\n{format_sources(sources)}")

        except Exception as e:
            print(f"Error generating answer: {e}")

        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()