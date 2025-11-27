from .embeddings import embed_texts
from .vectorstore import search
from .llm_client import generate_answer

def rag_answer(query: str):
    print("Create Embeddings")
    embeddings = embed_texts(query)
    print("Searching Results")
    results = search(embeddings)
    print("Joining Context")
    context = "\n".join([hit.payload["text"] for hit in results.points])

    final_prompt = f"""
    Use the following context to answer the question:

    {context}

    Question: {query}
    """
    print("Generating Answers")
    return generate_answer(final_prompt, context)
