import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

load_dotenv()


# Same embedding class as indexer.py
# Must be identical — same model, same settings
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def initialize_rag():
    """
    Connects to existing Pinecone index
    Returns vectorstore and llm ready to use
    """
    print("Connecting to Pinecone...")
    embeddings = SentenceTransformerEmbeddings()

    # connect to EXISTING index — don't create new one
    vectorstore = PineconeVectorStore(
        index_name="insurance-policies",
        embedding=embeddings
    )

    # connect to Groq AI
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    print("✅ Connected!")
    return vectorstore, llm


def answer_question(question, vectorstore, llm, top_k=3):
    """
    Takes a question → finds relevant chunks → gets AI answer
    """
    # search Pinecone for relevant chunks
    # similarity_search finds chunks closest to question
    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        return "No relevant information found.", []

    # build context from retrieved chunks
    context = ""
    sources = []

    for i, doc in enumerate(results):
        context += f"\nSection {i+1}:\n{doc.page_content}\n"

        # get source filename from metadata
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        sources.append(f"{source} (page {page})")

    # send context + question to AI
    messages = [
        SystemMessage(content="""You are an expert insurance and 
        financial policy analyst. Answer questions based ONLY on 
        the provided context from policy documents.
        
        Always:
        - Be precise and cite specific policy details
        - Use clear simple language
        - If something is not in the context say so
        - Structure your answer clearly"""),

        HumanMessage(content=f"""
Context from policy documents:
{context}

Question: {question}

Please provide a clear, accurate answer based on the context.""")
    ]

    response = llm.invoke(messages)

    return response.content, sources


if __name__ == "__main__":
    # quick test
    vectorstore, llm = initialize_rag()

    test_question = "What is the main topic of these documents?"
    print(f"\nQuestion: {test_question}")

    answer, sources = answer_question(test_question, vectorstore, llm)
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {sources}")