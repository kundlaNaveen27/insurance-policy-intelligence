import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
load_dotenv()

# ── CUSTOM EMBEDDING CLASS ──────────────────────────
# LangChain needs embeddings in a specific format
# We wrap our SentenceTransformer to work with LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # convert list of texts to list of number lists
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        # convert single question to numbers
        return self.model.encode([text])[0].tolist()


def clear_index(index_name="insurance-policies"):
    """
    Deletes all vectors from the Pinecone index.
    The index itself remains — only the stored documents are removed.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        try:
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            print(f"✅ Cleared all vectors from '{index_name}'")
        except Exception as e:
            # Pinecone throws NotFoundException when index is already empty
            print(f"Index was already empty or could not be cleared: {e}")
    else:
        print(f"Index '{index_name}' does not exist — nothing to clear")


def create_pinecone_index(index_name="insurance-policies"):
    """
    Creates a Pinecone index if it doesn't exist
    Think of index = a database table in Pinecone
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # get list of existing indexes
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,        # must match embedding size
            metric="cosine",      # similarity measurement
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created!")
    else:
        print(f"Index '{index_name}' already exists")

    return pc.Index(index_name)


def load_and_index_pdfs(documents_folder="documents"):
    """
    Reads all PDFs from folder and stores in Pinecone
    """
    embeddings = SentenceTransformerEmbeddings()

    # text splitter — same as before
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []

    # loop through every PDF in documents folder
    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(documents_folder, filename)
            print(f"Loading: {filename}")

            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()

                if not pages:
                    print(f"  ⚠️ No text extracted from {filename}")
                    print(f"  → PDF might be scanned or encrypted")
                    continue

                chunks = splitter.split_documents(pages)

                if not chunks:
                    print(f"  ⚠️ No chunks created from {filename}")
                    continue

                print(f"  → {len(chunks)} chunks created")

                for chunk in chunks:
                    chunk.metadata["source"] = filename

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
                continue

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Storing in Pinecone...")

    # store all chunks in Pinecone
    # this creates embeddings AND stores them
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        index_name="insurance-policies"
    )

    print("✅ All documents indexed in Pinecone!")
    return vectorstore


if __name__ == "__main__":
    print("🚀 Starting indexing process...\n")
    
    # Delete old index and create fresh one
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if "insurance-policies" in existing_indexes:
        print("Deleting old index...")
        pc.delete_index("insurance-policies")
        print("Deleted!")
    
    create_pinecone_index()
    load_and_index_pdfs()
    print("\n✅ Done! Your documents are now in Pinecone.")