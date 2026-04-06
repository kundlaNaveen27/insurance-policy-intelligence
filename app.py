import streamlit as st
import os
from rag_engine import initialize_rag, answer_question

# ── PAGE SETUP ──────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ── HEADER ──────────────────────────────────────────
st.title("🏦 Insurance Policy Intelligence System")
st.markdown("""
Ask any question about your insurance policies and get 
instant AI-powered answers with source citations.
""")

st.divider()

# ── INITIALIZE RAG (once per session) ───────────────
# st.session_state stores data between interactions
# Without this, RAG would reinitialize every question
if "vectorstore" not in st.session_state:
    with st.spinner("Connecting to policy database..."):
        vectorstore, llm = initialize_rag()
        st.session_state.vectorstore = vectorstore
        st.session_state.llm = llm
    st.success("✅ Connected to policy database!")

# ── CHAT HISTORY ────────────────────────────────────
# store conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.header("📤 Upload Policy Documents")

    # file uploader — accepts multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload insurance PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Documents"):
            with st.spinner("Indexing documents into Pinecone..."):

                import tempfile
                from indexer import clear_index, load_and_index_pdfs

                # clear old documents first so stale data doesn't mix in
                clear_index()

                # create temp folder
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # save each uploaded file
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(
                            tmp_dir,
                            uploaded_file.name
                        )
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                    # index new documents into Pinecone
                    load_and_index_pdfs(tmp_dir)

                    # refresh session state
                    st.session_state.vectorstore, st.session_state.llm = initialize_rag()

            st.success(f"✅ Indexed {len(uploaded_files)} documents!")

    st.divider()

    st.header("📋 Example Questions")
    examples = [
        "What does the travel insurance cover?",
        "What are the deductibles?",
        "Does life insurance cover pre-existing conditions?",
        "What is the out of pocket maximum?",
        "What services are excluded?",
        "How do I file a claim?"
    ]
    for example in examples:
        if st.button(example, key=example):
            st.session_state.current_question = example

    st.divider()

    st.header("📚 Loaded Documents")
    import os
    if os.path.exists("documents"):
        docs = os.listdir("documents")
        for doc in docs:
            if doc.endswith(".pdf"):
                st.markdown(f"- {doc}")
    else:
        st.markdown("No documents loaded yet")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗄️ Clear Index (Remove All Docs)"):
        with st.spinner("Clearing Pinecone index..."):
            from indexer import clear_index
            clear_index()
            st.session_state.vectorstore, st.session_state.llm = initialize_rag()
            st.session_state.messages = []
        st.success("✅ Index cleared! Upload new documents to get started.")

# ── DISPLAY CHAT HISTORY ────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"📍 Sources: {', '.join(message['sources'])}")

# ── QUESTION INPUT ───────────────────────────────────
question = st.chat_input("Ask a question about your policies...")

# handle example button clicks
if "current_question" in st.session_state:
    question = st.session_state.current_question
    del st.session_state.current_question

# ── PROCESS QUESTION ─────────────────────────────────
if question:
    # show user message
    with st.chat_message("user"):
        st.markdown(question)

    # add to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            answer, sources = answer_question(
                question,
                st.session_state.vectorstore,
                st.session_state.llm
            )

        st.markdown(answer)
        if sources:
            st.caption(f"📍 Sources: {', '.join(sources)}")

    # add to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })