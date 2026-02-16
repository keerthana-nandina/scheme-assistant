import os
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ==========================================================
# Configuration
# ==========================================================

SIMILARITY_THRESHOLD = 0.40
DATA_FOLDER = "data"
INDEX_FOLDER = "faiss_index"

SCHEME_DESCRIPTIONS = {
    "TREAD": "Trade Related Entrepreneurship Assistance and Development scheme for women providing 30 percent government grant and NGO routed loans.",
    "NSSH": "National Scheduled Caste and Scheduled Tribe Hub supporting SC ST entrepreneurs with 4 percent procurement target.",
    "CEGSSC": "Credit Enhancement Guarantee Scheme for Scheduled Castes providing loan guarantee cover through IFCI.",
    "NHDP_HANDLOOM": "National Handloom Development Programme supporting weavers including workshed and welfare schemes.",
    "CHCDS": "Comprehensive Handicrafts Cluster Development Scheme supporting artisan clusters with infrastructure."
}


# ==========================================================
# Initialization (Cached)
# ==========================================================

@st.cache_resource
def initialize_system():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = []

    for file_name in os.listdir(DATA_FOLDER):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file_name))
            pages = loader.load()

            scheme_name = file_name.replace(".pdf", "")
            for page in pages:
                page.metadata["scheme"] = scheme_name

            documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)

    if os.path.exists(INDEX_FOLDER):
        vector_store = FAISS.load_local(
            INDEX_FOLDER,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(split_docs, embedding_model)
        vector_store.save_local(INDEX_FOLDER)

    scheme_embeddings = embedding_model.embed_documents(
        list(SCHEME_DESCRIPTIONS.values())
    )

    llm = OllamaLLM(
        model="phi3",
        temperature=0.1,
        num_ctx=2048
    )

    return embedding_model, vector_store, np.array(scheme_embeddings), llm


# ==========================================================
# Scheme Detection
# ==========================================================

def detect_scheme(question, embedding_model, scheme_embeddings):

    question_lower = question.lower()

    # Direct match
    for scheme in SCHEME_DESCRIPTIONS.keys():
        if scheme.lower() in question_lower:
            return scheme

    # Semantic fallback
    query_embedding = embedding_model.embed_query(question)
    similarities = cosine_similarity(
        [query_embedding], scheme_embeddings
    )[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_scheme = list(SCHEME_DESCRIPTIONS.keys())[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_scheme

    return None


# ==========================================================
# Streamlit UI
# ==========================================================

st.set_page_config(page_title="Scheme Assistant", layout="wide")

st.title("Scheme Assistant")
st.markdown(
    "An AI-powered assistant for exploring Government Schemes "
    "using semantic routing and metadata-filtered retrieval."
)

# Sidebar
with st.sidebar:
    st.header("Available Schemes")
    for scheme in SCHEME_DESCRIPTIONS.keys():
        st.markdown(f"- **{scheme}**")
    st.markdown("---")
    st.caption("Powered by MiniLM + FAISS + Phi-3")

embedding_model, vector_store, scheme_embeddings, llm = initialize_system()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about any scheme...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    scheme = detect_scheme(user_input, embedding_model, scheme_embeddings)

    if scheme:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": {"scheme": scheme}}
        )
    else:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

    docs = retriever.invoke(user_input)

    if not docs:
        response = "Relevant information not found in provided document."
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Answer using ONLY the provided context.

If answer not found in context, reply exactly:
Relevant information not found in provided document.

Use bullet points.
Maximum 5 bullets.
Under 80 words.

Context:
{context}

Question:
{user_input}

Answer:
"""
        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})