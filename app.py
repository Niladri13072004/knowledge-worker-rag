import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
from time import sleep

# --- Page config ---
st.set_page_config(page_title="RAG Knowledge Worker", page_icon="📄")
st.subheader("📄 RAG Knowledge Worker Chatbot")

# --- Session state initialization ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "is_uploaded" not in st.session_state:
    st.session_state.is_uploaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LLM ---
llm = ChatGroq(model="openai/gpt-oss-20b")

# --- Functions ---
def process_document(path):
    loader = PyPDFLoader(path)
    doc = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(doc)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)
    st.session_state.vector_db = vector_db

def get_tools(db):
    @tool
    def retriever_tool(query: str):
        """Retrieve relevant document chunks based on the query from the knowledge base"""
        documents = db.similarity_search(query=query, k=2)
        context = ""
        for doc in documents:
            context += doc.page_content + "\n\n"
        return context
    return [retriever_tool]

def get_agent(db):
    system_prompt = """You are a helpful assistant that answers using retrieved context from the knowledge base. My knowledge base consists of the details from the uploaded documents. Always use the "retriever_tool" to retrieve the context before answering the user's query."""
    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=get_tools(db),
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    return agent

def call_agent(query):
    response = st.session_state.agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "rag-agent"}}
    )
    final_message = response["messages"][-1]
    return final_message.content

# --- File Upload ---
if not st.session_state.is_uploaded:
    file = st.file_uploader("Upload your document", type=["pdf"])
    if file:
        with open("document.pdf", "wb") as f:
            f.write(file.getvalue())
        with st.spinner("Processing document..."):
            process_document("./document.pdf")
            get_agent(st.session_state.vector_db)
        st.session_state.is_uploaded = True
        st.success("Document uploaded successfully!")
        sleep(1.5)
        st.rerun()

# --- Chat Interface ---
if st.session_state.is_uploaded and st.session_state.vector_db:
    # Display chat history
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        st.chat_message(role).markdown(content)

    # Chat input
    query = st.chat_input("Ask anything related to this document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        with st.spinner("Thinking..."):
            answer = call_agent(query)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
