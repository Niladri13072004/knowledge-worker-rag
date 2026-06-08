import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


# -----------------------------
# Global state
# -----------------------------
agent_global = None

# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# -----------------------------
# Document Processing
# -----------------------------
def process_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_db


# -----------------------------
# Tools
# -----------------------------
def get_tools(db):

    @tool
    def retriever_tool(query: str):
        """
        Retrieve relevant document chunks based on the query
        """

        docs = db.similarity_search(
            query=query,
            k=2
        )

        context = ""

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context

    return [retriever_tool]


# -----------------------------
# Agent
# -----------------------------
def get_agent(db):

    system_prompt = """
    You are a helpful assistant that answers using retrieved context
    from the knowledge base.

    My knowledge base consists of the details from the uploaded document.

    Always use the retriever_tool before answering.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=get_tools(db),
        system_prompt=system_prompt,
        checkpointer=memory
    )

    return agent


# -----------------------------
# Upload PDF
# -----------------------------
def upload_pdf(file):
    global agent_global

    if file is None:
        return "⚠️ Please upload a PDF first."

    try:
        vector_db = process_document(file.name)
        agent_global = get_agent(vector_db)
        return "✅ Document processed successfully! You can now ask questions below."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# -----------------------------
# Chat
# -----------------------------
def chat(message, history):
    global agent_global

    if agent_global is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please upload a PDF first."})
        return history, ""

    try:
        response = agent_global.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": "rag-agent"
                }
            }
        )

        answer = response["messages"][-1].content
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history, ""


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="RAG Knowledge Worker") as demo:

    gr.Markdown("# 📄 RAG Knowledge Worker Chatbot")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])

    upload_btn = gr.Button("Process Document")
    upload_status = gr.Textbox(label="Status", interactive=False)

    upload_btn.click(fn=upload_pdf, inputs=[pdf_file], outputs=[upload_status])

    chatbot = gr.Chatbot(height=500)
    user_input = gr.Textbox(placeholder="Ask anything related to the uploaded document...")
    send_btn = gr.Button("Send")

    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )

demo.launch()