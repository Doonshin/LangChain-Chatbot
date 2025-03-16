import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader  
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Directory to store FAISS indexes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_indexes")

# Dictionary to store PDF filenames and their FAISS indexes
uploaded_pdfs = {}

def process_pdfs(files):
    """Processes multiple uploaded PDFs and stores them in FAISS."""
    global uploaded_pdfs

    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR)

    processed_files = []
    for file in files:
        try:
            file_path = file.name
            file_name = os.path.basename(file_path)
            faiss_index_path = os.path.join(FAISS_INDEX_DIR, file_name.replace(".pdf", ""))

            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            vector_store = FAISS.from_documents(docs, embeddings)

            vector_store.save_local(faiss_index_path)
            uploaded_pdfs[file_name] = faiss_index_path  # Store file in dictionary

            processed_files.append(file_name)
        except Exception as e:
            return f"⚠️ Error processing {file_name}: {str(e)}", list(uploaded_pdfs.keys())

    return f"✅ Processed PDFs: {', '.join(processed_files)}", list(uploaded_pdfs.keys())  # Returns updated PDF list


def load_faiss_for_pdf(selected_pdf):
    """Loads the FAISS index for the selected PDF."""
    if selected_pdf in uploaded_pdfs:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return FAISS.load_local(uploaded_pdfs[selected_pdf], embeddings, allow_dangerous_deserialization=True)
    return None


def chat_with_selected_pdf(message, history, selected_pdf):
    """Handles chatbot interactions based on the selected PDF."""
    try:
        # Ensure selected_pdf is a string (if it's a list, take the first item)
        if isinstance(selected_pdf, list) and len(selected_pdf) > 0:
            selected_pdf = selected_pdf[0]
        
        # Validate selected PDF
        if not selected_pdf or selected_pdf not in uploaded_pdfs:
            return [(message, "⚠️ Please select a valid PDF before asking questions!")]

        vector_store = load_faiss_for_pdf(selected_pdf)

        if vector_store is None:
            return [(message, "⚠️ No FAISS index found for the selected PDF! Process it first.")]

        retriever = vector_store.as_retriever(search_kwargs={"k": 8})
        llm = ChatOpenAI(model="gpt-4")

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",
            return_source_documents=False,
            output_key="result"
        )

        response = rag_chain.run(message)
        history.append((message, response))
        return history
    except Exception as e:
        return [(message, f"⚠️ Error: {str(e)}")]


def summarize_selected_pdf(selected_pdf):
    """Generates a summary for the selected PDF."""
    if isinstance(selected_pdf, list) and len(selected_pdf) > 0:
        selected_pdf = selected_pdf[0]

    vector_store = load_faiss_for_pdf(selected_pdf)
    if vector_store is None:
        return "⚠️ No FAISS index found for the selected PDF! Process it first."

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    llm = ChatOpenAI(model="gpt-4")

    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = summarize_chain.run(retriever.get_relevant_documents("Summarize this document"))

    return summary


def export_chat(history):
    """Exports chat history as a text file."""
    chat_text = "\n".join([f"User: {q}\nAI: {a}\n" for q, a in history])
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        f.write(chat_text)
    return "chat_history.txt"


# Custom Theme
custom_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=["Arial", "sans-serif"]
)


# Gradio Interface
with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# 📄 Multi-PDF Q&A Chatbot & Summary Generator")

    with gr.Row():
        file_input = gr.Files(label="📂 Upload PDFs")
        process_button = gr.Button("🚀 Process PDFs", scale=2)
    
    status_output = gr.Textbox(label="📢 Status", interactive=False, lines=2)

    # Dropdown for selecting PDFs
    pdf_selector = gr.Dropdown(label="📄 Select PDF", choices=[], interactive=True)

    # Process PDFs & Update Dropdown
    def process_and_update(files):
        status_message, pdf_list = process_pdfs(files)
        return status_message, gr.update(choices=pdf_list)

    process_button.click(process_and_update, inputs=file_input, outputs=[status_output, pdf_selector])

    with gr.Row():
        chatbot = gr.Chatbot(label="🤖 Chatbot (Ask about selected PDF)", height=400)
    msg = gr.Textbox(label="💬 Enter your question", scale=3)
    submit_button = gr.Button("▶️ Send", scale=1)
    clear_button = gr.Button("🗑️ Clear Chat", scale=1)

    submit_button.click(chat_with_selected_pdf, inputs=[msg, chatbot, pdf_selector], outputs=chatbot)
    clear_button.click(lambda: [], None, chatbot)

    with gr.Row():
        summary_button = gr.Button("📜 Generate Summary")
        summary_output = gr.Textbox(label="📄 Summary")
        summary_button.click(summarize_selected_pdf, inputs=pdf_selector, outputs=summary_output)

    download_button = gr.Button("💾 Download Chat History")
    download_button.click(export_chat, inputs=chatbot, outputs=gr.File(label="📥 Download"))

demo.launch()
