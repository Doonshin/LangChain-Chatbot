import gradio as gr
import os
import json
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
PDF_INDEX_FILE = os.path.join(BASE_DIR, "uploaded_pdfs.json")

# Load previously uploaded PDFs
if os.path.exists(PDF_INDEX_FILE):
    with open(PDF_INDEX_FILE, "r", encoding="utf-8") as f:
        uploaded_pdfs = json.load(f)
else:
    uploaded_pdfs = {}

def save_uploaded_pdfs():
    """Save uploaded PDF information to a JSON file."""
    with open(PDF_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(uploaded_pdfs, f, indent=4)

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

            # Skip if already processed
            if file_name in uploaded_pdfs:
                continue

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

    # Save updated file list
    save_uploaded_pdfs()

    return f"✅ Processed PDFs: {', '.join(processed_files)}", list(uploaded_pdfs.keys())  # Returns updated PDF list

def load_faiss_for_pdf(selected_pdf):
    """Loads the FAISS index for the selected PDF."""
    if selected_pdf in uploaded_pdfs:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        return FAISS.load_local(uploaded_pdfs[selected_pdf], embeddings, allow_dangerous_deserialization=True)
    return None

def delete_selected_pdf(selected_pdf):
    """Deletes a selected PDF and its FAISS index."""
    global uploaded_pdfs

    if selected_pdf in uploaded_pdfs:
        index_path = uploaded_pdfs[selected_pdf]
        del uploaded_pdfs[selected_pdf]
        save_uploaded_pdfs()

        # Delete FAISS index directory
        if os.path.exists(index_path):
            os.system(f"rm -rf {index_path}")

        return f"🗑️ Deleted {selected_pdf}", list(uploaded_pdfs.keys())
    
    return "⚠️ No such file found!", list(uploaded_pdfs.keys())

def chat_with_selected_pdf(message, history, selected_pdf, chat_k):
    """Streaming-enabled chat function for PDF-based Q&A"""
    try:
        if not selected_pdf or selected_pdf not in uploaded_pdfs:
            yield history + [(message, "⚠️ Please select a valid PDF first.")]
            return

        vector_store = load_faiss_for_pdf(selected_pdf)
        if vector_store is None:
            yield history + [(message, "⚠️ No FAISS index found. Process the PDF first.")]
            return

        retriever = vector_store.as_retriever(search_kwargs={"k": chat_k})
        docs = retriever.get_relevant_documents(message)

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4-turbo", streaming=True)

        # Stream response directly from the model
        history.append((message, ""))  # Start with an empty response
        response_stream = llm.stream(f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer:")

        for chunk in response_stream:
            if hasattr(chunk, "content"):  # Extract text from AIMessageChunk
                history[-1] = (message, history[-1][1] + chunk.content)
                yield history  # Update the UI in real-time

    except Exception as e:
        yield history + [(message, f"⚠️ Error: {str(e)}")]



def summarize_selected_pdf(selected_pdf, summary_k):
    """Generates a summary for the selected PDF with streaming support."""
    try:
        if not selected_pdf or selected_pdf not in uploaded_pdfs:
            yield "⚠️ No FAISS index found for the selected PDF! Process it first."
            return

        vector_store = load_faiss_for_pdf(selected_pdf)
        if vector_store is None:
            yield "⚠️ No FAISS index found for the selected PDF! Process it first."
            return

        retriever = vector_store.as_retriever(search_kwargs={"k": summary_k})
        docs = retriever.get_relevant_documents("Summarize this document")
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4-turbo", streaming=True)

        # Start with an empty summary
        summary = ""
        yield summary  # Send an empty initial response

        # Stream the summary generation
        response_stream = llm.stream(f"Summarize the following document:\n\n{context}")

        for chunk in response_stream:
            if hasattr(chunk, "content"):  # Extract text from AIMessageChunk
                summary += chunk.content  # Append streamed content
                yield summary  # Update the UI in real time

    except Exception as e:
        yield f"⚠️ Error: {str(e)}"


custom_css = """
body, .gradio-container {
    background: linear-gradient(to right, #FFDEAD, #F4A460) !important;
    color: black !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 📄 Multi-PDF Q&A Chatbot & Summary Generator")

    with gr.Row():
        file_input = gr.Files(label="📂 Upload PDFs")
        process_button = gr.Button("🚀 Process PDFs", scale=2)

    
    status_output = gr.Textbox(label="📢 Status", interactive=False, lines=2)

    pdf_selector = gr.Dropdown(label="📄 Select PDF", choices=list(uploaded_pdfs.keys()), interactive=True)
    delete_button = gr.Button("🗑️ Delete Selected PDF")

    def process_and_update(files):
        status_message, pdf_list = process_pdfs(files)
        return status_message, gr.update(choices=pdf_list)

    def delete_and_update(selected_pdf):
        status_message, pdf_list = delete_selected_pdf(selected_pdf)
        return status_message, gr.update(choices=pdf_list, value="")

    process_button.click(process_and_update, inputs=file_input, outputs=[status_output, pdf_selector])
    delete_button.click(delete_and_update, inputs=pdf_selector, outputs=[status_output, pdf_selector])

    # Chatbot UI
    chatbot = gr.Chatbot(label="🤖 Chatbot (Ask about selected PDF)", height=400)
    msg = gr.Textbox(label="💬 Enter your question")

    # Chat Settings
    chat_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="🤖 Chatbot: Number of Retrieved Documents (k)")
    submit_button = gr.Button("▶️ Send")
    submit_button.click(chat_with_selected_pdf, inputs=[msg, chatbot, pdf_selector, chat_k_slider], outputs=chatbot)

    # Summary Settings
    summary_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="📜 Summary: Number of Retrieved Documents (k)")
    summary_button = gr.Button("📜 Generate Summary")
    summary_output = gr.Textbox(label="📄 Summary")
    summary_button.click(summarize_selected_pdf, inputs=[pdf_selector, summary_k_slider], outputs=summary_output)

demo.launch()
