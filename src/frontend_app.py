import gradio as gr
import os
from pathlib import Path
from retrieve_chunks_and_compare import compare_documents  # Import the comparison function

# Define paths to store uploaded documents
UPLOAD_DIR = "./uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_documents(doc1, doc2):
    """
    Process the uploaded documents and perform comparison.
    """
    # Save the uploaded documents locally
    doc1_path = Path(UPLOAD_DIR) / "document_1.pdf"
    doc2_path = Path(UPLOAD_DIR) / "document_2.pdf"
    with open(doc1_path, "wb") as f:
        f.write(doc1.read())
    with open(doc2_path, "wb") as f:
        f.write(doc2.read())

    # Call the backend pipeline to analyze and compare the documents
    # (Update the backend pipeline to handle two documents)
    service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = "insurance-docs-index"

    # Generate the comparison report
    query = "Compare the two uploaded documents and highlight the differences."
    comparison_result = compare_documents(query, service_endpoint, api_key, index_name)

    # Save the comparison result as a report
    report_path = Path(UPLOAD_DIR) / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(comparison_result)

    return comparison_result, str(report_path)

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Comparison Tool")
    gr.Markdown("Upload two documents to compare their content and generate a report.")

    with gr.Row():
        doc1 = gr.File(label="Upload Document 1")
        doc2 = gr.File(label="Upload Document 2")

    compare_button = gr.Button("Compare Documents")
    output_text = gr.Textbox(label="Comparison Result", lines=10)
    download_link = gr.File(label="Download Report")

    compare_button.click(
        process_documents,
        inputs=[doc1, doc2],
        outputs=[output_text, download_link],
    )

# Launch the Gradio app
demo.launch()