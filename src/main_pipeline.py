import os
import sys
import yaml
from dotenv import load_dotenv, find_dotenv

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import functions from custom modules
from get_custom_text.analyze_custom_doc_main import main as analyze_documents
from chunking.chunk_and_embed import chunk_and_embed
from indexing.index_to_azure import create_index, index_documents
from retrieve_chunks_and_compare import compare_documents

def load_config_and_env():
    """
    Load configuration and environment variables.
    """
    # Load the config file
    with open('./config.yaml') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load environment variables
    load_dotenv(find_dotenv())
    return config

def main_pipeline():
    """
    Main pipeline to analyze documents, chunk, embed, and store in Chroma.
    """
    # Step 1: Load configuration and environment variables
    print("Loading configuration and environment variables...")
    config = load_config_and_env()

    # Step 2: Create the index
    print("Creating the Azure Cognitive Search index...")
    service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = "insurance-docs-index"
    vector_dim = 384  # Adjust this to match your embedding dimensions

    create_index(service_endpoint, api_key, index_name, vector_dim)

    # Step 3: Analyze documents
    print("Analyzing documents...")
    model_id = config['doc_intelligence']['custom_models']['custom_insurance_model_1']
    training_folder_path = os.getenv("training_folder_SAS_URI")
    path_to_id_document = os.getenv("file_path_to_doc")

    if not os.path.exists(path_to_id_document):
        raise FileNotFoundError(f"The file at path '{path_to_id_document}' does not exist. Please check the file path.")

    # Analyze the document and get results
    document_results = analyze_documents(path_to_id_document, model_id, training_folder_path)
    print(f"Document analysis completed. Results: {document_results}")

    # Step 4: Chunk, embed, and store in Chroma
    print("Chunking, embedding, and storing in Chroma...")
    chunks = chunk_and_embed(document_results)  # Get the chunks with metadata and embeddings

    # Step 5: Index embeddings into Azure AI Search
    print("Indexing embeddings into Azure AI Search...")

    # Prepare documents for indexing
    documents = []
    for chunk in chunks:
        documents.append({
            "id": f"doc_{chunk['document_index']}_chunk_{chunk['chunk_id']}",
            "document_index": str(chunk["document_index"]),  # Convert to string
            "chunk_id": str(chunk["chunk_id"]),  # Convert to string
            "chunk_text": chunk["chunk"],
            "text_vector": chunk["embedding"]
        })

    # Index the documents
    index_documents(service_endpoint, api_key, index_name, documents)

    print("Pipeline completed successfully.")

    # Step 6: Compare documents using LLaMA
    print("Comparing documents using LLaMA...")
    query = "Compare the insurance policies for premium and deductible changes."
    response = compare_documents(query, service_endpoint, api_key, index_name)
    print("Comparison Results:")
    print(response)

if __name__ == "__main__":
    main_pipeline()