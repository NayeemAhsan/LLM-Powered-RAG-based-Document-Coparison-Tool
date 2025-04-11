from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import json

def flatten_fields(doc):
    """
    Convert extracted fields into a flat text blob for chunking.

    Args:
        doc (dict): A single document with extracted fields.

    Returns:
        str: Flattened text representation of the document fields.
    """
    fields = doc["fields"]
    text = "\n".join([f"{key.replace('_', ' ').title()}: {value['value']}" for key, value in fields.items()])
    return text

def chunk_and_embed(parsed_output, collection_name="insurance_docs", persist_directory="./data/chroma_db"):
    """
    Chunk extracted document fields, embed the chunks, and store them in a Chroma vector database.

    Args:
        parsed_output (list): List of parsed documents with extracted fields.
        collection_name (str): The name of the Chroma collection to store embeddings.
        persist_directory (str): Directory to persist the Chroma database.

    Returns:
        list: A list of dictionaries containing chunk metadata and embeddings.
    """
    all_chunks = []

    # Step 1: Flatten fields and chunk each document
    print("Flattening fields and chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Maximum size of each chunk
        chunk_overlap=50,  # Overlap between chunks to maintain context
        separators=["\n\n", "\n", ".", " "]  # Hierarchical splitting
    )

    for doc_index, doc in enumerate(parsed_output):
        raw_text = flatten_fields(doc)
        chunks = splitter.split_text(raw_text)

        # Store chunk with source info
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "document_index": doc_index,
                "chunk_id": i,
                "chunk": chunk,
                "embedding": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_query(chunk)
            })

    print(f"Document split into {len(all_chunks)} chunks.")
    
    # Step 2: Initialize the embedding model
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Initialize the Chroma vectorstore
    print("Initializing Chroma vectorstore...")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # Step 4: Add chunks and embeddings to the vectorstore
    print("Storing embeddings in Chroma...")
    for chunk_data in all_chunks:
        vectorstore.add_texts(
            texts=[chunk_data["chunk"]],
            metadatas=[{
                "document_index": chunk_data["document_index"],
                "chunk_id": chunk_data["chunk_id"]
            }],
            ids=[f"doc_{chunk_data['document_index']}_chunk_{chunk_data['chunk_id']}"]
        )

    # Persist the database to disk
    vectorstore.persist()
    print(f"Successfully stored {len(all_chunks)} chunks in the Chroma collection '{collection_name}'.")

    # Optional: Save chunks to JSON for debugging or future use
    with open("parsed_chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)
    print("✅ Chunks saved to 'parsed_chunks.json'.")
    return all_chunks

# Example usage
if __name__ == "__main__":
    # Simulated parsed output (replace with your actual parsed data)
    parsed_output = [{
        "document_index": 1,
        "fields": {
            "insured_carrier": {"value": "Morgan-Powers"},
            "no_of_accidents": {"value": "3"},
            "insured_address": {"value": "20636 Manuel Gateway, New Gregory, TX 98218"},
            "insured_premium": {"value": "$1,729.59"},
            "insured_deductible": {"value": "$573.97"},
            "insured_name": {"value": "David Coleman"},
            "insured_year": {"value": "2025"}
        }
    }]

    # Chunk, embed, and store in Chroma
    chunk_and_embed(parsed_output)