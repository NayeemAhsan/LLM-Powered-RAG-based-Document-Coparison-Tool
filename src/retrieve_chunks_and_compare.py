from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from azure.search.documents import SearchClient
import torch
import bitsandbytes as bnb
from sklearn.metrics.pairwise import cosine_similarity

# Load the LLaMA model and tokenizer
model_name = "/home/~/.llama/checkpoints/Llama-2-7B"  # Correct absolute path
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    # load_in_8bit=True,  # Enable 8-bit quantization
    device_map=None,  # Disable GPU mapping
).to("cpu")  # Explicitly move the model to CPU

# Load BERT model and tokenizer for similarity scoring
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name)

def retrieve_relevant_chunks(service_endpoint, api_key, index_name, query_embedding, top_k=5):
    """
    Retrieve relevant chunks from Azure Cognitive Search using query embeddings.

    Args:
        service_endpoint (str): Azure Cognitive Search service endpoint.
        api_key (str): Azure Cognitive Search API key.
        index_name (str): Name of the Azure Cognitive Search index.
        query_embedding (list): Embedding vector for the query.
        top_k (int): Number of top results to retrieve.

    Returns:
        list: List of relevant chunks retrieved from Azure Cognitive Search.
    """
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    # Perform vector search
    results = search_client.search(
        search_text="",  # Empty because we're using vector search
        vector=query_embedding,
        top=top_k,
        vector_fields="text_vector",  # Field containing the embeddings
    )

    # Extract the chunks from the results
    relevant_chunks = [result["chunk_text"] for result in results]
    return relevant_chunks

def compute_similarity(text1, text2):
    """
    Compute similarity between two texts using BERT embeddings.
    """
    # Tokenize and encode the texts
    inputs1 = bert_tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
    inputs2 = bert_tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)

    # Generate embeddings
    with torch.no_grad():
        embeddings1 = bert_model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = bert_model(**inputs2).last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings1.numpy(), embeddings2.numpy())
    return similarity[0][0]

def compare_documents(query, service_endpoint, api_key, index_name):
    """
    Compare two documents using Azure Cognitive Search and BERT similarity.
    """
    # Retrieve relevant chunks for both documents (update this logic as needed)
    # For simplicity, assume `retrieved_chunks_doc1` and `retrieved_chunks_doc2` are retrieved
    retrieved_chunks_doc1 = ["Sample text from Document 1"]
    retrieved_chunks_doc2 = ["Sample text from Document 2"]

    # Combine chunks into single texts
    text1 = " ".join(retrieved_chunks_doc1)
    text2 = " ".join(retrieved_chunks_doc2)

    # Compute similarity score
    similarity_score = compute_similarity(text1, text2)

    # Generate a comparison report
    report = f"Similarity Score: {similarity_score:.2f}\n\n"
    report += f"Differences:\n\nDocument 1:\n{text1}\n\nDocument 2:\n{text2}\n"

    return report