from transformers import AutoTokenizer, AutoModelForCausalLM
from azure.search.documents import SearchClient
import torch
import bitsandbytes as bnb

# Load the LLaMA model and tokenizer
model_name = "/home/mnahsan21/.llama/checkpoints/Llama-2-7B"  # Correct absolute path
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    # load_in_8bit=True,  # Enable 8-bit quantization
    device_map=None,  # Disable GPU mapping
).to("cpu")  # Explicitly move the model to CPU

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

def compare_documents(query, service_endpoint, api_key, index_name):
    """
    Compare documents using LLaMA and Azure Cognitive Search.

    Args:
        query (str): The query or context for comparison.
        service_endpoint (str): Azure Cognitive Search service endpoint.
        api_key (str): Azure Cognitive Search API key.
        index_name (str): Name of the Azure Cognitive Search index.

    Returns:
        str: The LLM's response comparing the documents.
    """
    # Generate query embedding using LLaMA
    inputs = tokenizer(query, return_tensors="pt").to("cpu")  # Ensure inputs are on CPU
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    # Retrieve relevant chunks from Azure Cognitive Search
    print("Retrieving relevant chunks from Azure Cognitive Search...")
    retrieved_chunks = retrieve_relevant_chunks(service_endpoint, api_key, index_name, query_embedding)

    # Combine the query and retrieved chunks into a single prompt
    prompt = f"Query: {query}\n\nRetrieved Chunks:\n"
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"{i + 1}. {chunk}\n"
    prompt += "\nCompare the documents and highlight the differences."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")  # Ensure inputs are on CPU

    # Generate the response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response