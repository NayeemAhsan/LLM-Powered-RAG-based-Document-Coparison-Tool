import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchIndex
)

def create_index(service_endpoint, api_key, index_name, vector_dim):
    """
    Create an Azure Cognitive Search index for storing embeddings and metadata.

    Args:
        service_endpoint (str): Azure Cognitive Search service endpoint.
        api_key (str): Azure Cognitive Search API key.
        index_name (str): Name of the index to create.
        vector_dim (int): Dimension of the embedding vectors.

    Returns:
        None
    """
    index_client = SearchIndexClient(endpoint=service_endpoint, credential=AzureKeyCredential(api_key))

    # Define the index schema
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True),
        SearchField(name="document_index", type=SearchFieldDataType.String, sortable=True, filterable=True),  # Changed to String
        SearchField(name="chunk_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),  # Changed to String
        SearchField(name="chunk_text", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=vector_dim,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    # Configure vector search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )
        ]
    )

    # Create the search index
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    result = index_client.create_or_update_index(index)
    print(f"✅ Index '{result.name}' created successfully.")

def index_documents(service_endpoint, api_key, index_name, documents):
    """
    Index documents into Azure Cognitive Search.

    Args:
        service_endpoint (str): Azure Cognitive Search service endpoint.
        api_key (str): Azure Cognitive Search API key.
        index_name (str): Name of the index to upload documents to.
        documents (list): List of documents to index. Each document should include precomputed embeddings.

    Returns:
        None
    """
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    # Upload documents
    result = search_client.upload_documents(documents=documents)
    print(f"✅ Indexed {len(result)} documents successfully.")