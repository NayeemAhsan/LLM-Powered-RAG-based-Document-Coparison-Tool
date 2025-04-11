# LLM-Powered RAG-Based Document Comparison Tool (Azure-Friendly)

This project is a **Retrieval-Augmented Generation (RAG)**-based document comparison tool powered by **Large Language Models (LLMs)** and **Azure Cognitive Services**. It enables users to analyze, chunk, embed, and compare documents, leveraging Azure's AI capabilities for document intelligence and search.

---

## **Key Features**

1. **Custom Model Training**:
   - Train custom models using Azure Document Intelligence to extract structured data from documents.
   - Supports training on insurance documents or other custom datasets.

2. **Document Analysis**:
   - Extract key fields from documents using trained custom models.
   - Analyze documents stored locally or in Azure Blob Storage.

3. **Chunking and Embedding**:
   - Split extracted document fields into smaller chunks for efficient processing.
   - Generate embeddings using **HuggingFace Sentence Transformers**.

4. **Vector Search with Azure Cognitive Search**:
   - Store embeddings in Azure Cognitive Search for fast retrieval.
   - Perform vector-based searches to find relevant document chunks.

5. **Document Comparison**:
   - Use **LLaMA (Large Language Model)** and **BERT similarity scoring** to compare documents and highlight differences.
   - Retrieve relevant chunks from Azure Cognitive Search and generate insights.

6. **Gradio Frontend**:
   - A user-friendly interface for uploading two documents and generating a comparison report.
   - Allows users to view or download the generated report.

7. **Azure Integration**:
   - Follows Azure best practices for security and scalability.
   - Uses Azure Blob Storage, Azure Cognitive Search, and Azure Document Intelligence.

---

## **Prerequisites**

1. **Azure Resources**:
   - Azure Document Intelligence resource.
   - Azure Cognitive Search resource.
   - Azure Blob Storage container with SAS URI for training data.

2. **Environment Variables**:
   - Configure `.env` with the following keys:
     ```plaintext
     DOCUMENTINTELLIGENCE_ENDPOINT=
     DOCUMENTINTELLIGENCE_API_KEY=
     AZURE_SEARCH_ENDPOINT=
     AZURE_SEARCH_API_KEY=
     training_folder_SAS_URI=
     file_path_to_doc=
     ```

3. **Python Dependencies**:
   - Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **LLaMA Model**:
   - Download and configure the **LLaMA-2-7B** model locally.

---

## **Usage**

### **1. Train a Custom Model**
Run the script to train a custom model using Azure Document Intelligence:
```bash
python [buildCustomModel.py](http://_vscodecontentref_/0)
```
### **2. Analyze Documents**
Analyze documents using the trained custom model:
```bash
python [analyze_custom_doc_main.py](http://_vscodecontentref_/6)
```

### **3. Chunk and Embed Documents**
Chunk and embed extracted document fields:
```bash
python [chunk_and_embed.py](http://_vscodecontentref_/7)
```

### **4. Index Embeddings into Azure Cognitive Search**
Create an index and upload embeddings:
```bash
python [index_to_azure.py](http://_vscodecontentref_/8)
```

### **5. Compare Documents**
Compare documents using LLaMA and Azure Cognitive Search:
```bash
python [retrieve_chunks_and_compare.py](http://_vscodecontentref_/9)
```

### **6. Run the Full Pipeline**
Execute the entire pipeline:
```bash
python [main_pipeline.py](http://_vscodecontentref_/10)
```
### **7. Use the Gradio Frontend**
Launch the Gradio-based frontend to upload and compare two documents:
```bash
python src/gradio_frontend.py
```
- Upload two documents (Document 1 and Document 2).
- Click the "Compare Documents" button to generate a comparison report.
- View the report in the interface or download it as a file.

---

## **Configuration**

- **Custom Models**: Define custom model IDs in `config.yaml` under `doc_intelligence.custom_models`.
- **Azure Search Index**: Configure the index name and vector dimensions in `src/indexing/index_to_azure.py`.

---

## **Example Workflow**

1. Generate sample insurance documents using `data/doc_generation_script.py`.
2. Train a custom model on the generated documents.
3. Analyze the documents to extract structured data.
4. Chunk and embed the extracted data.
5. Store embeddings in Azure Cognitive Search.
6. Compare documents and generate insights using LLaMA and BERT similarity scoring.
7. Use the Gradio frontend to upload and compare documents interactively

---

## **Technologies Used**

- **Azure Cognitive Services**:
  - Document Intelligence
  - Cognitive Search
- **HuggingFace Transformers**:
  - Sentence Transformers for embeddings
  - LLaMA for document comparison
  - BERT for similarity scoring
- **Gradio**:
  - Frontend for document upload and comparison
- **Chroma**:
  - Vector database for local embedding storage
- **Python Libraries**:
  - `langchain`, `dotenv`, `yaml`, `torch`, `faker`, `reportlab`

---

## **Troubleshooting**
### Common Issues

1. **Missing `.env` File**:
    - Ensure the `.env` file exists in the project directory and contains all  required environment variables.
2. **Incorrect File Paths**:
    - Verify that the paths to documents and configuration files are correct.
3. **Memory Issues**:
    - If running LLaMA on a CPU, ensure sufficient RAM or use quantization to reduce memory usage.
4. **Azure Authentication**:
    - Ensure that the Azure API keys and endpoints are correctly configured in the .env file.
5. **Gradio Frontend Errors**:
    - Ensure the gradio library is installed:
```bash
pip install gradio
```

---

## **License**

This project is licensed under the MIT License.

---

## **Acknowledgments**

- **Azure AI**: for providing robust document intelligence and search capabilities.
- **HuggingFace**: for state-of-the-art language models and embeddings.
- **Gradio**: For enabling a user-friendly interface for document comparison.
- **Llama**: for inspiring the use of LLMs in retrieval-augmented generation workflows.
