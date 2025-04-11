from langchain.text_splitter import RecursiveCharacterTextSplitter

# Simulated document fields from Azure Doc Intelligence
document = {
    "insured_carrier": "Morgan-Powers",
    "no_of_accidents": "3",
    "insured_address": "20636 Manuel Gateway, New Gregory, TX 98218",
    "insured_premium": "$1,729.59",
    "insured_deductible": "$573.97",
    "insured_name": "David Coleman",
    "insured_year": "2025"
}

# Convert to synthetic paragraph-style text for chunking
text_representation = f"""
Insurance Policy Document - {document['insured_year']}

Insured Name: {document['insured_name']}
Carrier: {document['insured_carrier']}
Premium: {document['insured_premium']}
Deductible: {document['insured_deductible']}
Address: {document['insured_address']}
Number of Accidents in Current Year: {document['no_of_accidents']}
"""

# Step 1: Initialize chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

# Step 2: Create chunks
chunks = text_splitter.split_text(text_representation)

# Output for inspection
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk}")
