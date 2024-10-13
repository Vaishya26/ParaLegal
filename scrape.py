import streamlit as st
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import json
import torch
import requests

# Load secrets from Streamlit (make sure you have stored your API keys in secrets)
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
kindo_api_key = st.secrets["KINDO_API_KEY"]

# Load LegalBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("legal")

# Simulated documents (replace with real document embeddings or document processing as per your use case)
documents = [
    {
        "ID": "1",
        "Case Name": "Smith vs Doe",
        "Court": "Supreme Court",
        "Jurisdiction": "United States",
        "Decision Date": "2021-09-01",
        "Name Abbreviation": "Smith v. Doe",
        "Citations": ["123 U.S. 456"],
        "Opinions": [{"author": "Justice X", "type": "Majority", "text": "This is the case opinion text"}]
    },
    # Add more cases if needed
]

# Function to generate embeddings
def generate_embeddings(documents, tokenizer, model, index):
    for doc in documents:
        id = doc["ID"]
        doc_str = str(doc)
        inputs = tokenizer(doc_str, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Ensure the embedding size matches Pinecone index dimension (assuming 768 here)
        assert embedding.shape[0] == 768, "Embedding size mismatch with Pinecone index dimension."

        # Upload embeddings to Pinecone
        index.upsert(vectors=[(f"doc_{id}", embedding.tolist())])

    st.write(f"{len(documents)} documents embedded and uploaded to Pinecone.")

# Generate embeddings once, you can uncomment this to enable it:
# generate_embeddings(documents, tokenizer, model, index)

# Function to generate query embedding
def generate_query_embedding(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return query_embedding

# Define a function to format the retrieved JSON into structured text for RAG
def format_case_data(case):
    structured_text = f"""
    Case ID: {case['ID']}
    Case Name: {case['Case Name']}
    Court: {case['Court']}
    Jurisdiction: {case['Jurisdiction']}
    Decision Date: {case['Decision Date']}
    Name Abbreviation: {case['Name Abbreviation']}
    Citations: {', '.join(case['Citations'])}

    Opinions:
    """
    for opinion in case['Opinions']:
        structured_text += f"\n- Opinion Author: {opinion['author']}\n  Type: {opinion['type']}\n  Text: {opinion['text'][:500]}..."  # Limiting to 500 characters for summary
    
    return structured_text

# User input for the query
user_query = st.text_input("Enter your legal question/query:")
if st.button("Submit"):
    if user_query:
        with st.spinner("Generating response..."):
            # Generate embedding for the user query
            query_embedding = generate_query_embedding(user_query, tokenizer, model)

            # Query Pinecone with the query embedding
            query_response = index.query(vector=query_embedding.tolist(), top_k=5)
            retrieved_docs_ids = [match['id'] for match in query_response['matches']]

            # Convert document IDs back to retrieve cases
            retrieved_docs_ids = [int(doc_id[4:]) for doc_id in retrieved_docs_ids]
            
            # Create a dictionary for quick lookup
            documents_dict = {case['ID']: case for case in documents}

            # Retrieve cases by their ID
            retrieved_docs = [documents_dict[doc_id] for doc_id in retrieved_docs_ids if doc_id in documents_dict]

            # Format each retrieved document
            structured_documents = [format_case_data(doc) for doc in retrieved_docs]

            # Combine into a single text document
            final_document = "\n\n".join(structured_documents)

            # Display retrieved documents
            st.write(final_document)

            # API request to Kindo AI
            url = "https://llm.kindo.ai/v1/chat/completions"
            headers = {
                "api-key": kindo_api_key,
                "content-type": "application/json"
            }
            
            data = {
                "model": "gemini-1.5-pro",
                "messages": [
                    {"role": "system", "content": "You are a highly skilled and experienced legal advisor..."},
                    {"role": "system", "content": "Here is the case reference document" + final_document},
                    {"role": "user", "content": user_query}
                ]
            }

            # Send the request
            response = requests.post(url, headers=headers, data=json.dumps(data))

            # Check the response and display it
            if response.status_code == 200:
                st.write(response.json()['choices'][0]['message']['content'])
            else:
                st.write(f"Error: {response.status_code}")
