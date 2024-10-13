import json
import torch
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone

# Streamlit app title
st.title("Legal Query Assistant")

# Step 1: Load LegalBERT tokenizer and model
@st.cache_resource
def load_legalbert_model():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    return tokenizer, model

# Step 2: Extract text from JSON case file
def extract_text_from_json(data):
    """Extract and combine relevant parts of the JSON into a single text."""
    text_parts = []
    
    # Extract name and court information
    if 'name' in data:
        text_parts.append(f"Case Name: {data['name']}")
    if 'court' in data:
        text_parts.append(f"Court: {data['court']['name']}")
    if 'jurisdiction' in data:
        text_parts.append(f"Jurisdiction: {data['jurisdiction']['name_long']}")
    
    # Extract citations
    if 'citations' in data:
        citations = [cite['cite'] for cite in data['citations']]
        text_parts.append(f"Citations: {', '.join(citations)}")
    
    # Extract case opinions text
    if 'casebody' in data and 'opinions' in data['casebody']:
        opinions_text = " ".join([opinion['text'] for opinion in data['casebody']['opinions']])
        text_parts.append(f"Opinions: {opinions_text}")
    
    # Combine all parts into one large text
    combined_text = "\n".join(text_parts)
    return combined_text

# Step 3: Load and prepare the JSON document
@st.cache_data
def load_and_prepare_document(file_path):
    """Load a single JSON file and extract relevant text."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        combined_text = extract_text_from_json(data)
    return combined_text

# Step 4: Initialize Pinecone
@st.cache_resource
def initialize_pinecone(api_key, index_name):
    """Initialize Pinecone."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

# Step 5: Generate query embedding using LegalBERT
def generate_query_embedding(query, tokenizer, model):
    """Generate embedding for the user query."""
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token embedding
    return query_embedding

# Step 6: Query Pinecone for similar cases
def query_pinecone(index, query_embedding):
    """Query Pinecone for similar documents."""
    query_response = index.query(vector=query_embedding.tolist(), top_k=5)
    retrieved_docs = [match['id'] for match in query_response['matches']]
    return retrieved_docs

# Step 7: Make a legal query via Kindo API
def send_legal_query_to_kindo(query, retrieved_docs, api_key):
    """Send legal query to the Kindo API."""
    
    # Prepare API endpoint and headers
    url = "https://llm.kindo.ai/v1/chat/completions"
    headers = {
        "api-key": api_key,
        "content-type": "application/json"
    }
    
    # Prepare data payload
    data = {
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "system", "content": """You are a highly skilled and experienced legal advisor specializing in analyzing legal case data and providing clear, actionable legal advice. Your goal is to assist the user by analyzing the key fields from their legal dataset and understanding how these details influence their case. You will need to:
            1. Analyze Legal References
            2. Understand Case Reasoning and Precedents
            3. Provide Case-Specific Analysis
            4. Make a Judgment
            5. Explain in Simple Terms"""},
            {"role": "system", "content": "Here is the case reference document" + retrieved_docs},
            {"role": "user", "content": query}
        ]
    }
    
    # Make the API request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Return the response
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}"

# Main logic for Streamlit app
if __name__ == "__main__":
    # Step 1: Load the model and tokenizer
    tokenizer, model = load_legalbert_model()
    
    # Step 2: Load and prepare the JSON document
    json_file = "doc-5.json"
    doc_5 = load_and_prepare_document(json_file)
    
    # Step 3: Initialize Pinecone
    pinecone_api_key = "78c6c4a1-bd01-4568-86e7-c05cc4c69756"
    pinecone_index_name = "legal"
    index = initialize_pinecone(pinecone_api_key, pinecone_index_name)

    # Step 4: Take user query from Streamlit input
    user_query = st.text_input("Enter your legal question:")
    
    # Step 5: Query only if user provides input
    if user_query:
        # Generate query embedding
        query_embedding = generate_query_embedding(user_query, tokenizer, model)
        
        # Query Pinecone for similar documents
        retrieved_docs = query_pinecone(index, query_embedding)
        
        # Send detailed query to Kindo API and get the response
        kindo_api_key = "c744f000-a237-4dd9-8fc4-200bfc7486b5-85e2b3111db14629"
        api_response = send_legal_query_to_kindo(user_query, doc_5, kindo_api_key)
        
        # Display the result in Streamlit
        st.write("Legal Response:")
        st.write(api_response)
