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

# Initialize session state to store conversation and chatId
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

if 'chatId' not in st.session_state:
    st.session_state['chatId'] = None  # Set to None initially

# Function to check if the user has asked the same question before
def check_repeated_question(user_query):
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user' and message['content'].strip().lower() == user_query.strip().lower():
            return True
    return False

# Function to send message to Kindo with conversation history and chatId
def send_message_to_kindo(user_query):
    url = "https://llm.kindo.ai/v1/chat/completions"
    
    headers = {
        "api-key": kindo_api_key,
        "content-type": "application/json"
    }

    # Check if there's a chatId stored for the session
    if st.session_state['chatId'] is None:
        # If no chatId, this is the first message and a new chat is started
        data = {
            "model": "azure/gpt-4o",
            "messages": st.session_state['conversation_history'] + [{"role": "user", "content": user_query}]
        }
    else:
        # If chatId exists, continue the conversation with the existing chatId
        data = {
            "model": "azure/gpt-4o",
            "messages": [{"role": "user", "content": user_query}],
            "chatId": st.session_state['chatId']
        }

    # Send the request to Kindo API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        kindo_response = response_data['choices'][0]['message']['content']
        
        # If it's the first message, store the chatId
        if st.session_state['chatId'] is None:
            st.session_state['chatId'] = response_data.get('chatId', None)

        return kindo_response
    else:
        st.write(f"Error: {response.status_code}")
        return None

# User input for the query
user_query = st.text_input("Enter your legal question/query:")
col1, col2, col3 = st.columns(3)

# Add submit, stop, edit, and clear history buttons in a single line using columns
with col1:
    submit_button = st.button("Submit")
with col2:
    stop_button = st.button("Stop")
with col3:
    clear_button = st.button("Clear History")

# Handle the stop button: reset the submission state
if stop_button:
    st.session_state['submitted'] = False
    st.write("Process stopped.")

# Handle the clear button: clear the conversation history and chatId
if clear_button:
    st.session_state['conversation_history'] = []
    st.session_state['chatId'] = None
    st.write("Chat history cleared.")

# If the submit button is clicked and the query is provided
if submit_button and user_query:
    st.session_state['submitted'] = True

    # Check if the question has been asked before
    if check_repeated_question(user_query):
        st.write("You've already asked this question. Refer to the previous response.")
    else:
        # Store user query in conversation history
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})

        # Generating response after submission
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

            # Store model response in conversation history
            st.session_state['conversation_history'].append({"role": "system", "content": final_document})

            # Send the query and conversation history to Kindo AI
            kindo_response = send_message_to_kindo(user_query)

            # Check the response and display it
            if kindo_response:
                st.write(kindo_response)

                # Store Kindo response in conversation history
                st.session_state['conversation_history'].append({"role": "system", "content": kindo_response})

# Display the full conversation history in a structured "You" and "System" chat format
st.write("Conversation History:")
for message in st.session_state['conversation_history']:
    if message['role'] == 'user':
        st.markdown(f"**You**: {message['content']}")  # User's message
    else:
        st.markdown(f"**System**: {message['content']}")  # System's message
