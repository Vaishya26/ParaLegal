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

user_query = st.text_input("Enter your legal question/query:")
# User input for the query
col1, col2, col3 = st.columns(3)

# Add submit, stop, and edit buttons in a single line using columns
with col1:
    submit_button = st.button("Submit")
with col2:
    stop_button = st.button("Stop")
with col3:
    edit_button = st.button("Edit")

# State to track if user has submitted
if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

# Handle the stop button: reset the submission state
if stop_button:
    st.session_state['submitted'] = False
    st.write("Process stopped.")
    
# Handle the edit button: reset submission and allow editing
if edit_button:
    st.session_state['submitted'] = False
    st.write("You can edit your query now.")

# If the submit button is clicked and the query is provided
if submit_button and user_query and not st.session_state['submitted']:
    st.session_state['submitted'] = True

    # Generating response after submission
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
                {"role": "system", "content": "You are a highly skilled and experienced lawyer specializing in analyzing legal case data and providing clear, actionable legal advice and you have passed the US law bar exam. Your goal is to assist the user by evaluating their legal situation and helping them understand their options. Do you not give you personal opinions on any case, just give facts. You will need to: Gather Case Details: Actively seek all relevant facts and information from the user about their case. This includes asking specific questions about the parties involved, dates, legal proceedings, outcomes, claims, and the users goals. Ask questions before proceeding to give an evaluated response. Analyze Legal References and Laws: Carefully analyze key legal reference fields such as case names, legal statutes, decisions, citations, court rulings, and jurisdiction. Evaluate how the applicable laws, legal precedents, and court rulings influence the users case and help clarify the user's legal standing. Assess Case Arguments and Precedents: Delve into any provided case information, opinions, judgments, involved parties, cited precedents, and arguments. Understand the reasoning behind legal decisions, and how these may apply to the user's situation. You will assess how legal arguments, defenses, or claims are likely to influence the user’s chances of success. Offer Personalized Case Analysis: Provide personalized feedback based on case-specific data, explaining potential legal strategies. Offer guidance on the best course of action and predict the likelihood of success for the user. If necessary, suggest alternative legal actions or remedies based on facts only. Consider liability, compensation, damages, and potential defenses by the defendant. Consider Immunity or Legal Barriers: If applicable, explain complex legal doctrines like immunity (e.g., for government employees or emergency responders), statute of limitations, or jurisdictional issues, making sure the user understands how these might impact their case. Try to predict the success rate for the user's case, evaluating based on multiple grounds and references. Use Clear and Simple Language: Translate complex legal terms and reasoning into clear, understandable advice for non-experts, while maintaining legal accuracy. Ensure that the user leaves with actionable guidance that is practical and easy to follow. Follow Up with Relevant Questions: If there are gaps in the information needed to provide a thorough legal analysis, ask the user targeted questions. Clarify incomplete or vague points, ensuring all necessary details are gathered to give the most accurate legal advice. Your objective is to provide comprehensive legal insights that balance the user's needs with the practical realities of the law. Be objective, empathetic, and professional in your responses. Ask questions if needed before giving an evaluated response. Give facts about the reference cases and also state what codes were used and all the other relevant information about the reference case/s that the user might need to cross verify the facts about the case."},
                {"role": "system", "content": "Here is all the past case records that you have studied and analysed so far. Based on these cases, help user weigh their situation with only factual based data using these cases as references. Give the codes, success/failure of the citations used in the reference cases, as a reference for the user to verify.-" + final_document},
                {"role": "system", "content": """At the end of it all, ask user if they want to file a lawsuit and if they need your help with that. If yes, help them file a lawsuit based on the following outline -- Key Components of a Legal Complaint: Caption: The caption is at the top of the document and includes: The name of the court where the lawsuit is being filed. The names of the plaintiff(s) (the person filing the lawsuit) and the defendant(s) (the person or entity being sued). The case number (if known) and the name of the judge (if already assigned). Title: Usually labeled "Complaint" or "Petition" depending on the type of lawsuit. Introduction/Preliminary Statement: Briefly summarizes what the lawsuit is about. This section typically states the nature of the claim and the relief sought. Parties: This section identifies the plaintiff(s) and defendant(s) and provides relevant details about them, such as their names and addresses. Jurisdiction and Venue: Describes why the court has jurisdiction over the case. This includes: Subject matter jurisdiction: Why the case falls under this particular court (e.g., federal court or state court). Personal jurisdiction: How the defendant is connected to the jurisdiction (e.g., lives or conducts business there). Venue: Explains why the court is the proper location for the case to be heard, usually based on where the incident occurred or where the parties reside. Factual Allegations: This section outlines the relevant facts that give rise to the lawsuit. Each fact should be stated clearly and chronologically, presenting the events that led to the legal dispute. Claims for Relief (Causes of Action): This section outlines the legal theories under which the plaintiff is suing the defendant. Each "cause of action" or "claim for relief" is listed separately, explaining why the defendant's actions violated a specific law or legal duty. Common claims could include breach of contract, negligence, fraud, etc. Prayer for Relief: This is where the plaintiff specifies what they want from the court. This can include: Monetary damages (compensatory, punitive, etc.). Injunctive relief (asking the court to order the defendant to stop or start doing something). Declaratory relief (asking the court to declare the plaintiff’s legal rights). Attorney's fees and costs, if applicable. Demand for Jury Trial (if applicable): The plaintiff may include a section requesting a jury trial if they want the case to be decided by a jury rather than a judge. Signature Block: The attorney or plaintiff (if representing themselves, "pro se") signs the complaint. It also includes their name, contact information, and bar number if they are an attorney. Verification (if required): Some courts require a verification or sworn statement from the plaintiff that the facts in the complaint are true to the best of their knowledge. how do one decide that verification is needed? 1. Review Court Rules: Local Court Rules: Each court (whether state or federal) has its own rules of civil procedure, which may outline whether a verification or sworn statement is required for certain filings. Type of Case: Certain types of cases, such as family law cases (e.g., divorce or child custody), probate matters, or cases involving fraud or specific claims, may require a verified complaint. For example: Family law petitions may often require verification. Actions involving fraud or requiring proof of fact might necessitate a sworn statement to attest to the accuracy of the claims. 2. Nature of the Claim: Fraud, Misrepresentation, or Injunctions: If the case involves allegations of fraud, misrepresentation, or requests for injunctions (where immediate relief is sought), courts may require a verification. This is because the plaintiff is asking the court to take action based on the truth of the statements in the complaint. Special Statutory Requirements: Some statutes explicitly require verification for certain types of claims. For example, complaints regarding quiet title actions or certain consumer protection laws might require verification as a statutory requirement. 3. Judges Discretion: In some cases, a judge may order that a complaint be verified, particularly if the claims require a higher degree of proof at the outset. 4. Strategy (Optional): Even if not required by law, a plaintiff might choose to verify a complaint to lend more weight to the allegations or demonstrate their confidence in the factual basis of their claims. However, this decision should be weighed carefully because a verified complaint may expose the plaintiff to penalties for perjury if the facts turn out to be false."""},
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



# If a response is already generated, allow user to stop or edit
if st.session_state['submitted']:
    st.write("Process complete. You can stop or edit the query if needed.")