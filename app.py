import os
import pandas as pd
import re
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
import io
import pdfplumber  # Added for better PDF extraction of tables

# Load environment variables
load_dotenv()

# Function to extract text from PDF (updated for Streamlit's UploadedFile)
def extract_text_from_pdf(uploaded_file):
    pdf_file = io.BytesIO(uploaded_file.read())  # Read file into memory
    with pdfplumber.open(pdf_file) as pdf:  # Use pdfplumber for better extraction of structured data
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''  # Handle potential None values
    return text

# Function to preprocess P&L text into structured format
def preprocess_pnl_text(text):
    lines = text.split("\n")
    data = []
    pattern = re.compile(r"([\w\s]+)\s+(\d{1,3}(?:,\d{3})*\.\d{2})")  # Extract financial key-value pairs
    
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            key, value = match.groups()
            data.append((key.strip(), float(value.replace(',', ''))))
    
    # Handle missing or incomplete data more gracefully
    df = pd.DataFrame(data, columns=["Metric", "Value"])
    df = df.dropna(subset=["Metric", "Value"])  # Drop rows with missing data
    return df

# Function to embed and store P&L data in Pinecone
def embed_and_store_pnl_data(pnl_df, index):
    """Generate embeddings and store financial data in Pinecone."""
    to_upsert = []
    for _, row in pnl_df.iterrows():
        text = f"{row['Metric']}: {row['Value']}"
        embedding = embedding_model.encode(text).tolist()
        to_upsert.append((str(row['Metric']), embedding, {"text": text}))
    
    # Batch upsert for optimization
    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"Stored {len(to_upsert)} entries in Pinecone.")

# Function to query the financial data using Pinecone
# Function to query the financial data using Pinecone
def query_pnl(question, index, pnl_df):
    """Query P&L data using RAG model and show relevant rows."""
    query_embedding = embedding_model.encode(question).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    if results and results.get("matches"):
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        # Retrieve the relevant rows based on the context (use the metric names as the key)
        relevant_rows = []
        for match in results["matches"]:
            metric_name = match["metadata"]["text"].split(":")[0].strip()
            relevant_rows.append(pnl_df[pnl_df['Metric'].str.contains(metric_name, case=False)])
        
        # Combine the relevant rows into a single DataFrame for display
        relevant_rows_df = pd.concat(relevant_rows).drop_duplicates().sort_values(by="Metric")
    else:
        context = "No relevant data found."
        relevant_rows_df = pd.DataFrame(columns=["Metric", "Value"])

    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"

    # Corrected to use the OpenAI Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" for the newer model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\nQuestion: {question}"}
        ],
        max_tokens=100
    )

    return response['choices'][0]['message']['content'].strip(), context, relevant_rows_df

# Initialize OpenAI API key and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Pinecone index with proper configuration
index_name = "financial-pnl-index"

# Check if index exists, if not, create one
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Update dimension based on the chosen embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created. Waiting for initialization...")
    time.sleep(60)  # Allow time for index creation

# Connect to the Pinecone index
index = pc.Index(index_name)

# Streamlit app interface
st.title("Financial P&L Query Bot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract and preprocess the PDF
    try:
        pdf_text = extract_text_from_pdf(uploaded_file)
        pnl_df = preprocess_pnl_text(pdf_text)

        st.write("Processed P&L Data:")
        st.dataframe(pnl_df)

        # Embed and store the P&L data in Pinecone
        embed_and_store_pnl_data(pnl_df, index)

        # Query input
        question = st.text_input("Ask a question about the financial data:")

        if question:
            answer, context, relevant_rows = query_pnl(question, index, pnl_df)
            st.write(f"Context:\n{context}")
            st.write(f"Answer: {answer}")

            if not relevant_rows.empty:
                st.write("Relevant Data Segments from the P&L Table:")
                st.dataframe(relevant_rows)
            else:
                st.write("No relevant data found in the P&L table.")
    
    except Exception as e:
        st.write(f"Error occurred: {str(e)}")
        st.write("Please ensure the PDF is formatted correctly.")
