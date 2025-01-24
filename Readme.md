# Financial P&L Query Bot

## Overview
The **Financial P&L Query Bot** is an AI-powered tool designed to extract, process, and query financial data from PDF profit and loss (P&L) statements. Using OpenAI's GPT models and Pinecone for vector-based similarity search, this bot allows users to ask natural language questions about their financial data and receive accurate insights.

## Features
- **Extract financial data** from PDF P&L statements
- **Preprocess financial data** into structured formats
- **Embed and store data** in Pinecone for efficient retrieval
- **Query financial data** using natural language
- **Interactive web interface** built with Streamlit

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip (Python package manager)

### Clone the Repository
```bash
git clone https://github.com/Saikeerthansa/Financial-QA-Bot.git
cd Financial-QA-Bot
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the project root directory and add the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
> **Important:** Do not share or commit your `.env` file to the repository.

---

## Usage

### Running the Application
Start the Streamlit web application using the command:
```bash
streamlit run app.py
```

### Uploading Financial Statements
1. Open the app in your web browser at `http://localhost:8501`
2. Upload a PDF financial statement.
3. View the processed data.

### Querying Financial Data
- Enter a natural language query in the input box (e.g., "What was the net income?")
- The bot will provide answers based on extracted data

---

## File Structure
```
.
├── .dockerignore          # Ignore files for Docker
├── .env                   # Environment variables (excluded from Git)
├── app.py                 # Main application script
├── collab_notebook.ipynb   # Jupyter Notebook for experimentation
├── Dockerfile              # Dockerfile for containerization
├── Financial-QA-Bot documentation.docx  # Project documentation
├── requirements.txt        # Python dependencies
├── Sample Financial Statement.pdf  # Sample input file
└── README.md               # Project documentation
```

---

## Deployment with Docker

### Build Docker Image
```bash
docker build -t financial-qa-bot .
```

### Run Docker Container
```bash
docker run -p 8501:8501 financial-qa-bot
```
