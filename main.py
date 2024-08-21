import os  # For environment variable management
import streamlit as st  # For the web interface
import pinecone  # For vector database management
from langchain.embeddings import SentenceTransformerEmbeddings  # For creating embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone  # For integrating Pinecone with Langchain
from langchain.llms import OpenAI  # For using OpenAI's language models
from langchain.chains import RetrievalQA  # For retrieval-based question answering

# Load API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Pinecone API key
openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API key

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="YOUR_PINECONE_ENVIRONMENT")  # Replace with your Pinecone environment
index_name = "vitafy-products"  # Replace with your index name
index = pinecone.Index(index_name)  # Connect to the Pinecone index

# Initialize OpenAI with GPT-4o Mini
llm = OpenAI(model="gpt-4o-mini", temperature=0.7)  # Set up the OpenAI model

# Create a vector store with embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Load the embedding model
vectorstore = LangchainPinecone(index, embeddings)  # Create a Langchain-compatible Pinecone vector store

# Set up the RetrievalQA chain
qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())  # Create the QA chain

# Streamlit UI
st.title("VitafyAI")  # Set the title of the web app
user_input = st.text_input("Ask a question about Vitafy products:")  # Input field for user questions

if st.button("Submit"):  # Button to submit the question
    if user_input:  # Check if user input is not empty
        try:
            # Get the answer from the QA chain
            answer = qa_chain.run(user_input)  # Run the QA chain with the user input
            st.write(answer)  # Display the answer to the user
        except Exception as e:  # Handle any errors that occur
            st.error(f"An error occurred: {e}")  # Display an error message
    else:
        st.warning("Please enter a question.")  # Prompt user to enter a question if input is empty
