# chatbot.py

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
            # Add other parameters if needed
        )

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Only return the helpful answer. Answer must be detailed and well explained.

Context: {context}
"""

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )


        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            # Initialize the prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.prompt_template),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            retriever_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            response = retriever_chain.invoke({"input": query})
            return response['answer']
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e} - Source: chatbot.py")
            return "Sorry, I couldn't process your request at the moment."