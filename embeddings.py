# vectors.py

import os
import base64

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document

import fitz
import re
import pytesseract
from PIL import Image
import io



def extract_text(pdf_document):
    """
    Extracts text from a PDF document, handling both text-based and image-based pages.

    Args:
        pdf_document (str): The file path to the PDF document.

    Returns:
        list: 
            - Extracted text from the PDF.
            - Metadata (placeholder, currently None).
            - A structure mapping the start indexes of text chunks for each page.
    """
    # Initialize variables to store results
    text = ""  
    metadata = None
    start_indexes = []

    # Open the PDF document using PyMuPDF (fitz)
    with fitz.open(pdf_document) as doc:
        if not doc:
            raise ValueError("The PDF document could not be opened.")  # Handle invalid PDF
        
        start_index = 0  # Initialize the start index for the first chunk of text

        # Iterate through all pages in the document
        for page_num in range(len(doc)):
            # Load the current page and extract text
            page = doc.load_page(page_num)
            page_text = page.get_text()

            # Check if the page contains extractable text
            if page_text.strip():  
                extracted_text = page_text
            else:  
                # If the page is image-based, process it with OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                extracted_text = pytesseract.image_to_string(img)

            # Clean the extracted text by removing excessive newlines
            current_text = re.sub(r'\s*\n\s*\n+', '\n', extracted_text)

            # Record and update the start index of the text
            start_indexes.append(start_index)
            start_index += len(current_text)
            text += current_text

    return [text, metadata, [[1], start_indexes]]

def split_chunks_document(docs):
    """
    Splits a document into smaller chunks based on the provided indexes and attaches metadata to each chunk.

    Args:
        documents (str): The full text of the document to be split.
        metadata (dict): Metadata associated with the document (can be None).
        indexes (list of lists): A list of start indexes for splitting the document.
                                 - If an inner list has a single index, the document is treated as a whole.
                                 - If the inner list has multiple indexes, the document is split into chunks.
        digest (str): A unique identifier or hash for the document, used for metadata.

    Returns:
        list: A list of `Document` objects, each containing a chunk of text and associated metadata.
    """
    documents, metadata, indexes = docs
    chunks = []

    # Iterate through each set of start indexes in the provided indexes list
    for start_index in indexes:
        # If document as whole wants to be processed
        if len(start_index) == 1:
            document = Document(
                page_content=documents,
                metadata=(
                    {**metadata,"start_index": 0, "document": "whole"}
                    if metadata 
                    else {"start_index": 0, "document": "whole"} 
                ),
            )
            # Append the whole document to the chunks list
            chunks.append(document)
        else:
            # Process document based on each page
            for i in range(len(start_index)):
                start = start_index[i]
                if i < len(start_index) - 1:
                    end = start_index[i + 1]
                    doc = documents[start:end]
                else:
                    doc = documents[start:]

                if doc:
                    document = Document(
                        page_content=doc,
                        metadata=(
                            {**metadata, "start_index": start, "document": "specific"} 
                            if metadata
                            else {"start_index": start, "document": "specific"}
                        ),
                    )
                    # Append the chunk to the chunks list
                    chunks.append(document)
    return chunks


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the EmbeddingsManager with the specified model and Qdrant settings.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        """
        Processes the PDF, creates embeddings, and stores them in Qdrant.

        Args:
            pdf_path (str): The file path to the PDF document.

        Returns:
            str: Success message upon completion.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load and preprocess the document
        docs = extract_text(pdf_path)
        
        # Split the text into chunks based on page
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=250
        # )
        # splits = text_splitter.split_documents(docs)
        splits = split_chunks_document(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        # Create and store embeddings in Qdrant
        try:
            qdrant = Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "Vector DB Successfully Created and Stored in Qdrant!"
