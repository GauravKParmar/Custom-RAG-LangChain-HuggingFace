## Import Libraries

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


## Constants


# PDF Document path
DOCUMENT_PATH = "./inputs/ConceptsofBiology-WEB.pdf"

# Define the path to the pre-trained model for embeddings
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"

# Define the path to save FAISS vectorstore.
FAISS_SAVE_PATH = "./faiss_db/faiss_index"

# Define the path to the pre-trained LLM.
LLM_MODEL_PATH = "Intel/dynamic_tinybert"


## Functions


def get_pdf_document(path: str) -> list:
    """
    Loads a PDF document from the specified path using PyMuPDFLoader and returns a list of documents.

    Parameters:
    path (str): The file path to the PDF document.

    Returns:
    documents (list): A list of documents loaded from the PDF file. Each document typically corresponds to a page or section
          of the PDF.
    """

    # Create an instance of PyMuPDFLoader pointing to document path.
    loader = PyMuPDFLoader(path)

    # Using the instance load the pdf file.
    documents = loader.load()

    # Return the extracted document list.
    return documents


def preprocess_document(document: list) -> list:
    """
    Preprocesses a list of documents by splitting them into smaller chunks based on specified Markdown-style separators.

    Parameters:
    document (list): A list of strings representing documents to be preprocessed.

    Returns:
    final_documents (list): A list of preprocessed documents where each document is split into smaller chunks based on the provided Markdown-style separators.
    """

    # Defining markdown style separators.
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    # Create instance of RecursiveCharacterTextSplitter to split the text in the documents.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The maximum number of characters in a chunk.
        chunk_overlap=100,  # The number of characters to overlap between chunks.
        add_start_index=True,  # If 'True', includes chunk's start index in metadata
        strip_whitespace=True,  # If 'True', strips whitespace from the start and end of every document
        separators=MARKDOWN_SEPARATORS,
    )

    # Split the texts in the document using the instance.
    final_documents = text_splitter.split_documents(document)

    # Return the processed documents.
    return final_documents


def get_embeddings(model_path: str):
    """
    Initialize and return an instance of HuggingFaceEmbeddings for a specified pre-trained model.

    Parameters:
    model_path (str): The file path or model identifier specifying the pre-trained model to be loaded.

    Returns:
    HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
    """

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {"device": "cpu"}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {"normalize_embeddings": True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs,  # Pass the encoding options
    )

    return embeddings


def create_vector_store_retriever(final_documents: list, embeddings):
    """
    Creates and returns a retriever object from a vector store created using FAISS,
    initialized with embeddings generated from the provided final_documents.

    Parameters:
    final_documents (list): A list of preprocessed documents or text splits.
    embeddings: Embeddings used to represent the final_documents in a vector space.

    Returns:
    retriever: A retriever object configured to retrieve up to 4 relevant splits/documents
               from the vector store based on similarity search.
    """
    # VectorStore Creation
    vectorstore = FAISS.from_documents(final_documents, embeddings)

    # Save vectors locally
    vectorstore.save_local(FAISS_SAVE_PATH)

    # Create a retriever object from the 'vectorstore' with a search configuration that retrieves up to 4 relevant splits/documents.
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return retriever


def create_qa_model_pipeline(model_path: str):
    """
    Creates a question-answering model pipeline using a specified pre-trained model.

    Parameters:
    model_path (str): The file path or model identifier specifying the pre-trained QA model to be loaded.

    Returns:
    HuggingFacePipeline: A pipeline object configured for question-answering tasks, using the
                         specified model_path for both tokenizer and model initialization.
    """
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding=True, truncation=True, max_length=512
    )

    # Load the QA model associated with the specified model
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    # Define a question-answering pipeline using the model and tokenizer
    pipe = pipeline(
        task="question-answering", model=model, tokenizer=tokenizer, return_tensors="pt"
    )

    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm_model = HuggingFacePipeline(
        pipeline=pipe, model_kwargs={"temperature": 0.7, "max_length": 512}
    )

    return llm_model


def get_qa_instance(llm_model, retriever):
    """
    Creates a question-answering instance using the RetrievalQA class.

    Parameters:
    llm_model (HuggingFacePipeline): A HuggingFacePipeline instance configured for question-answering tasks.
    retriever: A retriever object capable of retrieving relevant documents or segments based on similarity.

    Returns:
    RetrievalQA: An instance of RetrievalQA configured with the provided llm_model, retriever, and chain_type='refine',
                 and option to not return source documents.
    """
    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine," the retriever, and an option to not return source documents.
    qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa


def write_to_db():
    """
    Retrieves a retriever object configured with preprocessed documents and embeddings, ready to be used for document retrieval.

    This function orchestrates several steps:
    1. Retrieves a PDF document from the specified DOCUMENT_PATH using get_pdf_document.
    2. Preprocesses the document into smaller chunks using preprocess_document.
    3. Obtains embeddings for the preprocessed document using get_embeddings.
    4. Creates a retriever object using create_vector_store_retriever with the preprocessed_document and embeddings.

    Returns:
    retriever: A retriever object configured with preprocessed documents and embeddings, suitable for document retrieval tasks.
    """
    # Retrieve PDF document from DOCUMENT_PATH
    document = get_pdf_document(DOCUMENT_PATH)

    # Preprocess the document into smaller chunks
    preprocessed_document = preprocess_document(document)

    # Obtain embeddings for the preprocessed document
    embeddings = get_embeddings(EMBEDDING_MODEL_PATH)

    # Create a retriever object using preprocessed_document and embeddings
    retriever = create_vector_store_retriever(preprocessed_document, embeddings)

    return retriever


def get_answer(qa, query: str) -> str:
    """
    Retrieves an answer to a given query using the provided question-answering instance (qa).

    Parameters:
    qa (RetrievalQA): An instance of RetrievalQA configured for question-answering tasks.
    query (str): The query or question for which an answer is sought.

    Returns:
    output (str): The answer to the query as generated by the qa instance.
    """
    try:
        answer = qa.invoke({"query": query})
        output = answer["result"]
        return output
    except ValueError as ve:
        output = str(ve).split("------------")[1]
        return output
