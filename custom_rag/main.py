from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


## CONSTANTS

# PDF Document path
DOCUMENT_PATH = "./inputs/ConceptsofBiology-WEB.pdf"
# Define the path to the pre-trained model for embeddings
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"
# Define the path to the pre-trained LLM.
LLM_MODEL_PATH = "Intel/dynamic_tinybert"


## FUNCTIONS


def get_pdf_document(path: str):
    loader = PyMuPDFLoader(path)
    document = loader.load()
    return document


def preprocess_document(document):
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    final_documents = text_splitter.split_documents(document)
    return final_documents


def get_embeddings(model_path: str):

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


def create_vector_store_retriever(final_documents, embeddings):
    ## VectorStore Creation
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    # Create a retriever object from the 'vectorstore' with a search configuration
    # where it retrieves up to 4 relevant splits/documents.
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return retriever


def create_qa_model_pipeline(model_path):
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding=True, truncation=True, max_length=512
    )
    # Load the QA model associated with the specified model
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    # Define a question-answering pipeline using the model and tokenizer
    pipe = pipeline(
        "question-answering", model=model, tokenizer=tokenizer, return_tensors="pt"
    )
    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm_model = HuggingFacePipeline(
        pipeline=pipe, model_kwargs={"temperature": 0.7, "max_length": 512}
    )

    return llm_model


def get_qa_instance(llm_model, retriever):
    # Create a question-answering instance (qa) using the RetrievalQA class.
    # It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
    qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=False,
    )
    return qa


def init_custom_rag():
    document = get_pdf_document(DOCUMENT_PATH)
    preprocessed_document = preprocess_document(document)
    embeddings = get_embeddings(EMBEDDING_MODEL_PATH)
    retriever = create_vector_store_retriever(preprocessed_document, embeddings)
    llm_model = create_qa_model_pipeline(LLM_MODEL_PATH)
    qa = get_qa_instance(llm_model, retriever)
    return qa


def get_answer(qa, query: str) -> str:
    try:
        answer = qa.invoke({"query": query})
        return answer["result"]
    except ValueError as ve:
        return str(ve).split("------------")[1]


## MAIN


if __name__ == "__main__":
    qa = init_custom_rag()
    query = "Evolution"
    answer = get_answer(qa, query)
    print(answer)
