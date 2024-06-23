## Import Libraries

from fastapi import FastAPI, APIRouter
import uvicorn
from langchain_community.vectorstores import FAISS
from rag import (
    get_qa_instance,
    write_to_db,
    get_answer,
    get_embeddings,
    create_qa_model_pipeline,
    EMBEDDING_MODEL_PATH,
    LLM_MODEL_PATH,
    FAISS_SAVE_PATH,
)


app = FastAPI(
    title="Langchain Server", version="1.0", description="A simple API server"
)
router = APIRouter()


@router.get("/")
async def home():
    return {"output": "Custom RAG service"}


@router.post("/custom_rag")
async def data(data: dict) -> dict:
    """
    Endpoint to handle POST requests for custom RAG (Retrieval-Augmented Generation) operations.

    Retrieves input text from the request data, initializes a retriever and question-answering pipeline,
    performs a query using the input text, and returns the generated answer.

    Parameters:
    data (dict): A dictionary containing input data from the POST request. Assumes the structure {'input': input_text},
                 where 'input_text' is the text for which an answer is sought.

    Returns:
    output (dict): A dictionary containing the generated answer as {'output': answer}.
    """

    input_text = data["input"]

    # Load FAISS vectorstore and create retriever
    vectorstore = FAISS.load_local(
        FAISS_SAVE_PATH,
        get_embeddings(EMBEDDING_MODEL_PATH),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # Create question-answering pipeline and instance
    llm_model = create_qa_model_pipeline(LLM_MODEL_PATH)
    qa = get_qa_instance(llm_model, retriever)

    # Retrieve answer using the QA instance and input_text
    output = {"output": get_answer(qa, input_text)}
    return output


app.include_router(router)


if __name__ == "__main__":
    # Stores the embeddings to FAISS vector store.
    retriever = write_to_db()

    # Run the Uvicorn server with the specified configurations
    uvicorn.run("api_app:app", reload=True, port=8000, host="localhost")
