import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from groq import Groq
from langchain_core.tools import tool
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
TOKENIZERS_PARALLELISM = os.getenv('TOKENIZERS_PARALLELISM')

# Initialize vector store as a global variable to avoid reloading
_vector_store = None


def initialize_vector_store(persist_directory="./chroma_langchain_db", collection_name="document_collection"):
    """Initialize the vector store without using the tool wrapper"""
    global _vector_store
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        return True
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return False


@tool
def load_vector_store(persist_directory: str = "./chroma_langchain_db",
                      collection_name: str = "document_collection") -> str:
    """
    Load the vector store with embeddings.

    Args:
        persist_directory: Directory where the vector store is persisted
        collection_name: Name of the collection in the vector store

    Returns:
        A message indicating whether the vector store was loaded successfully
    """
    global _vector_store
    success = initialize_vector_store(persist_directory, collection_name)
    if success:
        doc_count = _vector_store.get().get('ids', []).__len__()
        return f"Vector store loaded successfully with {doc_count} documents"
    return "Failed to load vector store"


@tool
def get_available_documents(documents_dir: str = "./documents") -> List[str]:
    """
    Get a list of available PDF documents.

    Args:
        documents_dir: Directory containing the PDF documents

    Returns:
        A list of PDF document filenames
    """
    if os.path.exists(documents_dir):
        document_files = [doc for doc in os.listdir(documents_dir) if doc.lower().endswith('.pdf')]
        return document_files
    return []


@tool
def search_documents(query: str, k: int = 1) -> List[Dict[str, Any]]:
    """
    Search documents based on query similarity.

    Args:
        query: The search query
        k: Number of results to return

    Returns:
        A list of search results with source and content
    """
    global _vector_store
    if not _vector_store:
        if not initialize_vector_store():
            return [{"error": "Vector store could not be initialized."}]

    try:
        search_results = _vector_store.similarity_search(query, k=k)
        formatted_results = []
        for doc in search_results:
            formatted_results.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "content": doc.page_content
            })
        return formatted_results
    except Exception as e:
        return [{"error": f"Error searching documents: {str(e)}"}]


@tool
def get_llm_response(context: str, query: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Get LLM response based on context and query.

    Args:
        context: Document context to provide to the LLM
        query: The user query
        model: The LLM model to use

    Returns:
        The LLM's response
    """
    try:
        # Initialize Groq client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Create chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant that answers questions based on PDF content."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ],
            model=model,
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"


@tool
def search_and_answer(query: str, k: int = 1) -> str:
    """
    Search documents and get LLM answer based on search results.

    Args:
        query: The user query
        k: Number of search results to consider

    Returns:
        The LLM's answer based on the search results
    """
    global _vector_store
    if not _vector_store:
        if not initialize_vector_store():
            return "Failed to initialize vector store."

    try:
        results = search_documents(query, k)

        if not results or (len(results) == 1 and "error" in results[0]):
            return f"No results found for query: '{query}'"

    except Exception as e:
        return f"Error searching documents: {str(e)}"
    # Process results
    context = ""
    for result in results:
        source = result.get("source", "Unknown")
        content = result.get("content", "")
        context += f"Source: {source}\nContent: {content}\n\n"

    # Get LLM response by calling the function directly
    try:
        if hasattr(get_llm_response, "func"):
            return get_llm_response.func(context, query)
        else:
            # Fallback if .func attribute doesn't exist
            return get_llm_response(context=context, query=query)
    except Exception as e:
        return f"Error getting LLM response: {str(e)}"


@tool
def get_comprehensive_answer(query: str) -> str:
    """
    Generate a comprehensive answer about a topic without searching documents.

    Args:
        query: The topic or question to answer

    Returns:
        A comprehensive explanation about the topic
    """
    try:
        # Initialize Groq client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Create chat completion with a comprehensive prompt
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system",
                 "content": "You are a knowledgeable assistant that provides comprehensive explanations on various topics."},
                {"role": "user",
                 "content": f"Please provide a detailed explanation on the following topic or question: {query}"}
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating comprehensive answer: {str(e)}"


def create_agent():
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_groq import ChatGroq
    from langchain import hub

    # Initialize vector store at the start
    initialize_vector_store()

    # Define the tools for the agent
    tools = [
        load_vector_store,
        get_available_documents,
        search_documents,
        get_llm_response,
        search_and_answer,
        get_comprehensive_answer  # Added new tool for direct answers
    ]

    # Load the LLM
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    # Get the prompt template
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return agent_executor


# Example usage
if __name__ == "__main__":
    # Create the agent
    agent = create_agent()

    # Run the agent
    try:
        response = agent.invoke({"input": "What is an Attention Mechanism?"})
        print("\nFinal Answer:")
        print(response["output"])
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        import traceback

        traceback.print_exc()
