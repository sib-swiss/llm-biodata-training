from langchain_mistralai import ChatMistralAI
from langchain_core.language_models import BaseChatModel

def load_chat_model(model: str) -> BaseChatModel:
    provider, model_name = model.split("/", maxsplit=1)
    if provider == "mistral":
        # https://python.langchain.com/docs/integrations/chat/mistralai/
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=model_name,
            temperature=0,
            max_retries=2,
            random_seed=42,
        )
    if provider == "google":
        # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


llm = load_chat_model("mistral/mistral-large-latest")
# llm = load_chat_model("google/gemini-2.0-flash")


SYSTEM_PROMPT = """You are an assistant that helps users to navigate the resources and databases from the SIB Swiss Institute of Bioinformatics.
Here is the description of resources available at the SIB:
{context}
Use it to answer the question"""

from index import vectordb, embedding_model, collection_name

def ask(question: str) -> str:
    # Generate embeddings for the user question
    question_embeddings = next(iter(embedding_model.embed([question])))
    # Find similar embeddings in the vector database
    retrieved_docs = vectordb.query_points(
        collection_name=collection_name,
        query=question_embeddings,
        limit=10,
    )
    print(f"📚️ Retrieved {len(retrieved_docs.points)} documents")
    formatted_docs = '\n------\n'.join(doc.payload["description"] for doc in retrieved_docs.points)
    messages = [
        ("system", SYSTEM_PROMPT.format(context=formatted_docs)),
        ("human", question),
    ]
    print(formatted_docs)
    print("START MESSAGE\n\n")
    for resp in llm.stream(messages):
        print(resp.content, end="")
        if resp.usage_metadata:
            print(f"\n\n{resp.usage_metadata}")

ask("Which tools can I use for comparative genomics?")

