from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# System prompt for RAG-powered microwave assistant
SYSTEM_PROMPT = """You are a RAG-powered assistant specialized in microwave usage and troubleshooting.

Your responses are based on context retrieved from a microwave manual.

User messages will be structured as follows:
1. RAG Context - relevant information retrieved from the microwave manual
2. User Question - the actual question from the user

Instructions:
- Use the provided RAG Context to answer the User Question accurately
- Only answer questions related to microwave usage, operation, and troubleshooting
- If the question is not related to microwave usage or the provided context, politely decline and explain that you can only help with microwave-related questions
- If the context doesn't contain relevant information, say that you don't have enough information to answer
- Be concise and helpful in your responses
"""

# User prompt template with RAG Context and User Question sections
USER_PROMPT = """
### RAG Context:
{context}

### User Question:
{question}
"""


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

# Initialize clients
embeddings_client = DialEmbeddingsClient(
    deployment_name='text-embedding-3-small-1',
    api_key=API_KEY
)

chat_client = DialChatCompletionClient(
    deployment_name='gpt-4o',
    api_key=API_KEY
)

text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config=DB_CONFIG
)


def run_console_chat():
    """Run an interactive console chat with RAG capabilities."""
    print("🔧 Microwave Assistant (RAG-powered)")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    # Initialize conversation with system prompt
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    text_processor.process_text_file('embeddings/microwave_manual.txt', truncate_table=True)

    while True:
        # Get user input from console
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Retrieve context from vector database
        context_results = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=5,
            min_score_threshold=0.8
        )

        # Perform augmentation - combine context with user question
        context_text = "\n".join(context_results) if context_results else "No relevant context found."
        augmented_prompt = USER_PROMPT.format(
            context=context_text,
            question=user_input
        )

        # Add augmented user message to conversation
        conversation.add_message(Message(Role.USER, augmented_prompt))

        # Perform generation
        try:
            response = chat_client.get_completion(conversation.get_messages())
            conversation.add_message(response)
            print(f"\nAssistant: {response.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    # NOTE: Make sure to run docker-compose.yml first to start PostgreSQL with pgvector extension!
    # docker-compose up -d
    run_console_chat()
