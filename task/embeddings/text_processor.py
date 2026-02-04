from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
        self,
        file_name: str,
        chunk_size: int = 500,
        overlap: int = 50,
        dimensions: int = 1536,
        truncate_table: bool = False
    ) -> None:
        """
        Process a text file: chunk it, generate embeddings, and store in DB.

        Args:
            file_name: Path to the text file to process
            chunk_size: Size of each text chunk
            overlap: Overlap between chunks
            dimensions: Embedding dimensions
            truncate_table: Whether to truncate the vectors table before inserting
        """
        # Truncate table if needed
        if truncate_table:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("TRUNCATE TABLE vectors")
                conn.commit()

        # Load content from file and generate chunks
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = chunk_text(content, chunk_size, overlap)

        if not chunks:
            return

        # Generate embeddings from chunks
        embeddings_dict = self.embeddings_client.get_embeddings(chunks)

        # Save embeddings and chunks to DB
        document_name = file_name.split('/')[-1]

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for index, chunk in enumerate(chunks):
                    embedding = embeddings_dict[index]
                    embedding_str = str(embedding)

                    cursor.execute(
                        """
                        INSERT INTO vectors (document_name, text, embedding)
                        VALUES (%s, %s, %s::vector)
                        """,
                        (document_name, chunk, embedding_str)
                    )
            conn.commit()

    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int = 5,
        min_score_threshold: float = 0.5,
        dimensions: int = 1536
    ) -> list[str]:
        """
        Search for relevant context based on user request.

        Args:
            search_mode: The search mode (Euclidean or Cosine distance)
            user_request: The user's query text
            top_k: Number of top results to return
            min_score_threshold: Minimum distance threshold for filtering results
            dimensions: Embedding dimensions

        Returns:
            List of relevant text chunks
        """
        # Generate embeddings from user request
        embeddings_dict = self.embeddings_client.get_embeddings([user_request])
        query_embedding = embeddings_dict[0]
        query_embedding_str = str(query_embedding)

        # Determine distance operator based on search mode
        if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
            distance_operator = "<->"
        else:  # COSINE_DISTANCE
            distance_operator = "<=>"

        # Search in DB for relevant context
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = f"""
                    SELECT text, embedding {distance_operator} %s::vector AS distance
                    FROM vectors
                    WHERE embedding {distance_operator} %s::vector < %s
                    ORDER BY distance
                    LIMIT %s
                """
                cursor.execute(query, (query_embedding_str, query_embedding_str, min_score_threshold, top_k))
                results = cursor.fetchall()

        return [row['text'] for row in results]


