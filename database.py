import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
import numpy as np
import os

# --- Configuration ---
# Replace with your actual database connection details
# Consider using environment variables for security
DB_NAME = os.getenv("PG_DB", "face_auth_db")
DB_USER = os.getenv("PG_USER", "user")
DB_PASSWORD = os.getenv("PG_PASSWORD", "password")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")

# Cosine distance threshold for recognizing a face
# You might need to tune this value (0.0 to 2.0, lower means stricter)
# Typical values are around 0.4 to 0.6 for cosine distance
# L2 distance might use different thresholds
DISTANCE_THRESHOLD = 0.5 # Adjust as needed based on testing

# --- Database Connection ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            row_factory=dict_row  # Return rows as dictionaries
        )
        register_vector(conn) # Register the vector type handler
        return conn
    except psycopg.OperationalError as e:
        print(f"Error connecting to database: {e}")
        print(f"Please ensure PostgreSQL is running, the database '{DB_NAME}' exists,")
        print(f"the user '{DB_USER}' exists with the correct password,")
        print(f"and the pgvector extension is enabled ('CREATE EXTENSION IF NOT EXISTS vector;').")
        return None

# --- Table Initialization ---

def initialize_database(conn):
    """Creates the face_embeddings table if it doesn't exist."""
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension if not already enabled
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Create the table
            # The embedding dimension depends on the insightface model used.
            # 'buffalo_l' (default) uses 512 dimensions.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    embedding VECTOR(512) NOT NULL
                );
            """)
            # Optional: Create an index for faster similarity search
            # Using IVF Flat index, adjust parameters as needed
            # Or use HNSW for potentially better recall/speed trade-off
            # cur.execute("CREATE INDEX ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
            # Or HNSW:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw ON face_embeddings USING hnsw (embedding vector_cosine_ops);")

            conn.commit()
            print("Database initialized successfully (or already exists).")
            return True
    except psycopg.Error as e:
        print(f"Error initializing database: {e}")
        conn.rollback() # Rollback changes on error
        return False

# --- Database Operations ---

def add_face(conn, name: str, embedding: np.ndarray):
    """Adds a new face embedding to the database."""
    if not conn:
        return False
    if embedding is None or not isinstance(embedding, np.ndarray):
        print("Error: Invalid embedding provided.")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
                (name, embedding)
            )
            conn.commit()
            print(f"Successfully added face embedding for '{name}'.")
            return True
    except psycopg.Error as e:
        print(f"Error adding face embedding: {e}")
        conn.rollback()
        return False

def find_similar_face(conn, embedding_to_check: np.ndarray):
    """
    Finds the most similar face in the database using cosine distance.

    Args:
        conn: Active database connection.
        embedding_to_check: The new face embedding (NumPy array).

    Returns:
        A tuple (name, distance) if a similar face is found within the threshold,
        otherwise (None, None). Returns (None, float('inf')) if db error.
    """
    if not conn:
        return None, float('inf')
    if embedding_to_check is None or not isinstance(embedding_to_check, np.ndarray):
        print("Error: Invalid embedding provided for comparison.")
        return None, float('inf')

    try:
        with conn.cursor() as cur:
            # Using <=> for cosine distance with pgvector
            # (can also use <-> for L2 distance or <#> for inner product)
            cur.execute(
                """
                SELECT name, embedding <=> %s AS distance
                FROM face_embeddings
                ORDER BY distance ASC
                LIMIT 1;
                """,
                (embedding_to_check,)
            )
            result = cur.fetchone()

            if result and result['distance'] is not None and result['distance'] < DISTANCE_THRESHOLD:
                print(f"Match found: {result['name']} (Distance: {result['distance']:.4f})")
                return result['name'], result['distance']
            elif result:
                print(f"Closest match: {result['name']} (Distance: {result['distance']:.4f}) - Threshold not met.")
                return None, result['distance'] # Return distance even if no match
            else:
                print("No faces found in the database for comparison.")
                return None, None # No faces in DB

    except psycopg.Error as e:
        print(f"Error querying for similar faces: {e}")
        return None, float('inf') # Indicate error with infinite distance
    except ValueError as e:
        # Handle potential issues if the vector format is incorrect during comparison
        print(f"Vector comparison error: {e}")
        return None, float('inf')

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Testing database module...")
    connection = get_db_connection()
    if connection:
        if initialize_database(connection):
            # Example: Add a dummy face (replace with actual embedding later)
            # dummy_embedding = np.random.rand(512).astype(np.float32)
            # add_face(connection, "Test Person", dummy_embedding)

            # Example: Find a similar face (replace with actual embedding later)
            # query_embedding = np.random.rand(512).astype(np.float32)
            # name, distance = find_similar_face(connection, query_embedding)
            # if name:
            #     print(f"Found similar face: {name} with distance {distance}")
            # else:
            #     print("No similar face found within the threshold.")
            pass # Keep pass here, examples commented out

        connection.close()
        print("Database connection closed.")
    else:
        print("Failed to establish database connection.") 