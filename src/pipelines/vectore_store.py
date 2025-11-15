from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os


class TextChunker:
    """
    A utility class for splitting text documents into smaller chunks
      with optional overlap.

    Attributes:
        chunk_size (int): The maximum size of each text chunk. Defaults to 250.
        chunk_overlap (int): The number of overlapping characters between
          consecutive chunks. Defaults to 50.

    Methods:
        split(docs)

    Example:
        text_chunker = TextChunker(chunk_size=300, chunk_overlap=75)
        chunks = text_chunker.split(docs)
    """

    def __init__(self, chunk_size=250, chubk_overlap=50):
        """
        Initializes the vector store with specified chunk size and overlap.

        Args:
            chunk_size (int, optional): The size of each chunk. Defaults to 250.
            chunk_overlap (int, optional): The overlap size between chunks.
              Defaults to 50.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chubk_overlap

    def split(self, docs):
        """
        Splits a list of documents into smaller chunks based on the specified
          chunk size and overlap.

        Args:
            docs (list): A list of documents to be split. Each document is expected to
                         be a string or a similar text-based object.

        Returns:
            list: A list of smaller document chunks obtained by splitting the input
                  documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs_split = splitter.split_documents(docs)
        return docs_split


class Embedder:
    """
    A class for generating embeddings using a specified model.

    Attributes:
        embedder (SentenceTransformerEmbeddings): An instance of the embedding model.

    Methods:
        get()
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the vector store with a specified embedding model.

        Args:
            model_name (str): The name of the embedding model to use.
              Defaults to "all-MiniLM-L6-v2".
        """
        self.embedder = SentenceTransformerEmbeddings(model_name=model_name)

    def get(self):
        """
        Retrieve the embedder instance.

        Returns:
            object: The embedder instance.
        """
        return self.embedder


class VectorStore:
    """
    A class to manage a vector store for document embeddings and retrieval.

    Attributes:
        embedder: An embedding model used to convert documents into
          vector representations.
        db: The vector store database, initialized as None.

    Methods:
        create_vectore_store(docs_split)
        retriever()
    """

    def __init__(self, embedder, store_path="vector_store.faiss"):
        """
        Initializes the vector store with the given embedder.

        Args:
            embedder: An object responsible for generating embeddings.
            store_path (str): The file path to store the vector database.
              Defaults to "vector_store.faiss".
        """
        self.embedder = embedder
        self.store_path = store_path
        if os.path.exists(store_path):
            self.db = FAISS.load_local(
                store_path, self.embedder, allow_dangerous_deserialization=True
            )
        else:
            self.db = None

    def create_vector_store(self, docs_split):
        """
        Creates a vector store using the FAISS library from the provided document splits.

        Args:
            docs_split (list): A list of document splits to be embedded
              and stored in the vector store.

        Returns:
            FAISS: The created FAISS vector store instance.
        """
        self.db = FAISS.from_documents(docs_split, self.embedder)
        self.db.save_local(self.store_path)
        return self.db

    def retriever(self):
        """
        Retrieve a retriever object from the vector store.

        This method returns a retriever object that can be used to query
          the vector store.
        If the vector store has not been initialized, an exception is raised.

        Returns:
            Any: A retriever object for querying the vector store.

        Raises:
            ValueError: If the vector store (`db`) has not been created yet.
        """
        if self.db is None:
            raise ValueError("Vector store has not been created yet.")
        return self.db.as_retriever()
