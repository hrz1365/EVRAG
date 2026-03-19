from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever

import os
import pickle
from typing import List
# from langchain.schema import Document

class WeightedEnsembleRetriever:
    def __init__(self, retrievers: List, weights: List):
        self.retrievers = retrievers
        self.weights = weights
        print("Hybrid Reranker")

    def invoke(self, query: str) -> List:
        scored_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            # must use similarity_search_with_score if FAISS
            if hasattr(retriever, "similarity_search_with_score"):
                docs_and_scores = retriever.similarity_search_with_score(query, k=5)
                for doc, score in docs_and_scores:
                    doc.metadata["weighted_score"] = score * weight
                    scored_docs.append(doc)
            else:
                # fallback for retrievers without score
                docs = retriever._get_relevant_documents(query,run_manager=None)
                for doc in docs:
                    doc.metadata["weighted_score"] = 1.0 * weight
                    scored_docs.append(doc)
        # optional: sort by score
        scored_docs.sort(key=lambda d: getattr(d, "score", 0), reverse=True)
        print(scored_docs)
        exit()
        return scored_docs


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

    def __init__(self, chunk_size=400, chunk_overlap=50):
        """
        Initializes the vector store with specified chunk size and overlap.

        Args:
            chunk_size (int, optional): The size of each chunk. Defaults to 250.
            chunk_overlap (int, optional): The overlap size between chunks.
              Defaults to 50.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n"],
        )
        docs_split = splitter.split_documents(docs)
        print(f"Split into {len(docs_split)} chunks.")
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

    def __init__(self, embedder, search_type="hybrid", store_path_faiss="vector_store.faiss",  store_path_bm25="bm25_retriever.pkl"):
        """
        Initializes the vector store with the given embedder.

        Args:
            embedder: An object responsible for generating embeddings.
            store_path (str): The file path to store the vector database.
              Defaults to "vector_store.faiss".
        """
        if not search_type in ["hybrid", "mmr", "similarity"]:
            raise("Search type is unacceptable", search_type)
        self.search_type=search_type
        self.embedder = embedder
        self.store_path_faiss = store_path_faiss
        self.store_path_bm25 =store_path_bm25
        if os.path.exists(store_path_faiss):
            self.db = FAISS.load_local(
                self.store_path_faiss, self.embedder, allow_dangerous_deserialization=True
            )
            
            with open(self.store_path_bm25, "rb") as f:
                self.bm25_retriever =  pickle.load(f)
        else:
            self.db = None
            self.bm25_retriever = None

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
        self.db.save_local(self.store_path_faiss)

        self.bm25_retriever = BM25Retriever.from_documents(docs_split)
        with open(self.store_path_bm25, "wb") as f:
            pickle.dump(self.bm25_retriever , f)
        
        # return self.db, self.bm25_retriever

    def retriever(self, k=7):
        """
        Retrieve the top-k most relevant items from the vector store.

        This method uses the vector store to find and return the top-k items
        that are most relevant to a given query. The number of items to retrieve
        can be specified using the `k` parameter.

        Args:
            k (int, optional): The number of most relevant items to retrieve. Defaults to 3.

        Returns:
            Callable: A retriever object configured with the specified search parameters.

        Raises:
            ValueError: If the vector store (`db`) has not been created yet.
        """

        if self.db is None:
            raise ValueError("Vector store has not been created yet.")
        if self.bm25_retriever is None and self.search_type=="hybrid":
            raise ValueError("BM25 store has not been created yet.")

        
        if self.search_type =="hybrid":
            self.bm25_retriever.k = k

            hybrid_retriever = WeightedEnsembleRetriever(
                retrievers=[self.bm25_retriever, self.db.as_retriever(search_kwargs={"k": k, "fetch_k": 2*k})], 
                weights=[0.5, 0.5] 
            )
            return  hybrid_retriever
        if self.search_type=="similarity" or self.search_type=="mmr":
            return self.db.as_retriever(
                search_type=self.search_type, # Options: "similarity", "mmr", "similarity_score_threshold"
                search_kwargs={"k": k, "fetch_k": 2*k}) # Fetch 2k, return best k )
