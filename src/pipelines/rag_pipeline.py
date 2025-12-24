from src.pipelines import PDFLoader, TextChunker, Embedder, VectorStore, LLMEngineGemini
import glob


class RAGPipeline:
    """
    A pipeline for implementing a Retrieval-Augmented Generation (RAG) system.

    This class provides functionality to load a PDF document, process
      and chunk its content, create a vector store for efficient retrieval,
      and query the system using a language model.

    Attributes:
        pdf_loader (PDFLoader): An instance for loading PDF documents.
        chunker (TextChunker): An instance for splitting text into chunks.
        embedder (object): The embedding model used for creating vector representations.
        vector_store (VectorStore): A storage system for managing vectorized document
          chunks.
        llm_engine (LLMEngine): A language model engine for generating responses.

    Methods:
        build_index()
        query(question: str, k=3) -> str

    """

    def __init__(self, pdf_path, embed_model="all-MiniLM-L6-v2"):
        """
        Initializes the pipeline with the necessary components for processing PDFs,
        embedding text, and interacting with a language model.

        Args:
            pdf_path (str): The directory path with PDF documents to be processed.
            embed_model (str, optional): The name of the embedding model to use.
                Defaults to "all-MiniLM-L6-v2".
        """
        self.pdf_path = pdf_path
        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker()
        self.embedder = Embedder(embed_model).get()
        self.vector_store = VectorStore(self.embedder)
        self.llm_engine = LLMEngineGemini()

    def build_index(self):
        """
        Builds the index for the document retrieval system.

        This method performs the following steps:
        1. Loads documents using the PDF loader.
        2. Splits the loaded documents into smaller chunks using the chunker.
        3. Creates a vector store from the document chunks for efficient retrieval.

        Raises:
            Exception: If any step in the process fails.
        """
        if not self.vector_store.db:
            docs = []
            for pdf_file in glob.glob(f"{self.pdf_path}/*.pdf"):
                docs.extend(self.pdf_loader.load(pdf_file))
            docs_split = self.chunker.split(docs)
            self.vector_store.create_vector_store(docs_split)
        else:
            print("Vector store already exists. Skipping index building.")

    def query(self, question: str, k: int = 3):
        """
        Query the vector store and generate a response using the language model.

        Args:
            question (str): The input question to query the vector store.
            k (int, optional): The number of relevant documents to retrieve.
              Defaults to 3.

        Returns:
            str: The generated response from the language model based on
              the retrieved context.
        """
        retriever = self.vector_store.retriever(k=k)
        results = retriever.invoke(question)

        context = "\n".join([doc.page_content for doc in results])
        prompt = (
            f"Answer the following question concisely using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        answer = self.llm_engine.generate(prompt)

        # Return both answer and sources
        sources = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
            for doc in results
        ]
        return answer, sources
