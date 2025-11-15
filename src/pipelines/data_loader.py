from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    """
    PDFLoader is a class designed to handle the loading and parsing of PDF files.
    It provides an interface to extract document objects from a specified PDF file.

    Attributes:
        file_path (str): The path to the PDF file to be loaded.

    Methods:
        __init__(file_path)
        load()
    """

    def __init__(self):
        """Initializes the PDFLoader with the given file path."""
        # self.file_path = file_path

    def load(self, file_path):
        """Loads and parses the PDF file.

        Args:
            file_path (str): Path to the PDF file to be loaded.

        Returns:
            list: A list of document objects extracted from the PDF.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If loading fails for any other reason.
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs
