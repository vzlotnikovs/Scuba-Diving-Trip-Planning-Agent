import os
import structlog
from typing import Any, List, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from constants import (
    USER_AGENT,
    DOTENV_PATH,
    PDF_FILENAME_1,
    PDF_FILENAME_2,
    SUB_DIR,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_MODEL,
    COLLECTION_NAME,
    PERSIST_DIR,
    K_CONSTANT,
    RAG_PROMPT,
    TIMEOUT,
    MAX_RETRIES,
)

os.environ["USER_AGENT"] = USER_AGENT
load_dotenv(dotenv_path=DOTENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

log = structlog.get_logger()


class RAGSystem:
    """Class managing the Retrieval-Augmented Generation (RAG) system.

    Handles loading PDF documents, creating/loading Chroma vector stores,
    and initializing the retrieval agent.
    """

    _instance = None

    def __init__(self) -> None:
        """Initialize the RAGSystem with configuration constants.

        Note: The actual initialization of models and vector stores is deferred
        to the `_initialize` method for lazy loading.
        """
        self.sources: Optional[List[Any]] = None
        self.vector_store: Optional[Chroma] = None
        self.agent: Any = None

        self.SUB_DIR = SUB_DIR
        self.PDF_FILENAME_1 = PDF_FILENAME_1
        self.PDF_FILENAME_2 = PDF_FILENAME_2
        self.LLM_MODEL = LLM_MODEL
        self.EMBEDDINGS_MODEL = EMBEDDINGS_MODEL
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP
        self.COLLECTION_NAME = COLLECTION_NAME
        self.PERSIST_DIR = PERSIST_DIR
        self.K_CONSTANT = K_CONSTANT
        self.RAG_PROMPT = RAG_PROMPT

    @classmethod
    def get_instance(cls) -> "RAGSystem":
        """Get the instance of the RAGSystem.

        Returns:
            RAGSystem: The initialized instance.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Lazy initialization of system resources.

        Loads PDF sources, initializes the Chroma vector store, and creates
        the retrieval agent on first use.
        """
        if self.sources is None:
            self.sources = self.load_source_content(
                self.SUB_DIR, self.PDF_FILENAME_1, self.PDF_FILENAME_2
            )
        if self.vector_store is None:
            self.vector_store = self.get_or_create_vector_store(
                self.EMBEDDINGS_MODEL,
                self.COLLECTION_NAME,
                self.PERSIST_DIR,
                self.CHUNK_SIZE,
                self.CHUNK_OVERLAP,
            )
        if self.agent is None:
            self.agent = create_agent(
                model=ChatOpenAI(
                    model=self.LLM_MODEL,
                    streaming=True,
                    timeout=TIMEOUT,
                    max_retries=MAX_RETRIES,
                ),
                tools=[retrieve_context],
                system_prompt=self.RAG_PROMPT,
                checkpointer=MemorySaver(),
            )
            log.info(
                "rag_agent_initialized",
                model=self.LLM_MODEL,
                collection=self.COLLECTION_NAME,
            )

    def load_source_content(
        self,
        SUB_DIR: str,
        PDF_FILENAME_1: str,
        PDF_FILENAME_2: str,
    ) -> List[Any]:
        """Load source content from web and PDF documents.

        Loads content from two specified PDF files, adding metadata to each
        source indicating its origin.

        Args:
            SUB_DIR (str): The subdirectory where the PDF is located.
            PDF_FILENAME_1 (str): The filename of the first PDF to load.
            PDF_FILENAME_2 (str): The filename of the second PDF to load.

        Returns:
            list: A list of Document objects loaded from the sources.

        Raises:
            RuntimeError: If document loading fails or network request fails.
        """
        try:
            BASE_DIR = Path(__file__).resolve().parent
            PDF_PATH_1 = BASE_DIR / SUB_DIR / PDF_FILENAME_1
            PDF_PATH_2 = BASE_DIR / SUB_DIR / PDF_FILENAME_2

            pdf_loader_1 = PyPDFLoader(str(PDF_PATH_1))
            pdf_loader_2 = PyPDFLoader(str(PDF_PATH_2))

            log.info("pdf_loading_started")
            pdf_sources_1 = pdf_loader_1.load()
            pdf_sources_2 = pdf_loader_2.load()
            if not pdf_sources_1 or not pdf_sources_2:
                raise RuntimeError("No PDF documents loaded.")

            for source in pdf_sources_1:
                source.metadata["source"] = str(PDF_FILENAME_1)
            for source in pdf_sources_2:
                source.metadata["source"] = str(PDF_FILENAME_2)

            pdf_char_count_1 = sum(len(doc.page_content) for doc in pdf_sources_1)
            pdf_char_count_2 = sum(len(doc.page_content) for doc in pdf_sources_2)

            log.info(
                "pdf_loading_complete",
                pdf_1=PDF_FILENAME_1,
                pdf_1_chars=pdf_char_count_1,
                pdf_2=PDF_FILENAME_2,
                pdf_2_chars=pdf_char_count_2,
                total_chars=pdf_char_count_1 + pdf_char_count_2,
            )

            self.sources = pdf_sources_1 + pdf_sources_2
            if len(self.sources) == 0:
                raise RuntimeError("Error loading source content.")

            log.info("source_content_loaded", total_pages=len(self.sources))
            return self.sources
        except FileNotFoundError as e:
            raise RuntimeError(f"[ERROR] PDF file not found: {e}")
        except Exception as e:
            log.exception("source_content_load_error", error=str(e))
            raise RuntimeError(f"[ERROR] Error loading source content: {e}")

    def get_or_create_vector_store(
        self,
        EMBEDDINGS_MODEL: str,
        COLLECTION_NAME: str,
        PERSIST_DIR: str,
        CHUNK_SIZE: int,
        CHUNK_OVERLAP: int,
    ) -> Chroma:
        """Load an existing vector store or create a new one.

        Args:
            EMBEDDINGS_MODEL (str): The name of the embeddings model.
            COLLECTION_NAME (str): The name of the collection.
            PERSIST_DIR (str): The directory to load from or save to.
            CHUNK_SIZE (int): Chunk size for text splitting (if creating).
            CHUNK_OVERLAP (int): Chunk overlap for text splitting (if creating).

        Returns:
            Chroma: The initialized vector store.
        """
        persist_path = Path(PERSIST_DIR)
        embed_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

        if persist_path.exists() and any(persist_path.iterdir()):
            log.info("vector_store_loaded", persist_dir=PERSIST_DIR)
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embed_model,
                persist_directory=PERSIST_DIR,
            )

        log.info("vector_store_creating", persist_dir=PERSIST_DIR)
        return self.create_vector_store(
            CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_MODEL, COLLECTION_NAME, PERSIST_DIR
        )

    def create_vector_store(
        self,
        CHUNK_SIZE: int,
        CHUNK_OVERLAP: int,
        EMBEDDINGS_MODEL: str,
        COLLECTION_NAME: str,
        PERSIST_DIR: str,
    ) -> Chroma:
        """Create and persist a Chroma vector store from loaded documents.

        Splits the globally loaded source documents into chunks and stores them
        in a Chroma vector database using OpenAI embeddings.

        Args:
            CHUNK_SIZE (int): The maximum size of each text chunk.
            CHUNK_OVERLAP (int): The overlap size between chunks.
            EMBEDDINGS_MODEL (str): The name of the OpenAI embeddings model to use.
            COLLECTION_NAME (str): The name of the Chroma collection.
            PERSIST_DIR (str): The directory where the vector store will be saved.

        Returns:
            Chroma: The initialized and populated Chroma vector store.

        Raises:
            RuntimeError: If an error occurs during vector store creation.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )

            if self.sources is None:
                raise RuntimeError(
                    "Sources must be loaded before creating vector store"
                )
            all_splits = text_splitter.split_documents(self.sources)

            embed_model = OpenAIEmbeddings(
                model=EMBEDDINGS_MODEL,
            )

            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embed_model,
                persist_directory=PERSIST_DIR,
            )

            vector_store.add_documents(documents=all_splits)
            log.info(
                "vector_store_created",
                collection=COLLECTION_NAME,
                persist_dir=PERSIST_DIR,
                chunk_count=len(all_splits),
            )
            return vector_store
        except Exception as e:
            log.exception("vector_store_creation_error", error=str(e))
            raise RuntimeError(f"[ERROR] Error creating vector store: {e}")


@tool
def retrieve_context(query: str) -> str:
    """Retrieve information to help answer a query.

    Args:
        query (str): The user query.

    Returns:
        str: A formatted string containing retrieved documents with their sources.
    """
    try:
        rag = RAGSystem.get_instance()
        if rag.vector_store is None:
            raise RuntimeError("vector_store must be initialized")
        retrieved_sources = rag.vector_store.similarity_search(query, k=rag.K_CONSTANT)
        log.info(
            "retrieve_context_complete",
            results_count=len(retrieved_sources),
        )
        bullet_points = []
        for source in retrieved_sources:
            src = source.metadata.get("source", "Unknown")
            content = source.page_content.replace("\n", " ")
            bullet_points.append(f"Source: {src}\n {content}")
        return "\n".join(bullet_points)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Error retrieving context: {e}")
