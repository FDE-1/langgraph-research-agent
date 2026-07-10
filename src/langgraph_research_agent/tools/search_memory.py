import chromadb
from langchain_core.tools import StructuredTool, ToolException

from ..utils.logger import logger
from ..utils.setting import get_settings

settings = get_settings()


def search_memory_func(query: str, nb_results: int = 5) -> list[str]:
    """Search the given chroma db for the most connected element.
    Use when searching the memory.
    Do not use for calculation or to search the web.
    """
    try:
        client = chromadb.PersistentClient(path=settings.client_path)
        collection = client.get_collection(name=settings.collection_name)
        logger.info(
            "Tool usage: search_memory with following arguments "
            f"query={query} and nb_results={nb_results}"
        )
        results = collection.query(query_texts=[query], n_results=nb_results)
        documents = results["documents"]
        logger.debug(f"[search_memory] documents={documents}")
        if not documents:
            return []
        return documents[0]

    except chromadb.errors.ChromaError as e:
        raise ToolException(f"Erreur interne Chroma : {e}") from e
    except ValueError as e:
        raise ToolException(f"Erreur de paramètre : {e}") from e
    except Exception as e:
        raise ToolException(f"Erreur d'embedding ou inattendue : {e}") from e


search_memory = StructuredTool.from_function(
    func=search_memory_func, name="search_memory", handle_tool_error=True
)
