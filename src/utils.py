"""
Adapted from https://github.com/minimaxir/simpleaichat/blob/main/simpleaichat/utils.py
"""
import json
import os

import httpx
import tiktoken
from loguru import logger
from pydantic import Field

from src.config import OPENAI_API_KEY

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


async def get_embedding(
    input: str,
    model: str = "text-embedding-ada-002",
) -> list[float]:
    """Get vector representation of text input from OpenAI /embeddings \
        endpoint.

    Args:
        input (str): The text to encode.
        model (str): The model to use for encoding.
    Returns:
        list[float]: The embedding vector of dimensionality 1536.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {
        "input": input,
        "model": model,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            data=json.dumps(payload),
        )
        logger.debug(f"[{os.getpid()}] Embeddings response: [{response.status_code}]")
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return embedding


def wikipedia_search(query: str, n: int = 1) -> str | list[str]:
    """Searches Wikipedia for a query and returns the title of \
        the first result.

    Args:
        query (str): The query to search Wikipedia for.
        n (int, optional): The number of results to return. \
            Defaults to 1.

    Returns:
        Union[str, List[str]]: The title of the first result \
            if n == 1, else a list of titles.
    """
    logger.info(f"Searching Wikipedia for [{query}]")
    search_params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": n,
        "srsearch": query,
        "srwhat": "text",
        "srprop": "",
    }

    response = httpx.get(WIKIPEDIA_API_URL, params=search_params)
    logger.debug(f"[{os.getpid()}] Wiki response: [{response.status_code}]")
    results = [x["title"] for x in response.json()["query"]["search"]]

    return results[0] if n == 1 else results


def wikipedia_lookup(query: str, sentences: int = 1) -> str:
    """Looks up a query on Wikipedia and returns the first \
        sentence of the result.

    Args:
        query (str): The query to look up on Wikipedia.
        sentences (int, optional): The number of sentences \
            to return. Defaults to 1.

    Returns:
        str: The first sentence of the result.
    """
    logger.info(f"Looking up [{query}] on Wikipedia")
    lookup_params = {
        "action": "query",
        "prop": "extracts",
        "exsentences": sentences,
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
        "titles": query,
    }

    response = httpx.get(WIKIPEDIA_API_URL, params=lookup_params)
    logger.debug(f"[{os.getpid()}] Wiki response: [{response.status_code}]")
    return response.json()["query"]["pages"][0]["extract"]


def wikipedia_search_lookup(query: str, sentences: int = 1) -> str:
    """Searches Wikipedia for a query and returns the first \
        sentence of the first result.

    Args:
        query (str): The query to search Wikipedia for.
        sentences (int, optional): The number of sentences to return. \
            Defaults to 1.

    Returns:
        str: The first sentence of the first result.
    """
    return wikipedia_lookup(wikipedia_search(query, 1), sentences)


async def wikipedia_search_async(
    query: str,
    n: int = 1,
) -> str | list[str]:
    """Searches Wikipedia for a query and returns the \
        title of the first result.

    Args:
        query (str): The query to search Wikipedia for.
        n (int, optional): The number of results to return. \
            Defaults to 1.

    Returns:
        Union[str, List[str]]: The title of the first result \
            if n == 1, else a list of titles.
    """
    logger.info(f"Searching Wikipedia for [{query}]")
    search_params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": n,
        "srsearch": query,
        "srwhat": "text",
        "srprop": "",
    }

    async with httpx.AsyncClient(proxies=os.getenv("https_proxy")) as client:
        response = await client.get(WIKIPEDIA_API_URL, params=search_params)
    logger.debug(f"[{os.getpid()}] RPC response: [{response.status_code}]")
    results = [x["title"] for x in response.json()["query"]["search"]]

    return results[0] if n == 1 else results


async def wikipedia_lookup_async(query: str, sentences: int = 1) -> str:
    """Looks up a query on Wikipedia and returns the \
        first sentence of the result.

    Args:
        query (str): The query to look up on Wikipedia.
        sentences (int, optional): The number of sentences \
            to return. Defaults to 1.

    Returns:
        str: The first sentence of the result.
    """
    logger.info(f"Looking up [{query}] on Wikipedia")
    lookup_params = {
        "action": "query",
        "prop": "extracts",
        "exsentences": sentences,
        "exlimit": "1",
        "explaintext": "1",
        "formatversion": "2",
        "format": "json",
        "titles": query,
    }

    async with httpx.AsyncClient(proxies=os.getenv("https_proxy")) as client:
        response = await client.get(WIKIPEDIA_API_URL, params=lookup_params)
    logger.debug(f"[{os.getpid()}] RPC response: [{response.status_code}]")
    return response.json()["query"]["pages"][0]["extract"]


async def wikipedia_search_lookup_async(
    query: str,
    sentences: int = 1,
) -> str:
    """Searches Wikipedia for a query and returns the first sentence \
        of the first result.

    Args:
        query (str): The query to search Wikipedia for.
        sentences (int, optional): The number of sentences to return. \
            Defaults to 1.

    Returns:
        str: The first sentence of the first result.
    """
    return await wikipedia_lookup_async(
        await wikipedia_search_async(query, 1),
        sentences,
    )


def fd(description: str, **kwargs) -> Field:
    """Creates a Pydantic Field with a description.

    Args:
        description (str): The description of the field.
        **kwargs: The keyword arguments to pass to the Field constructor.

    Returns:
        Field: The Pydantic Field with a description.
    """
    return Field(description=description, **kwargs)


# https://stackoverflow.com/a/58938747
def remove_a_key(d: dict, remove_key: str) -> None:
    """Recursively removes a key from a dictionary.

    Args:
        d (dict): The dictionary to remove a key from.
        remove_key (str): The key to remove from the dictionary.

    Returns:
        None
    """
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)


def num_tokens_from_string(
    string: str,
    model: str = "gpt-3.5-turbo",
) -> int:
    """Returns the number of tokens in a text string.

    Args:
        string (str): The text string to count tokens in.
        model (str): The model to use for tokenization. \
            Defaults to CONFIG.model.chat.

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model)

    num_tokens = len(encoding.encode(string))

    return num_tokens


def get_encoding(model="gpt-3.5-turbo") -> str:
    """Returns the encoding of a model.

    Args:
        model (str): The model to use for encoding. \
            Defaults to CONFIG.model.chat.

    Returns:
        str: The encoding of the model.
    """
    encoding = tiktoken.encoding_for_model(model).name

    logger.debug(f"MODEL: [{model}] ENCODING: [{encoding}]")

    return encoding
