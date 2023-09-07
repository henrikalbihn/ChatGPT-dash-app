"""
Adapted from https://github.com/minimaxir/simpleaichat/blob/main/simpleaichat/utils.py
"""
import json

# import pandas as pd
# import multiprocessing as mp
import os

import httpx

# import numpy as np
import tiktoken
from loguru import logger
from pydantic import Field

from src.config import OPENAI_API_KEY

# import traceback


# , SUPABASE_KEY, SUPABASE_URL

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


# async def match_query(
#     input: str,
#     match_count: int = 10,
# ) -> list[dict[str, any]]:
#     """Matches a query to documents in the database. Performs \
#         cosine similarity semantic search over the embeddings of \
#             the query and the documents.

#     Args:
#         input (str): The query to match.
#         match_count (int, optional): The number of matches to return. \
#             Defaults to 10.

#     Returns:
#         list[dict[str, any]]: The response from the database.
#     """
#     query_embedding = await get_embedding(input=input)
#     matches = await match_documents(
#         query_embedding=query_embedding,
#         match_count=match_count,
#     )
#     logger.info(f"Returned [{len(matches)}] matches...")
#     return matches


# async def tool_match_query(
#     input: str,
#     match_count: int = 10,
# ) -> dict[list[dict[str, any]]]:
#     """Matches a query to documents in the database. \
#         Performs cosine similarity semantic search over the \
#             embeddings of the query and the documents. \
#                 Useful to search your internal knowledge base.

#     Args:
#         input (str): The query to match.
#         match_count (int, optional): The number of matches to return. \
#             Defaults to 10.

#     Returns:
#         dict[list[dict[str, any]]]: The response from the database.
#     """
#     results = await match_query(input, match_count)

#     # only return the content and metadata.url fields
#     results = [
#         {
#             "content": result["content"],
#             "url": result["metadata"].get("url", None),
#         }
#         for result in results
#     ]
#     return {
#         "context": json.dumps(results),
#     }


# def str_2_vec(x: dict[str, any]) -> dict[str, any]:
#     """Converts the embedding from a string to a list of floats."""

#     logger.debug(
#         f"[{os.getpid()}] Match! Doc id: [{x['id']}] \
#                  Similarity: [{x['similarity']:0.4f}]"
#     )

#     try:
#         # embedding is returned from Supabase RPC as a string
#         x["embedding"] = np.fromstring(
#             x["embedding"].strip("[]"),
#             np.float32,
#             sep=",",
#         ).tolist()

#         return x
#     except Exception as e:
#         logger.error(f"Error processing item: {e}")
#         logger.error(traceback.format_exc())
#         return None  # or some other way to indicate failure


# async def match_documents(
#     query_embedding: list[float],
#     match_count: int = 10,
# ) -> list[dict[str, any]]:
#     """Matches documents in the database to a query embedding.

#     Args:
#         query_embedding (list[float]): The embedding of the query.
#         match_count (int, optional): The number of matches to return. \
#             Defaults to 10.
#     Returns:
#         List[float]: A list of match scores.
#     """
#     logger.info(f"Matching [{match_count}] documents...")
#     data = {
#         "match_count": match_count,
#         "query_embedding": query_embedding,
#     }

#     headers = {
#         "Authorization": f"Bearer {SUPABASE_KEY}",
#         "apiKey": SUPABASE_KEY,
#     }
#     response = httpx.post(
#         f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
#         json=data,
#         headers=headers,
#     )

#     logger.debug(f"[{os.getpid()}] RPC response: [{response.status_code}]")
#     response_json = response.json()
#     logger.info(f"# results: {len(response_json)}")

#     # converting to a pandas dataframe first is a workaround
#     # and honestly not ideal, would be good to consider other options

#     # df = pd.DataFrame(response_json)
#     # df["embedding"] = df["embedding"].apply(
#     #     lambda x: np.fromstring(x.strip("[]"), np.float32, sep=",").tolist()
#     # )
#     # return df.to_dict(orient="records")

#     num_workers = mp.cpu_count()
#     logger.debug(f"Workers:processes [{num_workers}:{len(response_json)}]")

#     with mp.Pool(num_workers) as pool:
#         response_json = pool.map(str_2_vec, response_json)

#     return response_json


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
