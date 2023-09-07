# import json
# import logging

# # import os
# # from datetime import datetime
# from uuid import uuid4

# # import httpx

# # from app.config import CONFIG, SEARXNG_HOST
# # from app.schemas.search import SearchResult, SearchResults

# # from rich import print_json


# logger = logging.getLogger(__name__)


# class SearchEngine:
#     """Online search engine wrapper."""

#     def __init__(self, host: str = f"{SEARXNG_HOST}/search"):
#         self.host = host

#     async def search(self, query: str, k: int = 4) -> SearchResults:
#         """Search the online search engine

#         Args:
#             query (str): The query to search for.
#             k (int, optional): The number of results to return. Defaults to 4.

#         Returns:
#             SearchResults: The results of the search.
#         """
#         return await call_searxng(
#             query=query,
#             k=k,
#             host=self.host,
#         )

#     async def searchxng(
#         self,
#         query: str,
#         k: int = 4,
#     ) -> dict[list[dict[str, any]]]:
#         """Perform online search of the Search engine. Useful for current and up-to-date information.

#         Args:
#             query (str): The query to search for.
#             k (int, optional): The number of results to return. Defaults to 4.

#         Returns:
#             dict[list[dict[str, any]]]: The results of the search.
#         """
#         logger.info(f"Searching for [{query}]...")
#         results = await self.search(
#             query=query,
#             k=k,
#         )
#         results_json = results.model_dump()

#         # get only the url, title and content
#         results_json = [
#             {
#                 "url": result["url"],
#                 "title": result["title"],
#                 "content": result["content"],
#             }
#             for result in results_json["results"]
#         ]

#         return {"context": json.dumps(results_json)}


# async def call_searxng(
#     query: str,
#     k: int = 10,
#     host: str = SEARXNG_HOST,
# ) -> SearchResults:
#     """
#     Call the SearXNG API.
#     """
#     logger.info("Calling SearXNG API...")
#     # query = CONFIG.search.params.q
#     logger.debug(f"Query: '{query}'")
#     params = CONFIG["search"]["params"]

#     params["q"] = query
#     params["count"] = k
#     logger.debug(host)
#     logger.debug(params)
#     # Make the request to the SearXNG API
#     response = httpx.get(host, params=params)

#     # If the request was successful, log the results
#     if response.status_code == 200:
#         logger.info(f"Request was successful {response.status_code}.")
#         # logger.info(response.json())

#     else:
#         logger.error(f"Request failed. Status code: {response.status_code}.")
#     response = response.json()
#     suggestions = response["suggestions"]
#     results = response["results"]
#     res = []
#     for result in results[:k]:
#         fmt_res = {
#             "id": str(uuid4()),
#             "title": result["title"],
#             "url": result["url"],
#             "content": result["content"],
#             "category": result["category"],
#             "score": result["score"],
#         }
#         res.append(fmt_res)
#         # print_json(data=fmt_res)
#     return SearchResults(
#         id=str(uuid4()),
#         query=query,
#         suggestions=suggestions,
#         results=[SearchResult(**r).model_dump() for r in res],
#     )
