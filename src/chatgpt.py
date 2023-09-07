"""
Adapted from:
https://github.com/minimaxir/simpleaichat/blob/main/simpleaichat/chatgpt.py
"""


from collections.abc import Callable
from typing import Any

from httpx import AsyncClient, Client
from loguru import logger
from orjson import JSONDecodeError as OJDecodeError
from orjson import loads as ojloads
from pydantic import ConfigDict, HttpUrl

from src.config import CONFIG
from src.schemas import ChatMessage, ChatSession
from src.utils import num_tokens_from_string, remove_a_key

tool_prompt = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response \
    to the user's last message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""

sys_msg = CONFIG["model"]["system_message"] or "You are a helpful assistant."


class ChatGPTSession(ChatSession):
    """A ChatSession for the OpenAI API."""

    api_url: HttpUrl = "https://api.openai.com/v1/chat/completions"
    input_fields: set[str] = {"role", "content", "name"}
    system: str = sys_msg
    params: dict[str, Any] = {
        "temperature": CONFIG["model"]["temperature"],
        # "max_tokens": CONFIG.model.max_tokens,
        # "top_p": CONFIG.model.top_p,
        # "frequency_penalty": CONFIG.model.frequency_penalty,
        # "presence_penalty": CONFIG.model.presence_penalty,
        # "stop": CONFIG.model.stop,
        # "logit_bias": CONFIG.model.logit_bias,
    }
    model_config: ConfigDict(arbitrary_types_allowed=True)

    def prepare_request(
        self,
        prompt: str,
        system: str = None,
        params: dict[str, Any] = None,
        stream: bool = False,
        input_schema: Any = None,
        output_schema: Any = None,
        is_function_calling_required: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], ChatMessage]:
        """Prepares the request to the API.

        Args:
            prompt (str): The prompt to send to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.
            stream (bool, optional): Whether to stream the response \
                from the API. Defaults to False.
            input_schema (Any, optional): The input schema to send to \
                the API. Defaults to None.
            output_schema (Any, optional): The output schema to send \
                to the API. Defaults to None.
            is_function_calling_required (bool, optional): Whether function \
                calling is required. Defaults to True.

        Raises:
            AssertionError: If the prompt is not an instance of the \
                input schema.
            AssertionError: If the prompt is not an instance of the \
                output schema.

        Returns:
            Dict[str, Any]: The headers, data, and user message to send \
                to the API.
        """

        tokens = num_tokens_from_string(prompt)
        logger.info(f"Processing [{tokens}] tokens/[{len(prompt)}] characters")

        if tokens >= 4097:
            raise ValueError(
                f"Prompt is too long. Max tokens is 4096. \
                    You have {tokens} tokens / {len(prompt)} characters."
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
        }

        system_message = ChatMessage(
            role="system",
            content=system or self.system,
        )
        if not input_schema:
            user_message = ChatMessage(
                role="user",
                content=prompt,
            )
        else:
            assert isinstance(
                prompt,
                input_schema,
            ), f"prompt must be an instance of {input_schema.__name__}"
            user_message = ChatMessage(
                role="function",
                content=prompt.model_dump_json(),
                name=input_schema.__name__,
            )

        gen_params = params or self.params
        data = {
            "model": self.model,
            "messages": self.format_input_messages(
                system_message,
                user_message,
            ),
            "stream": stream,
            **gen_params,
        }

        # Add function calling parameters if a schema is provided
        if input_schema or output_schema:
            functions = []
            if input_schema:
                input_function = self.schema_to_function(input_schema)
                functions.append(input_function)
            if output_schema:
                output_function = self.schema_to_function(output_schema)
                functions.append(output_function) if output_function not in functions else None
                if is_function_calling_required:
                    data["function_call"] = {
                        "name": output_schema.__name__,
                    }
            data["functions"] = functions

        return headers, data, user_message

    def schema_to_function(self, schema: Any) -> dict[str, Any]:
        """Converts a schema to a function.

        Args:
            schema (Any): The schema to convert to a function.

        Returns:
            Dict[str, Any]: The function.
        """
        assert schema.__doc__, f"{schema.__name__} is missing a docstring."
        schema_dict = schema.model_json_schema()
        remove_a_key(schema_dict, "title")

        return {
            "name": schema.__name__,
            "description": schema.__doc__,
            "parameters": schema_dict,
        }

    def gen(
        self,
        prompt: str,
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ) -> dict[str, Any]:
        """Generates a response from the API.

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.
            input_schema (Any, optional): The input schema to send \
                to the API. Defaults to None.
            output_schema (Any, optional): The output schema to send \
                to the API. Defaults to None.

        Raises:
            KeyError: If no AI generation is returned from the API.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        logger.debug(f"prompt: {prompt}")

        headers, data, user_message = self.prepare_request(
            prompt,
            system,
            params,
            False,
            input_schema,
            output_schema,
        )

        r = client.post(
            self.api_url,
            json=data,
            headers=headers,
            timeout=None,
        )
        r = r.json()

        try:
            if not output_schema:
                content = r["choices"][0]["message"]["content"]
                assistant_message = ChatMessage(
                    role=r["choices"][0]["message"]["role"],
                    content=content,
                    finish_reason=r["choices"][0]["finish_reason"],
                    prompt_length=r["usage"]["prompt_tokens"],
                    completion_length=r["usage"]["completion_tokens"],
                    total_length=r["usage"]["total_tokens"],
                )
                self.add_messages(user_message, assistant_message, save_messages)
            else:
                content = r["choices"][0]["message"]["function_call"]["arguments"]
                content = ojloads(content)

            self.total_prompt_length += r["usage"]["prompt_tokens"]
            self.total_completion_length += r["usage"]["completion_tokens"]
            self.total_length += r["usage"]["total_tokens"]
        except KeyError as e:
            raise KeyError(f"No AI generation: {r}") from e

        return content

    def stream(
        self,
        prompt: str,
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        timeout: int = None,
    ) -> dict[str, Any]:
        """Streams a response from the API.

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send \
                to the API. Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.
            input_schema (Any, optional): The input schema to send \
                to the API. Defaults to None.
            timeout (int, optional): The timeout to use for the request. \
                Defaults to None.

        Yields:
            Dict[str, Any]: The response from the API.

        Raises:
            KeyError: If no AI generation is returned from the API.
        """
        logger.info(f"Generating response for prompt: {prompt}")

        headers, data, user_message = self.prepare_request(
            prompt,
            system,
            params,
            True,
            input_schema,
        )

        with client.stream(
            "POST",
            self.api_url,
            json=data,
            headers=headers,
            timeout=timeout,
        ) as r:
            content = []
            logger.debug("Begin token stream...")
            for i, chunk in enumerate(r.iter_lines()):
                if len(chunk) > 0:
                    chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
                    logger.debug(chunk)
                    _type = "token"
                    if i == 0:
                        _type = "start"
                    if chunk == "[DONE]":
                        logger.debug("End token stream...")
                        yield {
                            "delta": chunk,
                            "response": "".join(content),
                            "type": "end",
                        }
                    if chunk != "[DONE]":
                        if chunk != "":
                            try:
                                chunk_dict = ojloads(chunk)
                                delta = chunk_dict["choices"][0]["delta"].get("content")
                                if delta:
                                    content.append(delta)
                                    yield {
                                        "delta": delta,
                                        "response": "".join(content),
                                        "type": _type,
                                    }
                            except OJDecodeError as e:
                                logger.error(e)
                                logger.error(chunk)
                                pass

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )

        self.add_messages(user_message, assistant_message, save_messages)

        return assistant_message

    def gen_with_tools(
        self,
        prompt: str,
        tools: list[Any],
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Generates a response from the API with tools.

        Args:
            prompt (str): The prompt to send to the API.
            tools (List[Any]): The tools to use.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.

        Raises:
            KeyError: If no AI generation is returned from the API.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = tool_prompt.format(tools=tools_list)

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        tool_idx = int(
            self.gen(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": self.gen(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        context_dict = selected_tool(prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\n\
            You MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = self.gen(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(
            role="user",
            content=prompt,
        )
        assistant_message = ChatMessage(
            role="assistant",
            content=context_dict["response"],
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict

    async def gen_async(
        self,
        prompt: str,
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
        timeout: int = None,
    ) -> dict[str, Any]:
        """Generates a response from the API asynchronously.

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.
            input_schema (Any, optional): The input schema to send \
                to the API. Defaults to None.
            output_schema (Any, optional): The output schema to send \
                to the API. Defaults to None.
            timeout (int, optional): The timeout to use for the request. \
                Defaults to None.

        Raises:
            KeyError: If no AI generation is returned from the API.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        headers, data, user_message = self.prepare_request(
            prompt,
            system,
            params,
            False,
            input_schema,
            output_schema,
        )

        logger.info(f"Generating response for prompt: {prompt}")

        r = await client.post(
            self.api_url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        r = r.json()

        try:
            if not output_schema:
                content = r["choices"][0]["message"]["content"]
                assistant_message = ChatMessage(
                    role=r["choices"][0]["message"]["role"],
                    content=content,
                    finish_reason=r["choices"][0]["finish_reason"],
                    prompt_length=r["usage"]["prompt_tokens"],
                    completion_length=r["usage"]["completion_tokens"],
                    total_length=r["usage"]["total_tokens"],
                )
                self.add_messages(user_message, assistant_message, save_messages)
            else:
                content = r["choices"][0]["message"]["function_call"]["arguments"]
                content = ojloads(content)

            self.total_prompt_length += r["usage"]["prompt_tokens"]
            self.total_completion_length += r["usage"]["completion_tokens"]
            self.total_length += r["usage"]["total_tokens"]
        except KeyError as e:
            raise KeyError(f"No AI generation: {r}") from e

        return content

    async def stream_async(
        self,
        prompt: str,
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        timeout: int = None,
    ) -> dict[str, Any]:
        """Streams a response from the API asynchronously

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.
            input_schema (Any, optional): The input schema to send \
                to the API. Defaults to None.
            timeout (int, optional): The timeout to use for the request. \
                Defaults to None.

        Yields:
            Dict[str, Any]: The response from the API.
        """
        headers, data, user_message = self.prepare_request(
            prompt,
            system,
            params,
            True,
            input_schema,
        )
        async with client.stream(
            "POST",
            self.api_url,
            json=data,
            headers=headers,
            timeout=timeout,
        ) as r:
            content = []
            logger.debug("Begin token stream...")
            async for chunk in r.aiter_lines():
                if len(chunk) > 0:
                    chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
                    logger.debug(chunk)
                    _type = "token"
                    if chunk == "[DONE]":
                        logger.debug("End token stream...")
                        yield {
                            "delta": chunk,
                            "response": "".join(content),
                            "type": "end",
                        }
                    if chunk != "[DONE]":
                        if chunk != "":
                            try:
                                chunk_d = ojloads(chunk)
                                delta = chunk_d["choices"][0]["delta"].get("content")
                                if delta:
                                    content.append(delta)
                                    yield {
                                        "delta": delta,
                                        "response": "".join(content),
                                        "type": _type,
                                    }
                            except OJDecodeError as e:
                                logger.error(e)
                                logger.error(chunk)
                                pass
        # async with client.stream(
        #     "POST",
        #     self.api_url,
        #     json=data,
        #     headers=headers,
        #     timeout=timeout,
        # ) as r:
        #     content = []
        #     # i = 0
        #     # _type = "start"
        #     async for chunk in r.aiter_lines():
        #         if len(chunk) > 0:
        #             # i += 1
        #             chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
        #             # if i > 0:
        #             _type = "token"

        #             # if chunk == "[DONE]":
        #             # logger.debug("End token stream...")
        #             # _type = "end"
        #             if chunk != "[DONE]":
        #                 chunk_dict = ojloads(chunk)
        #                 delta = chunk_dict["choices"][0]["delta"].get("content")
        #                 if delta:
        #                     content.append(delta)
        #                     yield {
        #                         "delta": delta,
        #                         "response": "".join(content),
        #                         "type": _type,
        #                         # "count": i,
        #                     }

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )

        self.add_messages(user_message, assistant_message, save_messages)

    async def gen_with_tools_async(
        self,
        prompt: str,
        tools: list[Any],
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Generates a response from the API asynchronously with tools.

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.

        Raises:
            KeyError: If no AI generation is returned from the API.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = tool_prompt.format(tools=tools_list)

        logger.debug(f"TOOLS:\n\n{tool_prompt_format}")

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        logger.debug(f"LOGIT BIAS {logit_bias}")

        logger.debug(f"PROMPT:\n\n{prompt}")
        tool_idx = int(
            await self.gen_async(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": await self.gen_async(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        context_dict = await selected_tool(prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\n\
            You MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = await self.gen_async(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(
            role="user",
            content=prompt,
        )
        assistant_message = ChatMessage(
            role="assistant",
            content=context_dict["response"],
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict

    async def gen_targeted_tool_prompt(
        self,
        selected_tool: Callable,
        prompt: str,
        client: Client | AsyncClient,
    ) -> str:
        formatted_prompt = f"""
        TOOL:
        {selected_tool.__doc__}

        QUERY:
        {prompt}

        TARGETED_QUERY:"""
        response = await self.gen_async(
            formatted_prompt,
            client=client,
            system="You are a search query context enhancer. \
                Given the above tool and query, convert the query \
                    to a more targeted search query to SEO optimize \
                        the query and enhance results.",
            save_messages=False,
        )
        return response

    async def gen_with_tools_stream_async(
        self,
        prompt: str,
        tools: list[Any],
        client: Client | AsyncClient,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Generates a response from the API asynchronously with tools.

        Args:
            prompt (str): The prompt to send to the API.
            client (Union[Client, AsyncClient]): The client to use to send \
                the request to the API.
            system (str, optional): The system message to send to the API. \
                Defaults to None.
            save_messages (bool, optional): Whether to save the messages. \
                Defaults to None.
            params (Dict[str, Any], optional): The parameters to send \
                to the API. Defaults to None.

        Raises:
            KeyError: If no AI generation is returned from the API."""
        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = tool_prompt.format(tools=tools_list)
        logger.debug(f"TOOLS:\n\n{tool_prompt_format}")
        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}
        logger.debug("BEGIN TOOL SELECTION TASK...")
        tool_idx = int(
            await self.gen_async(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )
        logger.debug("END TOOL SELECTION TASK...")

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            logger.debug("BEGIN FINAL GENERATION TASK...")
            async for chunk in self.stream_async(
                prompt,
                client=client,
                system=system,
                save_messages=save_messages,
                params=params,
            ):
                yield chunk
            logger.debug("END FINAL GENERATION TASK...")

        selected_tool = tools[tool_idx - 1]
        logger.debug("BEGIN TARGETED PROMPT GENERATION TASK...")
        targeted_prompt = await self.gen_targeted_tool_prompt(selected_tool, prompt, client=client)
        logger.debug("END TARGETED PROMPT GENERATION TASK...")
        context_dict = await selected_tool(targeted_prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\n\
            You MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"
        content = []
        logger.debug("BEGIN FINAL GENERATION TASK...")
        logger.debug(f"PROMPT:\n\n{new_prompt}")

        async for chunk in self.stream_async(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=save_messages,
            params=params,
        ):
            content.append(chunk["delta"])
            yield chunk
        logger.debug("END FINAL GENERATION TASK...")
        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )
        user_message = ChatMessage(role="user", content=prompt)
        self.add_messages(user_message, assistant_message, save_messages)
