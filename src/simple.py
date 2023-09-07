"""
Adapted from:
https://github.com/minimaxir/simpleaichat/blob/main/\
    simpleaichat/simpleaichat.py
"""
import csv
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from uuid import UUID, uuid4

# import dateutil
import orjson

# import pandas as pd
from httpx import AsyncClient, Client
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rich.console import Console

from src.chatgpt import ChatGPTSession
from src.config import OPENAI_API_KEY
from src.schemas import ChatSession
from src.utils import wikipedia_search_lookup


class AIChat(BaseModel):
    """Implements the AI Chat."""

    client: Any
    default_session: ChatSession | None
    sessions: dict[str | UUID, ChatSession] = {}
    model_config: ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        character: str = None,
        character_command: str = None,
        system: str = None,
        id: str | UUID | None = None,
        prime: bool = True,
        default_session: bool = True,
        console: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the AI Chat.

        Args:
            character (str, optional): The character to use. Defaults to \
                None.
            character_command (str, optional): The command to use for the \
                character. Defaults to None.
            system (str, optional): The system to use. Defaults to None.
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to uuid4().
            prime (bool, optional): Whether to prime the session. \
                Defaults to True.
            default_session (bool, optional): Whether to set the \
                session as the default. Defaults to True.
            console (bool, optional): Whether to use the interactive \
                console. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if id is None:
            id = uuid4()

        logger.info(f"Initializing AI Chat... 🧠: [{id}]")

        client = Client(proxies=os.getenv("https_proxy"))
        system_format = self.build_system(
            character,
            character_command,
            system,
        )

        sessions = {}
        new_default_session = None
        if default_session:
            new_session = self.new_session(
                return_session=True,
                system=system_format,
                id=id,
                **kwargs,
            )

            new_default_session = new_session
            sessions = {new_session.id: new_session}

        super().__init__(
            client=client,
            default_session=new_default_session,
            sessions=sessions,
        )

        if not system and console:
            character = "Atom" if not character else character
            new_default_session.title = character
            self.interactive_console(character=character, prime=prime)

    def new_session(
        self,
        return_session: bool = False,
        **kwargs,
    ) -> ChatGPTSession | None:
        """Creates a new session.

        Args:
            return_session (bool, optional): Whether to return the session. \
                Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[ChatGPTSession]: The new session.
        """
        if "model" not in kwargs:  # set default
            kwargs["model"] = "gpt-3.5-turbo"
        # TODO: Add support for more models (PaLM, Claude)
        if "gpt-" in kwargs["model"]:
            gpt_api_key = OPENAI_API_KEY
            assert (
                gpt_api_key
            ), f"An API key for {kwargs['model'] } \
                was not defined."
            sess = ChatGPTSession(
                auth={
                    "api_key": gpt_api_key,
                },
                **kwargs,
            )

        if return_session:
            return sess
        else:
            self.sessions[sess.id] = sess

    def get_session(self, id: str | UUID = None) -> ChatSession:
        """Gets a session by id.

        Args:
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.

        Raises:
            KeyError: If no session by that key exists.

        Returns:
            ChatSession: The session.
        """
        try:
            sess = self.sessions[id] if id else self.default_session
        except KeyError as e:
            raise KeyError("No session by that key exists.") from e
        if not sess:
            raise ValueError("No default session exists.")
        return sess

    def reset_session(self, id: str | UUID = None) -> None:
        """Resets a session by id.

        Args:
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.

        Returns:
            None
        """
        logger.debug(f"Resetting session {id}...")
        sess = self.get_session(id)
        sess.messages = []

    def delete_session(self, id: str | UUID = None) -> None:
        """Deletes a session by id.

        Args:
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.

        Returns:
            None
        """
        logger.debug(f"Deleting session {id}...")
        sess = self.get_session(id)
        if self.default_session:
            if sess.id == self.default_session.id:
                self.default_session = None
        del self.sessions[sess.id]
        del sess

    @contextmanager
    def session(self, **kwargs) -> ChatSession:
        """Creates a new session.

        Args:
            **kwargs: Additional keyword arguments.

        Yields:
            ChatSession: The new session.
        """
        sess = self.new_session(return_session=True, **kwargs)
        self.sessions[sess.id] = sess
        try:
            yield sess
        finally:
            self.delete_session(sess.id)

    def __call__(
        self,
        prompt: str | Any,
        id: str | UUID = None,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        tools: list[Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ) -> str:
        """Generates a response to a prompt.

        Args:
            prompt (Union[str, Any]): The prompt to generate a response to.
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.
            system (str, optional): The system to use. \
                Defaults to None.
            save_messages (bool, optional): Whether to save messages. \
                Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. \
                Defaults to None.
            tools (List[Any], optional): The tools to use. \
                Defaults to None.
            input_schema (Any, optional): The input schema to use. \
                Defaults to None.
            output_schema (Any, optional): The output schema to use. \
                Defaults to None.

        Raises:
            AssertionError: If a tool does not have a docstring.
            AssertionError: If there are more than 9 tools.

        Returns:
            str: The response.
        """
        sess = self.get_session(id)
        if tools:
            logger.debug(f"Using tools: {tools}")
            for tool in tools:
                assert tool.__doc__, f"Tool {tool} does not have a docstring."
            assert len(tools) <= 9, "You can only have a maximum of 9 tools."
            return sess.gen_with_tools(
                prompt,
                tools,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
            )
        else:
            return sess.gen(
                prompt,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
                input_schema=input_schema,
                output_schema=output_schema,
            )

    def stream(
        self,
        prompt: str,
        id: str | UUID = None,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        timeout: int = None,
    ) -> str:
        """Streams a response to a prompt.

        Args:
            prompt (str): The prompt to generate a response to.
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.
            system (str, optional): The system to use. \
                Defaults to None.
            save_messages (bool, optional): Whether to save messages. \
                Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. \
                Defaults to None.
            input_schema (Any, optional): The input schema to use. \
                Defaults to None.

        Returns:
            str: The response.
        """
        sess = self.get_session(id)
        return sess.stream(
            prompt,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
            input_schema=input_schema,
            timeout=timeout,
        )

    def build_system(
        self,
        character: str = None,
        character_command: str = None,
        system: str = None,
    ) -> str:
        """Builds the system message.

        If a character is provided, the system message will be built \
            from that character's Wikipedia page - therefore the \
                character must be a public figure or fictional \
                    character with a Wikipedia page.

        Args:
            character (str, optional): The character to use. \
                Defaults to None.
            character_command (str, optional): The command to \
                use for the character. Defaults to None.
            system (str, optional): The system to use. \
                Defaults to None.

        Returns:
            str: The system message.
        """
        default = "You are a helpful assistant."
        if character:
            character_prompt = """
            You must follow ALL these rules in all responses:
            - You are the following character and should \
                ALWAYS act as them: {0}
            - NEVER speak in a formal tone.
            - Concisely introduce yourself first in character.
            """
            prompt = character_prompt.format(wikipedia_search_lookup(character)).strip()
            if character_command:
                character_system = """
                - {0}
                """
                prompt = (
                    prompt
                    + "\n"
                    + character_system.format(
                        character_command,
                    ).strip()
                )
            return prompt
        elif system:
            return system
        else:
            return default

    def interactive_console(
        self,
        character: str = None,
        prime: bool = True,
    ) -> None:
        """Starts an interactive console.

        Args:
            character (str, optional): The character to use. \
                Defaults to None.
            prime (bool, optional): Whether to prime the session. \
                Defaults to True.

        Returns:
            None
        """
        console = Console(highlight=False)
        sess = self.default_session
        ai_text_color = "bright_magenta"

        # prime with a unique starting response to the user
        if prime:
            console.print(
                f"[b]{character}[/b]: ",
                end="",
                style=ai_text_color,
            )
            for chunk in sess.stream("Hello!", self.client):
                console.print(
                    chunk["delta"],
                    end="",
                    style=ai_text_color,
                )

        while True:
            console.print()
            try:
                user_input = console.input("[b]You:[/b] ").strip()
                if not user_input:
                    break

                console.print(f"[b]{character}[/b]: ", end="", style=ai_text_color)
                for chunk in sess.stream(user_input, self.client):
                    console.print(
                        chunk["delta"],
                        end="",
                        style=ai_text_color,
                    )
            except KeyboardInterrupt:
                break

    def __str__(self) -> str:
        if self.default_session:
            return self.default_session.model_dump_json(
                exclude={"api_key", "api_url"},
                exclude_none=True,
                option=orjson.OPT_INDENT_2,
            )

    def __repr__(self) -> str:
        return ""

    # def to_frame(self, id: Union[str, UUID] = None) -> pd.DataFrame:
    #     """Returns a dataframe of the session.

    #     Args:
    #         id (Union[str, UUID], optional): The id of the session. \
    # Defaults to None.

    #     Returns:
    #         pd.DataFrame: The dataframe of the session.
    #     """
    #     logger.info(f"Converting session {id} to dataframe...")
    #     sess = self.get_session(id)
    #     sess_dict = sess.model_dump(
    #         exclude={"auth", "api_url", "input_fields"},
    #         exclude_none=True,
    #     )
    #     df = pd.DataFrame(sess_dict["messages"])
    #     df.insert(1, "session_id", id)
    #     logger.debug(f"Session {id} contains [{df.shape[0]}] messages.")
    #     return df

    # Save/Load Chats given a session id
    def save_session(
        self,
        output_path: str = None,
        id: str | UUID = None,
        format: str = "csv",
        minify: bool = False,
    ) -> None:
        """Saves a session.

        Args:
            output_path (str, optional): The output path. \
                Defaults to None.
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to None.
            format (str, optional): The format to save the session in. \
                Defaults to "csv".
            minify (bool, optional): Whether to minify the output.\
                Defaults to False.

        Returns:
            None
        """
        sess = self.get_session(id)
        sess_dict = sess.model_dump(
            exclude={"auth", "api_url", "input_fields"},
            exclude_none=True,
        )
        output_path = output_path or f"chat_session.{format}"
        logger.debug(f"Saving session to {output_path} in {format} format...")
        if format == "csv":
            with open(output_path, "w", encoding="utf-8") as f:
                fields = [
                    "role",
                    "content",
                    "received_at",
                    "prompt_length",
                    "completion_length",
                    "total_length",
                ]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for message in sess_dict["messages"]:
                    # datetime must be in common format to be loaded \
                    # into spreadsheet
                    # for human-readability, the timezone is set to \
                    # local machine
                    local_datetime = message["received_at"].astimezone()
                    message["received_at"] = local_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    w.writerow(message)
        elif format == "json":
            with open(output_path, "wb") as f:
                f.write(
                    orjson.dumps(
                        sess_dict,
                        option=orjson.OPT_INDENT_2 if not minify else None,
                    )
                )

    def load_session(
        self,
        input_path: str,
        id: str | UUID | None = None,
        **kwargs,
    ) -> None:
        """Loads a session.

        Args:
            input_path (str): The input path.
            id (Union[str, UUID], optional): The id of the session. \
                Defaults to uuid4().
            **kwargs: Additional keyword arguments.

        Raises:
            AssertionError: If the input path does not end with .csv or .json.

        Returns:
            None
        """
        if id is None:
            id = uuid4()
        assert input_path.endswith(".csv") or input_path.endswith(".json"), "Only CSV and JSON imports are accepted."

        logger.debug(f"Loading session from {input_path}...")
        # if input_path.endswith(".csv"):
        #     with open(input_path, "r", encoding="utf-8") as f:
        #         r = csv.DictReader(f)
        #         messages = []
        #         for row in r:
        #             # need to convert the datetime back to UTC
        #             local_datetime = datetime.datetime.strptime(
        #                 row["received_at"], "%Y-%m-%d %H:%M:%S"
        #             ).replace(tzinfo=dateutil.tz.tzlocal())
        #             row["received_at"] = local_datetime.astimezone(
        #                 datetime.timezone.utc
        #             )
        #             # https://stackoverflow.com/a/68305271
        #             row = {k: (None if v == "" else v) for k, v in row.items()}
        #             messages.append(ChatMessage(**row))

        #     self.new_session(id=id, **kwargs)
        #     self.sessions[id].messages = messages

        if input_path.endswith(".json"):
            with open(input_path, "rb") as f:
                sess_dict = orjson.loads(f.read())
            # update session with info not loaded, e.g. auth/api_url
            for arg in kwargs:
                sess_dict[arg] = kwargs[arg]
            self.new_session(**sess_dict)

    # Tabulators for returning total token counts
    def message_totals(self, attr: str, id: str | UUID = None) -> int:
        """Returns the total token count for a given attribute.

        Args:
            attr (str): The attribute to get the total token count for.
            id (Union[str, UUID], optional): The id of the session. Defaults to None.

        Returns:
            int: The total token count.
        """
        sess = self.get_session(id)
        return getattr(sess, attr)

    @property
    def total_prompt_length(self, id: str | UUID = None) -> int:
        """Returns the total prompt token count.

        Args:
            id (Union[str, UUID], optional): The id of the session. Defaults to None.

        Returns:
            int: The total prompt token count.
        """
        return self.message_totals("total_prompt_length", id)

    @property
    def total_completion_length(self, id: str | UUID = None) -> int:
        """Returns the total completion token count.

        Args:
            id (Union[str, UUID], optional): The id of the session. Defaults to None.

        Returns:
            int: The total completion token count.
        """
        return self.message_totals("total_completion_length", id)

    @property
    def total_length(self, id: str | UUID = None) -> int:
        """Returns the total token count.

        Args:
            id (Union[str, UUID], optional): The id of the session. Defaults to None.

        Returns:
            int: The total token count.
        """
        return self.message_totals("total_length", id)

    # alias total_tokens to total_length for common use
    @property
    def total_tokens(self, id: str | UUID = None) -> int:
        """Returns the total token count. Alias for total_length.

        Args:
            id (Union[str, UUID], optional): The id of the session. Defaults to None.

        Returns:
            int: The total token count.
        """
        return self.total_length(id)


class AsyncAIChat(AIChat):
    """Implements the asynchronous AI Chat."""

    async def __call__(
        self,
        prompt: str,
        id: str | UUID = None,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        tools: list[Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ) -> str:
        """Generates a response to a prompt.

        Args:
            prompt (str): The prompt to generate a response to.
            id (Union[str, UUID], optional): The id of the session. Defaults to None.
            system (str, optional): The system to use. Defaults to None.
            save_messages (bool, optional): Whether to save messages. Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. Defaults to None.
            tools (List[Any], optional): The tools to use. Defaults to None.
            input_schema (Any, optional): The input schema to use. Defaults to None.
            output_schema (Any, optional): The output schema to use. Defaults to None.

        Raises:
            AssertionError: If a tool does not have a docstring.
            AssertionError: If there are more than 9 tools.

        Returns:
            str: The response.
        """
        # TODO: move to a __post_init__ in Pydantic 2.0
        if isinstance(self.client, Client):
            self.client = AsyncClient(proxies=os.getenv("https_proxy"))
        sess = self.get_session(id)
        if tools:
            for tool in tools:
                assert tool.__doc__, f"Tool {tool} does not have a docstring."
            assert len(tools) <= 9, "You can only have a maximum of 9 tools."
            return await sess.gen_with_tools_async(
                prompt,
                tools,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
            )
        else:
            return await sess.gen_async(
                prompt,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
                input_schema=input_schema,
                output_schema=output_schema,
            )

    async def stream(
        self,
        prompt: str,
        id: str | UUID = None,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        input_schema: Any = None,
        timeout: int = None,
    ) -> str:
        """Streams a response to a prompt.

        Args:
            prompt (str): The prompt to generate a response to.
            id (Union[str, UUID], optional): The id of the session. Defaults to None.
            system (str, optional): The system to use. Defaults to None.
            save_messages (bool, optional): Whether to save messages. Defaults to None.
            params (Dict[str, Any], optional): Additional parameters. Defaults to None.
            input_schema (Any, optional): The input schema to use. Defaults to None.
            timeout (int, optional): The timeout. Defaults to None.

        Returns:
            str: The response.
        """
        # TODO: move to a __post_init__ in Pydantic 2.0
        if isinstance(self.client, Client):
            self.client = AsyncClient(proxies=os.getenv("https_proxy"))
        sess = self.get_session(id)
        return sess.stream_async(
            prompt,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
            input_schema=input_schema,
            timeout=timeout,
        )

    async def tool_stream(
        self,
        prompt: str,
        id: str | UUID = None,
        system: str = None,
        save_messages: bool = None,
        params: dict[str, Any] = None,
        tools: list[Any] = None,
    ) -> str:
        # TODO: move to a __post_init__ in Pydantic 2.0
        if isinstance(self.client, Client):
            self.client = AsyncClient(proxies=os.getenv("https_proxy"))
        sess = self.get_session(id)
        return sess.gen_with_tools_stream_async(
            prompt,
            tools=tools,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
        )

    @asynccontextmanager
    async def session(self, **kwargs) -> ChatSession:
        """Creates a new session.

        Args:
            **kwargs: Additional keyword arguments.

        Yields:
            ChatSession: The new session.
        """
        sess = self.new_session(return_session=True, **kwargs)
        self.sessions[sess.id] = sess
        try:
            yield sess
        finally:
            self.delete_session(sess.id)
            self.delete_session(sess.id)
