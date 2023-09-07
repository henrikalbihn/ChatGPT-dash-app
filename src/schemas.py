"""
Adapted from https://github.com/minimaxir/simpleaichat/blob/main/\
    simpleaichat/models.py
"""

import datetime
from typing import Any
from uuid import UUID, uuid4

import orjson

# from pydantic import root_validator
# from pydantic import validator
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

config = ConfigDict(from_attributes=True)


def orjson_dumps(v, *, default, **kwargs):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default, **kwargs).decode()


def now_tz():
    # Need datetime w/ timezone for cleanliness
    # https://stackoverflow.com/a/24666683
    return datetime.datetime.now(datetime.UTC)


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
    function_call: str | None = None
    received_at: datetime.datetime = Field(default_factory=now_tz)
    finish_reason: str | None = None
    prompt_length: int | None = None
    completion_length: int | None = None
    total_length: int | None = None

    def __str__(self) -> str:
        return str(self.model_dump(exclude_none=True))


class ChatSession(BaseModel):
    id: str | UUID = Field(default_factory=uuid4)
    created_at: datetime.datetime = Field(default_factory=now_tz)
    auth: dict[str, SecretStr]
    api_url: HttpUrl
    model: str
    system: str
    params: dict[str, Any] = {}
    messages: list[ChatMessage] = []
    input_fields: set[str] = {}
    recent_messages: int | None = None
    save_messages: bool | None = True
    total_prompt_length: int = 0
    total_completion_length: int = 0
    total_length: int = 0
    title: str | None = None

    def __str__(self) -> str:
        sess_start_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        last_message_str = self.messages[-1].received_at.strftime("%Y-%m-%d %H:%M:%S")
        return f"""Chat session started at {sess_start_str}:
        - {len(self.messages):,} Messages
        - Last message sent at {last_message_str}"""

    def format_input_messages(
        self,
        system_message: ChatMessage,
        user_message: ChatMessage,
    ) -> list:
        recent_messages = self.messages[-self.recent_messages :] if self.recent_messages else self.messages
        return (
            [
                system_message.model_dump(
                    include=self.input_fields,
                    exclude_none=True,
                )
            ]
            + [
                m.model_dump(
                    include=self.input_fields,
                    exclude_none=True,
                )
                for m in recent_messages
            ]
            + [
                user_message.model_dump(
                    include=self.input_fields,
                    exclude_none=True,
                )
            ]
        )

    def add_messages(
        self,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        save_messages: bool = None,
    ) -> None:
        # if save_messages is explicitly defined, always use that choice
        # instead of the default
        to_save = isinstance(save_messages, bool)

        if to_save:
            if save_messages:
                self.messages.append(user_message)
                self.messages.append(assistant_message)
        elif self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)


class ExtractedMetadata(BaseModel):
    """Metadata extracted from text."""

    short_description: str = Field(
        description="Short but salient summary of the event \
            in a single sentence assertion.",
    )
    long_description: str = Field(
        description="Highly detailed, salient & descriptive, \
            summary of the event in a paragraph.",
    )
    city: str = Field(
        description="City where event occured. If unknown, use '?'",
    )
    year: int | str = Field(
        description="Year when event occured. If unknown, use '?'",
    )
    month: int | str = Field(
        description="Integer value of the month when event occured. \
            If unknown, use '?'",
    )
    source: str = Field(
        description="Source of event information - url if available. \
            If unknown, use '?'",
    )
    confidence: float = Field(
        description="Confidence of event information on a scale of \
            0.00 to 1.00 rounded to 2 decimal places. \
                0.00 is no confidence, 0.99 is high confidence"
    )
    tags: list[str] = Field(
        description="One or two word tags (lower kebab-case) associated \
            with event. Example: ['machine-learning', 'classification']"
    )
    entities: list[str] = Field(
        description="Named entities associated with event \
            (e.g. people, places, organizations, authors). \
                As a list of strings (lower kebab-case), where \
                    each is 'entity:entity_type'. \
                        Example: ['elon-musk:person', 'tesla:organization']"
    )


class ChatInputSchema(BaseModel):
    model_config = config
    session_id: str = Field(
        title="Chat session id",
        description="Chat session id",
        default="7337dfef-0ab0-404f-be29-1717944739c5",
    )
    user_id: UUID = Field(
        title="User id",
        description="User id",
        default="536f3e27-48a8-4ed9-9b2e-7711b74d532d",
    )
    query: str = Field(
        title="Query",
        description="Query",
        default="Hello, how are you?",
    )
