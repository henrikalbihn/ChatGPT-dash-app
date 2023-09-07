"""
Adapted from: https://medium.com/@erickleppen/the-beginners-guide-to-building-a-chatgpt-powered-writing-assistant-web-app-using-python-9a9b3a54f55
"""
import asyncio
from uuid import uuid4

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, callback_context, dcc, html
from loguru import logger

from app.src.config import OUTPUT_STYLES, PROMPTS
from app.src.simple import AsyncAIChat

AI: AsyncAIChat = None
INIT_SESSION_ID = f"chatgpt-default-{uuid4().hex[:8]}"


def system_prompts() -> html.Div:
    """Create system prompts dropdown.

    Returns:
        html.Div: Div containing system prompts dropdown.
    """
    return html.Div(
        [
            html.H3("Select your writing genre:"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="sys-prompt",
                            options=list(PROMPTS.keys()),
                            value="Academic Essay",
                        ),
                        width=4,
                    ),  # end col 1
                    dbc.Col(
                        html.P("Sets the system level prompt to improve the style of the output."),
                        width=6,
                    ),  # end col 2
                ]
            ),  # end row
        ]
    )  # end div


def output_style() -> html.Div:
    """Create output style dropdown.

    Returns:
        html.Div: Div containing output style dropdown.
    """
    return html.Div(
        [
            html.H3("Select your output style:"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="output-style",
                            options=list(OUTPUT_STYLES.keys()),
                            value="Outline",
                        ),
                        width=4,
                    ),  # end col 1
                    dbc.Col(
                        html.P("Sets the style of output returned (outline, paragraph, or list)"),
                        width=6,
                    ),  # end col 2
                ]
            ),  # end row
        ]
    )  # end div


def text_areas() -> html.Div:
    """Create text areas for user input.

    Returns:
        html.Div: Div containing text areas.
    """
    return html.Div(
        [
            html.H3("Enter your prompt:"),
            dbc.Textarea(
                id="my-input",
                size="lg",
                placeholder="Enter your text",
            ),
            dbc.Button(
                "Generate Text",
                id="gen-button",
                className="me-2",
                n_clicks=0,
            ),
        ]
    )


# instantiate dash
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR],
    server=True,
    title="ChatGPT Writing Assitant 🧠",
)  # create layout

app.layout = html.Div(
    [
        dbc.Container(
            [
                html.H1("ChatGPT Writing Assitant 🧠"),
                html.Br(),
                system_prompts(),
                html.Br(),
                output_style(),
                html.Br(),
                text_areas(),
                html.Br(),
                html.H3("Output:"),
                html.Div(id="my-output"),
            ]
        )  # end container
    ]
)  # end div


@app.callback(
    Output(component_id="my-output", component_property="children"),
    Input(component_id="gen-button", component_property="n_clicks"),
    Input(component_id="sys-prompt", component_property="value"),
    Input(component_id="output-style", component_property="value"),
    State(component_id="my-input", component_property="value"),
)
def update_output_div(gen: int, sp: str, os: str, input_value: str) -> html.P:
    """Update the output div.

    Args:
        gen (int): The number of times the button has been clicked.
        sp (str): The system prompt.
        os (str): The output style.
        input_value (str): The input value.

    Returns:
        html.P: The output div.
    """
    # print(input_value) #debug

    # set text to sample
    text = "This is a \nsample"

    # listen for button clicks
    changed_id = [p["prop_id"] for p in callback_context.triggered][0]

    assert sp in PROMPTS, f"System prompt {sp} not found in {list(PROMPTS.keys())}"

    system_prompt = PROMPTS[sp]

    assert os in OUTPUT_STYLES, f"Output style {os} not found in {list(OUTPUT_STYLES.keys())}"

    style = OUTPUT_STYLES[os]

    # build messages payload
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": style},
    #     {"role": "assistant", "content": "On what topic?"},
    #     {"role": "user", "content": input_value},
    # ]
    global AI
    AI = AsyncAIChat(
        console=False,
        system=system_prompt,
        model="gpt-3.5-turbo",
        temperature=0.8,
        top_p=1,
        presence_penalty=0.5,
        frequency_penalty=0.4,
        id=INIT_SESSION_ID,
    )

    # button logic to submit to chatGPT API
    if "gen-button" in changed_id:
        logger.info(input_value)
        if input_value is None or input_value == "":
            input_value = ""
            text = html.P("hello <br> this is </br> a <br> test ")

        else:
            text = run_coroutine(input_value, style, INIT_SESSION_ID)

    # return html.P(text, style={"white-space": "pre-wrap"})
    return html.P(text, style={"white-space": "pre-wrap"})


def run_coroutine(
    input_value: str,
    style: str,
    session_id: str = INIT_SESSION_ID,
) -> str:
    """Run the coroutine.

    Args:
        input_value (str): The input value.
        style (str): The style.
        session_id (str): The session ID.

    Returns:
        str: The text.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    text = ""

    async def inner_coroutine() -> None:
        """Run the inner coroutine."""
        nonlocal text
        async for chunk in await AI.stream(
            input_value + " " + style,
            id=session_id,
            save_messages=True,
        ):
            delta = chunk["delta"]
            logger.debug(f"Delta: {delta}")
            text += delta

    loop.run_until_complete(inner_coroutine())
    return text


# run app server
if __name__ == "__main__":
    app.run(debug=True)
