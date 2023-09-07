"""
Configuration file for the project.
"""
from dotenv import dotenv_values
from omegaconf import OmegaConf

OPENAI_API_KEY = dotenv_values(".env")["OPENAI_API_KEY"]

CONFIG = OmegaConf.load("config.yaml")
"""The configuration for the project."""

PROMPTS = {
    "Academic Essay": "You are a world class academic professor and a world class technical writer that wants to produce coherent academic essays. Produce content step by step.",  # noqa: E501
    "Horror Story": "You are a world class horror author with a style similar to Stephen King and Anne Rice. Generate content that contains logical twists and builds suspense. Produce content step by step.",  # noqa: E501
    "Romance Novel": "You are a world class romance novelist with a style similar to Nora Roberts and Jane Austen. Generate content that is tantilizing, lustrious, and very erotic. Produce content step by step.",  # noqa: E501
}
"""The prompts to use for the chat API."""

OUTPUT_STYLES = {
    "Outline": "Respond with an outline no longer than 500 words",
    "Paragraph": "Respond with at least one paragraph. Output at least 200 words.",
    "List": "Respond with a top 15 list. Output is limited to 150 words.",
}
"""The output styles to use for the chat API."""
