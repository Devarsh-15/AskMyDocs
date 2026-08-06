import os
import groq
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]


def get_config_value(name):
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def get_groq_models():
    primary_model = get_config_value("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    fallback_models = get_config_value("GROQ_FALLBACK_MODELS")

    if fallback_models:
        model_names = [primary_model]
        model_names.extend(
            model.strip()
            for model in fallback_models.split(",")
            if model.strip()
        )
    else:
        model_names = [primary_model, *DEFAULT_GROQ_FALLBACK_MODELS]

    deduplicated_models = []

    for model_name in model_names:
        if model_name not in deduplicated_models:
            deduplicated_models.append(model_name)

    return deduplicated_models


def get_llm_client():
    """
    Initialize Groq client
    """
    api_key = get_config_value("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. For Streamlit Community Cloud, add it under "
            "App Settings -> Secrets as GROQ_API_KEY."
        )

    return groq.Groq(api_key=api_key)


# ------------------------------
# Context Normalization
# ------------------------------

def normalize_context(context_input):
    """
    Convert different context formats into clean text.

    Accepts:
    - List[Document]
    - List[str]
    - str
    - None
    """

    if context_input is None:
        return ""

    # Case 1: already a string
    if isinstance(context_input, str):
        return context_input

    # Case 2: list
    if isinstance(context_input, list):

        context_parts = []

        for item in context_input:

            # LangChain Document
            if hasattr(item, "page_content"):
                context_parts.append(item.page_content)

            # plain string
            elif isinstance(item, str):
                context_parts.append(item)

            else:
                context_parts.append(str(item))

        return "\n\n".join(context_parts)

    # fallback
    return str(context_input)


# ------------------------------
# Answer Generation
# ------------------------------

def generate_answer(client, query, context_input):
    """
    Generate answer using Groq LLM with strict RAG guardrails.
    """

    context = normalize_context(context_input)

    # ------------------------------
    # HARDENING GUARD
    # Prevent LLM call if context empty
    # ------------------------------

    if not context.strip():
        return "The information is not available in the documents."

    prompt = f"""
You are an AI assistant answering questions strictly from company documents.

Rules:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If the answer is not found in the context, respond exactly with:
"The information is not available in the documents."

Context:
{context}

Question:
{query}

Answer:
"""

    last_error = None

    for model_name in get_groq_models():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0  ## FOR DETERMINISTIC ANSWERS
            )

            return response.choices[0].message.content

        except Exception as error:
            last_error = error

    raise RuntimeError("All configured Groq models failed.") from last_error
                    