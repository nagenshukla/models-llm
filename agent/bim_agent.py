"""
BIM assistant as a Microsoft Agent Framework ChatAgent, backed by the
fine-tuned Phi-4-mini model served over an OpenAI-compatible endpoint, with
the knowledge-base lookup tools (kb_tool.py) wired in for grounding.

CAVEAT: agent-framework is a young, fast-moving SDK (the unification of
Semantic Kernel + AutoGen). The import paths and client constructor
signatures below match the documented pattern as of this writing, but
verify them against `pip show agent-framework` / the current docs in your
Colab session before trusting this blindly - a point-release could have
renamed something.

Serving setup (do this first, in Colab):

    pip install vllm
    python -m vllm.entrypoints.openai.api_server \
        --model train/output/merged_model \
        --port 8000 &

This exposes an OpenAI-compatible /v1/chat/completions endpoint. The same
serving pattern (an OpenAI-compatible container) is what you'd put behind
the Azure AI Foundry Managed Online Endpoint later - swap MODEL_BASE_URL
below to the Foundry endpoint URL and nothing else in this file changes.

Run:

    pip install -r agent/requirements.txt
    python agent/bim_agent.py "What are the parts of ISO 19650?"
"""

import asyncio
import os
import sys

from kb_tool import lookup_bim_term, lookup_structured_fact

MODEL_BASE_URL = os.environ.get("BIM_MODEL_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("BIM_MODEL_NAME", "bim-assistant")

INSTRUCTIONS = (
    "You are a BIM and digital construction assistant for civil engineers. "
    "For any defined term, standard, acronym, or numbered/structured fact "
    "(e.g. ISO 19650 parts, the LOD scale, CDE states), call lookup_bim_term "
    "or lookup_structured_fact first and base your answer on its result - "
    "do not state specifics from memory alone, since you have been observed "
    "inventing plausible-sounding but wrong numbers in the past. If a lookup "
    "returns no match, say you're not certain rather than guessing."
)


async def main():
    # Import here, not at module load, so kb_tool.py and this file's
    # docstring/help still work even before agent-framework is installed.
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient

    chat_client = OpenAIChatClient(
        base_url=MODEL_BASE_URL,
        api_key=os.environ.get("BIM_MODEL_API_KEY", "not-needed"),
        model=MODEL_NAME,
    )

    agent = ChatAgent(
        chat_client=chat_client,
        instructions=INSTRUCTIONS,
        tools=[lookup_bim_term, lookup_structured_fact],
    )

    query = " ".join(sys.argv[1:]) or "What are the parts of ISO 19650?"
    result = await agent.run(query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
