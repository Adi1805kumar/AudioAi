"""
tools.py — Gemini Function Call Implementations
─────────────────────────────────────────────────
When Gemini decides to call a tool, `execute_tool()` is the single
dispatch point. Add new tools here and declare them in websocket_manager.py.

Available tools:
  • google_search     — live web search via SerpAPI
  • search_documents  — local PDF/notes RAG retrieval
"""

from loguru import logger
from app.config import SERPAPI_KEY
from app.rag_engine import search as rag_search


def execute_tool(tool_name: str, args: dict) -> dict:
    """
    Dispatch a Gemini function call to the right implementation.
    Always returns a dict (Gemini requires a JSON-serialisable response).
    """
    handlers = {
        "google_search": _google_search,
        "search_documents": _search_documents,
    }

    handler = handlers.get(tool_name)
    if not handler:
        logger.warning("[Tools] Unknown tool called: {}", tool_name)
        return {"error": f"Tool '{tool_name}' is not registered."}

    try:
        return handler(**args)
    except Exception as e:
        logger.error("[Tools] Tool '{}' raised: {}", tool_name, e)
        return {"error": str(e)}


# ── Tool implementations ───────────────────────────────────────────────────────

def _google_search(query: str) -> dict:
    """
    Search the web using SerpAPI (Google Search).
    Falls back to a helpful message if no API key is set.
    """
    if not SERPAPI_KEY:
        return {
            "error": (
                "SERPAPI_KEY is not configured. "
                "Add it to your .env file to enable live web search."
            )
        }

    try:
        from serpapi import GoogleSearch  # pip install google-search-results
    except ImportError:
        return {"error": "serpapi package not installed. Run: pip install google-search-results"}

    logger.info("[Tools] Google Search: '{}'", query)
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5,
        "hl": "en",
    })
    result = search.get_dict()

    # Extract clean answer snippets
    organic = result.get("organic_results", [])
    if not organic:
        answer_box = result.get("answer_box", {})
        if answer_box:
            return {
                "answer": answer_box.get("answer") or answer_box.get("snippet", ""),
                "source": answer_box.get("link", ""),
            }
        return {"results": "No results found."}

    formatted = [
        {
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "link": r.get("link", ""),
        }
        for r in organic[:3]
    ]
    return {"results": formatted}


def _search_documents(query: str) -> dict:
    """
    Search locally ingested PDFs/documents via RAG.
    """
    logger.info("[Tools] RAG search: '{}'", query)
    context = rag_search(query)
    return {"context": context}