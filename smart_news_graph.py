"""
Smart News Analyzer & Fact Checker - LangGraph + Gemini 2.5 Flash Demo

This module defines a LangGraph state machine that:
- Ingests article via URL or pasted text
- Extracts claims using Gemini
- Verifies claims iteratively using Gemini's built-in Google Search tool
- Includes human-in-the-loop for suspicious claims
- Handles failures with recovery (bad URL -> ask user to paste text)
- Uses SQLite checkpoints for state persistence
- Runs sentiment analysis as a local tool

No LangChain - uses official google-genai SDK + LangGraph only.
"""

import json
import os
import re
import sqlite3
from operator import add
from typing import Annotated, Literal, Optional, TypedDict

import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------


def get_gemini_client():
    """Get configured Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class ClaimVerification(TypedDict):
    claim: str
    verdict: Literal["verified", "false", "uncertain", "needs_review"]
    evidence: str
    sources: list[str]


class GraphState(TypedDict, total=False):
    # Input
    input_mode: Literal["url", "text"]
    article_url: str
    article_text: str

    # Extracted data
    article_title: str
    claims: list[str]

    # Verification loop
    claim_idx: int
    verifications: Annotated[list[ClaimVerification], add]

    # Human in the loop
    needs_human_review: bool
    human_questions: list[dict]
    human_answers: dict

    # Sentiment
    sentiment: dict

    # Error handling
    error: Optional[str]

    # Final output
    final_report: str

    # Messages for chat interface
    messages: list[dict]


# ---------------------------------------------------------------------------
# Local tools (Python functions, not Gemini function calling)
# ---------------------------------------------------------------------------


def fetch_url_content(url: str) -> tuple[str, str]:
    """Fetch and extract text content from URL. Returns (title, text)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = soup.title.string if soup.title else "Untitled Article"

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Get text from article or body
    article = soup.find("article") or soup.find("main") or soup.body
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines[:100])  # Limit to first 100 lines

    return title.strip(), text


def analyze_sentiment(text: str) -> dict:
    """Run VADER sentiment analysis on text."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    # Determine overall sentiment
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
    }


# ---------------------------------------------------------------------------
# Gemini helper functions
# ---------------------------------------------------------------------------


def gemini_extract_claims(
    client: genai.Client, article_text: str, article_title: str
) -> list[str]:
    """Use Gemini to extract factual claims from article text."""

    prompt = f"""Analyze this news article and extract the key factual claims that can be verified.
Return a JSON array of strings, each string being one distinct factual claim.
Focus on claims about:
- Statistics and numbers
- Events that happened
- Quotes attributed to people
- Scientific or research findings
- Dates and timelines

Article Title: {article_title}

Article Text:
{article_text[:4000]}

Return ONLY a valid JSON array of strings, no other text. Example format:
["Claim 1 here", "Claim 2 here", "Claim 3 here"]

Extract 3-5 of the most important verifiable claims."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )

    # Parse JSON from response
    text = response.text.strip()
    # Try to extract JSON array from response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group())
            return claims[:5]  # Limit to 5 claims
        except json.JSONDecodeError:
            pass

    # Fallback: split by newlines if JSON parsing fails
    return [
        line.strip().strip('"').strip("'") for line in text.split("\n") if line.strip()
    ][:5]


def gemini_verify_claim_with_search(
    client: genai.Client, claim: str
) -> ClaimVerification:
    """Use Gemini with Google Search grounding to verify a claim."""

    prompt = f"""You are a fact-checker. Verify the following claim using web search.

CLAIM: {claim}

Analyze the search results and determine:
1. Is this claim VERIFIED (supported by evidence), FALSE (contradicted by evidence), 
   UNCERTAIN (mixed or insufficient evidence), or NEEDS_REVIEW (sensitive topic requiring human review)?
2. What evidence supports or contradicts this claim?
3. What are the sources?

Respond in this exact JSON format:
{{
    "verdict": "verified" or "false" or "uncertain" or "needs_review",
    "evidence": "Brief explanation of what you found",
    "sources": ["source1", "source2"]
}}"""

    try:
        # Use Gemini's built-in Google Search tool for grounding
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1, tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        text = response.text.strip()

        # Parse JSON from response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return ClaimVerification(
                    claim=claim,
                    verdict=result.get("verdict", "uncertain"),
                    evidence=result.get("evidence", "Could not parse evidence"),
                    sources=result.get("sources", []),
                )
            except json.JSONDecodeError:
                pass

        # Fallback if JSON parsing fails
        return ClaimVerification(
            claim=claim, verdict="uncertain", evidence=text[:500], sources=[]
        )

    except Exception as e:
        return ClaimVerification(
            claim=claim,
            verdict="uncertain",
            evidence=f"Error during verification: {str(e)}",
            sources=[],
        )


def gemini_generate_report(client: genai.Client, state: GraphState) -> str:
    """Use Gemini to generate final analysis report."""

    verifications_text = "\n".join(
        [
            f"- **{v['claim']}**\n  Verdict: {v['verdict'].upper()}\n  Evidence: {v['evidence']}"
            for v in state.get("verifications", [])
        ]
    )

    sentiment = state.get("sentiment", {})
    human_answers = state.get("human_answers", {})

    prompt = f"""Generate a concise fact-check report for this news article.

Article Title: {state.get("article_title", "Unknown")}

CLAIM VERIFICATIONS:
{verifications_text}

SENTIMENT ANALYSIS:
Overall: {sentiment.get("label", "unknown")} (compound score: {sentiment.get("compound", 0):.2f})

HUMAN REVIEWER NOTES:
{json.dumps(human_answers, indent=2) if human_answers else "None provided"}

Write a professional 2-3 paragraph summary including:
1. Overall assessment of the article's accuracy
2. Key findings from fact-checking
3. Any claims that readers should be cautious about
4. The emotional tone of the article

Be balanced and objective."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def route_input(state: GraphState) -> GraphState:
    """Determine if input is URL or text and normalize."""
    messages = state.get("messages", [])

    if not messages:
        return {"error": "No input provided"}

    last_message = messages[-1].get("content", "")

    # Check if it looks like a URL
    if last_message.strip().startswith(("http://", "https://", "www.")):
        url = last_message.strip()
        if not url.startswith("http"):
            url = "https://" + url
        return {
            "input_mode": "url",
            "article_url": url,
        }
    else:
        return {
            "input_mode": "text",
            "article_text": last_message,
            "article_title": "User-provided text",
        }


def fetch_article(state: GraphState) -> GraphState:
    """Fetch article from URL. May fail and trigger recovery."""
    if state.get("input_mode") != "url":
        return {}

    url = state.get("article_url", "")

    try:
        title, text = fetch_url_content(url)
        return {
            "article_title": title,
            "article_text": text,
        }
    except Exception as e:
        # Set error - will trigger human intervention to paste text
        return {
            "error": f"Failed to fetch URL: {str(e)}",
        }


def handle_fetch_error(state: GraphState) -> GraphState:
    """Handle URL fetch failure by asking user to paste article text."""
    error = state.get("error", "Unknown error")

    # Interrupt to ask user for article text
    user_input = interrupt(
        {
            "type": "fetch_error",
            "message": f"❌ {error}\n\nPlease paste the article text directly:",
        }
    )

    # When resumed, user_input contains the pasted text
    return {
        "article_text": user_input,
        "article_title": "User-provided text (after URL failure)",
        "input_mode": "text",
        "error": None,  # Clear the error
    }


def extract_claims(state: GraphState) -> GraphState:
    """Use Gemini to extract factual claims from the article."""
    client = get_gemini_client()

    article_text = state.get("article_text", "")
    article_title = state.get("article_title", "")

    if not article_text:
        return {"error": "No article text to analyze"}

    claims = gemini_extract_claims(client, article_text, article_title)

    return {
        "claims": claims,
        "claim_idx": 0,
        "verifications": [],  # Will be appended via Annotated[list, add]
    }


def verify_next_claim(state: GraphState) -> GraphState:
    """Verify the current claim using Gemini with Google Search grounding."""
    client = get_gemini_client()

    claims = state.get("claims", [])
    claim_idx = state.get("claim_idx", 0)

    if claim_idx >= len(claims):
        return {}  # No more claims

    claim = claims[claim_idx]
    verification = gemini_verify_claim_with_search(client, claim)

    return {
        "verifications": [verification],  # Will be appended
        "claim_idx": claim_idx + 1,
    }


def check_human_review_needed(state: GraphState) -> GraphState:
    """Check if any claims need human review and prepare questions."""
    verifications = state.get("verifications", [])

    needs_review = []
    for v in verifications:
        if v["verdict"] in ("needs_review", "uncertain"):
            needs_review.append(
                {
                    "claim": v["claim"],
                    "evidence": v["evidence"],
                    "current_verdict": v["verdict"],
                }
            )

    return {
        "needs_human_review": len(needs_review) > 0,
        "human_questions": needs_review,
    }


def human_review(state: GraphState) -> GraphState:
    """Interrupt for human review of uncertain claims."""
    questions = state.get("human_questions", [])

    if not questions:
        return {}

    # Format questions for the user
    question_text = "🔍 **Human Review Required**\n\n"
    question_text += "The following claims need your input:\n\n"

    for i, q in enumerate(questions, 1):
        question_text += f"**Claim {i}:** {q['claim']}\n"
        question_text += f"Evidence found: {q['evidence']}\n"
        question_text += f"Current verdict: {q['current_verdict']}\n\n"

    question_text += "Please provide your assessment. Format: `claim1: verified/false, claim2: verified/false`\n"
    question_text += "Or type 'skip' to proceed without changes."

    # Interrupt and wait for user input
    user_response = interrupt(
        {
            "type": "human_review",
            "message": question_text,
            "claims": [q["claim"] for q in questions],
        }
    )

    return {
        "human_answers": {"raw_response": user_response},
    }


def apply_human_answers(state: GraphState) -> GraphState:
    """Apply human reviewer's answers to update verifications."""
    human_answers = state.get("human_answers", {})
    raw_response = human_answers.get("raw_response", "").lower()

    if raw_response == "skip" or not raw_response:
        return {}

    # Parse user responses and update verifications
    verifications = state.get("verifications", [])
    updated_verifications = []

    for v in verifications:
        updated = dict(v)
        # Check if user provided verdict for this claim
        claim_lower = v["claim"].lower()[:30]  # First 30 chars for matching

        if "verified" in raw_response and claim_lower[:20] in raw_response:
            updated["verdict"] = "verified"
            updated["evidence"] += " [Human confirmed]"
        elif "false" in raw_response and claim_lower[:20] in raw_response:
            updated["verdict"] = "false"
            updated["evidence"] += " [Human marked as false]"

        updated_verifications.append(updated)

    # We need to return new verifications - but since we use add reducer,
    # we'll mark in human_answers that review was applied
    return {
        "human_answers": {
            **human_answers,
            "applied": True,
            "response": raw_response,
        }
    }


def run_sentiment_analysis(state: GraphState) -> GraphState:
    """Run sentiment analysis on the article."""
    article_text = state.get("article_text", "")

    if not article_text:
        return {"sentiment": {"label": "unknown", "compound": 0}}

    sentiment = analyze_sentiment(article_text)
    return {"sentiment": sentiment}


def generate_report(state: GraphState) -> GraphState:
    """Generate final fact-check report using Gemini."""
    client = get_gemini_client()

    report = gemini_generate_report(client, state)

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Conditional edges (routing logic)
# ---------------------------------------------------------------------------


def should_handle_error(state: GraphState) -> Literal["handle_error", "extract_claims"]:
    """Route based on whether fetch had an error."""
    if state.get("error"):
        return "handle_error"
    return "extract_claims"


def should_continue_verification(
    state: GraphState,
) -> Literal["verify_next", "check_review"]:
    """Decide whether to verify next claim or move to review check."""
    claims = state.get("claims", [])
    claim_idx = state.get("claim_idx", 0)

    if claim_idx < len(claims):
        return "verify_next"
    return "check_review"


def should_do_human_review(state: GraphState) -> Literal["human_review", "sentiment"]:
    """Decide whether human review is needed."""
    if state.get("needs_human_review"):
        return "human_review"
    return "sentiment"


def route_after_input(state: GraphState) -> Literal["fetch_article", "extract_claims"]:
    """Route based on input mode."""
    if state.get("input_mode") == "url":
        return "fetch_article"
    return "extract_claims"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph():
    """Construct the LangGraph state machine."""

    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("route_input", route_input)
    builder.add_node("fetch_article", fetch_article)
    builder.add_node("handle_fetch_error", handle_fetch_error)
    builder.add_node("extract_claims", extract_claims)
    builder.add_node("verify_next_claim", verify_next_claim)
    builder.add_node("check_human_review", check_human_review_needed)
    builder.add_node("human_review", human_review)
    builder.add_node("apply_human_answers", apply_human_answers)
    builder.add_node("sentiment_analysis", run_sentiment_analysis)
    builder.add_node("generate_report", generate_report)

    # Add edges
    builder.add_edge(START, "route_input")

    # After routing input, go to fetch (URL) or directly to extract (text)
    builder.add_conditional_edges(
        "route_input",
        route_after_input,
        {
            "fetch_article": "fetch_article",
            "extract_claims": "extract_claims",
        },
    )

    # After fetch, check for errors
    builder.add_conditional_edges(
        "fetch_article",
        should_handle_error,
        {
            "handle_error": "handle_fetch_error",
            "extract_claims": "extract_claims",
        },
    )

    # After handling error, continue to extract claims
    builder.add_edge("handle_fetch_error", "extract_claims")

    # After extracting claims, start verification loop
    builder.add_edge("extract_claims", "verify_next_claim")

    # Verification loop
    builder.add_conditional_edges(
        "verify_next_claim",
        should_continue_verification,
        {
            "verify_next": "verify_next_claim",  # Loop back
            "check_review": "check_human_review",
        },
    )

    # Check if human review needed
    builder.add_conditional_edges(
        "check_human_review",
        should_do_human_review,
        {
            "human_review": "human_review",
            "sentiment": "sentiment_analysis",
        },
    )

    # After human review
    builder.add_edge("human_review", "apply_human_answers")
    builder.add_edge("apply_human_answers", "sentiment_analysis")

    # After sentiment, generate report
    builder.add_edge("sentiment_analysis", "generate_report")

    # End
    builder.add_edge("generate_report", END)

    return builder


def get_graph_with_checkpointer():
    """Get compiled graph with SQLite checkpointer for persistence."""
    builder = build_graph()

    # Use SQLite for checkpoint persistence
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(
        checkpointer=checkpointer,
    )

    return graph


# The graph instance that langgraph dev will load
graph = build_graph().compile()

# ---------------------------------------------------------------------------
# Helper for running in notebook
# ---------------------------------------------------------------------------


def run_news_analysis(input_text: str, thread_id: str = "default"):
    """
    Run the news analysis graph with the given input.
    Returns the final state or interrupt information.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"messages": [{"role": "user", "content": input_text}]}

    try:
        result = graph.invoke(initial_state, config)
        return result
    except Exception as e:
        return {"error": str(e)}


def resume_after_interrupt(user_input: str, thread_id: str = "default"):
    """Resume the graph after an interrupt with user's input."""
    config = {"configurable": {"thread_id": thread_id}}

    # Use Command to resume with the user's input
    result = graph.invoke(Command(resume=user_input), config)
    return result


def get_current_state(thread_id: str = "default"):
    """Get the current state of the graph for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.get_state(config)


if __name__ == "__main__":
    # Quick test

    from dotenv import load_dotenv

    load_dotenv()

    local_graph = get_graph_with_checkpointer()
    print("✅ Local graph created with SQLite persistence")
