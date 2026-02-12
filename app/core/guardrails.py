import re
import logging

logger = logging.getLogger(__name__)


def strong_retrieval_guardrail(results, distance_threshold=1.5):
    if not results:
        return False
    return results[0]["distance"] <= distance_threshold


def keyword_intent_guardrail(question: str, text: str):
    question = question.lower()
    text = text.lower()

    critical_keywords = ["ifsc", "swift", "iban", "routing"]

    for kw in critical_keywords:
        if kw in question and kw not in text:
            return False

    return True


def question_coverage_guardrail(question: str, text: str, min_match_ratio=0.2):
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "because", "as", "until", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "many", "much",
        "tell", "give", "show", "find", "get", "list", "describe",
    }

    question_words = [
        w for w in re.findall(r"\w+", question.lower())
        if w not in stop_words and len(w) > 1
    ]
    text_words = set(re.findall(r"\w+", text.lower()))

    if not question_words:
        return True

    matched = [w for w in question_words if w in text_words]
    return (len(matched) / len(question_words)) >= min_match_ratio


def rate_guardrail(question: str, text: str) -> bool:
    question = question.lower()
    text = text.lower()

    if "rate" in question or "charge" in question or "amount" in question:
        has_number = bool(re.search(r"\d", text))
        has_currency = any(c in text for c in ["$", "usd", "inr", "eur", "gbp", "cad"])
        return has_number and has_currency

    return True


def final_guardrail(question: str, results) -> bool:
    if not strong_retrieval_guardrail(results):
        logger.debug("Guardrail blocked: strong_retrieval")
        return False

    top_text = results[0]["chunk"]["text"]

    if not keyword_intent_guardrail(question, top_text):
        logger.debug("Guardrail blocked: keyword_intent")
        return False

    if not question_coverage_guardrail(question, top_text):
        logger.debug("Guardrail blocked: question_coverage")
        return False

    if not rate_guardrail(question, top_text):
        logger.debug("Guardrail blocked: rate_guardrail")
        return False

    return True
