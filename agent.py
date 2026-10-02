def choose_action(question: str) -> str:
    question = question.lower()

    if "summarize" in question or "summary" in question:
        return "summarize"

    if any(word in question for word in [
        "what",
        "which",
        "who",
        "where",
        "when",
        "how"
    ]):
        return "search"

    return "unsupported"