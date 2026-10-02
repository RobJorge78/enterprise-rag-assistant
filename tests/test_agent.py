from agent import choose_action


def test_search_action():
    question = "What programming languages does this candidate know?"

    action = choose_action(question)

    assert action == "search"


def test_summarize_action():
    question = "Summarize this document."

    action = choose_action(question)

    assert action == "summarize"


def test_unsupported_action():
    question = "Tell me a joke."

    action = choose_action(question)

    assert action == "unsupported"