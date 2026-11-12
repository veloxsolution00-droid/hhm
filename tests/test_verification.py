"""Unit tests for the isolated claim-verification boundary."""

from app.rag.verification import verify_claims


def test_missing_claim_citation_fails_without_calling_model():
    calls = []
    evidence = "[1] policy.pdf (Page 1):\nThe policy allows five days."

    passed = verify_claims(
        "What does the policy allow?",
        evidence,
        "The policy allows five days.",
        generate_fn=lambda *args, **kwargs: calls.append((args, kwargs)) or "SUPPORTED ANSWERED",
        enabled=True,
    )

    assert passed is False
    assert calls == []


def test_verifier_receives_only_passages_cited_by_each_claim():
    captured = {}
    evidence = (
        "[1] a.pdf (Page 1):\nThe first policy states five days.\n\n"
        "[2] b.pdf (Page 2):\nThe second policy states ten days.\n\n"
        "[3] c.pdf (Page 3):\nUnrelated passage."
    )

    def generate(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "SUPPORTED ANSWERED"

    passed = verify_claims(
        "Compare the policies.",
        evidence,
        "The first policy states five days [1]. The second states ten days [2].",
        generate_fn=generate,
        enabled=True,
        max_tokens=9,
    )

    assert passed is True
    assert "The first policy states five days." in captured["prompt"]
    assert "The second policy states ten days." in captured["prompt"]
    assert "Unrelated passage." not in captured["prompt"]
    assert captured["kwargs"] == {"num_predict": 9}


def test_verifier_rejects_unsupported_model_verdict():
    passed = verify_claims(
        "What does the policy allow?",
        "[1] policy.pdf (Page 1):\nThe policy allows five days.",
        "The policy allows five days [1].",
        generate_fn=lambda *args, **kwargs: "UNSUPPORTED ANSWERED",
        enabled=True,
    )

    assert passed is False


def test_disabled_verification_does_not_call_generation():
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Verifier must not call the model when disabled.")

    assert verify_claims(
        "Question", "", "Answer [1].",
        generate_fn=unexpected_call,
        enabled=False,
    ) is True
