"""Secrets never reach logs, the database or a comment (issue #15)."""

import logging

import pytest
from pydantic import SecretStr

from review_agents.config import ModelEndpoint, Observability, Settings
from review_agents.credentials import TokenVault
from review_agents.redact import MASK, LogFilter, Redactor, register_settings

# Fake values in the shapes real ones take, assembled so no scanner takes them for real.
GITHUB_TOKEN = "ghs_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FINE_GRAINED = "github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"
MODEL_KEY = "sk-" + "proj-Zq7Xw2Vb9Nm4Lk8Jh3Gf6Ds1"


def test_a_known_secret_is_masked_wherever_it_appears() -> None:
    redact = Redactor()
    redact.add("our-shared-secret-value-0123456789")

    text = redact("signing with our-shared-secret-value-0123456789!")

    assert text == f"signing with {MASK}!"


def test_a_short_value_is_not_taken_for_a_secret() -> None:
    # A key set to "test" in a dev setup must not mask every "test" in a report.
    redact = Redactor()
    redact.add("test")
    redact.add(None)

    assert redact("a test") == "a test"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"token {GITHUB_TOKEN} used", f"token {MASK} used"),
        (f"pat={FINE_GRAINED}", f"pat={MASK}"),
        (f"key: {MODEL_KEY}", f"key: {MASK}"),
        ("Authorization: Bearer abcdefghijklmnop0123", f"Authorization: Bearer {MASK}"),
        ("https://user:hunter2hunter2@proxy.internal:3128", f"https://{MASK}@proxy.internal:3128"),
    ],
    ids=["github", "fine-grained", "model-key", "bearer", "url-userinfo"],
)
def test_secrets_in_well_known_shapes_are_masked_unregistered(text: str, expected: str) -> None:
    assert Redactor()(text) == expected


def test_ordinary_text_is_left_alone() -> None:
    text = "refund(order, 10_000) returns sk-ip and ghs_ is a prefix; see https://x.test/a:b"

    assert Redactor()(text) == text


FINDER_KEY = "finder-key-" + "0123456789"
VERIFIER_KEY = "verifier-key-" + "0123456789"
OBSERVABILITY_KEY = "b2JzZXJ2YWJp" + "bGl0eS1rZXk="


def test_the_service_s_own_secrets_are_registered() -> None:
    redact = Redactor()
    endpoint = {"base_url": "https://m.test/v1", "model": "m"}
    settings = Settings(
        shared_secret=SecretStr("s" * 40),
        finder_model=ModelEndpoint.model_validate({**endpoint, "api_key": FINDER_KEY}),
        verifier_model=ModelEndpoint.model_validate({**endpoint, "api_key": VERIFIER_KEY}),
        observability=Observability.model_validate(
            {
                "enabled": True,
                "endpoint": "https://o.test",
                "headers": {"Authorization": f"Basic {OBSERVABILITY_KEY}"},
            }
        ),
    )

    register_settings(redact, settings)

    text = redact(" ".join(["s" * 40, FINDER_KEY, VERIFIER_KEY, OBSERVABILITY_KEY]))
    assert text == " ".join([MASK] * 4)


def test_a_review_s_github_token_is_masked_once_the_vault_holds_it() -> None:
    redact = Redactor()
    vault = TokenVault(redact)

    vault.put(7, "plain-looking-token-0123456789")

    assert redact("git said plain-looking-token-0123456789") == f"git said {MASK}"


def test_log_records_are_masked_with_their_arguments_and_tracebacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    redact = Redactor()
    redact.add("our-shared-secret-value-0123456789")
    logger = logging.getLogger("review_agents.test_redact")
    caplog.handler.addFilter(LogFilter(redact))

    try:
        raise RuntimeError("bad key our-shared-secret-value-0123456789")
    except RuntimeError:
        logger.exception("review %s failed with %s", 7, GITHUB_TOKEN)

    [record] = caplog.records
    assert record.getMessage() == f"review 7 failed with {MASK}"
    assert record.exc_text is not None and MASK in record.exc_text
    assert "our-shared-secret-value-0123456789" not in caplog.text
    assert GITHUB_TOKEN not in caplog.text


def test_only_the_newest_review_tokens_are_kept() -> None:
    redact = Redactor(recent=2)
    redact.add("service-secret-0123456789")
    for n in range(3):
        redact.add_recent(f"review-token-{n}-0123456789")

    text = redact("service-secret-0123456789 review-token-0-0123456789 review-token-2-0123456789")

    assert text == f"{MASK} review-token-0-0123456789 {MASK}"
