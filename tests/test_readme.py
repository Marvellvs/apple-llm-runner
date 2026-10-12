"""US 7.1: README documents install, HF_TOKEN, download, run and curl examples."""

from __future__ import annotations

import re

from conftest import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"

REQUIRED_SECTIONS = (
    "Quick start",
    "Configuration",
    "API",
    "Make targets",
)

ENV_VARS = (
    "LLM_HOST",
    "LLM_PORT",
    "LLM_MODEL",
    "LLM_MODELS_DIR",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "HF_TOKEN",
)

MAKE_TARGETS = (
    "setup",
    "download",
    "list",
    "delete",
    "run",
    "test",
    "coverage",
    "clean",
)


def _readme() -> str:
    assert README.is_file(), "README.md must exist at the project root"
    return README.read_text()


def _section(text: str, title: str) -> str:
    """Return the body of a `## <title>` section (up to the next `##`)."""
    match = re.search(
        rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"README must contain a '## {title}' section"
    return match.group(1)


def test_readme_has_required_sections() -> None:
    text = _readme()
    for title in REQUIRED_SECTIONS:
        assert re.search(
            rf"^## {re.escape(title)}\s*$", text, re.MULTILINE
        ), f"missing section: '## {title}'"


def test_quick_start_covers_install_hf_token_download_and_run() -> None:
    quick_start = _section(_readme(), "Quick start")
    assert "make setup" in quick_start
    assert "HF_TOKEN" in quick_start
    assert "make download" in quick_start
    assert "LLM_MODEL" in quick_start
    assert "make run" in quick_start


def test_configuration_documents_every_env_variable() -> None:
    configuration = _section(_readme(), "Configuration")
    for var in ENV_VARS:
        assert var in configuration, f"env variable {var} missing from Configuration"


def test_configuration_includes_hf_token_description() -> None:
    configuration = _section(_readme(), "Configuration")
    assert "HF_TOKEN" in configuration
    assert re.search(r"\btoken\b", configuration, re.IGNORECASE)


def test_api_section_documents_chat_completions_endpoint() -> None:
    api = _section(_readme(), "API")
    assert "/v1/chat/completions" in api
    assert "stream" in api
    assert "messages" in api
    assert "temperature" in api
    assert "max_tokens" in api


def test_readme_contains_plain_curl_example() -> None:
    text = _readme()
    assert re.search(r"curl\b", text), "README must include a curl example"
    example = re.search(
        r"```bash\s*\n(curl\b.*?)```", text, re.DOTALL
    )
    assert example is not None
    block = example.group(1)
    assert "/v1/chat/completions" in block
    assert "Content-Type: application/json" in block
    assert re.search(r'"stream"\s*:\s*true', block) is None, (
        "the plain example must not enable streaming"
    )


def test_readme_contains_streaming_curl_example() -> None:
    blocks = re.findall(r"```bash\s*\n(curl\b.*?)```", _readme(), re.DOTALL)
    streaming = [b for b in blocks if re.search(r'"stream"\s*:\s*true', b)]
    assert streaming, "README must include a streaming curl example"
    assert "/v1/chat/completions" in streaming[0]
    assert re.search(r"curl -N\b", streaming[0]), (
        "streaming example should disable curl buffering with -N"
    )


def test_make_targets_section_lists_all_targets() -> None:
    make_section = _section(_readme(), "Make targets")
    for target in MAKE_TARGETS:
        assert re.search(rf"\bmake {target}\b", make_section), (
            f"make target '{target}' missing from Make targets"
        )


def test_readme_mentions_hf_token_is_optional_and_not_logged() -> None:
    text = _readme()
    configuration = _section(text, "Configuration")
    assert re.search(
        r"optional", configuration, re.IGNORECASE
    ), "Configuration must say HF_TOKEN is optional"
    assert re.search(
        r"never.*(log|print)", configuration, re.IGNORECASE | re.DOTALL
    ), "Configuration must say the token is never logged"


def test_readme_mentions_macos_apple_silicon_and_mlx() -> None:
    text = _readme()
    assert re.search(r"macOS", text)
    assert re.search(r"Apple Silicon", text)
    assert re.search(r"\bMLX\b", text)
