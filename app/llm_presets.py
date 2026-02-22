from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LLMPreset:
    slug: str
    label: str
    default_base_url: str
    default_api_key_env_var: str
    fallback_models: tuple[str, ...]
    required_fields: tuple[str, ...]
    known_features_by_model: dict[str, tuple[str, ...]]

    def as_public_dict(self) -> dict[str, object]:
        return {
            "slug": self.slug,
            "label": self.label,
            "default_base_url": self.default_base_url,
            "default_api_key_env_var": self.default_api_key_env_var,
            "fallback_models": list(self.fallback_models),
            "required_fields": list(self.required_fields),
            "known_features_by_model": {
                model: list(features) for model, features in self.known_features_by_model.items()
            },
        }


LLM_REQUIRED_FIELDS: tuple[str, ...] = ("base_url", "model", "api_key_env_var")


_PRESETS: tuple[LLMPreset, ...] = (
    LLMPreset(
        slug="chatgpt",
        label="ChatGPT (OpenAI)",
        default_base_url="https://api.openai.com/v1",
        default_api_key_env_var="OPENAI_API_KEY",
        fallback_models=(
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
        ),
        required_fields=LLM_REQUIRED_FIELDS,
        known_features_by_model={
            "gpt-4o": ("json_output", "vision", "streaming"),
            "gpt-4o-mini": ("json_output", "vision", "streaming"),
            "gpt-4.1": ("json_output", "vision", "streaming"),
            "gpt-4.1-mini": ("json_output", "vision", "streaming"),
        },
    ),
    LLMPreset(
        slug="claude",
        label="Claude (Anthropic)",
        default_base_url="https://api.anthropic.com/v1",
        default_api_key_env_var="ANTHROPIC_API_KEY",
        fallback_models=(
            "claude-sonnet-4-5",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ),
        required_fields=LLM_REQUIRED_FIELDS,
        known_features_by_model={
            "claude-sonnet-4-5": ("json_output", "vision", "long_context"),
            "claude-3-7-sonnet-latest": ("json_output", "vision", "long_context"),
            "claude-3-5-haiku-latest": ("json_output", "fast_inference"),
        },
    ),
    LLMPreset(
        slug="gemini",
        label="Gemini (Google)",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_api_key_env_var="GOOGLE_API_KEY",
        fallback_models=(
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ),
        required_fields=LLM_REQUIRED_FIELDS,
        known_features_by_model={
            "gemini-2.5-pro": ("json_output", "vision", "long_context"),
            "gemini-2.5-flash": ("json_output", "vision", "fast_inference"),
            "gemini-2.0-flash": ("json_output", "vision", "fast_inference"),
        },
    ),
    LLMPreset(
        slug="manual",
        label="Manual",
        default_base_url="",
        default_api_key_env_var="",
        fallback_models=(),
        required_fields=LLM_REQUIRED_FIELDS,
        known_features_by_model={},
    ),
)

_PRESETS_BY_SLUG: dict[str, LLMPreset] = {preset.slug: preset for preset in _PRESETS}


def get_llm_presets() -> list[LLMPreset]:
    return list(_PRESETS)


def get_llm_preset(slug: str | None) -> LLMPreset | None:
    cleaned_slug = (slug or "").strip().lower()
    if not cleaned_slug:
        return None
    return _PRESETS_BY_SLUG.get(cleaned_slug)


def infer_preset_slug(base_url: str) -> str:
    cleaned_base_url = (base_url or "").strip()
    if not cleaned_base_url:
        return "manual"
    try:
        parsed = urlsplit(cleaned_base_url)
    except ValueError:
        return "manual"
    host = (parsed.hostname or "").strip().lower()
    path = parsed.path.rstrip("/").lower()
    if "api.openai.com" in host:
        return "chatgpt"
    if "api.anthropic.com" in host:
        return "claude"
    if "generativelanguage.googleapis.com" in host or "googleapis.com" in host:
        if "/openai" in path or path.endswith("openai"):
            return "gemini"
    return "manual"

