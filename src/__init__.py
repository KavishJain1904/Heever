"""Heever -- AI support agent for Twitter customer support.

Module contracts are specified in docs/12-module-contracts.md.
On-disk schemas are specified in docs/11-data-contracts.md.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Config(dict):
    """dict with attribute access, so `config.thresholds.tau_intent` reads naturally.

    Config-as-policy is a stated demonstration (docs/02 §1), so the ergonomics of
    reading a threshold from config must be better than the ergonomics of hardcoding
    one. Mappings are wrapped recursively on access.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - attribute errors are programmer error
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@functools.lru_cache(maxsize=1)
def load_env() -> None:
    """Read .env into the process environment, once. Never overrides a variable
    that is already set -- including set-but-empty, which is how tests and a
    reviewer's `make eval` say "no key" on purpose."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is pinned in requirements.txt
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


@functools.lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml. Cached: the file is read once per process."""
    import yaml

    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(resolved, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


@functools.lru_cache(maxsize=8)
def load_taxonomy(path: str | Path | None = None) -> Config:
    """Load taxonomy/intents.yaml."""
    import yaml

    if path is None:
        path = REPO_ROOT / load_config().taxonomy["path"]
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def intent_names(path: str | Path | None = None) -> list[str]:
    """Every declared intent name, in declaration order."""
    return [i["name"] for i in load_taxonomy(path)["intents"]]
