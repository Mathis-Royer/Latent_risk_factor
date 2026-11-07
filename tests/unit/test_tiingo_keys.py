"""
Tests for Tiingo API key resolution (env var → Colab Secrets → gitignored file).
"""

import sys
import types

import pytest

from src.data_pipeline.tiingo_keys import load_tiingo_keys


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from keys set in the developer's shell or a Colab runtime."""
    monkeypatch.delenv("TIINGO_API_KEYS", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.userdata", raising=False)


def _fake_colab(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]) -> None:
    """
    Install a stand-in for google.colab.userdata (only importable on Colab).

    :param monkeypatch (MonkeyPatch): pytest monkeypatch fixture
    :param secrets (dict[str, str]): Secrets the fake userdata.get() can return
    """

    class SecretNotFoundError(Exception):
        pass

    def get(name: str) -> str:
        if name not in secrets:
            raise SecretNotFoundError(name)
        return secrets[name]

    userdata = types.ModuleType("google.colab.userdata")
    userdata.get = get  # type: ignore[attr-defined]
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)


def test_env_var_comma_separated(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TIINGO_API_KEYS", " k1, k2 ,,k1 ")
    assert load_tiingo_keys(keys_file=str(tmp_path / "missing.txt")) == ["k1", "k2"]


def test_env_var_wins_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("file_key\n")
    monkeypatch.setenv("TIINGO_API_KEYS", "env_key")
    assert load_tiingo_keys(keys_file=str(keys_file)) == ["env_key"]


def test_colab_secret_used_when_env_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _fake_colab(monkeypatch, {"TIINGO_API_KEYS": "c1,c2"})
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("file_key\n")
    assert load_tiingo_keys(keys_file=str(keys_file)) == ["c1", "c2"]


def test_missing_colab_secret_falls_back_to_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _fake_colab(monkeypatch, {})
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("file_key\n")
    assert load_tiingo_keys(keys_file=str(keys_file)) == ["file_key"]


def test_file_one_key_per_line(tmp_path) -> None:
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("a\n\n  b  \na\n")
    assert load_tiingo_keys(keys_file=str(keys_file)) == ["a", "b"]


def test_no_source_returns_empty_list(tmp_path) -> None:
    assert load_tiingo_keys(keys_file=str(tmp_path / "missing.txt")) == []
