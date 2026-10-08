"""The .env loader: parsing, precedence, and the token aliases. No network."""

import os

from worldoptbench.env import load_env, parse_env


def test_parse_handles_comments_quotes_and_export():
    text = "# note\n\nA=1\nexport B = 'two'\nC=\"three\"\nnot a pair\nD=a=b\n"
    assert parse_env(text) == {"A": "1", "B": "two", "C": "three", "D": "a=b"}


def test_token_alias_is_exported_as_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("token=hf_example\n", encoding="utf-8")
    load_env(env)
    assert os.environ["HF_TOKEN"] == "hf_example"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_example"
    assert "token" not in os.environ  # the generic key is not leaked into the environment


def test_a_variable_already_in_the_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    env = tmp_path / ".env"
    env.write_text("token=from_file\nOTHER_KEY=x\n", encoding="utf-8")
    monkeypatch.delenv("OTHER_KEY", raising=False)
    load_env(env)
    assert os.environ["HF_TOKEN"] == "from_shell"
    assert os.environ["OTHER_KEY"] == "x"
    monkeypatch.delenv("OTHER_KEY")


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}
