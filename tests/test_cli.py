from __future__ import annotations

import json

from vipii.cli import main


def test_cli_text_output(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "sample.txt"
    path.write_text("Số điện thoại 0912345678.", encoding="utf-8")

    assert main(["scan", str(path)]) == 0

    output = capsys.readouterr().out
    assert "PHONE_NUMBER" in output
    assert "0912345678" in output


def test_cli_plain_text_output(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["scan", "Số điện thoại 0912345678."]) == 0

    output = capsys.readouterr().out
    assert "PHONE_NUMBER" in output
    assert "0912345678" in output


def test_cli_explicit_text_input_does_not_read_matching_file(tmp_path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "0912345678"
    path.write_text("Xin chào.", encoding="utf-8")

    assert main(["scan", "0912345678", "--input-type", "text"]) == 0

    assert "PHONE_NUMBER" in capsys.readouterr().out


def test_cli_explicit_file_input(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "sample.txt"
    path.write_text("Mã số thuế 0312345678.", encoding="utf-8")

    assert main(["scan", str(path), "--input-type", "file"]) == 0

    assert "MST" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "sample.txt"
    path.write_text("Mã số thuế 0312345678.", encoding="utf-8")

    assert main(["scan", str(path), "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"] == "MST"


def test_cli_uses_custom_yaml_config(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "recognizers.yml"
    config.write_text(
        """
recognizers:
  - name: customer_id
    label: CUSTOMER_ID
    patterns:
      - regex: '\\bKH-\\d{6}\\b'
        context_words: ["mã khách hàng"]
        base_score: 0.6
""",
        encoding="utf-8",
    )

    assert main(["scan", "Mã khách hàng KH-123456.", "--config", str(config)]) == 0

    assert "CUSTOMER_ID" in capsys.readouterr().out


def test_cli_redact_output(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "sample.txt"
    path.write_text("CCCD 001203000123.", encoding="utf-8")

    assert main(["scan", str(path), "--redact"]) == 0

    assert "001203000123" not in capsys.readouterr().out


def test_cli_redacts_plain_text(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["scan", "CCCD 001203000123.", "--redact"]) == 0

    assert "001203000123" not in capsys.readouterr().out


def test_cli_empty_and_no_match_files(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    empty = tmp_path / "empty.txt"
    plain = tmp_path / "plain.txt"
    empty.write_text("", encoding="utf-8")
    plain.write_text("Xin chào.", encoding="utf-8")

    assert main(["scan", str(empty), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["scan", str(plain)]) == 0
    assert capsys.readouterr().out == ""
