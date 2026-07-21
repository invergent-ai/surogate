"""Tests for the mergekit-tokensurgeon transplant wrapper (subprocess mocked)."""

from __future__ import annotations

import json
import os
import shutil
from unittest import mock

import pytest

from surogate.transplant import TRANSPLANT_MANIFEST, run_transplant, run_transplant_back
from surogate.transplant import transplant as transplant_mod


def _fake_output_model(out_dir: str, tie_word_embeddings: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"vocab_size": 129280, "tie_word_embeddings": tie_word_embeddings}, f)
    open(os.path.join(out_dir, "tokenizer.json"), "w").write("{}")
    open(os.path.join(out_dir, "model.safetensors"), "wb").write(b"\0")


def _completed(returncode=0, stderr=""):
    return mock.Mock(returncode=returncode, stdout="", stderr=stderr)


@pytest.fixture()
def student_teacher(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "teacher"
    student.mkdir()
    teacher.mkdir()
    (student / "config.json").write_text("{}")
    (teacher / "config.json").write_text("{}")
    return str(student), str(teacher)


class TestRunTransplant:
    def test_invocation_and_manifest(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        out = str(tmp_path / "out")

        def fake_run(cmd, capture_output, text):
            _fake_output_model(out)
            return _completed()

        with (
            mock.patch.object(shutil, "which", return_value="/usr/bin/mergekit-tokensurgeon"),
            mock.patch.object(transplant_mod.subprocess, "run", side_effect=fake_run) as run_mock,
        ):
            result = run_transplant(student, teacher, out, method="omp", k=64, device="cuda")

        assert result == out
        cmd = run_mock.call_args.args[0]
        assert cmd[0] == "/usr/bin/mergekit-tokensurgeon"
        assert cmd[1:4] == [student, teacher, out]
        assert cmd[cmd.index("--approximation-method") + 1] == "omp"
        assert cmd[cmd.index("--k") + 1] == "64"
        assert cmd[cmd.index("--device") + 1] == "cuda"
        assert "--trust-remote-code" not in cmd

        manifest = json.load(open(os.path.join(out, TRANSPLANT_MANIFEST)))
        assert manifest["student_model"] == student
        assert manifest["teacher_model"] == teacher
        assert manifest["method"] == "omp"
        assert manifest["k"] == 64

    def test_trust_remote_code_and_extra_args(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        out = str(tmp_path / "out")

        def fake_run(cmd, capture_output, text):
            _fake_output_model(out)
            return _completed()

        with (
            mock.patch.object(shutil, "which", return_value="mergekit-tokensurgeon"),
            mock.patch.object(transplant_mod.subprocess, "run", side_effect=fake_run) as run_mock,
        ):
            run_transplant(
                student, teacher, out, trust_remote_code=True, extra_args=["--byte-match", "yes"]
            )
        cmd = run_mock.call_args.args[0]
        assert "--trust-remote-code" in cmd
        assert cmd[cmd.index("--byte-match") + 1] == "yes"

    def test_missing_mergekit_is_actionable(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        with mock.patch.object(shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="pip install mergekit"):
                run_transplant(student, teacher, str(tmp_path / "out"))

    def test_subprocess_failure_surfaces_stderr(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        with (
            mock.patch.object(shutil, "which", return_value="mergekit-tokensurgeon"),
            mock.patch.object(
                transplant_mod.subprocess,
                "run",
                return_value=_completed(returncode=1, stderr="boom: donor vocab unreadable"),
            ),
        ):
            with pytest.raises(RuntimeError, match="donor vocab unreadable"):
                run_transplant(student, teacher, str(tmp_path / "out"))

    def test_incomplete_output_rejected(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        out = str(tmp_path / "out")

        def fake_run(cmd, capture_output, text):
            os.makedirs(out, exist_ok=True)  # no config.json / tokenizer / weights
            return _completed()

        with (
            mock.patch.object(shutil, "which", return_value="mergekit-tokensurgeon"),
            mock.patch.object(transplant_mod.subprocess, "run", side_effect=fake_run),
        ):
            with pytest.raises(RuntimeError, match="config.json"):
                run_transplant(student, teacher, out)

    def test_unknown_method_rejected(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        with mock.patch.object(shutil, "which", return_value="mergekit-tokensurgeon"):
            with pytest.raises(ValueError, match="approximation method"):
                run_transplant(student, teacher, str(tmp_path / "out"), method="magic")
            with pytest.raises(ValueError, match="k must be"):
                run_transplant(student, teacher, str(tmp_path / "out"), k=0)


class TestRunTransplantBack:
    def test_reverse_uses_manifest_donor_and_method(self, tmp_path, student_teacher):
        student, teacher = student_teacher
        transplanted = str(tmp_path / "transplanted")
        _fake_output_model(transplanted)
        manifest_path = os.path.join(transplanted, TRANSPLANT_MANIFEST)
        json.dump(
            {
                "version": 1,
                "student_model": student,
                "student_dir": student,
                "teacher_model": teacher,
                "method": "common_interpolation",
                "k": 8,
            },
            open(manifest_path, "w"),
        )
        distilled = str(tmp_path / "distilled")
        _fake_output_model(distilled)
        out = str(tmp_path / "restored")

        def fake_run(cmd, capture_output, text):
            _fake_output_model(out)
            return _completed()

        with (
            mock.patch.object(shutil, "which", return_value="mergekit-tokensurgeon"),
            mock.patch.object(transplant_mod.subprocess, "run", side_effect=fake_run) as run_mock,
        ):
            run_transplant_back(distilled, manifest_path, out)

        cmd = run_mock.call_args.args[0]
        # distilled model is the base; the ORIGINAL STUDENT is the tokenizer donor
        assert cmd[1:4] == [distilled, student, out]
        assert cmd[cmd.index("--approximation-method") + 1] == "common_interpolation"
        assert cmd[cmd.index("--k") + 1] == "8"

    def test_manifest_without_student_rejected(self, tmp_path):
        manifest_path = str(tmp_path / "m.json")
        json.dump({"version": 1}, open(manifest_path, "w"))
        with pytest.raises(ValueError, match="student"):
            run_transplant_back(str(tmp_path), manifest_path, str(tmp_path / "out"))


def test_cli_parser_roundtrip():
    from surogate.cli.transplant import prepare_command_parser

    parser = prepare_command_parser()
    args = parser.parse_args(
        ["--student", "s", "--teacher", "t", "--output", "o", "--method", "omp", "--k", "32"]
    )
    assert (args.student, args.teacher, args.output, args.method, args.k) == ("s", "t", "o", "omp", 32)
    args = parser.parse_args(["--student", "d", "--output", "o", "--restore", "m.json"])
    assert args.restore == "m.json"


def test_cli_registered():
    from surogate.cli.main import COMMAND_MAPPING

    assert COMMAND_MAPPING["transplant-tokenizer"] == "surogate.cli.transplant"
