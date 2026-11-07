import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_teacher_reasoning_repetition.py"
SPEC = importlib.util.spec_from_file_location("teacher_reasoning_repetition", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in MODULE.words(text):
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 1))
        return {"input_ids": ids}


def record(messages, fingerprint="fingerprint"):
    return {
        "input": messages,
        "analysis_split": "train",
        "analysis_input_file": "/data/train.jsonl",
        "analysis_input_line": 1,
        "instance_id": "r2egym__repo__task",
        "repo": "repo",
        "seed": 0,
        "trajectory_fingerprint": fingerprint,
        "source_file": "batch-0000.jsonl",
        "source_line": 7,
    }


def test_repeated_ngram_positions_ignore_first_occurrence():
    tokens = list(range(12)) * 3
    repeated = MODULE.repeated_ngram_positions(tokens)
    assert min(repeated) == 12
    assert max(repeated) == 35
    assert len(repeated) == 24


def test_self_loop_and_literal_tool_call_are_flagged():
    repeated = " ".join(f"token{i}" for i in range(40))
    reasoning = ((repeated + " ") * 6) + "<tool_call>"
    trajectories, turns = MODULE.analyze_records(
        [record([{"role": "assistant", "reasoning_content": reasoning, "content": ""}])],
        MODULE.TokenizerAdapter(WordTokenizer()),
    )
    assert turns[0]["high_risk"] is True
    assert "reasoning_self_loop" in turns[0]["labels"]
    assert "literal_tool_call" in turns[0]["labels"]
    assert trajectories[0]["high_risk"] is True


def test_tool_copying_code_is_separated_from_reasoning_loop():
    code = "\n".join(f"def function_{i}(): return value_{i}" for i in range(70))
    trajectories, turns = MODULE.analyze_records(
        [record([
            {"role": "tool", "content": code},
            {"role": "assistant", "reasoning_content": code, "content": "inspect this code"},
        ])],
        MODULE.TokenizerAdapter(WordTokenizer()),
    )
    assert "tool_output_copying" in turns[0]["labels"]
    assert "legitimate_code_reference" in turns[0]["labels"]
    assert turns[0]["high_risk"] is False
    assert trajectories[0]["high_risk"] is False


def test_cross_turn_repetition_is_flagged():
    reasoning = " ".join(f"step{i}" for i in range(180))
    _, turns = MODULE.analyze_records(
        [record([
            {"role": "assistant", "reasoning_content": reasoning, "content": ""},
            {"role": "assistant", "reasoning_content": reasoning, "content": ""},
        ])],
        MODULE.TokenizerAdapter(WordTokenizer()),
    )
    assert "cross_turn_repetition" in turns[1]["labels"]
    assert turns[1]["best_previous_assistant_turn"] == 1


def test_summary_population_and_ratios():
    reasoning = " ".join(f"x{i}" for i in range(60))
    trajectories, turns = MODULE.analyze_records(
        [record([{"role": "assistant", "reasoning_content": reasoning, "content": ""}])],
        MODULE.TokenizerAdapter(WordTokenizer()),
    )
    summary = MODULE.build_summary(trajectories, turns)
    assert summary["population"] == {
        "trajectories": 1,
        "reasoning_turns": 1,
        "train": 1,
        "validation": 0,
    }
    assert summary["reasoning"]["total_tokens"] == 60
