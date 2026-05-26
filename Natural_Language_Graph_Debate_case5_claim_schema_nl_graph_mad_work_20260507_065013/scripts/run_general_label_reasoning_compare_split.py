from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset, load_from_disk

from _local_bootstrap import PROJECT_ROOT
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.scripts.run_compare_origin_vs_nl_graph_batch import (
    _build_mode_round_metrics,
    _build_summary,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.label_answer import extract_label_answer as shared_extract_label_answer
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.local_evaluator import extract_numeric_answer
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.runtime import close_runtime, load_runtime, run_prompts
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.mcq_answer import (
    canonical_mcq_answer,
    extract_option_map,
    mcq_answers_match,
    question_requests_option_text_answer,
)
from Natural_Language_Graph_Debate_case5_claim_schema_nl_graph_mad_work_20260507_065013.src.short_answer import (
    extract_short_answer,
    question_requests_short_answer,
    short_answers_match,
)


DEFAULT_MODEL = "/home/zihan/silver/model/qwen3"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "datasets"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    token_path = PROJECT_ROOT / "token"
    token = token_path.read_text().strip() if token_path.exists() else ""
    return SimpleNamespace(
        model=args.model,
        model_dir=args.model_dir,
        use_vllm=True,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        token=token,
        multi_persona=False,
        data="math",
        memory_for_model_activations_in_gb=2,
    )


def _round0_prompt(problem: str, style: str = "bare") -> str:
    style = str(style or "bare").strip().lower()
    lower_problem = problem.lower()
    if " yes or no" in lower_problem or "yes/no" in lower_problem:
        example = "{final answer: \\boxed{yes}}"
    elif " true or false" in lower_problem or "true/false" in lower_problem:
        example = "{final answer: \\boxed{true}}"
    elif question_requests_option_text_answer(problem):
        example = "{final answer: \\boxed{Nitrofurantoin}}"
    elif question_requests_short_answer(problem):
        example = "{final answer: \\boxed{Paris}}"
    else:
        example = "{final answer: \\boxed{1}}"
    base = (
        f'{problem.strip()} '
        'Make sure to state your final answer in curly brackets at the very end of your response, '
        f'just like: "{example}".'
    )
    if style == "bare":
        return base
    if style in {"cot", "zero_shot_cot", "think_step_by_step"}:
        return (
            f"{problem.strip()} "
            "Think step by step. "
            "Make sure to state your final answer in curly brackets at the very end of your response, "
            f'just like: "{example}".'
        )
    if style in {"bare_reason", "brief_reason", "minimal_reason"}:
        return (
            f"{problem.strip()} "
            "Give one brief reason that supports your answer, then state your final answer in curly brackets "
            f'at the very end of your response, just like: "{example}".'
        )
    if style in {"claim_atomic", "light_atomic", "predicate_atomic"}:
        return (
            f"{problem.strip()} "
            "Write your reasoning as atomic steps before the final answer. "
            "Each step should contain one complete fact, evidence claim, or inference about the question. "
            "Before the final answer, include one short step that connects the evidence to the exact question predicate. "
            "Use the ordinary sense of the question unless the question explicitly asks for a technical or special definition. "
            "Keep the asked relation separate from nearby, weaker, stronger, or historical relations. "
            "If you use a common-sense bridge, state it as a bridge instead of silently changing the question. "
            "Do not invent an extra condition, exception, identity, discount, or special permission unless it follows from the question or evidence. "
            "Do not write format notes or bare labels as reasoning steps. "
            "The last non-empty line must be the final answer in curly brackets, "
            f'just like: "{example}".'
        )
    if style in {"logic_atomic", "formal_logic_atomic"}:
        return (
            f"{problem.strip()} "
            "Write your reasoning as atomic proof steps before the final answer. "
            "Each step should contain one explicit premise, one rule application, or one inference about the queried conclusion. "
            "Use only the stated premises and their logical consequences; do not use ordinary-world assumptions, common-sense facts, or self-evidence. "
            "Do not use the converse or inverse of a conditional unless it is explicitly stated. "
            "For a disjunction, do not choose one branch unless the premises rule out the other branch. "
            "Before the final answer, include one short step saying whether the premises prove the conclusion, prove its negation, or leave it unknown/uncertain. "
            "Do not write format notes or bare labels as reasoning steps. "
            "The last non-empty line must be the final answer in curly brackets, "
            f'just like: "{example}".'
        )
    return (
        base
        + " Use short atomic reasoning lines. "
        + "Each non-empty line should contain exactly one local fact, check, inference, or conclusion. "
        + "The last non-empty line must be the final answer line."
    )


def _normalize_numeric_label(value) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        text = str(value).strip()
        return text or None
    if numeric.is_integer():
        return str(int(numeric))
    text = str(numeric)
    return text[:-2] if text.endswith(".0") else text


def _extract_explicit_label_answer(text: str) -> str | None:
    clean = str(text or "").replace("\n", " ").strip()
    if not clean:
        return None
    final_braced = re.findall(r"\{[^{}]*final answer[^{}]*\}", clean, flags=re.IGNORECASE)
    for seg in final_braced[::-1]:
        boxed = re.search(r"boxed\{\s*(-?\d{1,2})\s*\}", seg, flags=re.IGNORECASE)
        if boxed:
            return boxed.group(1)
        direct = re.search(r"final answer[^0-9-]*(-?\d{1,2})(?!\d)", seg, flags=re.IGNORECASE)
        if direct:
            return direct.group(1)
    tail = clean[-200:]
    boxed_tail = re.search(r"boxed\{\s*(-?\d{1,2})\s*\}\s*[\}\]\"'. ]*$", tail, flags=re.IGNORECASE)
    if boxed_tail:
        return boxed_tail.group(1)
    final_tail = re.search(r"final answer\s*[:\-]?\s*(?:\\boxed\{\s*)?(-?\d{1,2})(?:\s*\})?\s*[\]})\"'. ]*$", tail, flags=re.IGNORECASE)
    if final_tail:
        return final_tail.group(1)
    answer_tail = re.search(r"(?:therefore\s+)?answer\s*[:=\-]\s*(-?\d{1,2})\s*[\]})\"'. ]*$", tail, flags=re.IGNORECASE)
    if answer_tail:
        return answer_tail.group(1)
    return None


def _extract_label_answer(text: str, question: str = "") -> str | None:
    if question_requests_option_text_answer(question):
        return canonical_mcq_answer(text, question)
    if question_requests_short_answer(question):
        return extract_short_answer(text)
    shared = shared_extract_label_answer(text, question)
    if shared is not None:
        return shared
    explicit = _extract_explicit_label_answer(text)
    if explicit is not None:
        return explicit
    return None


def _majority_answer(answer_map: dict[str, str | None]) -> str | None:
    answers = [answer for answer in answer_map.values() if answer is not None]
    if not answers:
        return None
    counts = Counter(answers)
    best_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == best_count}
    for _, answer in answer_map.items():
        if answer in tied:
            return answer
    return None


def _all_answers_unanimous(answer_map: dict[str, str | None]) -> bool:
    answers = [answer for answer in answer_map.values() if answer is not None]
    return bool(answer_map) and len(answers) == len(answer_map) and len(set(answers)) == 1


def _is_correct(answer: str | None, gold_answer: str, question: str = "") -> bool:
    if extract_option_map(question) and mcq_answers_match(answer, gold_answer, question):
        return True
    if isinstance(gold_answer, list):
        return short_answers_match(answer, gold_answer)
    if answer is not None and str(answer).strip() == str(gold_answer).strip():
        return True
    return short_answers_match(answer, gold_answer)


def _stable_category_sample_indices(rows: list[dict], sample_ratio: float, seed: int) -> list[int]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        category = str(row.get("category", "uncategorized"))
        stable_key = str(row.get("question_id", row.get("unique_id", idx)))
        digest = hashlib.md5(f"{seed}:{category}:{stable_key}".encode("utf-8")).hexdigest()
        grouped[category].append((digest, idx))

    selected: list[int] = []
    for category, items in grouped.items():
        items.sort()
        keep = max(1, round(len(items) * sample_ratio))
        selected.extend(idx for _, idx in items[:keep])
    return sorted(selected)


def _build_strategyqa_cases(dataset_path: Path, split: str, limit: int | None) -> list[dict]:
    dataset = load_from_disk(str(dataset_path))[split]
    rows = dataset.select(range(limit)) if limit is not None else dataset
    cases = []
    for row in rows:
        problem = f"{str(row['question']).strip()} Answer yes or no. Use only yes or no as the final answer."
        cases.append(
            {
                "unique_id": f"strategyqa/{split}/{row['qid']}.json",
                "problem": problem,
                "answer": "yes" if bool(row["answer"]) else "no",
                "difficulty": None,
            }
        )
    return cases


def _build_prontoqa_cases(dataset_path: Path, split: str, limit: int | None) -> list[dict]:
    dataset = load_from_disk(str(dataset_path))[split]
    rows = dataset.select(range(limit)) if limit is not None else dataset
    cases = []
    for row in rows:
        problem = (
            f"Context:\n{str(row['context']).strip()}\n\n"
            f"Question:\n{str(row['question']).strip()}\n\n"
            "Answer true or false. Use only true or false as the final answer."
        )
        answer = "true" if str(row["answer"]).strip().upper() == "A" else "false"
        cases.append(
            {
                "unique_id": f"prontoqa/{split}/{row['id']}.json",
                "problem": problem,
                "answer": answer,
                "difficulty": None,
                "metadata": {
                    "source_answer": row["answer"],
                    "source_options": row["options"],
                },
            }
        )
    return cases


def _build_anli_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("anli", split=split)
    rows = dataset.select(range(limit)) if limit is not None else dataset
    cases = []
    label_map = {
        0: "1",  # entailment
        1: "2",  # neutral
        2: "3",  # contradiction
    }
    for row in rows:
        problem = (
            f"Premise:\n{str(row['premise']).strip()}\n\n"
            f"Hypothesis:\n{str(row['hypothesis']).strip()}\n\n"
            "Answer 1 for entailment, 2 for neutral, and 3 for contradiction. "
            "Use only 1, 2, or 3 as the final answer. "
            "If natural label words feel easier, you may answer with entailment, neutral, or contradiction; the evaluator will normalize them."
        )
        label = int(row["label"])
        cases.append(
            {
                "unique_id": f"anli/{split}/{row['uid']}.json",
                "problem": problem,
                "answer": label_map[label],
                "difficulty": None,
                "metadata": {
                    "label_name": ["entailment", "neutral", "contradiction"][label],
                },
            }
        )
    return cases


def _build_mmlu_pro_cases(
    dataset_path: Path,
    split: str,
    limit: int | None,
    sample_ratio: float,
    sample_seed: int,
) -> list[dict]:
    dataset = load_from_disk(str(dataset_path))[split]
    rows = [dict(item) for item in dataset]
    if sample_ratio < 1.0:
        selected_indices = _stable_category_sample_indices(rows, sample_ratio, sample_seed)
        rows = [rows[idx] for idx in selected_indices]
    if limit is not None:
        rows = rows[:limit]
    cases = []
    for row in rows:
        option_lines = [f"{idx + 1}. {str(option).strip()}" for idx, option in enumerate(row["options"])]
        problem = (
            f"Question:\n{str(row['question']).strip()}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            f"Answer with the option number only (1-{len(row['options'])}). "
            f"Use only a number from 1 to {len(row['options'])} as the final answer."
        )
        answer = str(int(row["answer_index"]) + 1)
        cases.append(
            {
                "unique_id": f"mmlu_pro/{split}/{row['question_id']}.json",
                "problem": problem,
                "answer": answer,
                "difficulty": None,
                "metadata": {
                    "category": row["category"],
                    "source_answer": row["answer"],
                    "num_options": len(row["options"]),
                },
            }
        )
    return cases


def _build_logiqa_cases(dataset_path: Path, split: str, limit: int | None) -> list[dict]:
    split_aliases = {
        "dev": ["dev", "validation", "val"],
        "validation": ["validation", "dev", "val"],
        "val": ["val", "validation", "dev"],
        "test": ["test"],
        "train": ["train"],
    }
    candidates = []
    for split_name in split_aliases.get(str(split or "test").lower(), [split or "test"]):
        candidates.extend(
            [
                dataset_path / f"{split_name}.jsonl",
                dataset_path / f"logiQA_{split_name}.jsonl",
                dataset_path / "logiQA" / f"logiQA_{split_name}.jsonl",
                dataset_path / "logiQA" / f"{split_name}.jsonl",
            ]
        )
    data_file = next((path for path in candidates if path.exists()), None)
    if data_file is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find LogiQA split {split!r}. Tried: {tried}")

    rows = _load_jsonl(data_file)
    if limit is not None:
        rows = rows[:limit]
    cases = []
    for index, row in enumerate(rows):
        choices = [str(choice).strip() for choice in row["choices"]]
        option_lines = [f"{idx + 1}. {choice}" for idx, choice in enumerate(choices)]
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        problem = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            f"Answer with the option number only (1-{len(choices)}). "
            f"Use only a number from 1 to {len(choices)} as the final answer."
        )
        answer_choice = row.get("answer_choice")
        answer = str(int(answer_choice) + 1)
        row_id = row.get("id_", row.get("id", index))
        cases.append(
            {
                "unique_id": f"logiqa/{split}/{row_id}.json",
                "problem": problem,
                "answer": answer,
                "difficulty": None,
                "metadata": {
                    "answer_text": row.get("answer_text"),
                    "question_type": row.get("question_type", ""),
                    "num_options": len(choices),
                    "source_file": str(data_file),
                },
            }
        )
    return cases


def _build_csqa_cases(dataset_path: Path, split: str, limit: int | None) -> list[dict]:
    split_aliases = {
        "dev": ["dev", "validation", "val", "test"],
        "validation": ["validation", "dev", "val", "test"],
        "val": ["val", "validation", "dev", "test"],
        "test": ["test", "validation", "dev"],
        "train": ["train", "validation", "dev", "test"],
        "": ["validation", "dev", "test"],
    }
    candidates = []
    if dataset_path.is_file():
        candidates.append(dataset_path)
    else:
        for split_name in split_aliases.get(str(split or "").lower(), [split or "validation"]):
            candidates.extend(
                [
                    dataset_path / f"csqa_{split_name}.json",
                    dataset_path / f"{split_name}.json",
                    dataset_path / f"csqa_{split_name}.jsonl",
                    dataset_path / f"{split_name}.jsonl",
                ]
            )
        candidates.extend(
            [
                dataset_path / "csqa_flattened.json",
                dataset_path / "csqa_dev.json",
            ]
        )
    data_file = next((path for path in candidates if path.exists()), None)
    if data_file is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find CSQA split {split!r}. Tried: {tried}")

    rows = _load_json(data_file)
    if isinstance(rows, dict):
        rows = rows.get(split) or rows.get("data") or list(rows.values())
    rows = [row for row in rows if row]
    if limit is not None:
        rows = rows[:limit]

    cases = []
    labels = ["A", "B", "C", "D", "E"]
    for index, row in enumerate(rows):
        stem = str(row.get("stem") or row.get("question", {}).get("stem") or "").strip()
        question_concept = row.get("question_concept") or row.get("concept") or ""
        option_texts = []
        question_obj = row.get("question") if isinstance(row.get("question"), dict) else {}
        if isinstance(question_obj, dict) and question_obj.get("choices"):
            option_texts = [str(choice.get("text", "")).strip() for choice in question_obj.get("choices", [])]
        if not option_texts:
            option_texts = [str(row.get(f"choice_{label}", "")).strip() for label in labels]
        option_texts = [text for text in option_texts if text]
        if len(option_texts) != 5:
            continue

        option_lines = [f"{label}. {text}" for label, text in zip(labels, option_texts)]
        problem = (
            f"Question:\n{stem}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            "Answer with the option text only. Use one option text as the final answer."
        )
        answer_key = str(row.get("answerKey", "")).strip().upper()
        if answer_key not in labels:
            continue
        answer_text = option_texts[labels.index(answer_key)]
        cases.append(
            {
                "unique_id": f"csqa/{split or data_file.stem}/{row.get('id', index)}.json",
                "problem": problem,
                "answer": answer_text,
                "difficulty": None,
                "metadata": {
                    "answer_key": answer_key,
                    "question_concept": question_concept,
                    "source_file": str(data_file),
                },
            }
        )
    return cases


def _first_existing_split(dataset, preferred: str):
    if preferred and preferred in dataset:
        return dataset[preferred]
    for name in ("test", "validation", "dev", "train"):
        if name in dataset:
            return dataset[name]
    return next(iter(dataset.values()))


def _build_bamboogle_cases(split: str, limit: int | None) -> list[dict]:
    try:
        dataset = load_dataset("cmriat/bamboogle")
    except Exception:
        dataset = load_dataset("corag/multihopqa", "bamboogle")
    rows = _first_existing_split(dataset, split or "test")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    cases = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or row.get("query") or row.get("input") or "").strip()
        if not question:
            continue
        answers = row.get("golden_answers", None)
        if answers is None:
            answers = row.get("answers", row.get("answer", row.get("gold_answer", "")))
        if isinstance(answers, str):
            gold_answers = [answers]
        else:
            gold_answers = [str(item) for item in answers]
        problem = (
            f"Question:\n{question}\n\n"
            "Answer with the short answer phrase. Use the shortest entity, name, date, number, or phrase that answers the question as the final answer."
        )
        cases.append(
            {
                "unique_id": f"bamboogle/{split or 'test'}/{row.get('id', idx)}.json",
                "problem": problem,
                "answer": gold_answers,
                "difficulty": None,
                "metadata": {"source": "bamboogle"},
            }
        )
    return cases


def _render_hotpot_context(context: dict) -> str:
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    lines = []
    for title, sent_list in zip(titles, sentences):
        sent_text = " ".join(str(sent).strip() for sent in sent_list if str(sent).strip())
        if sent_text:
            lines.append(f"{str(title).strip()}: {sent_text}")
    return "\n".join(lines)


def _build_hotpotqa_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("hotpot_qa", "distractor")
    rows = _first_existing_split(dataset, split or "validation")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    cases = []
    for idx, row in enumerate(rows):
        context_text = _render_hotpot_context(row["context"])
        problem = (
            f"Question:\n{str(row['question']).strip()}\n\n"
            f"Context:\n{context_text}\n\n"
            "Answer with the short answer phrase. Use the shortest exact answer phrase as the final answer."
        )
        cases.append(
            {
                "unique_id": f"hotpotqa/{split or 'validation'}/{idx}-{row.get('id', idx)}.json",
                "problem": problem,
                "answer": str(row["answer"]).strip(),
                "difficulty": row.get("type"),
                "metadata": {
                    "level": row.get("level"),
                    "supporting_facts": row.get("supporting_facts"),
                },
            }
        )
    return cases


def _render_musique_paragraphs(paragraphs) -> str:
    lines = []
    for idx, para in enumerate(paragraphs or []):
        if isinstance(para, dict):
            title = para.get("title") or para.get("heading") or para.get("section_title") or para.get("title_text")
            text = para.get("paragraph_text") or para.get("text") or para.get("paragraph") or para.get("content") or ""
            support = para.get("is_supporting")
            prefix = f"{title}: " if title else f"P{idx + 1}: "
            marker = " [support]" if support else ""
            text = str(text).strip()
            if text:
                lines.append(f"{prefix}{text}{marker}")
        else:
            text = str(para).strip()
            if text:
                lines.append(f"P{idx + 1}: {text}")
    return "\n".join(lines)


def _build_musique_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("bdsaglam/musique")
    rows = _first_existing_split(dataset, split or "validation")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    cases = []
    for idx, row in enumerate(rows):
        paragraphs = row.get("paragraphs") or row.get("context") or []
        context_text = _render_musique_paragraphs(paragraphs)
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        answer_aliases = row.get("answer_aliases", [])
        if isinstance(answer_aliases, str):
            answer_aliases = [answer_aliases]
        gold = [str(item).strip() for item in (list(answers) + list(answer_aliases)) if str(item).strip()]
        if not gold:
            gold = [str(row.get("answer", "")).strip()]
        problem = (
            f"Question:\n{str(row['question']).strip()}\n\n"
            f"Context:\n{context_text}\n\n"
            "Answer with the short answer phrase. Use the shortest exact answer phrase as the final answer."
        )
        cases.append(
            {
                "unique_id": f"musique/{split or 'validation'}/{idx}-{row.get('id', row.get('query_id', idx))}.json",
                "problem": problem,
                "answer": gold,
                "difficulty": None,
                "metadata": {
                    "query_id": row.get("query_id"),
                    "context_doc_ids": row.get("context_doc_ids"),
                },
            }
        )
    return cases


def _build_proofwriter_cases(split: str, limit: int | None) -> list[dict]:
    dataset_path = DEFAULT_DATASET_ROOT / "proofwriter"
    dataset = load_from_disk(str(dataset_path))[split or "validation"]
    rows = dataset.select(range(min(limit, len(dataset)))) if limit is not None else dataset
    label_map = {"True": "1", "False": "2", "Unknown": "3"}
    cases = []
    for idx, row in enumerate(rows):
        problem = (
            f"Premises:\n{str(row['theory']).strip()}\n\n"
            f"Question:\n{str(row['question']).strip()}\n\n"
            "Answer 1 for true, 2 for false, and 3 for unknown. "
            "Use only 1, 2, or 3 as the final answer."
        )
        answer = label_map.get(str(row["answer"]).strip(), str(row["answer"]).strip())
        cases.append(
            {
                "unique_id": f"proofwriter/{split or 'validation'}/{idx}-{row.get('id', row.get('example_id', idx))}.json",
                "problem": problem,
                "answer": answer,
                "difficulty": row.get("config"),
                "metadata": {
                    "example_id": row.get("example_id"),
                    "n_fact": row.get("NFact"),
                    "n_rule": row.get("NRule"),
                },
            }
        )
    return cases


def _build_folio_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("tasksource/folio")
    rows = _first_existing_split(dataset, split or "validation")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    label_map = {"True": "1", "False": "2", "Uncertain": "3"}
    cases = []
    for idx, row in enumerate(rows):
        problem = (
            f"Premises:\n{str(row['premises']).strip()}\n\n"
            f"Conclusion:\n{str(row['conclusion']).strip()}\n\n"
            "Answer 1 for true, 2 for false, and 3 for uncertain. "
            "Use only 1, 2, or 3 as the final answer."
        )
        answer = label_map.get(str(row["label"]).strip(), str(row["label"]).strip())
        cases.append(
            {
                "unique_id": f"folio/{split or 'validation'}/{idx}-{row.get('example_id', idx)}.json",
                "problem": problem,
                "answer": answer,
                "difficulty": None,
                "metadata": {
                    "story_id": row.get("story_id"),
                    "example_id": row.get("example_id"),
                },
            }
        )
    return cases


def _build_boolq_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("google/boolq")
    rows = _first_existing_split(dataset, split or "validation")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    cases = []
    for idx, row in enumerate(rows):
        problem = (
            f"Passage:\n{str(row['passage']).strip()}\n\n"
            f"Question:\n{str(row['question']).strip()}\n\n"
            "Answer yes or no. Use only yes or no as the final answer."
        )
        cases.append(
            {
                "unique_id": f"boolq/{split or 'validation'}/{row.get('idx', idx)}.json",
                "problem": problem,
                "answer": "yes" if bool(row["answer"]) else "no",
                "difficulty": None,
            }
        )
    return cases


def _build_medqa_cases(split: str, limit: int | None) -> list[dict]:
    dataset = load_dataset("truehealth/medqa")
    rows = _first_existing_split(dataset, split or "test")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    cases = []
    for idx, row in enumerate(rows):
        options = dict(row["options"])
        option_lines = [f"{letter}. {str(options[letter]).strip()}" for letter in sorted(options)]
        problem = (
            f"Question:\n{str(row['question']).strip()}\n\n"
            "Options:\n"
            + "\n".join(option_lines)
            + "\n\n"
            "Answer with the option text only. Use one option text as the final answer."
        )
        cases.append(
            {
                "unique_id": f"medqa/{split or 'test'}/{row.get('id', idx)}.json",
                "problem": problem,
                "answer": str(row["answer"]).strip(),
                "difficulty": row.get("meta_info"),
                "metadata": {
                    "answer_idx": row.get("answer_idx"),
                    "options": options,
                },
            }
        )
    return cases


def _build_cases(args: argparse.Namespace) -> list[dict]:
    dataset_name = str(args.dataset).strip().lower()
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    if dataset_name == "hotpotqa":
        return _build_hotpotqa_cases(args.split, args.limit)
    if dataset_name == "musique":
        return _build_musique_cases(args.split, args.limit)
    if dataset_name == "proofwriter":
        return _build_proofwriter_cases(args.split, args.limit)
    if dataset_name == "folio":
        return _build_folio_cases(args.split, args.limit)
    if dataset_name == "strategyqa":
        path = dataset_path or (DEFAULT_DATASET_ROOT / "strategyqa")
        return _build_strategyqa_cases(path, args.split, args.limit)
    if dataset_name == "prontoqa":
        path = dataset_path or (DEFAULT_DATASET_ROOT / "prontoqa")
        return _build_prontoqa_cases(path, args.split, args.limit)
    if dataset_name == "anli":
        return _build_anli_cases(args.split, args.limit)
    if dataset_name == "mmlu_pro":
        path = dataset_path or (DEFAULT_DATASET_ROOT / "mmlu_pro")
        return _build_mmlu_pro_cases(path, args.split, args.limit, args.sample_ratio, args.sample_seed)
    if dataset_name == "logiqa":
        path = dataset_path or (DEFAULT_DATASET_ROOT / "logiqa")
        return _build_logiqa_cases(path, args.split, args.limit)
    if dataset_name == "csqa":
        path = dataset_path or Path("/home/zihan/silver/DRF/CSQA")
        return _build_csqa_cases(path, args.split, args.limit)
    if dataset_name == "bamboogle":
        return _build_bamboogle_cases(args.split, args.limit)
    if dataset_name == "boolq":
        return _build_boolq_cases(args.split, args.limit)
    if dataset_name == "medqa":
        return _build_medqa_cases(args.split, args.limit)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _agent_key(dataset_name: str, model_name: str, agent_idx: int) -> str:
    return f"{dataset_name}__{model_name}__None__Agent{agent_idx}"


def _generate_round0_history(
    *,
    args: argparse.Namespace,
    cases: list[dict],
    history_path: Path,
) -> None:
    runtime = load_runtime(_build_runtime_args(args))
    try:
        prompt_records = []
        case_prompt_map = {}
        sample_index_map = {}
        model_name = Path(args.model).name if "/" in args.model else args.model
        dataset_name = str(args.dataset).strip().lower()
        for sample_index, case in enumerate(cases):
            prompt = _round0_prompt(case["problem"], getattr(args, "round0_prompt_style", "bare"))
            case_prompt_map[case["unique_id"]] = prompt
            sample_index_map[case["unique_id"]] = sample_index
            for agent_idx in range(1, 4):
                agent_id = _agent_key(dataset_name, model_name, agent_idx)
                prompt_records.append((f"{case['unique_id']}::{agent_id}", prompt))

        outputs = []
        for start in range(0, len(prompt_records), args.prompt_batch_size):
            outputs.extend(run_prompts(runtime, prompt_records[start : start + args.prompt_batch_size]))

        grouped: dict[str, dict[str, str]] = {}
        for (alias, _), output in zip(prompt_records, outputs):
            case_id, agent_id = alias.split("::", 1)
            grouped.setdefault(case_id, {})[agent_id] = output

        history_rows = []
        for case in cases:
            case_id = case["unique_id"]
            prompt = case_prompt_map[case_id]
            initial_responses = grouped.get(case_id, {})
            history_rows.append(
                {
                    "record_type": "debate_trace_v2",
                    "unique_id": case_id,
                    "sample_index": sample_index_map[case_id],
                    "mode": {
                        "mode_key": "ORIGINMAD",
                        "solver": "debate",
                        "debate_scheme": "legacy",
                        "num_agents": 3,
                        "debate_rounds": args.origin_rounds,
                        "model": model_name,
                        "data": dataset_name,
                        "data_size": len(cases),
                        "shard_id": 0,
                        "num_shards": 1,
                    },
                    "task": {
                        "unique_id": case_id,
                        "question": prompt,
                        "gold_answer": case["answer"],
                    },
                    "summary": {
                        "num_rounds_recorded": 1,
                        "final_round_index": 0,
                    },
                    "rounds": [
                        {
                            "round_index": 0,
                            "inputs": {
                                "initial_responses": initial_responses,
                            },
                        }
                    ],
                }
            )
        _write_jsonl(history_path, history_rows)
    finally:
        close_runtime(runtime)


def _wait_for_pid(pid: int) -> int:
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return status


def _run_compare_part(
    *,
    gpu: str,
    cases_path: Path,
    history_path: Path,
    output_path: Path,
    log_path: Path,
    model: str,
    model_dir: str,
    batch_size: int,
    origin_rounds: int,
    graph_rounds: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    prompt_profile: str,
    disable_target_check_landing: bool,
) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_compare_origin_vs_nl_graph_batch.py"),
        "--cases_path",
        str(cases_path),
        "--history_path",
        str(history_path),
        "--output_path",
        str(output_path),
        "--log_path",
        str(log_path),
        "--model",
        model,
        "--model_dir",
        model_dir,
        "--origin_rounds",
        str(origin_rounds),
        "--graph_rounds",
        str(graph_rounds),
        "--case_batch_size",
        str(batch_size),
        "--max_new_tokens",
        str(max_new_tokens),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--prompt_profile",
        prompt_profile,
    ]
    if disable_target_check_landing:
        cmd.append("--disable_target_check_landing")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    console_path = log_path.with_suffix(".console.log")
    with console_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"\nLauncher command: {' '.join(cmd)}\nCUDA_VISIBLE_DEVICES={gpu}\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc.pid


def _compute_round0_case_view(history_rows: list[dict], case_lookup: dict[str, dict]) -> tuple[list[dict], dict]:
    per_case = []
    for row in history_rows:
        case_id = row.get("unique_id") or row.get("task", {}).get("unique_id")
        if case_id not in case_lookup:
            continue
        responses = row.get("rounds", [{}])[0].get("inputs", {}).get("initial_responses", {})
        question = case_lookup[case_id]["problem"]
        answer_map = {_agent: _extract_label_answer(text, question) for _agent, text in responses.items()}
        normalized_answers = {f"A{idx + 1}": answer_map.get(agent_id) for idx, agent_id in enumerate(sorted(answer_map))}
        chosen = _majority_answer(normalized_answers)
        gold = case_lookup[case_id]["answer"]
        per_case.append(
            {
                "unique_id": case_id,
                "initial_answers": normalized_answers,
                "round0_chosen_answer": chosen,
                "gold_answer": gold,
                "round0_is_correct": _is_correct(chosen, gold, question),
                "is_unanimous": _all_answers_unanimous(normalized_answers),
            }
        )
    num_cases = len(per_case)
    round0_num_correct = sum(1 for item in per_case if item["round0_is_correct"])
    nonunanim_count = sum(1 for item in per_case if not item["is_unanimous"])
    return per_case, {
        "num_cases": num_cases,
        "round0_num_correct": round0_num_correct,
        "round0_accuracy": (round0_num_correct / num_cases) if num_cases else 0.0,
        "nonunanim_count": nonunanim_count,
        "nonunanim_fraction": (nonunanim_count / num_cases) if num_cases else 0.0,
    }


def _filter_history_rows(history_rows: list[dict], case_ids: set[str]) -> list[dict]:
    return [
        row
        for row in history_rows
        if (row.get("unique_id") or row.get("task", {}).get("unique_id")) in case_ids
    ]


def _round0_subset_summary(results: list[dict]) -> dict:
    num_cases = len(results)
    round0_correct = 0
    round0_to_graph_wrong = 0
    round0_wrong_to_graph_right = 0
    graph_only = 0
    origin_only = 0
    for item in results:
        gold = item["gold_answer"]
        question = item.get("problem", "")
        round0_answer = _majority_answer(item.get("initial_answers", {}))
        origin_answer = item.get("origin", {}).get("chosen_answer")
        graph_answer = item.get("graph", {}).get("chosen_answer")
        round0_ok = _is_correct(round0_answer, gold, question)
        origin_ok = _is_correct(origin_answer, gold, question)
        graph_ok = _is_correct(graph_answer, gold, question)
        if round0_ok:
            round0_correct += 1
        if round0_ok and not graph_ok:
            round0_to_graph_wrong += 1
        if (not round0_ok) and graph_ok:
            round0_wrong_to_graph_right += 1
        if graph_ok and not origin_ok:
            graph_only += 1
        if origin_ok and not graph_ok:
            origin_only += 1
    return {
        "num_cases": num_cases,
        "round0_num_correct": round0_correct,
        "round0_accuracy": (round0_correct / num_cases) if num_cases else 0.0,
        "round0_to_graph_wrong": round0_to_graph_wrong,
        "round0_wrong_to_graph_right": round0_wrong_to_graph_right,
        "graph_only": graph_only,
        "origin_only": origin_only,
    }


def _merge_outputs(
    *,
    dataset: str,
    split: str,
    args: argparse.Namespace,
    subset_cases: list[dict],
    part_outputs: list[Path],
    work_dir: Path,
    round0_full_summary: dict,
    round0_subset_prep_summary: dict,
) -> dict:
    results_by_id = {}
    part_payloads = []
    for path in part_outputs:
        payload = _load_json(path)
        part_payloads.append(payload)
        for item in payload.get("results", []):
            results_by_id[item["unique_id"]] = item
    results = [results_by_id[case["unique_id"]] for case in subset_cases]
    origin_num_correct, origin_accuracy = _build_summary(results, "origin")
    graph_num_correct, graph_accuracy = _build_summary(results, "graph")
    origin_round_metrics = _build_mode_round_metrics(results, "origin", args.origin_rounds)
    graph_round_metrics = _build_mode_round_metrics(results, "graph", args.graph_rounds)
    round0_subset_summary = _round0_subset_summary(results)
    return {
        "dataset": dataset,
        "split": split,
        "cases_path": str(work_dir / "nonunanim_cases.json"),
        "history_path": str(work_dir / "nonunanim.history.jsonl"),
        "num_cases": len(results),
        "origin_rounds": args.origin_rounds,
        "graph_rounds": args.graph_rounds,
        "case_batch_size": args.case_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_profile": args.prompt_profile,
        "round0_prompt_style": args.round0_prompt_style,
        "round0_full_summary": round0_full_summary,
        "round0_subset_prep_summary": round0_subset_prep_summary,
        "round0_subset_summary": round0_subset_summary,
        "origin_num_correct": origin_num_correct,
        "origin_accuracy": origin_accuracy,
        "graph_num_correct": graph_num_correct,
        "graph_accuracy": graph_accuracy,
        "origin_round_metrics": origin_round_metrics,
        "graph_round_metrics": graph_round_metrics,
        "part_outputs": [str(path) for path in part_outputs],
        "part_summaries": [
            {
                "path": str(path),
                "num_cases": payload.get("num_cases"),
                "origin_num_correct": payload.get("origin_num_correct"),
                "origin_accuracy": payload.get("origin_accuracy"),
                "graph_num_correct": payload.get("graph_num_correct"),
                "graph_accuracy": payload.get("graph_accuracy"),
            }
            for path, payload in zip(part_outputs, part_payloads)
        ],
        "results": results,
    }


def _render_summary_log(payload: dict) -> str:
    round0_full = payload.get("round0_full_summary", {})
    round0_subset = payload.get("round0_subset_summary", {})
    lines = [
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {payload.get('dataset')}",
        f"Split: {payload.get('split')}",
        f"Prompt profile: {payload.get('prompt_profile')}",
        f"Round0 prompt style: {payload.get('round0_prompt_style')}",
        f"Round0 full accuracy: {round0_full.get('round0_num_correct')}/{round0_full.get('num_cases')} = {round0_full.get('round0_accuracy', 0.0):.4f}",
        f"Round0 non-unanimous cases: {round0_full.get('nonunanim_count')}/{round0_full.get('num_cases')} = {round0_full.get('nonunanim_fraction', 0.0):.4f}",
        f"Subset round0 accuracy: {round0_subset.get('round0_num_correct')}/{round0_subset.get('num_cases')} = {round0_subset.get('round0_accuracy', 0.0):.4f}",
        f"Origin accuracy: {payload.get('origin_num_correct')}/{payload.get('num_cases')} = {payload.get('origin_accuracy', 0.0):.4f}",
        f"Graph accuracy: {payload.get('graph_num_correct')}/{payload.get('num_cases')} = {payload.get('graph_accuracy', 0.0):.4f}",
        f"Round0 correct -> Graph wrong: {round0_subset.get('round0_to_graph_wrong')}",
        f"Round0 wrong -> Graph right: {round0_subset.get('round0_wrong_to_graph_right')}",
        f"Graph only: {round0_subset.get('graph_only')}",
        f"Origin only: {round0_subset.get('origin_only')}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["strategyqa", "prontoqa", "anli", "mmlu_pro", "logiqa", "csqa", "bamboogle", "boolq", "medqa", "hotpotqa", "musique", "proofwriter", "folio"])
    parser.add_argument("--split", default="")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_dir", default="/data/yyr/model")
    parser.add_argument("--origin_rounds", type=int, default=3)
    parser.add_argument("--graph_rounds", type=int, default=3)
    parser.add_argument("--case_batch_size", type=int, default=100)
    parser.add_argument("--prompt_batch_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--gpus", default="1,3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_target_check_landing", action="store_true", default=True)
    parser.add_argument("--prompt_profile", default="universal")
    parser.add_argument("--round0_prompt_style", default="claim_atomic")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--sample_seed", type=int, default=7)
    parser.add_argument("--compare_all_cases", action="store_true")
    args = parser.parse_args()

    if not args.split:
        if args.dataset == "strategyqa":
            args.split = "train"
        elif args.dataset == "prontoqa":
            args.split = "test"
        elif args.dataset == "anli":
            args.split = "dev_r3"
        elif args.dataset == "mmlu_pro":
            args.split = "test"
        elif args.dataset == "logiqa":
            args.split = "test"
        elif args.dataset == "csqa":
            args.split = "validation"
        elif args.dataset == "bamboogle":
            args.split = "test"
        elif args.dataset == "boolq":
            args.split = "validation"
        elif args.dataset == "medqa":
            args.split = "test"
        elif args.dataset in {"hotpotqa", "musique", "proofwriter", "folio"}:
            args.split = "validation"
    if args.limit is None and args.dataset in {"boolq"}:
        args.limit = 1000
    if args.limit is None and args.dataset in {"medqa", "hotpotqa", "musique", "proofwriter", "folio"}:
        args.limit = 1000
    if args.prompt_profile == "universal":
        if args.dataset == "hotpotqa":
            args.prompt_profile = "hotpotqa_relation_v4"
        elif args.dataset == "musique":
            args.prompt_profile = "musique_relation_v3"
        elif args.dataset == "proofwriter":
            args.prompt_profile = "proofwriter_relation_v3"
        elif args.dataset == "folio":
            args.prompt_profile = "folio_relation_v3"
    if args.round0_prompt_style == "claim_atomic" and args.dataset in {"proofwriter", "folio"}:
        args.round0_prompt_style = "logic_atomic"

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    cases = _build_cases(args)
    case_lookup = {case["unique_id"]: case for case in cases}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"General compare run started: {datetime.now().isoformat(timespec='seconds')}",
                f"Dataset: {args.dataset}",
                f"Split: {args.split}",
                f"Output path: {args.output_path}",
                f"Log path: {args.log_path}",
                f"Work dir: {args.work_dir}",
                f"Origin rounds: {args.origin_rounds}",
                f"Graph rounds: {args.graph_rounds}",
                f"Case batch size: {args.case_batch_size}",
                f"Prompt batch size: {args.prompt_batch_size}",
                f"Max new tokens: {args.max_new_tokens}",
                f"Prompt profile: {args.prompt_profile}",
                f"Round0 prompt style: {args.round0_prompt_style}",
                f"GPUs: {args.gpus}",
                f"Dataset size before subset: {len(cases)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpus) not in {1, 2}:
        raise RuntimeError("--gpus must contain one GPU id or two comma-separated GPU ids.")
    if len(gpus) == 1:
        gpu = gpus[0]
        full_case_path = work_dir / "full.json"
        full_history_path = work_dir / "full.history.jsonl"
        _write_json(full_case_path, cases)

        prep_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_general_label_reasoning_compare_split.py"),
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--dataset_path",
            args.dataset_path,
            "--work_dir",
            str(work_dir),
            "--output_path",
            str(output_path),
            "--log_path",
            str(log_path),
            "--model",
            args.model,
            "--model_dir",
            args.model_dir,
            "--origin_rounds",
            str(args.origin_rounds),
            "--graph_rounds",
            str(args.graph_rounds),
            "--case_batch_size",
            str(args.case_batch_size),
            "--prompt_batch_size",
            str(args.prompt_batch_size),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--gpu_memory_utilization",
            str(args.gpu_memory_utilization),
            "--gpus",
            gpu,
            "--prompt_profile",
            args.prompt_profile,
            "--round0_prompt_style",
            args.round0_prompt_style,
            "--sample_ratio",
            str(args.sample_ratio),
            "--sample_seed",
            str(args.sample_seed),
            "--prepare_only",
            "--history_output_path",
            str(full_history_path),
            "--cases_input_path",
            str(full_case_path),
        ]
        if args.limit is not None:
            prep_cmd.extend(["--limit", str(args.limit)])
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        prep_log = full_history_path.with_suffix(".prepare.console.log")
        with prep_log.open("w", encoding="utf-8") as log_file:
            log_file.write(f"\nLauncher command: {' '.join(prep_cmd)}\nCUDA_VISIBLE_DEVICES={gpu}\n\n")
            log_file.flush()
            proc = subprocess.Popen(prep_cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        code = _wait_for_pid(proc.pid)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Round0 prep pid={proc.pid} exited with code {code}; log={prep_log}\n")
        if code != 0:
            raise RuntimeError(f"Round0 prep failed for {prep_log} with code {code}")

        full_history_rows = _load_jsonl(full_history_path)
        round0_cases, round0_full_summary = _compute_round0_case_view(full_history_rows, case_lookup)
        _write_json(work_dir / "round0_full_case_view.json", round0_cases)

        nonunanim_ids = [item["unique_id"] for item in round0_cases if not item["is_unanimous"]]
        subset_cases = [case_lookup[case_id] for case_id in nonunanim_ids]
        subset_case_lookup = {case["unique_id"]: case for case in subset_cases}
        subset_history_rows = _filter_history_rows(full_history_rows, set(nonunanim_ids))
        _, round0_subset_prep_summary = _compute_round0_case_view(subset_history_rows, subset_case_lookup)

        if args.compare_all_cases:
            compare_cases = cases
            compare_history_rows = full_history_rows
            round0_compare_summary = round0_full_summary
            compare_cases_path = work_dir / "compare_all_cases.json"
            compare_history_path = work_dir / "compare_all.history.jsonl"
        else:
            compare_cases = subset_cases
            compare_history_rows = subset_history_rows
            round0_compare_summary = round0_subset_prep_summary
            compare_cases_path = work_dir / "nonunanim_cases.json"
            compare_history_path = work_dir / "nonunanim.history.jsonl"
        _write_json(compare_cases_path, compare_cases)
        _write_jsonl(compare_history_path, compare_history_rows)

        part_output = work_dir / "compare.out.json"
        part_log = work_dir / "compare.run.log"
        pid = _run_compare_part(
            gpu=gpu,
            cases_path=compare_cases_path,
            history_path=compare_history_path,
            output_path=part_output,
            log_path=part_log,
            model=args.model,
            model_dir=args.model_dir,
            batch_size=args.case_batch_size,
            origin_rounds=args.origin_rounds,
            graph_rounds=args.graph_rounds,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prompt_profile=args.prompt_profile,
            disable_target_check_landing=args.disable_target_check_landing,
        )
        code = _wait_for_pid(pid)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Compare pid={pid} exited with code {code}; log={part_log}\n")
        if code != 0:
            raise RuntimeError(f"Compare run failed for {part_log} with code {code}")

        payload = _merge_outputs(
            dataset=args.dataset,
            split=args.split,
            args=args,
            subset_cases=compare_cases,
            part_outputs=[part_output],
            work_dir=work_dir,
            round0_full_summary=round0_full_summary,
            round0_subset_prep_summary=round0_compare_summary,
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n")
            log_file.write(_render_summary_log(payload))
            log_file.write("\n")
        return

    assigned_gpus = gpus
    run_parallel = True

    split_at = (len(cases) + 1) // 2
    full_case_parts = [cases[:split_at], cases[split_at:]]
    full_case_paths = [work_dir / "full_part1.json", work_dir / "full_part2.json"]
    full_history_paths = [work_dir / "full_part1.history.jsonl", work_dir / "full_part2.history.jsonl"]
    for path, payload in zip(full_case_paths, full_case_parts):
        _write_json(path, payload)

    prep_pids = []
    for gpu, cases_path, history_path in zip(assigned_gpus, full_case_paths, full_history_paths):
        prep_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_general_label_reasoning_compare_split.py"),
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--dataset_path",
            args.dataset_path,
            "--work_dir",
            str(work_dir),
            "--output_path",
            str(output_path),
            "--log_path",
            str(log_path),
            "--model",
            args.model,
            "--model_dir",
            args.model_dir,
            "--origin_rounds",
            str(args.origin_rounds),
            "--graph_rounds",
            str(args.graph_rounds),
            "--case_batch_size",
            str(args.case_batch_size),
            "--prompt_batch_size",
            str(args.prompt_batch_size),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--gpu_memory_utilization",
            str(args.gpu_memory_utilization),
            "--gpus",
            gpu,
            "--prompt_profile",
            args.prompt_profile,
            "--round0_prompt_style",
            args.round0_prompt_style,
            "--sample_ratio",
            str(args.sample_ratio),
            "--sample_seed",
            str(args.sample_seed),
            "--prepare_only",
            "--history_output_path",
            str(history_path),
            "--cases_input_path",
            str(cases_path),
        ]
        if args.limit is not None:
            prep_cmd.extend(["--limit", str(args.limit)])
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        prep_log = history_path.with_suffix(".prepare.console.log")
        with prep_log.open("w", encoding="utf-8") as log_file:
            log_file.write(f"\nLauncher command: {' '.join(prep_cmd)}\nCUDA_VISIBLE_DEVICES={gpu}\n\n")
            log_file.flush()
            proc = subprocess.Popen(prep_cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        if run_parallel:
            prep_pids.append((proc.pid, prep_log))
        else:
            code = _wait_for_pid(proc.pid)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"Round0 prep pid={proc.pid} exited with code {code}; log={prep_log}\n")
            if code != 0:
                raise RuntimeError(f"Round0 prep failed for {prep_log} with code {code}")

    for pid, prep_log in prep_pids:
        code = _wait_for_pid(pid)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Round0 prep pid={pid} exited with code {code}; log={prep_log}\n")
        if code != 0:
            raise RuntimeError(f"Round0 prep failed for {prep_log} with code {code}")

    full_history_rows = []
    for history_path in full_history_paths:
        full_history_rows.extend(_load_jsonl(history_path))
    _write_jsonl(work_dir / "full.history.jsonl", full_history_rows)

    round0_cases, round0_full_summary = _compute_round0_case_view(full_history_rows, case_lookup)
    _write_json(work_dir / "round0_full_case_view.json", round0_cases)

    nonunanim_ids = [item["unique_id"] for item in round0_cases if not item["is_unanimous"]]
    subset_cases = [case_lookup[case_id] for case_id in nonunanim_ids]
    subset_case_lookup = {case["unique_id"]: case for case in subset_cases}
    _write_json(work_dir / "nonunanim_cases.json", subset_cases)

    subset_history_rows = _filter_history_rows(full_history_rows, set(nonunanim_ids))
    _write_jsonl(work_dir / "nonunanim.history.jsonl", subset_history_rows)
    _, round0_subset_prep_summary = _compute_round0_case_view(subset_history_rows, subset_case_lookup)

    subset_split_at = (len(subset_cases) + 1) // 2
    subset_case_parts = [subset_cases[:subset_split_at], subset_cases[subset_split_at:]]
    subset_case_paths = [work_dir / "subset_part1.json", work_dir / "subset_part2.json"]
    subset_history_paths = [work_dir / "subset_part1.history.jsonl", work_dir / "subset_part2.history.jsonl"]
    part_outputs = [work_dir / "subset_part1.out.json", work_dir / "subset_part2.out.json"]
    part_logs = [work_dir / "subset_part1.run.log", work_dir / "subset_part2.run.log"]

    for path, payload in zip(subset_case_paths, subset_case_parts):
        _write_json(path, payload)

    for path, payload in zip(subset_history_paths, subset_case_parts):
        subset_ids = {case["unique_id"] for case in payload}
        _write_jsonl(path, _filter_history_rows(subset_history_rows, subset_ids))

    run_pids = []
    for gpu, cases_path, history_path, part_output, part_log in zip(assigned_gpus, subset_case_paths, subset_history_paths, part_outputs, part_logs):
        pid = _run_compare_part(
            gpu=gpu,
            cases_path=cases_path,
            history_path=history_path,
            output_path=part_output,
            log_path=part_log,
            model=args.model,
            model_dir=args.model_dir,
            batch_size=args.case_batch_size,
            origin_rounds=args.origin_rounds,
            graph_rounds=args.graph_rounds,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prompt_profile=args.prompt_profile,
            disable_target_check_landing=args.disable_target_check_landing,
        )
        if run_parallel:
            run_pids.append((pid, part_log))
        else:
            code = _wait_for_pid(pid)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"Compare pid={pid} exited with code {code}; log={part_log}\n")
            if code != 0:
                raise RuntimeError(f"Compare run failed for {part_log} with code {code}")

    for pid, part_log in run_pids:
        code = _wait_for_pid(pid)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Compare pid={pid} exited with code {code}; log={part_log}\n")
        if code != 0:
            raise RuntimeError(f"Compare run failed for {part_log} with code {code}")

    payload = _merge_outputs(
        dataset=args.dataset,
        split=args.split,
        args=args,
        subset_cases=subset_cases,
        part_outputs=part_outputs,
        work_dir=work_dir,
        round0_full_summary=round0_full_summary,
        round0_subset_prep_summary=round0_subset_prep_summary,
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n")
        log_file.write(_render_summary_log(payload))
        log_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--history_output_path", type=str, default="")
    parser.add_argument("--cases_input_path", type=str, default="")
    known_args, _ = parser.parse_known_args()
    if known_args.prepare_only:
        full_parser = argparse.ArgumentParser()
        full_parser.add_argument("--dataset", required=True)
        full_parser.add_argument("--split", default="")
        full_parser.add_argument("--dataset_path", default="")
        full_parser.add_argument("--work_dir", required=True)
        full_parser.add_argument("--output_path", required=True)
        full_parser.add_argument("--log_path", required=True)
        full_parser.add_argument("--model", default=DEFAULT_MODEL)
        full_parser.add_argument("--model_dir", default="/data/yyr/model")
        full_parser.add_argument("--origin_rounds", type=int, default=3)
        full_parser.add_argument("--graph_rounds", type=int, default=3)
        full_parser.add_argument("--case_batch_size", type=int, default=100)
        full_parser.add_argument("--prompt_batch_size", type=int, default=128)
        full_parser.add_argument("--max_new_tokens", type=int, default=512)
        full_parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
        full_parser.add_argument("--gpus", default="0")
        full_parser.add_argument("--limit", type=int, default=None)
        full_parser.add_argument("--prompt_profile", default="universal")
        full_parser.add_argument("--round0_prompt_style", default="claim_atomic")
        full_parser.add_argument("--sample_ratio", type=float, default=1.0)
        full_parser.add_argument("--sample_seed", type=int, default=7)
        full_parser.add_argument("--prepare_only", action="store_true")
        full_parser.add_argument("--history_output_path", required=True)
        full_parser.add_argument("--cases_input_path", required=True)
        prep_args = full_parser.parse_args()
        cases = _load_json(Path(prep_args.cases_input_path))
        _generate_round0_history(
            args=prep_args,
            cases=cases,
            history_path=Path(prep_args.history_output_path),
        )
    else:
        main()
