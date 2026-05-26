from __future__ import annotations

import json
import random
import re
from typing import Any

from .label_answer import extract_label_answer, question_uses_label_answers
from .local_evaluator import extract_math_answer, normalize_math_answer
from .local_runtime_backend import wrap_message


def _get_agent_answer(args, response, question: str = ""):
    del args
    if question_uses_label_answers(question):
        return extract_label_answer(response, question)
    return normalize_math_answer(extract_math_answer(response))


def _get_answer_map(args, responses, question: str = ""):
    return {agent: _get_agent_answer(args, resp, question=question) for agent, resp in responses.items()}


def _get_peer_map(args, agents, prev_answers):
    def differs(a, b):
        if a is None and b is None:
            return False
        return a != b

    peer_map = {}
    for i, agent in enumerate(agents):
        if len(agents) <= 1:
            peer_map[agent] = None
            continue
        if getattr(args, "centralized", False):
            peers = agents[:i] + agents[i + 1 :] if i == 0 else [agents[0]]
        elif getattr(args, "sparse", False):
            peers = [agents[(i - 1) % len(agents)], agents[(i + 1) % len(agents)]]
        else:
            peers = agents[:i] + agents[i + 1 :]
        diff_peers = [p for p in peers if differs(prev_answers.get(p), prev_answers.get(agent))]
        if getattr(args, "limit_one_peer", False):
            peer_map[agent] = random.choice(diff_peers) if diff_peers else None
        else:
            peer_map[agent] = diff_peers
    return peer_map


def _extract_answer(text: str, question: str = ""):
    if question_uses_label_answers(question):
        return extract_label_answer(text, question)
    return normalize_math_answer(extract_math_answer(text))


def strip_after_extracted_answer(text: str, question: str = "") -> str:
    if not text:
        return text
    answer = _extract_answer(text, question=question)
    if not answer:
        return text
    idx = text.find(str(answer))
    if idx == -1:
        return text
    cut_idx = max(text.rfind("\n", 0, idx), text.rfind(".", 0, idx), text.rfind(",", 0, idx))
    return text[:idx] if cut_idx == -1 else text[: cut_idx + 1]


def _build_critique_messages(args, sample, responses, peer_map, personas=None):
    new_message = {}
    for agent in responses:
        peer = peer_map.get(agent)
        if peer:
            if isinstance(peer, list):
                msg = "Compare the other agents' responses with yours. Decide whether your reasoning has an error or theirs does."
                msg += "\n\nQuestion:\n" + sample
                for other_agent in peer:
                    peer_resp = responses[other_agent]
                    if getattr(args, "finalfilter", False):
                        peer_resp = strip_after_extracted_answer(peer_resp)
                    msg += f"\n\nPeer: {other_agent}\nResponse:\n{peer_resp}\n"
                msg += f"\n\nYour response:\n{responses[agent]}\n"
                msg += "\n\nYour reason must cite a specific step or equation from either response (quote it or refer to the step number)."
                msg += "\n\nDo not mention any final answer; only identify the erroneous step(s)."
                msg += "\n\nIf you cannot cite a concrete error, choose INCONCLUSIVE and recommend KEEP."
                msg += "\n\nOutput format (repeat for each peer):\nPeer: <peer_id>\nDiagnosis: <OTHER_WRONG or SELF_WRONG or INCONCLUSIVE>\nReason: <brief error analysis with a cited step/equation>"
            else:
                peer_resp = responses[peer]
                if getattr(args, "finalfilter", False):
                    peer_resp = strip_after_extracted_answer(peer_resp)
                msg = "Compare the other agent's response with yours. Decide whether your reasoning has an error or theirs does."
                msg += "\n\nQuestion:\n" + sample
                msg += f"\n\nOther agent's response:\n{peer_resp}\n"
                msg += f"\n\nYour response:\n{responses[agent]}\n"
                msg += "\n\nYour reason must cite a specific step or equation from either response."
                msg += "\n\nDo not mention any final answer; only identify the erroneous step."
                msg += "\n\nOutput format:\nDiagnosis: <OTHER_WRONG or SELF_WRONG or INCONCLUSIVE>\nReason: <brief error analysis with a cited step/equation>"
        else:
            msg = f"This was your most recent opinion:\n{responses[agent]}\n"
            msg += "\n\nQuestion:\n" + sample
            msg += "\n\nBriefly check it for mistakes. Output format:\nDiagnosis: SELF_WRONG or SELF_CORRECT\nReason: <brief error analysis>"
        new_message[agent] = wrap_message(personas, agent, msg, getattr(args, "system_prompt_text", None))
    return new_message


def _build_revision_messages(
    args,
    sample,
    responses,
    critique_responses,
    peer_map,
    verifier_reports=None,
    personas=None,
    suffix=None,
    extract_json_obj_fn=None,
    is_no_answer_repair_report_fn=None,
    mindmap_shared_graph=None,
):
    del verifier_reports, extract_json_obj_fn, is_no_answer_repair_report_fn, mindmap_shared_graph
    new_message = {}
    agents_to_update = []
    critics_for = {}
    for critic, target in peer_map.items():
        if target is None:
            continue
        if isinstance(target, list):
            for target_agent in target:
                critics_for.setdefault(target_agent, []).append(critic)
        else:
            critics_for.setdefault(target, []).append(critic)

    for agent in responses:
        critics = critics_for.get(agent, [])
        if getattr(args, "limit_one_peer", False) and critics:
            critics = [random.choice(critics)]
        if not critics:
            continue
        msg = "One or more agents reviewed the response. Consider all feedback and solutions."
        for critic in critics:
            msg += f"\n\nCritic: {critic}\nBug Report:\n{critique_responses[critic]}\n"
        msg += f"\n\nThis was the most recent opinion to revise:\n{responses[agent]}\n"
        msg += "\n\nBefore deciding, independently verify the key claim(s) with a concrete equation, substitution, counterexample, or local step check."
        msg += "\nDo not justify any change by saying other agents agree or disagree."
        msg += "\n\nWhen revising, prefer the smallest corrected suffix that starts at the incorrect claim."
        msg += "\nKeep unaffected earlier reasoning verbatim when possible."
        msg += "\nIf you cannot verify a concrete error, choose KEEP and leave revised_response empty."
        msg += (
            '\n\nReturn STRICT JSON only:\n'
            '{"decision":"REVISE|KEEP","adopted_from":["critic_id_or_agent_alias"],'
            '"verification":"specific local check with no peer references","revised_subgraph_nodes":[],'
            '"incorrect_claim":"the concrete local claim you found to be wrong; empty when decision is KEEP",'
            '"corrected_claim":"the corrected replacement claim; empty when decision is KEEP",'
            '"revised_final_answer":"exact final answer expression for the final line; leave empty when decision is KEEP",'
            '"revised_response":"full revised solution text; leave empty when decision is KEEP"}'
        )
        msg += f"\n\nQuestion:\n{sample}"
        if suffix is not None:
            msg += suffix
        new_message[agent] = wrap_message(personas, agent, msg, getattr(args, "system_prompt_text", None))
        agents_to_update.append(agent)
    return new_message, agents_to_update


def _build_direct_response_update_messages(args, sample, responses, peer_map, personas=None):
    new_message = {}
    for agent, response in responses.items():
        peers = peer_map.get(agent)
        if peers is None:
            peers = []
        elif not isinstance(peers, list):
            peers = [peers]
        if not peers:
            continue
        msg = "Read the other agents' responses and then freely revise your own full solution."
        msg += "\n\nQuestion:\n" + sample
        for other_agent in peers:
            msg += f"\n\nOther agent: {other_agent}\nResponse:\n{responses[other_agent]}\n"
        msg += f"\n\nYour current response:\n{response}\n"
        msg += "\n\nOutput only your full updated solution."
        new_message[agent] = wrap_message(personas, agent, msg, getattr(args, "system_prompt_text", None))
    return new_message


def _extract_json_obj(text: str):
    if not text:
        return None
    text = str(text)
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except Exception:
                        break
    return None


def _normalize_revision_decision(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"REVISE", "KEEP"} else ""


def _extract_revision_fields_from_text(text: str):
    out = {}
    patterns = {
        "decision": r"^\s*Decision\s*:\s*(.+?)\s*$",
        "verification": r"^\s*Verification\s*:\s*(.+?)\s*$",
        "revised_final_answer": r"^\s*Revised final answer\s*:\s*(.+?)\s*$",
        "incorrect_claim": r"^\s*Incorrect claim\s*:\s*(.+?)\s*$",
        "corrected_claim": r"^\s*Corrected claim\s*:\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text or "", flags=re.IGNORECASE | re.MULTILINE)
        if match:
            out[key] = match.group(1).strip()
    match = re.search(r"^\s*Revised response\s*:\s*(.*)$", text or "", flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        out["revised_response"] = match.group(1).strip()
    match = re.search(r"^\s*Adopted_from\s*:\s*(.+?)\s*$", text or "", flags=re.IGNORECASE | re.MULTILINE)
    if match:
        out["adopted_from"] = [part.strip() for part in match.group(1).split(",") if part.strip()]
    return out


def _ensure_final_answer_line(response: str, revised_final_answer: str) -> str:
    response = str(response or "").strip()
    if not revised_final_answer:
        return response
    final_line = "{final answer: " + revised_final_answer + "}"
    lines = [line.rstrip() for line in response.splitlines() if line.strip()]
    if lines and "final answer" in lines[-1].lower():
        lines[-1] = final_line
    else:
        lines.append(final_line)
    return "\n".join(lines)


def _parse_revision_output(args, raw_output, previous_response):
    del args
    parsed = _extract_json_obj(raw_output)
    decision = ""
    adopted_from = []
    verification = ""
    revised_response = ""
    revised_final_answer = ""
    incorrect_claim = ""
    corrected_claim = ""

    if isinstance(parsed, dict):
        decision = _normalize_revision_decision(parsed.get("decision"))
        adopted_raw = parsed.get("adopted_from", [])
        if isinstance(adopted_raw, list):
            adopted_from = [str(item).strip() for item in adopted_raw if str(item).strip()]
        elif adopted_raw not in (None, ""):
            adopted_from = [str(adopted_raw).strip()]
        verification = str(parsed.get("verification", "")).strip()
        revised_response = str(
            parsed.get("revised_response") or parsed.get("response") or parsed.get("revised_opinion") or ""
        ).strip()
        revised_final_answer = str(parsed.get("revised_final_answer") or parsed.get("final_answer") or "").strip()
        incorrect_claim = str(parsed.get("incorrect_claim") or "").strip()
        corrected_claim = str(parsed.get("corrected_claim") or "").strip()

    if not decision or not revised_response:
        fallback = _extract_revision_fields_from_text(raw_output or "")
        if not decision:
            decision = _normalize_revision_decision(fallback.get("decision", ""))
        if not adopted_from:
            adopted_from = list(fallback.get("adopted_from", []))
        if not verification:
            verification = str(fallback.get("verification", "")).strip()
        if not revised_response:
            revised_response = str(fallback.get("revised_response", "")).strip()
        if not revised_final_answer:
            revised_final_answer = str(fallback.get("revised_final_answer", "")).strip()
        if not incorrect_claim:
            incorrect_claim = str(fallback.get("incorrect_claim", "")).strip()
        if not corrected_claim:
            corrected_claim = str(fallback.get("corrected_claim", "")).strip()

    if decision == "KEEP":
        revised_response = previous_response
    elif not revised_response:
        revised_response = previous_response if not revised_final_answer else _ensure_final_answer_line(previous_response, revised_final_answer)

    revised_response = _ensure_final_answer_line(revised_response, revised_final_answer)
    meta = {
        "decision": decision or "",
        "adopted_from": adopted_from,
        "verification": verification,
        "raw_output": raw_output,
        "revised_response": revised_response,
        "revised_final_answer": revised_final_answer,
        "revised_subgraph_nodes": [],
        "incorrect_claim": incorrect_claim,
        "corrected_claim": corrected_claim,
    }
    return revised_response, meta


def _parse_direct_update_output(raw_output, previous_response):
    cleaned = str(raw_output or "").strip()
    if not cleaned:
        return previous_response
    return cleaned
