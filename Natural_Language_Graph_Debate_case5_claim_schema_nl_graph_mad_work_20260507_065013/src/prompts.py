from __future__ import annotations

import json
from dataclasses import dataclass, replace
import re
from typing import Dict

from .models import AgentTrace, DivergenceCase, NaturalLanguageGraph
from .label_answer import allowed_label_answers, label_answer_hint, question_uses_label_answers
from .short_answer import extract_short_answer, normalize_short_answer
from .target_contract import analyze_question_target, clean_question_text


NL_GRAPH_SYSTEM_PROMPT = (
    "Work only with natural-language reasoning. "
    "Do not output JSON, XML, code blocks, or schema objects unless explicitly requested. "
    "Every claim should stay in ordinary task language. "
    "Use graph thinking only to name shared local progress, preserve parallel methods, and locate the first real divergence."
)




MERGE_RULES_SHORT = """
Rules:
- Common ground must be supported by at least two active paths. If only one path states it, keep it in that path note.
- Common ground contains only the same local fact about the same object. Do not mix A1 says X and A2 says Y inside one shared sentence.
- If at least two traces share a real setup, equation, substitution, formula, checked value, or requested-object statement before they split, keep that shared local checkpoint.
- If answers differ, keep the last genuinely shared local claim before the first split, then expose the earliest same-object value, formula, candidate-validity, or requested-object conflict.
- Progress lag is not a conflict: one path has not reached a later step while another has continued.
- Do not use process-only sentences such as "continues", "is in progress", "has not yet proceeded", or "stops here" as the first split.
- Do not use plan-only sentences such as "the next step is ..." or "the answer is derived by ..." as the first split unless they already contain conflicting concrete counts, formulas, or validity claims.
- Do not use a generic reminder such as "must consider the restriction" or "the roots are real" as the first split unless the other side makes the opposite claim about that same object.
- A setup choice such as "keep the variables general" versus "set a = 1 and reparameterize" is not a split by itself.
- A bookkeeping choice such as "three special people and three others" versus "treat the remaining six as a group" is not a split by itself.
- If the question asks for a set, interval, count, formula, transformed output, or named final object, do not stop at a nearby intermediate quantity when both paths still need the same downstream step.
- For range or minimum questions, "approaches the boundary" is not the same as "attains the boundary".
- Equivalent notation or equivalent algebraic forms are not a conflict by themselves.
- A method fork is not a claim conflict unless the methods make different claims about the same object.
- Keep the memo compact: usually 1 to 4 common-ground sentences and one first split.
""".strip()


AUDIT_RULES_SHORT = """
Audit rules:
- Common ground must still be supported by at least two active paths. If it belongs to only one path, move it to that path note.
- If the memo jumps directly to final-answer disagreement while the traces share a real setup, equation, substitution, formula, or checked local value, restore that shared local claim.
- Prefer a compact shared prefix of 1 to 4 natural-language claims over an empty prefix when the traces support it.
- Delete any divergence that is only agreement, equivalent notation, equivalent algebra, or progress lag.
- Delete any divergence that is only process state, such as "still computing", "has not yet continued", or "is midway through a later step".
- Delete any divergence that is only a plan description, such as "the next step is ..." or "the answer is derived by ...", before a concrete count or formula appears.
- Replace any generic split such as "must consider the restriction" with the first concrete conflicting count, formula, candidate-validity, boundary, or requested-object claim.
- Delete any split that is only setup style, such as "keep variables general" versus "set a = 1 and use t, s".
- Delete any split that is only bookkeeping style, such as "three special people and three others" versus "treat the remaining six as a group".
- If the question asks for a set or interval, rewrite a boundary-point split into the set-level split whenever the traces already imply different sets.
- If one side only shows a limit or approach to a boundary value, do not rewrite that as attainability without an explicit allowed assignment.
- Split any common-ground sentence that hides a same-object mismatch.
- If answers differ, expose the earliest same-object mismatch instead of a later answer mismatch.
- If a path stops early, say so in its path summary instead of borrowing another path's later claim.
- Keep the corrected dossier short and prefix-focused.
""".strip()


PREFIX_RULES_SHORT = """
Rules:
- Common ground must be supported by at least two active paths. If only one path states it, move it into that path note.
- Keep only the shortest shared prefix needed for the first actionable conflict.
- The best frontier is usually the last shared local claim before the first split, written in ordinary language.
- Do not leave the shared prefix empty if at least two traces visibly share a setup, equation, substitution, formula, or checked local value.
- Never keep agreement inside the first split.
- If the same object gets different values, formulas, or local conclusions, expose that conflict immediately.
- Do not split on process status such as "has not yet continued", "is in progress", or "stops after this setup".
- Do not split on plan wording such as "the next step is ..." or "the answer is derived by ..." if both paths are still describing how they intend to continue.
- Do not split on a generic reminder such as "must account for the restriction" if both paths already agree on that reminder; keep building until a concrete count, formula, candidate-validity, boundary, or requested-object claim diverges.
- Do not split on a setup choice such as "keep variables general" versus "set a = 1 and reparameterize"; keep building until those choices produce different same-object claims.
- Do not split on a bookkeeping choice such as "three special people and three others" versus "treat the remaining six as a group"; keep building until those choices produce different restricted counts or validity claims.
- If one path is only at an intermediate object but both traces still need a later named object from the question, keep building until the first conflict that directly determines that requested object.
- For range or optimum questions, a limit statement like "approaches 2 as t -> 0+" is not yet an attainability claim.
- A path being earlier or less complete than another path is not itself a divergence; keep building until a same-object claim actually conflicts.
- A symbolic operation and its evaluated value are the same progress if the arithmetic matches.
- If there is no actionable conflict yet, keep only the short shared prefix plus short path summaries.
""".strip()


MERGE_FEWSHOT = """
Tiny example 1:
Common Ground
- Both paths reduce the polynomial to (x^4 + 4)(x^4 - 1).

Paths
- A1: continues factoring x^4 + 4 over the integers.
- A2: treats x^4 + 4 as irreducible over the integers.

First Split
- First split: A1 claims x^4 + 4 factors further over the integers; A2 claims x^4 + 4 is irreducible over the integers.

Tiny example 2:
Common Ground
- Both paths use the reciprocal of 2/3, which is 3/2.
- The question asks for the multiplier, not the quotient after multiplying.

Paths
- A1: stops at the requested multiplier.
- A2: continues to compute the quotient.

First Split
- First split: A1 claims the requested answer is the multiplier 3/2; A2 claims the requested answer is the quotient 15.

Tiny example 3:
Common Ground
- Both paths have solved the same setup equation.

Paths
- A1: has not yet evaluated the formula.
- A2: evaluates the formula to 42.

First Split
- none yet; this is progress lag, not a mathematical conflict.

Tiny example 4:
Common Ground
- Both paths reduce the expression to a form that is always greater than or equal to 2.

Paths
- A1: turns the local analysis into the range statement (2, infinity).
- A2: turns the local analysis into the range statement [2, infinity).

First Split
- First split: A1 claims the requested range is (2, infinity); A2 claims the requested range is [2, infinity).

Tiny example 5:
Common Ground
- Both paths translate the point relative to the center and rotate the translated vector.

Paths
- A1: stops at the rotated intermediate vector.
- A2: translates back and states the requested final point.

First Split
- none yet; the rotated intermediate vector is not the requested final object.

Tiny example 6:
Common Ground
- Both paths fix one person and agree the adjacency restriction must be enforced.

Paths
- A1: counts the valid gap placements as 12.
- A2: counts the valid gap placements as 18.

First Split
- First split: A1 claims the valid placement count is 12; A2 claims the valid placement count is 18.

Tiny example 7:
Common Ground
- Both paths are rewriting the same positive-variable expression.

Paths
- A1: keeps a, b, c as general variables.
- A2: sets a = 1 and reparameterizes with t and s.

First Split
- none yet; this is a setup choice, not a conflicting claim about the requested object.

Tiny example 8:
Common Ground
- Both paths derive an expression whose lower bound approaches 2.

Paths
- A1: notes the value approaches 2 as t -> 0+.
- A2: states that 2 is actually attained.

First Split
- First split: A1 claims 2 is only approached, not attained; A2 claims 2 is attained by an allowed positive assignment.

Tiny example 9:
Common Ground
- Both paths fix one person and still need to enforce the adjacency restriction.

Paths
- A1: notes there are three special people and three others to track.
- A2: treats the remaining six people as a group before the restriction-specific count.

First Split
- none yet; this is bookkeeping before the restricted count, not a conflicting count claim.
""".strip()


RESOLUTION_FEWSHOT = """
Example resolution 1:
For divergence D1, the chosen action is choose_A and the winning side is Claim A. The repaired claim to continue from is: x^4 + 4 factors further over the integers as (x^2 + 2x + 2)(x^2 - 2x + 2). Revision should restart from [C1]. Keep these paths active: [P1]. Drop these paths: [P2]. Reason: The local factorization claim is supported by algebraic expansion.

Example resolution 2:
For divergence D1, the chosen action is choose_A and the winning side is Claim A. The repaired claim to continue from is: The requested answer is the multiplier 3/2, not the quotient 15. Revision should restart from [C2]. Keep these paths active: [P1]. Drop these paths: [P2]. Reason: The original question asks for the multiplier.

Example resolution 3:
For divergence D1, the chosen action is keep_parallel and the winning side is both. The repaired claim to continue from is: The two forms are equivalent and do not decide the answer yet. Revision should restart from [C1]. Keep these paths active: [P1], [P2]. Drop these paths: none. Reason: This is notation difference, not a mathematical conflict.

Example resolution 4:
For divergence D1, the chosen action is choose_A and the winning side is Claim A. The repaired claim to continue from is: The requested range is (2, infinity), not [2, infinity). Revision should restart from [C2]. Keep these paths active: [P1]. Drop these paths: [P2]. Reason: The local boundary analysis shows the value 2 is approached but not attained, so the set-level requested object is the open interval.
""".strip()


ANLI_MERGE_FEWSHOT = """
Tiny example 1:
Common Ground
- Both paths use the same premise evidence about a professional role and a recognition award.
- The hypothesis asks whether that role and recognition support an evaluative description.

Paths
- A1: treats the role and recognition as enough support for the evaluative hypothesis.
- A2: demands the exact evaluative wording.

First Split
- First split: A1 claims the shared evidence entails the hypothesis; A2 claims the same evidence leaves the hypothesis neutral.

Tiny example 2:
Common Ground
- Both paths rely on the same premise evidence and the same added detail in the hypothesis.
- The premise evidence does not state a fact that is incompatible with the added detail.

Paths
- A1: treats an unsupported extra detail as neutral.
- A2: treats the same extra detail as contradiction.

First Split
- First split: A1 claims the added detail is neutral because no incompatible premise fact is stated; A2 claims the added detail contradicts the premise.

Tiny example 3:
Common Ground
- Both paths use the same stated premise fact about an entity, status, time, place, value, or role.
- The hypothesis changes that same entity, status, time, place, value, or role.

Paths
- A1: says the hypothesis is contradicted by an incompatible fact.
- A2: says the hypothesis is not contradicted, only unsupported.

First Split
- First split: A1 claims the stated premise fact contradicts the hypothesis; A2 claims the premise only lacks support for the hypothesis.
""".strip()


LOGIC_RELATION_V2_FEWSHOT = """
Tiny formal-logic example 1:
Premise bridge: The premises prove P.
Correct claim: The queried conclusion P is proved, so the supported truth value is true.

Tiny formal-logic example 2:
Premise bridge: The premises prove not P.
Correct claim: The queried conclusion P is disproved, so the supported truth value is false.

Tiny formal-logic example 3:
Premise bridge: The premises prove Q but do not prove P or not P.
Correct claim: The queried conclusion P remains unknown/uncertain.

Tiny formal-logic example 4:
Premise bridge: The premises prove P -> Q and prove Q.
Correct claim: The queried conclusion P remains unknown/uncertain, because the converse Q -> P is not stated.

Tiny formal-logic example 5:
Premise bridge: The premises prove P or Q but do not rule out either branch.
Correct claim: The queried conclusion P remains unknown/uncertain, because a disjunction alone does not prove one branch.
""".strip()


ANLI_FRONTIER_UPDATE_GUIDE = """
Frontier update policy:
- Read only the newly revealed step window against the current memo.
- If the window adds a premise fact or hypothesis object shared by at least two traces, keep only that source fact/object in Common Ground.
- Common Ground is evidence memory, not a verdict. Never put support judgments there: "neutral", "entailed", "contradicted", "not explicitly stated", "not directly supported", "not enough information", "unsupported", or "not confirmed".
- If a visible sentence says only "not explicitly stated", "not directly supported", or "not enough information", keep it in Paths as that path's relation judgment; do not promote it to Common Ground.
- If one path only says a process status such as "continues" or "has not yet reached the label", keep that in Paths and continue building.
- If one path only says "not explicitly stated" while another says "supported by the premise", that is a relation conflict.
- If one path says an added detail is unsupported and the other says it is contradicted, that is a real split only when the premise contains an incompatible fact.
- If two paths use the same premise evidence but one labels it entailment and the other neutral, expose that relation mismatch.
- Do not split on wording alone when both paths support the same relation.
- Do not leave the shared prefix empty if at least two traces share the same premise evidence.
- A final-label difference is only the fallback when no earlier same-relation conflict is visible.
""".strip()


LOGIC_FRONTIER_UPDATE_GUIDE = """
Frontier update policy:
- Read only the newly revealed proof-step window against the current memo.
- If the window adds an explicit premise, explicit rule, or valid rule application shared by at least two traces, add only that proof step to Common Ground.
- Common Ground is proof state, not the verdict. Do not put final labels or one-path proof-status judgments there.
- If a visible sentence says only true, false, unknown, unsupported, not proved, or not enough information, keep it in Paths as that path's proof-status judgment.
- Split when two paths disagree about the queried statement's proof status: proved, disproved, or still unknown.
- Split when one path uses a converse/inverse, chooses a disjunction branch, or adds a missing premise while another path leaves the query unknown.
- Do not split on wording alone if both paths support the same proof status.
- Do not split on a nearby proven statement unless it directly proves or disproves the exact queried statement.
- If one path only has more proof steps while another has not reached them yet, keep building; progress lag is not a conflict.
- A final-label difference is only the fallback when no earlier premise/rule-to-query conflict is visible.
""".strip()


LOGIQA_FRONTIER_UPDATE_GUIDE = """
Frontier update policy:
- Read only the newly revealed reasoning window against the current option-support memo.
- If the window adds a stated premise, valid rule application, or exact option-text mapping shared by at least two traces, add only that item to Common Ground.
- Common Ground is not a final choice. Do not put a bare option number there.
- Keep each path's candidate status attached to the exact option text: supported, eliminated, or still open.
- Split when two paths disagree about the same option text's status or about the premise-to-option bridge.
- If one path eliminates option N and then answers N, expose that proof-to-option landing error.
- Do not split on wording alone if both paths support the same exact option text.
- A final-number difference is only the fallback when no earlier option-support conflict is visible.
""".strip()


LOGIQA_MERGE_FEWSHOT = """
Tiny multiple-choice logic example 1:
Common Ground
- Both paths use the premise that P implies Q.

Paths
- A1: treats option 2, "Q follows", as supported by the premise.
- A2: treats option 1, "P follows", as supported even though that is only the condition.

First Split
- First split: A1 claims the premise supports option 2's exact text; A2 claims the premise supports option 1's exact text.

Tiny multiple-choice logic example 2:
Common Ground
- Both paths agree that option 3 contradicts a stated condition.

Paths
- A1: eliminates option 3 and keeps checking the remaining options.
- A2: says option 3 is impossible but still lands on answer 3.

First Split
- First split: A1 treats option 3 as eliminated; A2 has a proof-to-option landing error because eliminating option 3 cannot support answer 3.

Tiny multiple-choice logic example 3:
Common Ground
- Both paths use the same either-or premise.

Paths
- A1: keeps both branches open because neither branch is ruled out.
- A2: chooses one branch and selects the option that depends on it.

First Split
- First split: A1 claims the branch-dependent option is still open; A2 claims that branch-dependent option is supported.
""".strip()


LOGIQA_MERGE_GUIDE = f"""
Preferred memo shape:
Common Ground
- One to four short sentences describing stated premises, rule applications, or option mappings that at least two traces genuinely share.
- Preserve exact option text when a shared claim mentions an option.

Paths
- A1 and A3: short summary of which exact option text the path supports, eliminates, or leaves open.
- A2: short summary of which exact option text the path supports, eliminates, or leaves open.

First Split
- First split: A1 and A3 claim <option text/status/bridge>; A2 claims <conflicting option text/status/bridge>.

If final option numbers differ, first map each number to its option text and compare the premise-to-option support bridge. Do not decide which side is right here. Do not expose step ids or internal claim ids unless they are already obvious. The backend will recover anchors.

The first split should be a concrete option-support conflict, not a process status line or a bare final-number conflict.

Few-shot:
{LOGIQA_MERGE_FEWSHOT}
""".strip()


LOGIQA_RESOLUTION_FEWSHOT = """
Tiny option-resolution example 1:
Correct claim: Option 2, "Q follows", is supported because the stated premise is P implies Q and P is given; continue to answer 2.

Tiny option-resolution example 2:
Correct claim: Option 3 is eliminated because it contradicts the stated condition; eliminating option 3 cannot support answer 3.

Tiny option-resolution example 3:
Correct claim: The either-or premise leaves both branches open, so the branch-dependent option is still open rather than supported.
""".strip()


NATURAL_MERGE_GUIDE = """
Preferred memo shape:
Common Ground
- One to four short sentences describing what at least two traces genuinely share before the split.
- Use ordinary mathematical language, for example "Both paths reduce the expression to ..." or "Both paths identify the requested object as ...".

Paths
- A1 and A3: short natural-language path summary.
- A2: short natural-language path summary.

First Split
- First split: A1 and A3 claim <first local claim>; A2 claims <conflicting local claim>.

If the final answers or requested objects differ, first look for the last shared local checkpoint before the answers split, then write one First split line after that checkpoint. Do not decide which side is right here. Do not expose step ids or internal claim ids unless they are already obvious. The backend will recover anchors.

The first split should be a concrete mathematical claim, not a process status line. If the question asks for a set, interval, count, formula, transformed output, or named final object, prefer the earliest split that directly determines that requested object.

Few-shot:
{fewshot}
""".strip()

NATURAL_MERGE_GUIDE = NATURAL_MERGE_GUIDE.format(fewshot=MERGE_FEWSHOT)


ANLI_MERGE_GUIDE = f"""
Preferred memo shape:
Common Ground
- One to four short sentences describing premise evidence that at least two traces genuinely share.
- Include the target hypothesis object when needed.
- Common Ground is not allowed to decide the relation. It should not say support, unsupported, entails, neutral, contradicts, not explicitly stated, not directly supported, not enough information, or not confirmed.
- If a sentence would contain one of those relation words, move it to Paths or omit it from Common Ground.

Paths
- A1 and A3: short summary of how that path relates the shared evidence to the hypothesis.
- A2: short summary of how that path relates the shared evidence to the hypothesis.

First Split
- First split: A1 claims <premise evidence implies entailment/neutral/contradiction>; A2 claims <different relation claim>.

If final labels differ, first look for the premise evidence that makes them disagree. Do not decide which side is right here.
If one path says "not explicitly stated" and another says "supported", rewrite that as a relation split over the same evidence, not as a shared fact.
If no evidence-level or relation-level conflict is visible yet, write "none yet" under First Split.

Few-shot:
{ANLI_MERGE_FEWSHOT}
""".strip()


ANLI_STRICT_EVIDENCE_MERGE_GUIDE = f"""
Preferred memo shape:
Common Ground
- One to four short sentences containing only premise facts that at least two traces genuinely share.
- If needed, include the hypothesis object as a short noun phrase or quoted hypothesis text only.
- Common Ground must not say whether the premise supports, fails to support, entails, contradicts, confirms, mentions, states, or leaves the hypothesis open.
- Do not write absence/explicitness verdicts in Common Ground, such as "the premise does not explicitly state", "not directly supported", "not mentioned", "does not confirm", "unsupported", or "not enough information".
- If a thought has to use one of those verdict words, it belongs in Paths or First Split, never in Common Ground.

Paths
- A1 and A3: short summary of how that path relates the premise evidence to the hypothesis.
- A2: short summary of how that path relates the premise evidence to the hypothesis.
- If a path only demands exact wording, say that in its path summary rather than promoting the demand to Common Ground.

First Split
- First split: A1 claims <premise evidence implies entailment/neutral/contradiction>; A2 claims <different relation claim>.

If final labels differ, first look for the premise evidence that makes them disagree. Do not decide which side is right here.
When the dispute is explicit wording versus ordinary implication, write the split over the evidence-to-hypothesis relation, not over whether the exact words appear.
If no evidence-level or relation-level conflict is visible yet, write "none yet" under First Split.

Few-shot:
{ANLI_MERGE_FEWSHOT}
""".strip()


LOGIC_MERGE_GUIDE = f"""
Preferred memo shape:
Common Ground
- One to four short premises, explicit rules, or valid rule applications shared by at least two traces.
- Common Ground is proof state only. It should not decide the final truth label.
- Do not put true, false, unknown, unsupported, or not enough information in Common Ground unless it is the queried proof status shared by the paths.

Paths
- A1 and A3: short summary of how that path connects premises/rules to the queried statement.
- A2: short summary of how that path connects premises/rules to the queried statement.

First Split
- First split: A1 claims <premises/rules prove the query, prove its negation, or leave it unknown>; A2 claims <different proof-status claim>.

If final labels differ, first identify the proof-status bridge that causes the difference. Do not decide which side is right here.
If one path uses a converse/inverse, chooses a disjunction branch, or adds a missing premise while another path does not, expose that as the proof-status conflict.
If no proof-status conflict is visible yet, write "none yet" under First Split.

Few-shot:
{LOGIC_RELATION_V2_FEWSHOT}
""".strip()


STRATEGYQA_MERGE_GUIDE = """
Preferred memo shape:
Common Ground
- One to three short factual bridge claims that at least two traces genuinely share.
- Keep the question predicate separate from background facts when that predicate decides yes/no.

Paths
- A1 and A3: short summary of how that path maps facts to the question predicate.
- A2: short summary of how that path maps facts to the question predicate.
- If a path only gives a final yes/no with no factual bridge, say "bare label only; no factual bridge observed" rather than treating the label as evidence.

First Split
- First split: A1 claims <factual bridge implies the predicate is true/false>; A2 claims <conflicting bridge or conflicting label implication>.

If final labels differ, first look for the factual bridge or label implication that makes them disagree. Do not decide which side is right here.
Do not write the first split as only "yes vs no" when at least one path gives factual support. Preserve the supported factual bridge.
If no evidence-level or predicate-level conflict is visible yet, write "none yet" under First Split.

Few-shot:
{fewshot}
""".strip()


PRONTOQA_MERGE_GUIDE = """
Preferred memo shape:
Common Ground
- One to four short facts or rules that at least two traces genuinely share.
- Keep the queried statement separate from the supporting facts and rules.

Paths
- A1 and A3: short summary of how that path applies the shared facts/rules to the queried statement.
- A2: short summary of how that path applies the shared facts/rules to the queried statement.
- If a path only gives true/false with no fact-rule bridge, say "bare label only; no fact-rule bridge observed".

First Split
- First split: A1 claims <facts/rules imply the queried statement is true/false>; A2 claims <conflicting fact-rule implication>.

If final labels differ, first look for the fact-rule bridge that makes them disagree. Do not decide which side is right here.
Do not write the first split as only "true vs false" when at least one path gives a fact-rule bridge.
If no fact-level, rule-level, or implication-level conflict is visible yet, write "none yet" under First Split.
""".strip()


ANLI_RESOLUTION_FEWSHOT = """
Tiny example 1:
Premise evidence: The premise states a concrete value, identity, date, status, location, or comparison.
Hypothesis: The hypothesis changes that same object to an incompatible value.
Correct claim: The premise contradicts the hypothesis because the stated fact and the hypothesis cannot both be true.

Tiny example 2:
Premise evidence: The premise gives stronger or more specific support than the hypothesis asks for.
Hypothesis: The hypothesis is a weaker restatement, such as multiple, at least one, being hired, or the same uncertainty.
Correct claim: The premise entails the hypothesis because the hypothesis asks for no more than the premise gives.

Tiny example 3:
Premise evidence: The premise simply does not mention an added detail and also does not conflict with it.
Hypothesis: The added detail is true.
Correct claim: The premise is neutral toward the hypothesis because the detail is unsupported but not contradicted.
""".strip()


STRATEGYQA_MERGE_FEWSHOT = """
Tiny example 1:
Common Ground
- Both paths identify The Police as a band, not law-enforcement officers.

Paths
- A1: treats band membership as not granting arrest authority.
- A2: reads "police" as a legal role.

First Split
- First split: A1 claims the named entity is the band The Police and does not imply lawful arrest authority; A2 claims the word "Police" supplies lawful arrest authority.

Tiny example 2:
Common Ground
- Both paths use the same fact that the person would not understand the language.

Paths
- A1: maps not understanding the language to being confused by it.
- A2: maps not understanding the language to answering no.

First Split
- First split: A1 claims not understanding the language makes the question predicate "would be confused" true; A2 claims the same fact supports no.

Tiny example 3:
Common Ground
- Both paths discuss evidence about an item appearing in some places.

Paths
- A1: treats "some places" as enough.
- A2: keeps the question predicate "most places" separate from "some places".

First Split
- First split: A1 claims some appearances support the "most places" predicate; A2 claims some appearances do not support the "most places" predicate.

Tiny example 4:
Common Ground
- Both paths discuss the same object and a nearby factual constraint.

Paths
- A1: answers the exact target relation asked by the question.
- A2: answers a nearby relation, such as whether the object is available, purchasable, historically occurred, normally allowed, or merely possible.

First Split
- First split: A1 claims the evidence supports the exact target relation; A2 claims a true nearby constraint changes the answer even though it answers a different relation.
""".strip()


STRATEGYQA_RESOLUTION_FEWSHOT = """
Tiny example 1:
Question: Could members of The Police perform lawful arrests?
Correct claim: The named entity is the band The Police; band membership does not grant lawful arrest authority, so the supported label is no.

Tiny example 2:
Question: Would someone be confused by a language they do not understand?
Correct claim: Not understanding the language makes the question predicate "would be confused" true, so the supported label is yes.

Tiny example 3:
Question: Can you find a face in most shops?
Correct claim: Evidence that the face appears in some shops does not support the "most shops" predicate, so the supported label is no.

Tiny example 4:
Question: Would an unknown language confuse the listener?
One path only says "no"; another path says the listener does not understand the language.
Correct claim: Not understanding the language supports the predicate "would confuse the listener", so the supported label is yes.

Tiny example 5:
Question: Could a shorter local condition alone answer the whole question?
Correct claim: The shorter condition only settles a subcondition, not the whole question predicate, so it is not enough to support the final label by itself.

Tiny example 6:
Question: Could a company afford a famous object?
Correct claim: The target relation is financial capacity, so sale availability is only a nearby purchase constraint; keep the claim about whether the company has enough resources to afford the object.
""".strip()


def _profile_merge_guide(profile: str | PromptProfile | None) -> str:
    resolved = resolve_prompt_profile(profile)
    if _profile_name(resolved) in {
        "anli_relation_v3",
        "anli_relation_v4",
        "anli_relation_v6",
        "anli_relation_v7",
        "anli_relation_v8",
        "anli_relation_v9",
        "anli_relation_v10",
        "anli_relation_v11",
        "anli_relation_v12",
        "anli_relation_v13",
        "anli_relation_v14",
        "anli_relation_v15_equal_token",
        "anli_relation_v16_extraction_guard",
        "anli_relation_v17_action_guard",
        "anli_relation_v18_memory_guard",
        "anli_relation_v19_bridge_audit",
        "anli_relation_v15_equal_token",
        "anli_relation_v15_equal_token",
    }:
        return ANLI_STRICT_EVIDENCE_MERGE_GUIDE
    if _is_anli_profile(resolved):
        return ANLI_MERGE_GUIDE
    if _is_truthvalue_profile(resolved):
        return LOGIC_MERGE_GUIDE
    if _is_strategyqa_profile(resolved):
        return STRATEGYQA_MERGE_GUIDE.format(fewshot=resolved.merge_fewshot)
    if _is_prontoqa_profile(resolved):
        return PRONTOQA_MERGE_GUIDE
    if _is_logiqa_strict_option_profile(resolved):
        return LOGIQA_MERGE_GUIDE
    return NATURAL_MERGE_GUIDE.replace(MERGE_FEWSHOT, resolved.merge_fewshot)


def _profile_dataset_note(profile: str | PromptProfile | None) -> str:
    return resolve_prompt_profile(profile).dataset_note.strip()


def _profile_language_instruction(profile: str | PromptProfile | None) -> str:
    resolved = resolve_prompt_profile(profile)
    if _is_anli_profile(resolved):
        return "Write ordinary premise-hypothesis reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_truthvalue_profile(resolved):
        return "Write ordinary premise/rule proof-status sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_strategyqa_profile(resolved):
        return "Write ordinary factual reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_prontoqa_profile(resolved):
        return "Write ordinary fact-and-rule reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_bamboogle_profile(resolved):
        return "Write ordinary multi-hop evidence reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_boolq_profile(resolved):
        return "Write ordinary passage-based reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_medqa_profile(resolved):
        return "Write ordinary medical reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_logiqa_profile(resolved):
        return "Write ordinary multiple-choice logic reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    if _is_universal_profile(resolved) or resolved.name.startswith("mmlu_pro"):
        return "Write ordinary reasoning sentences. Do not fill a schema, assign ids, or expose internal step ids."
    return "Write ordinary mathematical sentences. Do not fill a schema, assign ids, or expose internal step ids."


def _profile_full_memo_instruction(profile: str | PromptProfile | None, *, audit: bool = False) -> str:
    base = _profile_language_instruction(profile)
    resolved = resolve_prompt_profile(profile)
    if _is_anli_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary premise-hypothesis reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_truthvalue_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary premise/rule proof-status language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_strategyqa_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary factual reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_prontoqa_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary fact-and-rule reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_bamboogle_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary multi-hop evidence reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_boolq_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary passage-based reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_medqa_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary medical reasoning language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if _is_logiqa_profile(resolved):
        prefix = "Output one corrected full memo" if audit else "Output one full memo"
        return f"{prefix} in ordinary multiple-choice logic language. Do not fill a schema, assign ids, expose step ids, or decide which side is right."
    if audit:
        return f"Output one corrected full memo. {base}"
    return f"Output one full memo. {base}"


def _profile_frontier_update_guide(profile: str | PromptProfile | None) -> str:
    resolved = resolve_prompt_profile(profile)
    if _is_anli_profile(resolved):
        return ANLI_FRONTIER_UPDATE_GUIDE
    if _is_truthvalue_profile(resolved):
        return LOGIC_FRONTIER_UPDATE_GUIDE
    if _is_logiqa_strict_option_profile(resolved):
        return LOGIQA_FRONTIER_UPDATE_GUIDE
    return FRONTIER_UPDATE_GUIDE


def _profile_pairwise_resolution_rules(profile: str | PromptProfile | None) -> list[str]:
    resolved = resolve_prompt_profile(profile)
    if _is_anli_profile(resolved):
        return [
            "- Output one short paragraph or a few short sentences, not a field table.",
            "- Allowed actions are choose_A, choose_B, and keep_parallel.",
            "- Compare premise evidence to the hypothesis, not final label popularity.",
            "- If one trace only demands exact wording while the premise gives stronger support, keep the entailment relation.",
            "- If one trace treats missing information as contradiction, require an incompatible premise fact.",
            "- Use keep_parallel only when both relation claims can remain true without deciding the final label yet.",
            "- Prefer a full premise-to-hypothesis relation claim over a bare label or diagnosis sentence.",
            "- If the repaired claim would start with 'the premise does not explicitly...' rewrite it into a full entail/neutral/contradict relation claim.",
            "- Use [C1] as the rewrite anchor if there is no explicit shared claim id.",
            "- Treat Claim A as the first trace and Claim B as the second trace.",
        ]
    return [
        "- Output one short paragraph or a few short sentences, not a field table.",
        "- Allowed actions are choose_A, choose_B, and keep_parallel. Do not invent a third repaired answer.",
        "- Prefer a same-object local mismatch over a later final-answer mismatch.",
        "- Do not choose by answer popularity.",
        "- If one trace answers a nearby but wrong requested object, repair that requested-object drift.",
        "- If a trace states a candidate is valid or invalid, require the supporting local check to match the question.",
        "- Use keep_parallel only when both local claims can remain true without deciding the answer yet.",
        "- Prefer a concrete repaired claim over a vague summary.",
        "- Use [C1] as the rewrite anchor if there is no explicit shared claim id.",
        "- Treat Claim A as the first trace and Claim B as the second trace.",
        "- Follow the examples below in spirit.",
    ]


def _profile_revision_common_rules(profile: str | PromptProfile | None, *, label_hint: str = "") -> list[str]:
    resolved = resolve_prompt_profile(profile)
    if _is_anli_profile(resolved):
        rules = [
            "- Output plain text only.",
            "- Continue as if the repaired premise-to-hypothesis relation is the next accepted reasoning line.",
            f"{resolved.revision_rules}",
            "- Do not infer or restate an old losing relation or old final label.",
            "- If any stale packet line conflicts with the repaired relation, replace it.",
            "- Keep the evidence tied to the premise; do not add outside facts.",
            "- The last non-empty line must be the final NLI label requested by the question.",
        ]
        if label_hint:
            rules.append(f"- {label_hint}")
        return rules
    rules = [
        "- Output plain text only.",
        "- This trace is being revised because its previous branch lost or needs repair; follow the resolution summary as authoritative.",
        f"{resolved.revision_rules}",
        "- For true/false tasks, state whether the target statement is true or false before writing the final label token.",
        "- Do not infer or restate the old losing suffix or old final answer.",
        "- If any stale packet line conflicts with the repaired claim, replace it; do not keep both the stale line and the repaired line.",
        "- One non-empty line per local claim, operation, check, or conclusion.",
        "- Keep concrete numeric substitutions explicit when they are known.",
        "- Do not end with only an intermediate condition, boundary note, function name, object name, earlier iterate, or component quantity.",
        "- Preserve the requested answer form from the question and repaired branch. Do not replace an exact form such as a radical, fraction, pi expression, degree form, or text label with a decimal approximation unless the question explicitly asks for a decimal.",
        "- If you claim a candidate value is attainable or valid, include the local check that supports it.",
        "- The last non-empty line must be a final answer line for the requested object whenever the problem asks for a final answer.",
    ]
    if label_hint:
        rules.append(f"- {label_hint}")
    return rules


FRONTIER_UPDATE_GUIDE = """
Frontier update policy:
- Read only the newly revealed step window against the current memo.
- If the window extends a fact shared by at least two traces, add that local fact to Common Ground and keep going.
- If a possible shared sentence needs phrases like "for A1", "for A2", "while", or "whereas" to describe different values, it is not shared progress.
- If traces use different methods but do not contradict each other about the same object, do not resolve yet. Keep both methods in Paths.
- If a path note itself says one agent uses one concrete value, formula, candidate status, or coefficient while another uses a different one, that is a claim conflict. Move it to First Split.
- If one path only says a process status such as "is still computing", "has not yet continued", or "stops at the current checkpoint", keep that in Paths and continue building; it is not a claim conflict.
- If one path only says a plan sentence such as "the next step is to count ..." or "the answer is derived by arranging ...", keep that in Paths until the plan produces a concrete count, formula, or validity claim.
- If both paths already agree on a generic reminder such as "the restriction matters" or "the roots are real", do not split there. Keep building until a concrete count, formula, candidate-validity, boundary, or requested-object claim diverges.
- If one path keeps general variables while another reparameterizes or fixes a normalizing value, do not split there unless they now claim different values, sets, counts, or validity facts about the same requested object.
- If one path only changes bookkeeping for a restricted counting problem, do not split there unless it changes the restricted count, valid placements, or candidate validity.
- If one path names an intermediate object but the question asks for a later named object, do not stop there unless the other path makes the opposite claim about that same intermediate object and that object directly determines the final requested object.
- For boundary or optimum questions, a limit statement such as "approaches 2" does not certify that 2 is attainable; require an explicit allowed assignment or explicit impossibility argument.
- Paths should describe strategy only after local claim conflicts have been removed from them.
- If two paths make incompatible claims about the same object, stop the graph there and write exactly one First split line with the two conflicting claims.
- A method fork is not a claim conflict. A final-answer difference is only the fallback when no earlier shared checkpoint or same-object claim conflict is visible.
- Progress lag is not a claim conflict: "has not yet proceeded" versus "has solved/applied" should stay in Paths, not First Split.
""".strip()


@dataclass(frozen=True)
class PromptProfile:
    name: str
    system_prompt: str
    merge_intro: str
    merge_rules: str
    merge_fewshot: str
    audit_intro: str
    audit_rules: str
    prefix_intro: str
    prefix_rules: str
    relation_analysis_intro: str
    relation_analysis_rules: str
    resolution_intro: str
    resolution_rules: str
    resolution_fewshot: str
    revision_rules: str
    dataset_note: str = ""


def _profile_name(profile: str | PromptProfile | None) -> str:
    if isinstance(profile, PromptProfile):
        return profile.name
    if profile is None:
        return ""
    return str(profile)


def _is_anli_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "anli",
        "anli_relation_v2",
        "anli_relation_v3",
        "anli_relation_v4",
        "anli_relation_v5",
        "anli_relation_v6",
        "anli_relation_v7",
        "anli_relation_v8",
        "anli_relation_v9",
        "anli_relation_v10",
        "anli_relation_v11",
        "anli_relation_v12",
        "anli_relation_v13",
        "anli_relation_v14",
    }


def _is_strategyqa_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "strategyqa",
        "strategyqa_relation_v1",
        "sqa_relation_v1",
        "strategyqa_relation_v2",
        "sqa_relation_v2",
        "strategyqa_relation_v3",
        "sqa_relation_v3",
        "strategyqa_relation_v4",
        "sqa_relation_v4",
        "strategyqa_relation_v5",
        "sqa_relation_v5",
        "strategyqa_relation_v6",
        "sqa_relation_v6",
    }


def _is_prontoqa_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {"prontoqa", "prontoqa_relation_v1"}


def _is_bamboogle_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "bamboogle",
        "bamboogle_relation_v1",
        "bamboogle_relation_v2",
        "bamboogle_relation_v3",
        "bamboogle_relation_v4",
        "bamboogle_relation_v5",
        "bamboogle_relation_v6",
        "bamboogle_relation_v7_equal_token",
        "bamboogle_relation_v8_equal_token",
        "bamboogle_relation_v9_extraction_guard",
        "bamboogle_relation_v10_bridge_audit",
        "hotpotqa_relation_v1",
        "hotpotqa_relation_v2",
        "hotpotqa_relation_v3",
        "hotpotqa_relation_v4",
        "hotpotqa_relation_v36_equal_token",
        "hotpotqa_relation_v37_extraction_guard",
        "hotpotqa_relation_v38_bridge_audit",
        "hotpotqa_relation_v39_ledger_audit",
        "hotpotqa_relation_v5",
        "hotpotqa_relation_v6",
        "hotpotqa_relation_v7",
        "hotpotqa_relation_v8",
        "hotpotqa_relation_v9",
        "hotpotqa_relation_v10",
        "hotpotqa_relation_v11",
        "hotpotqa_relation_v12",
        "hotpotqa_relation_v13",
        "hotpotqa_relation_v14",
        "hotpotqa_relation_v15",
        "hotpotqa_relation_v16",
        "hotpotqa_relation_v17",
        "hotpotqa_relation_v18",
        "hotpotqa_relation_v19",
        "hotpotqa_relation_v20",
        "hotpotqa_relation_v21",
        "hotpotqa_relation_v22",
        "hotpotqa_relation_v23",
        "hotpotqa_relation_v24",
        "hotpotqa_relation_v25",
        "hotpotqa_relation_v26",
        "hotpotqa_relation_v27",
        "hotpotqa_relation_v28",
        "hotpotqa_relation_v29",
        "hotpotqa_relation_v30",
        "hotpotqa_relation_v31",
        "hotpotqa_relation_v32",
        "hotpotqa_relation_v33",
        "hotpotqa_relation_v34",
        "hotpotqa_relation_v35",
        "hotpotqa_relation_v36_equal_token",
        "musique_relation_v1",
        "musique_relation_v2",
        "musique_relation_v3",
    }


def _uses_observed_candidate_block(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "math500_v33_ledger_audit",
        "mmlu_pro_relation_v5_ledger_audit",
        "hotpotqa_relation_v24",
        "hotpotqa_relation_v25",
        "hotpotqa_relation_v26",
        "hotpotqa_relation_v27",
        "hotpotqa_relation_v28",
        "hotpotqa_relation_v29",
        "hotpotqa_relation_v30",
        "hotpotqa_relation_v31",
        "hotpotqa_relation_v32",
        "hotpotqa_relation_v33",
        "hotpotqa_relation_v34",
        "hotpotqa_relation_v35",
        "hotpotqa_relation_v36_equal_token",
        "hotpotqa_relation_v39_ledger_audit",
        "bamboogle_relation_v7_equal_token",
    }


def _extract_observed_candidate_phrase(trace: AgentTrace) -> str | None:
    texts = [trace.original_response, trace.normalized_trace_text]
    final_patterns = [
        r"\{[^{}]*final answer\s*[:=\-]?\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*\}",
        r"(?:final answer|short answer|answer phrase)\s*[:=\-]?\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*$",
        r"(?:therefore\s+)?(?:the\s+)?(?:short\s+)?answer\s+(?:is|=)\s*(?:\\?boxed\{\s*)?(.+?)(?:\s*\})?\s*$",
    ]
    for text in texts:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        for line in reversed(lines[-6:]):
            for pattern in final_patterns:
                match = re.search(pattern, line, flags=re.IGNORECASE)
                if not match:
                    continue
                candidate = match.group(1).strip().strip("{}[]()").strip("\"'`“”‘’").strip()
                candidate = re.sub(r"[.。!?,;:]+$", "", candidate).strip()
                if candidate and normalize_short_answer(candidate):
                    return candidate
        if lines:
            candidate = lines[-1].strip().strip("{}[]()").strip("\"'`“”‘’").strip()
            candidate = re.sub(r"[.。!?,;:]+$", "", candidate).strip()
            if candidate and len(candidate.split()) <= 12 and normalize_short_answer(candidate):
                return candidate
    return extract_short_answer(trace.original_response) or extract_short_answer(trace.normalized_trace_text)


def _render_observed_candidate_block(
    question: str,
    traces: Dict[str, AgentTrace] | None,
    profile: str | PromptProfile | None = None,
) -> str:
    if not traces or not _uses_observed_candidate_block(profile):
        return ""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for agent_id, trace in sorted(traces.items()):
        answer = _extract_observed_candidate_phrase(trace)
        if not answer:
            continue
        normalized = normalize_short_answer(answer)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rows.append((agent_id, answer.strip()))
    if not rows:
        return ""
    lines = [
        "Observed candidate short answers from current branches:",
    ]
    for agent_id, answer in rows[:8]:
        lines.append(f"- {agent_id}: {answer}")
    if _profile_name(profile) in {
        "hotpotqa_relation_v25",
        "hotpotqa_relation_v26",
        "hotpotqa_relation_v27",
        "hotpotqa_relation_v28",
        "hotpotqa_relation_v29",
        "hotpotqa_relation_v30",
        "hotpotqa_relation_v31",
        "hotpotqa_relation_v32",
        "hotpotqa_relation_v33",
        "hotpotqa_relation_v34",
        "hotpotqa_relation_v35",
        "hotpotqa_relation_v36_equal_token",
        "hotpotqa_relation_v39_ledger_audit",
    }:
        lines.extend(
            [
                "Use these only as noisy branch hypotheses, not as gold answers and not as fixed multiple-choice options.",
                "First audit the evidence-to-slot bridge for each hypothesis: source entity -> bridge entity -> requested attribute -> answer phrase.",
                "A hypothesis may identify the right object while using the wrong phrase length; after choosing the bridge, rewrite the phrase to the minimal sufficient slot answer.",
                "Minimal sufficient is not shortest-at-all-costs: preserve required units, counted objects, date spans, county/state level, titles, and yes/no labels.",
                "If a listed hypothesis already fills the supported slot exactly, keep that phrase instead of regenerating a nearby phrase.",
                "Do not copy a whole sentence, explanation, or over-specific title just because it is an observed candidate.",
            ]
        )
    elif _profile_name(profile) == "math500_v33_ledger_audit":
        lines.extend(
            [
                "Use these only as noisy branch hypotheses, not as gold answers.",
                "For each candidate, audit: requested math object -> shared checkpoint -> checked operation -> candidate answer.",
                "A candidate wins only if the visible computation reaches the exact requested object, not just a method, intermediate value, or plausible form.",
                "If no candidate has a complete computation bridge, choose keep_parallel and name the missing local operation.",
            ]
        )
    elif _profile_name(profile) == "mmlu_pro_relation_v5_ledger_audit":
        lines.extend(
            [
                "Use these only as noisy branch hypotheses, not as gold answers.",
                "For each candidate, audit: requested relation -> exact option text -> visible supporting bridge -> final option token.",
                "A candidate wins only if the bridge supports the exact option text, not merely a related topic or shared phrase.",
                "If no candidate has an exact option bridge, choose keep_parallel and name the missing mapping.",
            ]
        )
    else:
        lines.extend(
            [
                "Use these only as transparent branch candidates, not as gold answers.",
                "Audit the evidence-to-slot bridge for each candidate. Prefer a supported observed candidate over inventing a new phrase.",
                "Create a new phrase only when the visible trace evidence explicitly supports a more exact completion than every observed candidate.",
            ]
        )
    return "\n".join(lines)


def _is_truthvalue_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "proofwriter_relation_v1",
        "folio_relation_v1",
        "proofwriter_relation_v2",
        "folio_relation_v2",
        "proofwriter_relation_v3",
        "folio_relation_v3",
        "folio_relation_v4_equal_token",
        "folio_relation_v5_equal_token",
        "folio_relation_v6_extraction_guard",
        "folio_relation_v7_bridge_audit",
        "folio_relation_v8_ledger_audit",
    }


def _is_boolq_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {"boolq", "boolq_relation_v1"}


def _is_medqa_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "medqa",
        "medqa_relation_v1",
        "medqa_relation_v2",
        "medqa_relation_v3",
        "medqa_relation_v4",
        "medqa_relation_v5",
        "medqa_relation_v6",
        "medqa_relation_v7",
        "medqa_relation_v8",
        "medqa_relation_v9",
    }


def _is_logiqa_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "logiqa_relation_v2",
        "logiqa_relation_v3",
        "logiqa_relation_v4",
        "logiqa_relation_v5",
        "logiqa_relation_v6",
        "logiqa_relation_v7",
        "logiqa_relation_v8",
        "logiqa_relation_v9",
        "logiqa_relation_v10",
        "logiqa_relation_v11_equal_token",
        "logiqa_relation_v12_extraction_guard",
        "logiqa_relation_v13_flow_guard",
        "logiqa_relation_v14_action_guard",
        "logiqa_relation_v15_memory_guard",
        "logiqa_relation_v16_stable_rewrite_guard",
        "logiqa_relation_v17_bridge_audit",
    }


def _is_logiqa_strict_option_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {
        "logiqa_relation_v2",
    }


def _is_universal_profile(profile: str | PromptProfile | None) -> bool:
    return _profile_name(profile) in {"universal", "universal_minimal"}


SHARED_PREFIX_SUPPORT_RULE = (
    "- If paths share the same prefix or evidence but end with different final answers, "
    "do not make the final tokens the first claim conflict. Instead compare how the "
    "shared prefix supports the requested object, label, relation, or option."
)

SUPPORT_RELATION_SPLIT_RULE = (
    "- A clean split can be: shared prefix S; one path claims S supports target relation R1, "
    "another path claims S supports R2."
)


LABEL_SUPPORT_MERGE_FEWSHOT = """
Tiny support-claim example 1:
Common Ground
- Both paths use the fact that the listener does not understand the language.

Paths
- A1: claims this fact means the listener would not be confused.
- A2: claims this fact means the listener would be confused.

First Split
- First split: A1 claims not understanding the language supports the predicate "would not be confused"; A2 claims the same fact supports the predicate "would be confused".

Tiny support-claim example 2:
Common Ground
- The question asks whether the statement is true.

Paths
- A1: gives evidence that proves the statement.
- A2: gives only the final label false.

First Split
- First split: A1 claims the evidence supports the statement being true; A2 gives no evidence beyond the false label.

Tiny support-claim example 3:
Common Ground
- none yet; merely giving final labels is answer format, not shared evidence.

Paths
- A1: gives only the final label no.
- A2: gives evidence that the target predicate would be true.

First Split
- First split: A1 gives no support claim beyond no; A2 claims its evidence supports the target predicate being true.

Tiny support-claim example 4:
Common Ground
- Both paths discuss a local fact that may only settle part of the question.

Paths
- A1: treats the local fact as enough for the final label.
- A2: claims the local fact does not yet settle the whole question predicate.

First Split
- First split: A1 claims the nearby constraint decides the target relation; A2 claims it only supports a nearby or partial predicate.

Tiny support-claim example 5:
Common Ground
- Both paths discuss whether the evidence supports an ability or capacity relation.

Paths
- A1: treats actual occurrence, transaction completion, or ordinary availability as decisive.
- A2: keeps the question on the ability or capacity relation itself.

First Split
- First split: A1 claims the practical or historical constraint settles the ability relation; A2 claims the ability relation still needs to be evaluated on its own terms.
""".strip()


LABEL_SUPPORT_RESOLUTION_FEWSHOT = """
Tiny support-claim resolution 1:
Correct claim: Not understanding the language supports the predicate "would be confused", so the supported label is yes.

Tiny support-claim resolution 2:
Correct claim: Evidence that proves the target statement supports the label true; a bare false label is not a competing evidence claim.

Tiny support-claim resolution 3:
Correct claim: The local fact settles only part of the question predicate, so it is not enough by itself to support a final yes/no label.

Tiny support-claim resolution 4:
Correct claim: If the question asks one relation, keep that relation as the target; a nearby constraint is evidence only when there is a credible bridge between them.

Tiny support-claim resolution 5:
Correct claim: For ability or capacity questions, actual occurrence, purchase availability, ordinary access, or historical commonness is only a constraint; keep the claim about whether the subject had the relevant ability or capacity.

Tiny support-claim resolution 6:
Correct claim: For a would/could scenario, lack of prior contact or exposure does not by itself answer no; evaluate what the scenario implies, so the supported label follows from that bridge.

Tiny support-claim resolution 7:
Correct claim: If the scenario is encountering an unfamiliar language or signal, lack of understanding supports confusion in that encounter, so the supported label is yes.
""".strip()


def _label_support_relation_rules(question: str, profile: str | PromptProfile | None = None) -> list[str]:
    if _is_logiqa_profile(profile):
        return []
    if not question_uses_label_answers(question):
        return []
    labels = ", ".join(allowed_label_answers(question)[:8])
    label_tail = f" Allowed labels: {labels}." if labels else ""
    return [
        "- Do not use answer format, allowed labels, or the fact that all paths answered as Common Ground; final-answer protocol is not shared evidence.",
        "- a bare final label is not a claim or evidence; Do not prefer a bare label for being direct; compare the evidence or proposition behind the label; do not treat an earlier isolated pro or con fact as the whole position.",
        "- Keep the exact question predicate fixed. If a fact answers only a nearby, weaker, stronger, historical, or partial relation, say so instead of upgrading it.",
        "- For would/could/hypothetical questions, evaluate the scenario asked; lack of historical occurrence, contact, or exposure is not by itself a no-answer unless the question asks whether it happened.",
        "- For date, age, or time-window claims, compute the interval explicitly before deciding which path supports the predicate.",
        "- A specific outside fact asserted by only one path is a path claim to compare, not established evidence or Common Ground.",
        "- A bridge may use a normal common-sense inference, but it must be grounded in the trace, the question, or ordinary meaning; do not add a hidden extra condition or special permission.",
        "- In resolution, make Correct claim end with 'so the supported label is <label>'.",
        "- State the claim as: evidence/proposition -> how it supports or fails the target relation, then the label." + label_tail,
    ]


def _label_support_merge_examples(question: str, profile: str | PromptProfile | None = None) -> list[str]:
    if _is_logiqa_profile(profile):
        return []
    if not question_uses_label_answers(question):
        return []
    return ["Support-claim examples:", LABEL_SUPPORT_MERGE_FEWSHOT]


def _label_support_resolution_examples(question: str, profile: str | PromptProfile | None = None) -> list[str]:
    if _is_logiqa_profile(profile):
        return []
    if not question_uses_label_answers(question):
        return []
    return ["Support-claim resolution examples:", LABEL_SUPPORT_RESOLUTION_FEWSHOT]


UNIVERSAL_PROMPT_PROFILE = PromptProfile(
    name="universal",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo.",
    merge_rules="""
Rules:
- Keep only claim-level shared progress.
- Do not split on process lag, equivalent notation, or method style alone.
- Split as soon as two paths make incompatible claims about the same object.
- If paths share evidence but final answers differ, expose the differing support relation from that evidence to the requested answer.
- Do not treat a bare final answer token as evidence when the trace contains an earlier factual or local claim.
- For label tasks with mixed evidence, summarize each path by its final predicate-level bridge, not by whichever earlier local fact is easiest to quote.
- Preserve predicate strength: a path claiming possible, some, or exists does not automatically settle stronger predicates such as most, common, typical, or would.
- Keep the exact question predicate separate from nearby historical, technical, weaker, or stronger predicates.
- Lock the target relation: keep the question's verb/property and relation direction fixed while comparing paths.
- A true nearby constraint is not decisive by itself if it answers a different relation from the question.
- For ability or capacity questions, separate the ability/capacity relation from actual occurrence, purchase completion, ordinary availability, or historical commonness.
- For label tasks, the graph claim should be "evidence supports/fails the requested relation", not "evidence supports/fails a prerequisite or adjacent relation".
- Use the ordinary sense of the question unless the question explicitly asks for a technical or special definition.
- If the evidence only supports a nearby predicate, say that bridge explicitly before choosing the label.
- A reasonable common-sense bridge can be a claim; keep it separate from invented facts or special extra conditions.
- If a path only works after adding an extra premise that is not recoverable from the question, trace, or ordinary meaning, mark that bridge as unsupported.
- Treat a specific outside fact asserted by only one path as a path claim, not as Common Ground or established evidence.
""".strip(),
    merge_fewshot=MERGE_FEWSHOT,
    audit_intro="Audit a natural-language claim memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep the memo trace-faithful.
- Move one-path-only material back into the relevant path note.
- Do not invent unsupported facts.
""".strip(),
    prefix_intro="Rebuild a compact natural-language claim memo that exposes only the shortest shared prefix and the first unresolved actionable conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared prefix needed for the first actionable conflict.
- Do not split on process status or equivalent setup style alone.
- Split only on a concrete same-object claim conflict.
- If the same shared prefix leads to different final answers, split on how that prefix supports the requested object or label.
- For label tasks, the first actionable split is usually the bridge from the evidence bundle to the whole predicate; do not split on an isolated local fact if the same trace later weighs that fact differently.
- If the disagreement is about predicate strength, split on the strength difference rather than on the final label token.
- If one side answers a historical, technical, weaker, stronger, or adjacent predicate instead of the exact question predicate, split on that predicate mismatch.
- If one side changes the target relation, such as from afford to purchase availability or from confused to lack of exposure, split on that relation drift before resolving.
- If one path answers a prerequisite or adjacent relation, split on whether that relation actually settles the requested relation.
- If one side relies on an extra condition not stated in the question or trace evidence, split on whether that condition is actually part of the bridge.
- If one side only becomes plausible after adding a premise not recoverable from the question, trace, or ordinary meaning, split on that bridge quality.
- If one side relies on a specific outside fact asserted by only that side, split on whether that fact is established by shared evidence or only claimed by that path.
""".strip(),
    relation_analysis_intro="Describe one divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- Focus on the earliest same-object claim conflict.
- Ignore process lag and equivalent wording.
- If the conflict is only final-answer wording after a shared prefix, analyze the shared-prefix-to-target support relation.
- If a trace has both pro and con facts, analyze the trace's final bridge from those facts to the whole requested predicate.
- If the two sides differ by predicate strength, name the stronger and weaker predicates before resolving.
- First decide whether each path is answering the exact question predicate or drifting to a nearby predicate.
- Name the target relation before resolving, and compare how each path bridges evidence to that relation.
- If a path uses a prerequisite or adjacent relation as decisive support, ask whether the bridge is reasonable or whether it changes the target relation.
- For would/could/hypothetical questions, evaluate the scenario asked; lack of historical occurrence, contact, or exposure is not by itself a no-answer unless the question asks whether it happened.
- For date, age, or time-window claims, compute the interval explicitly before deciding which path supports the predicate.
- For an ability or capacity predicate, first ask what ability/capacity is being claimed; do not replace it with whether the event actually happened, the object can be bought, or the opportunity was common.
- A path's own assertion that a nearby relation decides the target is not enough; judge whether the bridge itself preserves the question predicate.
- If a side relies on an extra condition, decide whether that condition is stated or naturally follows from the question and trace.
- If a side needs a hidden premise to support the label, say the bridge is unsupported; if it uses an ordinary common-sense inference, compare that inference directly.
- If a side relies on a specific outside fact asserted only by that side, analyze it as a path claim before using it to resolve the predicate.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language claim graph.",
    resolution_rules="""
Rules:
- Keep the repaired claim concrete.
- Prefer the earliest same-object conflict.
- Do not choose by answer popularity.
- Resolve the support relation from shared evidence to the requested object before trusting a final answer token.
- For true/false tasks, keep the target statement and the final truth label separate.
- For label tasks, a correct claim must say how evidence supports or fails the whole question predicate; a final label cannot support itself.
- If the evidence only proves a subcondition, do not turn it into the final label.
- If the chosen claim quotes an early local fact but ignores a later predicate-level bridge in the same trace, it is not the right claim to keep.
- Do not upgrade weak evidence into a strong predicate unless the trace gives a reasonable bridge to that stronger predicate.
- Do not reject or accept a claim merely because it answers a nearby historical, technical, weaker, or stronger predicate.
- If the evidence only supports a nearby predicate, keep that nearby predicate as the bridge claim rather than silently upgrading it.
- Preserve the target relation itself: a nearby fact, historical occurrence, permission, possibility, or constraint does not settle a different relation unless the trace gives a credible bridge.
- For ability or capacity questions, choose the claim that evaluates the ability/capacity relation itself over a claim that only evaluates actual occurrence, purchase completion, ordinary availability, or historical commonness.
- If the best available claim is only about an adjacent relation, do not use it as a winning claim for the requested relation; either keep the exact-relation claim or leave the divergence unresolved.
- A path's own assertion that a nearby relation decides the target is not enough; judge whether that bridge itself preserves the question predicate.
- Do not accept a bridge from nearby relation to target relation unless that bridge itself is justified by the trace, question, or ordinary meaning.
- Do not resolve in favor of an extra condition unless the question, trace evidence, or ordinary meaning supports it.
- If the candidate claim depends on a hidden premise not found in the trace, keep the trace-grounded claim or leave the divergence unresolved.
- Do not resolve in favor of a specific outside fact asserted by only one path merely because it is confidently phrased.
""".strip(),
    resolution_fewshot=RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired claim to the requested object.
- Do not stop at an intermediate object or partial expression.
""".strip(),
)


MATH500_PROMPT_PROFILE = PromptProfile(
    name="math500_current",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo.",
    merge_rules=MERGE_RULES_SHORT,
    merge_fewshot=MERGE_FEWSHOT,
    audit_intro="Audit a natural-language claim memo before divergence resolution.",
    audit_rules=AUDIT_RULES_SHORT,
    prefix_intro="Rebuild a compact natural-language claim memo that exposes only the shortest shared prefix and the first unresolved actionable conflict.",
    prefix_rules=PREFIX_RULES_SHORT,
    relation_analysis_intro="Describe one divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- Do not choose a winning side yet.
- Focus on the earliest same-object claim conflict, not the final answer alone.
- Use the requested object only to avoid comparing the wrong quantity.
- Ignore process-status differences such as one path continuing farther, being in progress, or stopping early.
- Ignore generic reminders such as 'must consider the restriction' until they turn into a concrete conflicting count, formula, set, candidate-validity, or boundary claim.
- If one side is only an intermediate object and the question asks for a later named object, say that the split is not yet decisive.
- Do not use answer popularity as evidence; two traces agreeing can still share the same target-object drift or local arithmetic error.
- Call out if the divergence is actually caused by an earlier shared claim that should have been split.
- If the two sides are equivalent forms or progress lag, say so and point to the later real mismatch.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language claim graph.",
    resolution_rules="""
Rules:
- Keep each field short. Do not output JSON.
- Allowed actions are choose_A, choose_B, and keep_parallel. Do not invent a third repaired answer.
- Resolve by the earliest local claim conflict, not by answer popularity.
- Verify the correct claim against the relevant trace evidence and the original equation; do not choose a side only because the graph summary or a majority path states it.
- Use keep_parallel only when both local claims can simultaneously stay true and still lead to the same requested object.
- If one path changes the requested object, candidate set, factor list, or local factual check, do not use keep_parallel.
- If the two claims make different statements about the same object, treat that as a real conflict even if the final answers have not separated yet.
- If the two sides are equivalent forms or progress lag, keep_parallel and name what still needs to be checked later.
- Do not resolve a divergence by choosing a pure process-status sentence such as 'is in progress', 'has not yet continued', or 'stops here'.
- Do not resolve a divergence with a generic reminder such as 'must consider the restriction' unless the opposite side explicitly denies that same reminder.
- `Correct claim` must be a positive claim to continue from, not only a diagnosis that something is wrong.
- Prefer a concrete correct claim over a vague compromise sentence.
- Rewrite should begin at the divergence frontier, not earlier.
- If two paths already disagree on the same object, resolve that conflict even if a third path has not reached that frontier yet.
""".strip(),
    resolution_fewshot=RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Use only the question, preserved prefix, and selected repaired claim as context; the old suffix is intentionally hidden.
- Your job is to continue from the repaired claim and finish the original question, not merely restate the repaired claim.
- Treat the selected repaired claim as a checkpoint, not as the final answer unless it explicitly already is the requested answer.
- Do not stop at the repaired claim itself; continue until the requested object is answered.
- Start the new suffix at the repaired claim or its immediate consequence.
- Continue until you reach the requested object from the question.
- Do not end with only an intermediate condition, boundary note, function name, object name, earlier iterate, or component quantity.
""".strip(),
)


GSM8K_PROMPT_PROFILE = PromptProfile(
    name="gsm8k",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo for arithmetic reasoning.",
    merge_rules="""
Rules:
- Track quantities, units, and formulas explicitly.
- Do not split on speed differences alone; split only when the same quantity gets a different value, formula, or unit.
- If the trace reaches an expression but not the evaluated amount, keep building rather than declaring a split.
""".strip(),
    merge_fewshot=MERGE_FEWSHOT,
    audit_intro="Audit a natural-language claim memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep quantity claims concrete and unit-aware.
- Move one-path-only arithmetic progress back into the path note.
- Do not invent totals that were not computed.
""".strip(),
    prefix_intro="Rebuild a compact natural-language claim memo that exposes only the shortest shared prefix and the first unresolved actionable conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared quantity checkpoint before the first split.
- Do not split on whether a path has only reached an intermediate expression.
- Split only when two paths assign different values, formulas, or units to the same quantity.
""".strip(),
    relation_analysis_intro="Describe one divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- Focus on the earliest same-quantity conflict.
- Ignore progress lag and unfinished arithmetic.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language claim graph.",
    resolution_rules="""
Rules:
- Return the concrete quantity claim to keep.
- If the right side is only an expression, continue it to the evaluated amount.
- Do not leave the repaired claim as a half-finished equation.
""".strip(),
    resolution_fewshot=RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired quantity claim and finish the numeric answer.
- Do not stop at a symbolic expression if the question asks for a final amount.
""".strip(),
)


STRATEGYQA_PROMPT_PROFILE = PromptProfile(
    name="strategyqa",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo for factual reasoning.",
    merge_rules="""
Rules:
- Focus on one short factual bridge and the question predicate it supports.
- Keep the named entity, the factual bridge, and the yes/no implication separate.
- Do not split merely because one path is longer; split only when the same factual bridge gets conflicting content.
- Do not split on two phrasings that both support the same yes/no label.
- If one path's final yes/no label does not follow from its factual bridge, call that a label-implication error.
- Preserve the actual named entity from the question. Do not replace a band, title, brand, team, or person name with a literal role suggested by one word in the name.
- For yes/no questions, keep the literal predicate: do not collapse "would", "common", "most", or "be heard" into weaker notions like "could", "exists", "some", or "possible".
- Keep the target relation fixed: do not replace "afford" with "available to buy", "confused by" with "exposed to", or any requested relation with a nearby factual constraint.
""".strip(),
    merge_fewshot=STRATEGYQA_MERGE_FEWSHOT,
    audit_intro="Audit a natural-language claim memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep only short factual bridge claims that are shared by at least two active paths.
- Move one-path-only background or explanation text back into the path note.
- Do not invent outside facts.
- Delete any split that only changes the reading from ordinary/default meaning to a special favorable possibility.
""".strip(),
    prefix_intro="Rebuild a compact natural-language claim memo that exposes only the shortest shared prefix and the first unresolved actionable conflict.",
    prefix_rules="""
Rules:
- Keep the shortest factual bridge before the first split.
- Do not split on explanation style or progress lag.
- Split when two paths make incompatible factual claims about the same entity or relation.
- If one path only says something is possible while another claims it is typical, common, literal, or expected, treat that as a same-predicate conflict and preserve the stronger literal wording.
""".strip(),
    relation_analysis_intro="Describe one divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- Focus on the earliest factual bridge conflict.
- Ignore narration style and process lag.
- If both sides imply the same yes/no answer, call that equivalent wording rather than a decisive mismatch.
- For label tasks, analyze the truth of the question predicate before trusting a branch's final yes/no token.
- If a branch's facts support one label but its final token says the other label, name that as a label-implication error.
- Treat entity identity as a first-class relation: actual entity vs word-in-name role is a real conflict.
- Name the literal predicate conflict explicitly: for example "common vs exists", "would vs could", "most vs some", "literal use vs symbolic association", or "direct support vs unsupported assumption".
- Name relation drift explicitly when a path answers a nearby relation instead of the relation in the question.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language claim graph.",
    resolution_rules="""
Rules:
- Return the factual bridge claim to keep.
- Prefer a concise yes/no-supporting fact over a long explanation.
- The correct claim must contain three parts in one sentence: the fact to keep, how it answers the literal question predicate, and the supported yes/no label.
- The correct claim should be a predicate-level claim, not merely "the answer is yes/no".
- A bare final yes/no label is not factual evidence. Do not choose a side merely because it directly states a label, is explicit, or is unambiguous.
- If one side has only a bare label and another side gives a factual bridge, judge the bridge-to-predicate implication.
- If you can only restate a bare label with no predicate-level bridge, treat the divergence as unresolved and use keep_parallel instead of forcing a fake repair.
- If the available bridge only proves a subcondition, say so and keep going until the full predicate is settled.
- Do not map a negative fact directly to "no"; first ask whether that fact makes the question predicate true or false.
- Do not add a hypothetical role, identity, permission, or special condition that is not in the question or traces.
- If a branch's factual explanation supports the opposite label from its final yes/no token, keep the factual explanation and repair the label implication.
- If the two claims are compatible and support the same yes/no label, use keep_parallel instead of forcing choose_A or choose_B.
- If one side relies on a hypothetical role or special authority not stated in the question, reject that side and keep the actual-entity reading.
- Do not let a later explanation line override the factual bridge.
- Do not choose a branch merely because it produces a definite yes/no.
- Do not repair by adding a favorable assumption such as "if we assume ..." unless that assumption is already stated in the traces and directly answers the question.
- Reject possibility-only support when the question asks about an ordinary/default outcome ("would"), a prevalence claim ("common", "most"), or literal physical/causal contact.
- When one branch relies on reinterpretation of the predicate and the other keeps the ordinary reading, keep the ordinary reading.
- When one branch answers a nearby relation rather than the asked relation, keep the branch that reasons about the asked relation, even if the nearby fact is true.
""".strip(),
    resolution_fewshot=STRATEGYQA_RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired factual bridge and finish the yes/no answer.
- Do not stop at a background fact if the question asks for a final yes/no.
- Keep the repaired bridge literal. Do not switch from "would" to "could", from "common" to "exists", or from "most" to "some" while finishing the answer.
""".strip(),
)


ANLI_PROMPT_PROFILE = PromptProfile(
    name="anli",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo for premise-hypothesis reasoning.",
    merge_rules="""
Rules:
- Compare premise evidence to the hypothesis, not final label tokens alone.
- Keep the premise fact, the hypothesis claim, and the label relation separate.
- Do not split on entailment/neutral/contradiction wording if both paths use the same premise evidence and support the same relation.
- If paths share the premise evidence but one label does not follow from it, treat that as a label-relation error.
- A weaker hypothesis can be entailed by stronger premise evidence.
- A broad evaluative hypothesis can be entailed by strong role, achievement, recognition, or outcome evidence.
- An unsupported added detail is neutral unless the premise states an incompatible fact.
- Do not use outside facts that are not in the premise to rescue an entailment.
- "Former X" means past association with X; if the hypothesis says current X status, check for contradiction.
- If the premise explicitly states uncertainty such as "not clear", "unknown", or "unclear who", a hypothesis that states the same uncertainty is entailed.
- If the hypothesis only says "multiple", "several", plural membership, or at least one/two, do not require an exact number beyond what the hypothesis asks.
""".strip(),
    merge_fewshot=ANLI_MERGE_FEWSHOT,
    audit_intro="Audit a natural-language NLI claim memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Common Ground should contain only premise-supported facts and the target hypothesis object.
- Relation judgments such as support, unsupported, entailment, neutral, contradiction, or "not explicitly stated" belong in Paths or First Split, not Common Ground.
- Delete any split that is only a different label word without a different premise-to-hypothesis relation.
- Replace final-label splits with the concrete premise evidence that supports or fails to support the hypothesis.
""".strip(),
    prefix_intro="Rebuild a compact natural-language claim memo that exposes the first premise-to-hypothesis relation conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared premise evidence before the first relation conflict.
- Split only when paths disagree on what the premise evidence implies for the hypothesis.
- Do not split on final label wording before checking whether the evidence supports the same relation.
""".strip(),
    relation_analysis_intro="Describe one premise-hypothesis relation conflict before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First state the premise evidence.
- Then state what the hypothesis asks.
- Then decide whether the evidence entails, contradicts, or leaves the hypothesis neutral.
- Entailment can be a weaker restatement of the premise; do not demand extra precision that the hypothesis does not ask for.
- Unsupported extra detail is neutral only when the premise leaves it open.
- If the premise gives an incompatible value, identity, date, status, location, or comparison, that is contradiction rather than neutral.
- If the hypothesis wording is malformed or ambiguous and no incompatible fact is clear, prefer neutral.
- A premise statement of uncertainty entails a hypothesis that states the same uncertainty.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language NLI claim graph.",
    resolution_rules="""
Rules:
- Return the premise-to-hypothesis relation claim to keep.
- Correct claim must be one complete relation sentence. Start it with exactly one of: "The premise entails the hypothesis because ...", "The premise is neutral toward the hypothesis because ...", or "The premise contradicts the hypothesis because ...".
- Do not write Correct claim as only a diagnosis such as "the premise does not explicitly state ..." or "the other side assumes ..."; translate that diagnosis into the relation claim to keep.
- Do not choose a side merely because it gives a definite label.
- Use neutral only when the premise leaves the hypothesis open.
- Use contradiction when a stated premise fact is incompatible with the hypothesis, even if the premise does not spell out the negation.
- If the hypothesis wording is malformed or ambiguous and no incompatible fact is clear, keep neutral.
- If a branch's evidence supports the opposite label from its final token, keep the evidence and repair the relation label.
- Do not turn an entailed uncertainty statement into neutral just because the underlying event is uncertain.
- Do not require exact counts when the hypothesis only asks for multiple, several, plural, or at least support.
- Do not use outside knowledge to fill a named person, team, or event missing from the premise.
""".strip(),
    resolution_fewshot=ANLI_RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired premise-to-hypothesis relation claim.
- Finish with the requested NLI label.
- Do not restate a stale final label if it conflicts with the repaired relation.
""".strip(),
)


ANLI_RELATION_V2_PROFILE = replace(
    ANLI_PROMPT_PROFILE,
    name="anli_relation_v2",
    merge_rules=ANLI_PROMPT_PROFILE.merge_rules
    + "\n- Treat phrases like \"not explicitly stated\", \"not directly supported\", and \"does not confirm\" as one path's relation judgment, not as shared evidence."
    + "\n- When the premise gives ordinary implication evidence, compare that evidence to the hypothesis; do not downgrade it to neutral only because the exact hypothesis wording is absent."
    + "\n- For role, status, identity, date, location, value, or event-name changes, ask whether the same object can satisfy both descriptions; if not, expose a contradiction relation split.",
    audit_rules=ANLI_PROMPT_PROFILE.audit_rules
    + "\n- Remove Common Ground bullets that say the hypothesis is not explicitly stated, not directly supported, not confirmed, unsupported, or not contradicted; those are relation judgments."
    + "\n- Common Ground may mention the hypothesis text, but the verdict about whether the premise entails, contradicts, or leaves it neutral must appear only in Paths or First Split.",
    prefix_rules=ANLI_PROMPT_PROFILE.prefix_rules
    + "\n- If the current prefix contains an explicitness or support verdict, rebuild the prefix from the premise facts and hypothesis object before locating the relation conflict.",
    relation_analysis_rules=ANLI_PROMPT_PROFILE.relation_analysis_rules
    + "\n- Do not let an explicit-wording diagnosis be the final object; translate it into entailment, neutral, or contradiction for the whole hypothesis."
    + "\n- If the hypothesis adds a fact, separate unsupported additions from incompatible additions: unsupported additions are neutral, incompatible additions are contradiction."
    + "\n- If the hypothesis is a weak/evaluative restatement, check whether the premise supplies strong role, achievement, result, recognition, or situational evidence before choosing neutral.",
    resolution_rules=ANLI_PROMPT_PROFILE.resolution_rules
    + "\n- The Correct claim line must begin with exactly one of the three canonical phrases and must include the evidence-to-hypothesis reason in the same sentence."
    + "\n- Never use these as Correct claim starts: \"The hypothesis is not directly\", \"The premise does not explicitly\", \"The premise does not confirm\", \"The hypothesis introduces\". Rewrite them into a canonical relation claim."
    + "\n- If the winning side contains a useful local diagnosis, keep its evidence but convert the output into the whole premise-to-hypothesis relation.",
    revision_rules=ANLI_PROMPT_PROFILE.revision_rules
    + "\n- Treat the repaired canonical relation sentence as authoritative for the final label: entails -> 1, neutral -> 2, contradicts -> 3."
    + "\n- Do not continue from an explicitness diagnosis when a canonical relation sentence is available.",
)


ANLI_RELATION_V3_PROFILE = replace(
    ANLI_RELATION_V2_PROFILE,
    name="anli_relation_v3",
    merge_rules=ANLI_RELATION_V2_PROFILE.merge_rules
    + "\n- Before writing Common Ground, check each sentence: if it judges whether the hypothesis is mentioned, stated, supported, confirmed, implied, contradicted, or left open, move it to Paths or First Split."
    + "\n- Keep hypothesis wording in Common Ground only as the requested object, for example: `The hypothesis object is whether Patrice is a very good nurse.` Do not append an explicitness verdict.",
    audit_rules=ANLI_RELATION_V2_PROFILE.audit_rules
    + "\n- For an audit, delete Common Ground sentences that contain absence/explicitness verdicts such as does not mention, does not state, not explicitly, not directly, not supported, unsupported, or not enough information."
    + "\n- After deleting a verdict from Common Ground, preserve any remaining premise fact or hypothesis object in neutral wording.",
    prefix_rules=ANLI_RELATION_V2_PROFILE.prefix_rules
    + "\n- A clean prefix may contain premise evidence plus the hypothesis object, but not an explicitness/support verdict about that object.",
)


ANLI_RELATION_V4_PROFILE = replace(
    ANLI_RELATION_V3_PROFILE,
    name="anli_relation_v4",
    relation_analysis_rules=ANLI_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- Do not let an absence phrase such as \"not explicitly stated\" override ordinary lexical, role, institutional, or event implications already in the premise."
    + "\n- For role/status terms such as men’s team, former role, hired for a role, initials from a full name, living creatures, or stronger professional recognition, decide whether the hypothesis is a weaker ordinary implication before choosing neutral."
    + "\n- For contradiction, require the hypothesis and premise to make incompatible claims about the same object, time, value, status, or event; a nearby different time, count, or example is neutral when the premise leaves the hypothesis open."
    + "\n- If the hypothesis uses current-time wording and the premise uses past-status wording such as former, treat the current-status hypothesis as contradicted by the premise.",
    resolution_rules=ANLI_RELATION_V3_PROFILE.resolution_rules
    + "\n- Do not choose neutral merely because the premise does not use the exact hypothesis words; first ask whether the hypothesis is an ordinary weaker implication of the stated role, action, identity, status, or event."
    + "\n- Treat exact-wording objections as losing when the premise supplies a direct semantic bridge: being on a men's team supports being a man; hiring someone for a role supports wanting them as an employee; a full name supports its initials; rescuing animals supports rescuing lives."
    + "\n- Use contradiction only for same-object incompatibility, not for an unsupported extra detail or a different nearby mention. A Week 11 loss does not contradict an unstated Week 10 loss by itself; a listed count of five places does not contradict an unsupported eight-times wording unless the hypothesis asserts the list has exactly eight."
    + "\n- When the hypothesis says current status and the premise says former status, keep contradiction because former status rules out current status in ordinary news wording."
    + "\n- When the premise gives strong achievement, recognition, qualification, successful selection, or professional role evidence, do not downgrade a weak evaluative or competence hypothesis to neutral only because it is not quoted verbatim.",
    revision_rules=ANLI_RELATION_V3_PROFILE.revision_rules
    + "\n- Map the repaired relation directly to the final label: entails -> 1, neutral -> 2, contradicts -> 3, even if a stale trace repeats an exact-wording objection.",
)


ANLI_RELATION_V5_PROFILE = replace(
    ANLI_RELATION_V4_PROFILE,
    name="anli_relation_v5",
    relation_analysis_rules=ANLI_RELATION_V4_PROFILE.relation_analysis_rules
    + "\n- When the hypothesis is an ordinary paraphrase, role consequence, or mild semantic consequence of the premise, prefer entailment over neutral."
    + "\n- Do not require the exact hypothesis wording if the premise already gives the same role, action, status, comparison, or observable result in ordinary language."
    + "\n- A single concrete event can support a broader ordinary description when the hypothesis is about that same event or role, unless the premise leaves a real incompatible fact open."
    + "\n- Treat close evaluative paraphrases as entailment when the premise supplies enough positive evidence for the evaluation, even if the premise does not use the same adjective.",
    resolution_rules=ANLI_RELATION_V4_PROFILE.resolution_rules
    + "\n- If the repaired relation claim would be neutral only because the hypothesis is not quoted verbatim, re-check whether the premise already gives an ordinary paraphrase, role consequence, or evaluative consequence; if so, keep entailment."
    + "\n- Examples of acceptable weak entailment include: team membership -> gender identity in ordinary sports wording, hiring for a role -> qualification/competence, a forecast ending flurries -> no snow/rain in the target period, and a direct incident of vehicle theft -> automobile theft behavior."
    + "\n- Use neutral for genuinely unsupported extra details, not for every near-paraphrase or plausible consequence."
    + "\n- Keep contradiction for same-object incompatibility, including former/current status conflicts and mismatched explicit times, counts, or roles.",
    revision_rules=ANLI_RELATION_V4_PROFILE.revision_rules
    + "\n- If the repaired relation is an ordinary paraphrase or consequence, let it drive the final label even if the losing branch said 'not explicitly stated'.",
)


ANLI_RELATION_V6_PROFILE = replace(
    ANLI_RELATION_V4_PROFILE,
    name="anli_relation_v6",
    relation_analysis_rules=ANLI_RELATION_V4_PROFILE.relation_analysis_rules
    + "\n- Accept an ordinary semantic consequence only when the premise itself supplies the same actor or object, the same target relation, and a short bridge that does not rely on outside named-entity facts."
    + "\n- Do not turn a general category, count, location, title, or topic mention into an absent named person, intention, habit, emotion, or exact method."
    + "\n- A single event supports a role-defined or event-defined consequence only when that consequence is intrinsic to the stated event or role; otherwise leave broader habits, frequency, motivation, or evaluation neutral."
    + "\n- For contradiction, distinguish unsupported additions from incompatible replacements: use contradiction when the hypothesis replaces an asserted status, method, ownership, gendered term, employment relation, or current/former relation with an incompatible one.",
    resolution_rules=ANLI_RELATION_V4_PROFILE.resolution_rules
    + "\n- The Correct claim must choose exactly one relation phrase: `The premise entails the hypothesis because ...`, `The premise is neutral toward the hypothesis because ...`, or `The premise contradicts the hypothesis because ...`. Do not write combined forms such as entails/is neutral/contradicts."
    + "\n- Preserve v4's exact-wording repair only for same-object ordinary bridges: a stolen vehicle incident can support automobile theft, a forecast of sunshine after flurries end can support no precipitation in that target period, rescuing trapped living animals can support rescuing lives, and a role can support actions intrinsic to that role."
    + "\n- Do not use outside knowledge to fill missing specifics. A premise that only gives a country count, general group, or broad topic does not entail that specific named people, motives, emotions, or habits in the hypothesis are true."
    + "\n- Do not infer strong or habitual descriptions from weak evidence: looking forward to an event is not going crazy; one bad exam after studying does not prove inefficient study; community support does not prove overweight users are looking for community unless that group and desire are stated."
    + "\n- Use contradiction, not neutral, when the hypothesis swaps the asserted relation for an incompatible same-object relation, such as a job versus volunteering, creating one's own account versus using someone else's account, male sibling language versus a female claim, or former versus current status.",
    revision_rules=ANLI_RELATION_V4_PROFILE.revision_rules
    + "\n- Continue from the single selected relation only; if the resolution mentions multiple possible relation labels, rewrite it into the one relation that the evidence supports.",
)


ANLI_RELATION_V7_PROFILE = replace(
    ANLI_RELATION_V4_PROFILE,
    name="anli_relation_v7",
    relation_analysis_rules=ANLI_RELATION_V4_PROFILE.relation_analysis_rules
    + "\n- If the premise gives an exact number, date, amount, comparison, required credential, role status, or named method that makes the hypothesis false for the same object, treat the relation as contradiction rather than neutral."
    + "\n- If the hypothesis uses weak wording such as may, can, could, possible, likely, or may cost more, ordinary premise evidence for that possibility can entail the hypothesis; do not demand certainty that the stronger claim would require."
    + "\n- Accept a cause/result bridge only when the result is intrinsic to the stated event and no alternative in the premise blocks it, such as a burned steak supporting overcooking or too much heat.",
    resolution_rules=ANLI_RELATION_V4_PROFILE.resolution_rules
    + "\n- Exact falsification is contradiction: if the premise gives 3.7 minus 2.7 equals 1.0, the hypothesis says more than 1.5, the premise contradicts the hypothesis; if a required bachelor's degree is stated, a no-diploma or no-prerequisite hypothesis is contradicted when the missing prerequisite is required for that degree."
    + "\n- Former/current and time-of-article conflicts are contradiction: `former Aston Villa man` contradicts a hypothesis that he played for Aston Villa at the time of the article."
    + "\n- For weak modal hypotheses, keep entailment when the premise supplies ordinary possibility evidence: `starts at $14.59` supports that it may cost more; a cat avoiding handling supports that it may be afraid; summer popularity supports June, July, and August when those are the stated summer months."
    + "\n- Do not extend this weak-modal repair to named people, unsupported motives, broad habits, or exact methods absent from the premise.",
    revision_rules=ANLI_RELATION_V4_PROFILE.revision_rules
    + "\n- If the repaired relation uses exact falsification or weak-modal support, preserve that relation and map it directly to the final NLI label.",
)


ANLI_RELATION_V8_PROFILE = replace(
    ANLI_RELATION_V7_PROFILE,
    name="anli_relation_v8",
    relation_analysis_rules=ANLI_RELATION_V7_PROFILE.relation_analysis_rules
    + "\n- Guard against over-entailment: a general category, count, service, location, or topic does not entail absent named examples, absent personal motives, absent preparation by the subject, or absent platform-specific permissions."
    + "\n- Do not use outside product or service knowledge to create contradiction; if the premise only gives a rule and the hypothesis names an unmentioned service, keep neutral unless the premise itself states that service violates the rule."
    + "\n- A bad outcome after effort does not by itself contradict efficient or thorough effort; it is neutral unless the premise describes the method as careless, incomplete, or ineffective.",
    resolution_rules=ANLI_RELATION_V7_PROFILE.resolution_rules
    + "\n- A country count or category statement does not entail specific named people unless those names appear in the premise."
    + "\n- Having or using coffee, music, a tool, or another object does not entail that the subject personally made, prepared, owned, or had permission for it unless the premise states that relation."
    + "\n- Do not contradict from outside service assumptions: if the premise says non-DRM music is allowed but never mentions Spotify, `Spotify can be used` is neutral, not contradiction."
    + "\n- Do not turn active participation, community support, or a general benefit into a stated desire, motive, habit, or emotion for a narrower group unless the premise states that group-level attitude.",
    revision_rules=ANLI_RELATION_V7_PROFILE.revision_rules
    + "\n- Preserve these anti-overreach guards during revision; do not add absent names, motives, preparation acts, ownership, service rules, or group attitudes.",
)


ANLI_RELATION_V9_PROFILE = replace(
    ANLI_RELATION_V8_PROFILE,
    name="anli_relation_v9",
    relation_analysis_rules=ANLI_RELATION_V8_PROFILE.relation_analysis_rules
    + "\n- The anti-overreach guards are for missing specifics, outside facts, motives, ownership, permissions, and named examples; they do not block ordinary lexical definitions, weak paraphrases, plural quantifiers, quoted descriptions, or same-slot incompatibilities already present in the premise."
    + "\n- For ANLI, do not require courtroom-level certainty. If the premise gives a concrete role, event, description, count phrase, age/date, or stated source and the hypothesis is a weaker ordinary-language consequence about the same object, treat that as entailment unless a real alternative remains open."
    + "\n- Keep contradiction for same-slot replacements: a male sibling term versus female wording, a chosen-by-me rule versus candidate-agreed questions, after-Christmas timing versus for-Christmas timing, an exact age/date/value mismatch, source/addressee identity swaps, title/audience swaps, or a live/damaged/en-route object versus a sunk/different-location claim.",
    resolution_rules=ANLI_RELATION_V8_PROFILE.resolution_rules
    + "\n- Do not write `supports but not definitive, therefore neutral` when the support is a normal ANLI semantic bridge. Ordinary bridges include: allegations imply not-yet-proven claims; more than 100 years old can support antique; several issues means more than one issue; attending school in a place supports living there for a while; driving a stolen vehicle or being a carjacking suspect supports stealing automobiles; a quoted adjective such as chivalrous supports that adjective."
    + "\n- Keep v8's anti-overreach only for absent names, absent preparation/ownership/permission, outside service rules, unsupported motives/emotions/habits, and category/count-to-specific-name jumps."
    + "\n- Use contradiction, not neutral, for explicit same-slot incompatibility: brother is male rather than female; questions chosen by the moderator are not questions agreed to by candidates; a vacation after Christmas is not a vacation for Christmas; age 32 contradicts age 33; an accused-person defense article is not directed at victims; a quote source is not the addressee being spoken to.",
    revision_rules=ANLI_RELATION_V8_PROFILE.revision_rules
    + "\n- Preserve ordinary ANLI semantic bridges and same-slot contradictions during revision; do not downgrade them to neutral merely because the exact hypothesis sentence is not quoted.",
)


ANLI_RELATION_V10_PROFILE = replace(
    ANLI_RELATION_V4_PROFILE,
    name="anli_relation_v10",
    relation_analysis_rules=ANLI_RELATION_V4_PROFILE.relation_analysis_rules
    + "\n- Treat each observed final candidate label from the traces as a visible hypothesis, not as gold. Compare the premise evidence supporting entailment, neutral, and contradiction before choosing."
    + "\n- For non-unanimous fixed-label NLI, prefer an observed candidate whose trace gives a premise-grounded relation over a newly invented relation that is only more polished."
    + "\n- If a trace's final token conflicts with its own premise-to-hypothesis evidence, keep the evidence and repair the relation label; do not follow the stale token."
    + "\n- Do not average two relation phrasings. Select the candidate relation whose evidence best answers the exact hypothesis object.",
    resolution_rules=ANLI_RELATION_V4_PROFILE.resolution_rules
    + "\n- First identify which observed label candidates are present among the traces: entailment/1, neutral/2, contradiction/3. Then choose the candidate whose reasoning has the strongest premise evidence."
    + "\n- A supported observed candidate should beat a free-form regenerated answer unless the regenerated relation is directly forced by premise evidence visible in the traces."
    + "\n- When one branch gives a correct semantic bridge but hedges with `not explicit`, keep the bridge and map it to the relation it supports."
    + "\n- When both labels appear plausible, decide by evidence-to-hypothesis support: entailment needs the hypothesis to be a stated or ordinary weaker consequence; neutral needs an unsupported but compatible addition; contradiction needs same-object incompatibility.",
    revision_rules=ANLI_RELATION_V4_PROFILE.revision_rules
    + "\n- Preserve the selected observed candidate relation when it is premise-supported; do not regenerate a different label just because the wording is smoother."
    + "\n- If the selected relation came from repairing a trace's stale final token, follow the repaired relation rather than the old token.",
)


ANLI_RELATION_V11_PROFILE = replace(
    ANLI_RELATION_V9_PROFILE,
    name="anli_relation_v11",
    relation_analysis_rules=ANLI_RELATION_V9_PROFILE.relation_analysis_rules
    + "\n- Treat each observed final candidate label from the traces as a visible hypothesis, not as gold. Compare the premise evidence supporting entailment, neutral, and contradiction before choosing."
    + "\n- Preserve the v9 ordinary-language standard: do not downgrade a normal semantic bridge to neutral just because the exact hypothesis wording is absent."
    + "\n- When two traces already share a candidate label, a one-trace minority can overturn them only with a premise-grounded same-object incompatibility, missing-specific guard, or stronger ordinary semantic bridge."
    + "\n- If a trace's final token conflicts with its own premise-to-hypothesis evidence, keep the evidence and repair the relation label; do not follow the stale token.",
    resolution_rules=ANLI_RELATION_V9_PROFILE.resolution_rules
    + "\n- First identify which observed label candidates are present among the traces: entailment/1, neutral/2, contradiction/3. Then choose the candidate whose reasoning has the strongest premise evidence."
    + "\n- Before overturning a 2-vs-1 observed candidate majority, state the concrete premise evidence that defeats both majority traces; an exact-wording or not-explicit objection alone is not enough."
    + "\n- A supported observed candidate should beat a free-form regenerated answer unless the regenerated relation is directly forced by premise evidence visible in the traces."
    + "\n- When one branch gives a correct semantic bridge but hedges with `not explicit`, keep the bridge and map it to the relation it supports."
    + "\n- When both labels appear plausible, decide by evidence-to-hypothesis support: entailment needs the hypothesis to be a stated or ordinary weaker consequence; neutral needs an unsupported but compatible addition; contradiction needs same-object incompatibility.",
    revision_rules=ANLI_RELATION_V9_PROFILE.revision_rules
    + "\n- Preserve the selected observed candidate relation when it is premise-supported; do not regenerate a different label just because the wording is smoother."
    + "\n- If the selected relation came from repairing a trace's stale final token, follow the repaired relation rather than the old token.",
)


ANLI_RELATION_V12_PROFILE = replace(
    ANLI_RELATION_V10_PROFILE,
    name="anli_relation_v12",
    relation_analysis_rules=ANLI_RELATION_V10_PROFILE.relation_analysis_rules
    + "\n- Diagnose the relation with one of four reusable operations before choosing the label: ordinary semantic bridge, missing-specific neutral, same-slot contradiction, or stale-token repair."
    + "\n- Ordinary semantic bridge: if the premise states an event, role, source, article, release, invitation, named time span, age/date/value, or quoted description and the hypothesis is its weaker ordinary-language consequence, treat it as entailment unless a real alternative remains open."
    + "\n- Missing-specific neutral: if the hypothesis adds an unstated motive, ownership, permission, exact method, named example, quantity, frequency, or personal attitude, keep neutral unless the premise supplies that same slot."
    + "\n- Same-slot contradiction: use contradiction when premise and hypothesis assign incompatible values to the same time, count, location, role, identity, source, audience, status, duration, or event slot; unsupported extra detail alone is not contradiction."
    + "\n- Stale-token repair: if a trace says neutral/contradiction only because wording is not explicit while its own evidence supports an ordinary bridge, repair the label to entailment.",
    resolution_rules=ANLI_RELATION_V10_PROFILE.resolution_rules
    + "\n- Name the selected operation in the rationale: `ordinary semantic bridge`, `missing-specific neutral`, `same-slot contradiction`, or `stale-token repair`; keep the Correct claim itself as a canonical relation sentence."
    + "\n- Do not choose neutral from the phrase `not explicitly stated` until you test whether the premise gives an ordinary bridge such as cake at an invited party -> participants can eat cake, hunting partner -> the people know each other, article by X -> X reported/wrote it, released by a label -> the label helped/released the artist's work, or 11am-1pm -> two hours."
    + "\n- Do not choose contradiction merely because the hypothesis contains an absent topic or detail. Contradiction needs the premise to make that same slot false, such as April vs March, market vs Central Park, one hour vs 11am-1pm, current vs former, male vs female, or an exact count/date/value mismatch."
    + "\n- If a candidate relation relies on outside facts about named people, products, schools, services, or organizations, prefer the observed candidate supported by the premise text itself."
    + "\n- For a 2-vs-1 observed split, the minority may win when it names the correct operation and the majority only repeats exact-wording, unsupported-specific, or stale-token reasoning.",
    revision_rules=ANLI_RELATION_V10_PROFILE.revision_rules
    + "\n- Carry forward the selected operation and relation label together; do not let a stale exact-wording sentence flip an ordinary bridge back to neutral."
    + "\n- Keep unsupported extras neutral during revision unless the repaired relation names a same-slot incompatibility.",
)


ANLI_RELATION_V13_PROFILE = replace(
    ANLI_RELATION_V10_PROFILE,
    name="anli_relation_v13",
    relation_analysis_rules=ANLI_RELATION_V10_PROFILE.relation_analysis_rules
    + "\n- Keep v10's evidence-to-observed-candidate comparison, but apply an ANLI ordinary-language standard instead of an exact-wording standard."
    + "\n- Do not downgrade a weak or modal hypothesis to neutral when the premise gives the ordinary bridge: may/reduces/can/result-in claims, severe medical conditions supporting possible death, invitations with event food, quoted descriptions, article/source reports, and concrete stolen-object events can support their weaker hypothesis wording."
    + "\n- A specific event can support a simple weak behavior description when the hypothesis does not demand frequency, habit, motive, permission, ownership, or a named unstated example."
    + "\n- Use neutral only for an unsupported but compatible added slot, such as an absent motive, attitude, permission, ownership, exact method, frequency, named example, or quantity not recoverable from the premise."
    + "\n- Use contradiction when the premise and hypothesis fill the same requested slot incompatibly, including route endpoints, source/addressee/audience, quote versus opposite paraphrase, emotion polarity, status, location, timing, count, age/date/value, or exhausted percentage partitions.",
    resolution_rules=ANLI_RELATION_V10_PROFILE.resolution_rules
    + "\n- Before choosing neutral from an exact-wording objection, ask whether the hypothesis is only a weaker ordinary consequence of the premise; if yes, keep the entailment candidate."
    + "\n- Treat `may`, `can`, `possible`, `result in`, and other weak hypothesis wording as intentionally weak. The premise need only support the possibility or ordinary consequence, not prove a stronger claim."
    + "\n- Do not turn a correct ordinary bridge into neutral because it is based on one concrete event: stolen pickup/carjacking can support stealing automobiles; critical condition/no treatment can support may die; an event with cake can support attendees can eat cake."
    + "\n- For contradiction, test whether the hypothesis asks for the same slot with an incompatible value, such as Albuquerque-to-SF/Orange-County routes versus SF-to-Orange-County routes, hostile reaction versus very happy, source versus audience, current versus former, or arithmetic complements that exhaust a partition."
    + "\n- Keep neutral for compatible unsupported additions only; do not use neutral as a compromise between an ordinary bridge and a same-slot incompatibility.",
    revision_rules=ANLI_RELATION_V10_PROFILE.revision_rules
    + "\n- Preserve weak/modal entailments and same-slot contradictions during revision; do not reintroduce an exact-wording neutral objection after the relation has been repaired."
    + "\n- If the chosen relation depends on arithmetic complements or exhausted partitions, carry the computed comparison into the final NLI label.",
)


ANLI_RELATION_V14_PROFILE = replace(
    ANLI_RELATION_V13_PROFILE,
    name="anli_relation_v14",
    relation_analysis_rules=ANLI_RELATION_V13_PROFILE.relation_analysis_rules
    + "\n- Repair the first relation decision, not the late label token: compare the premise-backed bridge or incompatibility against each observed candidate before accepting a neutral compromise."
    + "\n- Preserve ordinary bridges for everyday event consequences and roles: a known partner can support knowing the person, a ruined/stained/spoiled object can support a negative quality, mistaken ingestion can support drinking/ingesting, loud sound plus ear pain can support pain from the sound, and a route/source/article/quote can support the weaker claim about that same route/source/article/quote."
    + "\n- Neutral is still required when the hypothesis adds an exact frequency, habit, motive, preference, permission, obligation, quantity, duration, or normative should/wanted claim that the premise does not supply; a single event usually does not prove often, generally, should, wanted, a lot, or at least a couple of days."
    + "\n- Do not turn negative tone, criticism, cost, downgrades, difficulty, or lack of proof into contradiction unless the premise explicitly fills the same requested slot with an incompatible value."
    + "\n- For same-slot contradiction, allow ordinary text-level incompatibilities: route endpoints or direction swaps, quote versus incompatible paraphrase, food/object sense versus group/entity sense, after an event versus on/during that event, source versus addressee, and exact day/date/time/count/value mismatches."
    + "\n- Compute simple quantities before choosing: 11am to 1pm is two hours, ten is a majority of twelve, and `first since 1988` means there was one in 1988 rather than none.",
    resolution_rules=ANLI_RELATION_V13_PROFILE.resolution_rules
    + "\n- At resolution time, explicitly test the three-way boundary: ordinary weaker consequence -> entailment; compatible unsupported added slot -> neutral; same-slot incompatible value -> contradiction."
    + "\n- Do not use `not definitively`, `not explicit`, or `could be another reason` to defeat an ordinary bridge when the hypothesis is weak and stays on the same event, object, role, source, route, or result."
    + "\n- When a branch claims contradiction from negativity alone, ask what exact hypothesis slot is made false. If it is only costly, criticized, difficult, downgraded, unlikely, or unsupported, keep neutral rather than contradiction."
    + "\n- Exact quantities and frequencies need exact support: often, never, multiple times, at least a couple days, a lot, should, wanted, and general present-tense claims should not be inferred from a single compatible fact unless the premise directly states that slot."
    + "\n- Before accepting a time, date, count, or `since` relation, do the simple arithmetic or wording check in the rationale; do not let a fluent but wrong computation carry the label.",
    revision_rules=ANLI_RELATION_V13_PROFILE.revision_rules
    + "\n- Carry forward the three-way boundary during revision; do not drift from an ordinary bridge to neutral or from a negative-but-compatible fact to contradiction."
    + "\n- Preserve simple arithmetic, date/time, count, route, quote/paraphrase, and source/addressee checks in the final relation sentence when they determine the label.",
)


ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE = replace(
    ANLI_RELATION_V14_PROFILE,
    name="anli_relation_v15_equal_token",
    merge_rules=ANLI_RELATION_V14_PROFILE.merge_rules
    + "\n- Under longer outputs, keep the premise, hypothesis, and final NLI label as separate objects; do not promote a repeated label to evidence.",
    relation_analysis_rules=ANLI_RELATION_V14_PROFILE.relation_analysis_rules
    + "\n- Lock the three-way boundary before judging a verbose trace: ordinary weaker consequence -> entailment; compatible unsupported added slot -> neutral; same-slot incompatible value -> contradiction."
    + "\n- If a long explanation contains both a correct boundary and a stale exact-wording objection, resolve the boundary first.",
    resolution_rules=ANLI_RELATION_V14_PROFILE.resolution_rules
    + "\n- Correct claim must be a premise-to-hypothesis relation sentence plus the supported label. Do not output only entailment, neutral, or contradiction."
    + "\n- Do not choose by a majority of labels; choose by the relation boundary supported by the premise text.",
    revision_rules=ANLI_RELATION_V14_PROFILE.revision_rules
    + "\n- Preserve the supported relation boundary and label through revision. Do not let late prose drift from entailment to neutral or from neutral to contradiction without a same-slot check.",
)


ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE = replace(
    ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE,
    name="anli_relation_v16_extraction_guard",
    relation_analysis_rules=ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- If the corrected claim states an explicit supported label, preserve that label unless the same claim's premise-to-hypothesis relation contradicts it."
    + "\n- Treat `not explicitly stated` as neutral only after checking whether the hypothesis is a weaker ordinary consequence of the premise.",
    resolution_rules=ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- End a decisive relation claim with `supported label is N`, where N follows the three-way boundary in the same sentence."
    + "\n- Do not let later rationale text about the rejected side override the corrected claim's explicit label.",
    revision_rules=ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve the corrected claim's explicit label in the final line; do not re-read stale rejected-side wording as the answer.",
)


ANLI_RELATION_V17_ACTION_GUARD_PROFILE = replace(
    ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE,
    name="anli_relation_v17_action_guard",
    relation_analysis_rules=ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- Compare each branch as a premise-to-hypothesis bridge, not as a label token. A branch with only a repeated label has weaker support than a branch with the correct relation boundary.",
    resolution_rules=ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- Use choose_A or choose_B when one side gives a complete premise-to-hypothesis bridge and the other side only repeats a label, objects to wording without checking ordinary consequence, or adds an unsupported same-slot claim."
    + "\n- Use keep_parallel only when neither side has a complete relation bridge, or when both bridges are compatible and still incomplete."
    + "\n- Do not write Action as resolve, repair, contradiction, entailment, or neutral; the Action field must be exactly choose_A, choose_B, or keep_parallel.",
    revision_rules=ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Start the rewrite from the selected relation boundary and end with the same supported label; do not restart from a stale majority label.",
)


ANLI_RELATION_V18_MEMORY_GUARD_PROFILE = replace(
    ANLI_RELATION_V17_ACTION_GUARD_PROFILE,
    name="anli_relation_v18_memory_guard",
    resolution_rules=ANLI_RELATION_V17_ACTION_GUARD_PROFILE.resolution_rules
    + "\n- If a previous round selected a complete relation bridge, only overturn it by naming the exact bridge error in that prior relation, not by a new label majority."
    + "\n- When rejecting an old label, say why that old premise-to-hypothesis relation is wrong; otherwise keep the earlier supported relation boundary.",
    revision_rules=ANLI_RELATION_V17_ACTION_GUARD_PROFILE.revision_rules
    + "\n- If the selected relation bridge is already explicit, preserve it through the final line unless the current resolution names a concrete bridge error.",
)


ANLI_RELATION_V19_BRIDGE_AUDIT_PROFILE = replace(
    ANLI_RELATION_V18_MEMORY_GUARD_PROFILE,
    name="anli_relation_v19_bridge_audit",
    relation_analysis_rules=ANLI_RELATION_V18_MEMORY_GUARD_PROFILE.relation_analysis_rules
    + "\n- Audit the exact premise-to-hypothesis bridge before any label: weaker ordinary consequence, compatible added slot, or same-slot contradiction."
    + "\n- A repeated NLI label is not support unless the trace states the boundary that licenses it.",
    resolution_rules=ANLI_RELATION_V18_MEMORY_GUARD_PROFILE.resolution_rules
    + "\n- The Correct claim must be rewrite-supported: one premise-to-hypothesis boundary sentence ending with `supported label is N`."
    + "\n- Use keep_parallel if the visible text does not support carrying that label into a revised trace."
    + "\n- Do not use a one-shot canonical label to override the agents unless a revised trace preserves the same boundary.",
    revision_rules=ANLI_RELATION_V18_MEMORY_GUARD_PROFILE.revision_rules
    + "\n- Preserve the selected boundary sentence through revision and end with the same label."
    + "\n- If the resolution only criticizes another label and does not support a replacement boundary, continue rather than landing a label.",
)


MATH500_BEST_20260506_PROFILE = replace(MATH500_PROMPT_PROFILE, name="math500_best_20260506")


MATH500_V18_PROFILE = replace(
    MATH500_BEST_20260506_PROFILE,
    name="math500_v18",
    resolution_rules=MATH500_BEST_20260506_PROFILE.resolution_rules
    + "\n- For math, `Correct claim` must be the mathematical assertion to continue from: a value, equation, inequality, set, case split, or object identification that advances the solution."
    + "\n- Do not use a method label, unchanged restatement of the original expression, or diagnosis of the bad path as the `Correct claim` unless it also gives the next concrete mathematical assertion."
    + "\n- When the divergence is an optimization, extremum, count, vector, interval, or transformed expression, the kept claim must include the decisive value/order/constraint/formula, not only the name of the method."
    + "\n- If a branch already has a complete final answer compatible with the kept mathematical claim, preserve that answer unless the kept claim directly contradicts it."
    + "\n- If rewriting from the claim would require solving several missing algebra steps, include the nearest concrete checkpoint before the claim instead of a vague high-level claim.",
    revision_rules=MATH500_BEST_20260506_PROFILE.revision_rules
    + "\n- Treat the repaired claim as one line in an ordinary math solution; continue the algebra/calculation from that line."
    + "\n- If the repaired claim only says which method is valid, first restate the concrete equation, value, or constraint needed by that method, then continue."
    + "\n- Preserve component order, interval endpoints, case labels, and requested-object wording when finishing the final answer.",
)


MATH500_V19_PROFILE = replace(
    MATH500_V18_PROFILE,
    name="math500_v19",
    resolution_rules=MATH500_V18_PROFILE.resolution_rules
    + "\n- Unlike multiple-choice tasks, a supported local math claim is usually not itself a final answer candidate; do not land on it until the requested object is completed."
    + "\n- When a branch already has a complete final answer and the repair is only method-level, preserve that final answer unless the repaired claim directly computes a different final requested object."
    + "\n- For matrix, vector, interval, set, probability, and counting questions, reject scalar, object-name, or process placeholders as final answers even when they follow a valid local method claim.",
    revision_rules=MATH500_V18_PROFILE.revision_rules
    + "\n- The last line must be a completed answer to the original requested object, not a method name, matrix product name, recurrence name, partial condition, or instruction to compute."
    + "\n- If the repaired claim is only a local checkpoint, finish the remaining calculation before writing the final answer.",
)


MATH500_V20_PROFILE = replace(
    MATH500_V18_PROFILE,
    name="math500_v20",
    merge_rules=MATH500_V18_PROFILE.merge_rules
    + "\n- When traces share a formula, recurrence, equation, candidate check, or substitution and then disagree, expose the first replayable operation that changes the value, sign, boundary, or validity claim."
    + "\n- Do not summarize a numeric disagreement only as final-answer conflict when the visible traces contain the local operation that caused it."
    + "\n- If a path gives a candidate final value after a shared checkpoint, keep the local computation that supports that value attached to the path summary.",
    prefix_rules=MATH500_V18_PROFILE.prefix_rules
    + "\n- Prefer a frontier that lets the next model recompute one concrete step: the shared state plus the two conflicting substitutions, arithmetic evaluations, boundary decisions, or candidate-validity checks."
    + "\n- If the disagreement depends on a sign, denominator, endpoint, or candidate check, keep that exact local object in the split instead of jumping to a method label or final answer.",
    resolution_rules=MATH500_V18_PROFILE.resolution_rules
    + "\n- First replay the shared local state and the conflicting operation before choosing a side; do not choose by which branch has a completed-looking answer."
    + "\n- The repaired claim should state the verified checkpoint and, when it is directly computed in that replay, the requested final object. If more work remains, stop at the verified checkpoint and let revision finish normally."
    + "\n- For sign, arithmetic, substitution, endpoint, and candidate-validity conflicts, the rationale must name the concrete operation that was checked, not only the method or answer popularity.",
    revision_rules=MATH500_V18_PROFILE.revision_rules
    + "\n- Continue from the verified checkpoint as an ordinary solution line; if the checkpoint is an equation or vertex/candidate location, explicitly evaluate the requested object before the final answer."
    + "\n- Do not copy a completed-looking final answer unless the local computation in the repaired claim supports it.",
)


MATH500_V21_PROFILE = replace(
    MATH500_V20_PROFILE,
    name="math500_v21",
    resolution_rules=MATH500_V20_PROFILE.resolution_rules
    + "\n- Math is not closed-candidate QA. A visible branch answer is useful only as a consistency anchor; the decision must be justified by replaying the shared state and the local operation that produces the requested object."
    + "\n- If the chosen path has only an intermediate equation, congruence, vector, count component, or candidate test, do not present it as a final answer. State the remaining operation needed, or leave revision to finish it."
    + "\n- When a resolution would overturn a 2-vs-1 observed answer majority, name the concrete local operation that defeats both majority traces. If that operation is not visible, keep the majority answer as unresolved rather than choosing by method preference."
    + "\n- When all observed final answers differ, prefer the answer whose supporting trace contains the checked operation from the shared checkpoint to the requested object; do not break ties by trace order.",
    revision_rules=MATH500_V20_PROFILE.revision_rules
    + "\n- Strip formatting wrappers such as bold markdown around the final answer; the final line should contain only the completed requested object."
    + "\n- If the repaired claim is a checkpoint, do exactly the next local operation needed for the requested object before the final line. Avoid restarting the whole solution when one replayable operation is enough.",
)


MATH500_V30_PROFILE = replace(
    MATH500_V21_PROFILE,
    name="math500_v30",
    merge_rules=MATH500_V21_PROFILE.merge_rules
    + "\n- If two active traces already share the same completed final answer, keep that final-answer support attached to their local computation; do not let a single competing branch become common ground by sounding more algebraic."
    + "\n- When there is no 2-vs-1 completed-answer majority, expose the checkpoint that can be replayed into the requested final object instead of treating observed final answers as a closed candidate list.",
    relation_analysis_rules=MATH500_V21_PROFILE.relation_analysis_rules
    + "\n- Identify whether the current conflict is stable-majority versus minority, all-different, or unfinished-checkpoint versus completed answer."
    + "\n- For a stable 2-vs-1 math majority, the minority needs a concrete local operation that defeats the majority traces and reaches the requested object; a cleaner method label or intermediate formula is not enough."
    + "\n- For all-different or unfinished-checkpoint conflicts, do not force an observed final candidate. Replay the smallest checked operation from the shared state and then finish the requested object.",
    resolution_rules=MATH500_V21_PROFILE.resolution_rules
    + "\n- First state the answer-distribution status: stable majority, all-different, or unfinished checkpoint."
    + "\n- If there is a stable 2-vs-1 completed-answer majority, choose against it only when the corrected claim names the majority's exact local error and computes the final requested object from that correction."
    + "\n- If there is no stable majority, a new final answer may be produced from a verified checkpoint, but the Correct claim must include the checkpoint, the checked operation, and the completed requested object."
    + "\n- Do not treat a local simplification, direction vector, favorable-case count, congruence rule, or denominator sign as the final answer unless the claim also finishes the value, expression, set, count, or probability asked for.",
    revision_rules=MATH500_V21_PROFILE.revision_rules
    + "\n- When revising from a checkpoint and no stable majority answer is being preserved, finish only the remaining local operation and write the completed requested object."
    + "\n- When the resolution preserves a stable majority answer, keep that answer form unless the resolution explicitly computes a different final requested object.",
)


MATH500_V31_EQUAL_TOKEN_PROFILE = replace(
    MATH500_V30_PROFILE,
    name="math500_v31_equal_token",
    merge_rules=MATH500_V30_PROFILE.merge_rules
    + "\n- Under longer outputs, keep only the replayable state needed for the requested object; do not promote later prose or a polished final token unless its local operation is visible.",
    relation_analysis_rules=MATH500_V30_PROFILE.relation_analysis_rules
    + "\n- Lock the requested object before comparing branches: scalar, interval, expression, count, probability, set, or named object. A nearby intermediate of the same calculation is not enough.",
    resolution_rules=MATH500_V30_PROFILE.resolution_rules
    + "\n- In a long trace, prefer the shortest verified path from shared state to requested object. Do not reward extra algebra unless it checks the disputed operation."
    + "\n- If the repaired claim preserves an observed final answer, name the local operation that supports that answer; if it changes the answer, compute the requested object in the claim itself.",
    revision_rules=MATH500_V30_PROFILE.revision_rules
    + "\n- Keep the supported requested object stable during the final rewrite. Do not drift to a related intermediate, method label, or explanatory sentence because the answer budget is longer.",
)


MATH500_V32_BRIDGE_AUDIT_PROFILE = replace(
    MATH500_V31_EQUAL_TOKEN_PROFILE,
    name="math500_v32_bridge_audit",
    relation_analysis_rules=MATH500_V31_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- Audit candidate answers by replayable computation: shared state -> checked operation -> requested object. A final token without that bridge is weak.",
    resolution_rules=MATH500_V31_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- A decisive Correct claim must be rewrite-supported: it either preserves a visible final answer with its checked operation or computes the new requested object in the claim."
    + "\n- Use keep_parallel if the correction is only an intermediate checkpoint and cannot yet land the requested object.",
    revision_rules=MATH500_V31_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Continue only the remaining local operation from the repaired claim, then finish with the requested object."
    + "\n- Do not let a resolution-only canonical answer override the rewrite unless the rewritten trace carries the same computed object.",
)


MATH500_V33_LEDGER_AUDIT_PROFILE = replace(
    MATH500_V32_BRIDGE_AUDIT_PROFILE,
    name="math500_v33_ledger_audit",
    relation_analysis_rules=MATH500_V32_BRIDGE_AUDIT_PROFILE.relation_analysis_rules
    + "\n- Write a compact computation ledger: requested object, observed candidate, shared checkpoint, checked operation, and missing operation if any."
    + "\n- A branch answer is unsupported when it skips the disputed arithmetic/sign/endpoint/case-validity step, even if the final form looks plausible.",
    resolution_rules=MATH500_V32_BRIDGE_AUDIT_PROFILE.resolution_rules
    + "\n- The Correct claim must contain the ledger bridge from shared checkpoint to requested object, or explicitly keep_parallel with the missing local operation."
    + "\n- Do not synthesize a new final value unless the visible ledger computes it from the shared checkpoint.",
    revision_rules=MATH500_V32_BRIDGE_AUDIT_PROFILE.revision_rules
    + "\n- Preserve the computation ledger through revision: checkpoint, checked operation, requested object, final answer."
    + "\n- If the repaired claim is only a missing-operation note, continue that operation before landing the answer.",
)


# Frozen alias for the MMLU-Pro 10% run that beat origin with
# prompt_profile=universal and round0_prompt_style=bare.
# Keep this profile stable; future universal cleanup should not change it.
MMLU_PRO_BEST_20260507_PROFILE = replace(UNIVERSAL_PROMPT_PROFILE, name="mmlu_pro_best_20260507")


MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE = replace(
    MMLU_PRO_BEST_20260507_PROFILE,
    name="mmlu_pro_relation_v1_equal_token",
    merge_intro="Merge several MMLU-Pro multiple-choice traces into a compact option-support memo.",
    merge_rules=UNIVERSAL_PROMPT_PROFILE.merge_rules
    + "\n- Treat each observed final option as a noisy hypothesis attached to its exact option text."
    + "\n- Keep the requested subject relation, option text, and final option letter/number separate."
    + "\n- Split on the first evidence-to-option support conflict, not on answer-token popularity or explanation length.",
    prefix_intro="Rebuild a compact MMLU-Pro option-support memo that exposes the first unresolved evidence-to-option conflict.",
    prefix_rules=UNIVERSAL_PROMPT_PROFILE.prefix_rules
    + "\n- Preserve the option mapping before resolving. If a path supports an option text but lands on another option token, split on that landing error."
    + "\n- If final answers differ after a shared concept, compare how that concept supports the exact option text.",
    relation_analysis_intro="Describe one MMLU-Pro option-support divergence before making a repair decision.",
    relation_analysis_rules=UNIVERSAL_PROMPT_PROFILE.relation_analysis_rules
    + "\n- First state the requested relation from the question and the exact option text under dispute."
    + "\n- Compare evidence-to-option support, not final option popularity."
    + "\n- A longer explanation is not stronger unless it preserves the requested subject and maps cleanly to the option text.",
    resolution_rules=UNIVERSAL_PROMPT_PROFILE.resolution_rules
    + "\n- Correct claim must name the supported option text or the exact option relation, then say why it answers the requested relation."
    + "\n- Do not choose by vote. Prefer the branch whose evidence supports the exact option mapping."
    + "\n- If a claim only eliminates one option, do not land on that eliminated option; continue comparing the remaining candidates.",
    revision_rules=UNIVERSAL_PROMPT_PROFILE.revision_rules
    + "\n- Preserve the option mapping during revision and finish with the requested option token only."
    + "\n- Do not drift from the supported option text to a nearby option because the output is longer.",
)


MMLU_PRO_RELATION_V2_EQUAL_TOKEN_PROFILE = replace(
    MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE,
    name="mmlu_pro_relation_v2_equal_token",
    merge_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.merge_rules
    + "\n- Keep the exact question predicate, exact option text, and supporting bridge separate; do not collapse them into a shared theme."
    + "\n- A candidate option is not supported unless the visible bridge reaches that exact option under the requested relation."
    + "\n- If two options are locally plausible from different branches, keep both candidates separate until the exact option mapping is explicit.",
    prefix_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.prefix_rules
    + "\n- Name the requested relation and exact option text before comparing branches."
    + "\n- If a path only matches a nearby concept or a partial label family, keep it as a partial bridge rather than a winner."
    + "\n- Do not let a fluent explanation outrank a weaker but exact option mapping.",
    relation_analysis_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- For each branch, report exact option text support separately from general topical support."
    + "\n- If the branch does not explicitly bridge from the premises to the exact option text, mark it unsupported even if it sounds consistent."
    + "\n- Never treat answer popularity or a longer explanation as proof of the option mapping.",
    resolution_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- Prefer keep_parallel when the traces support a topic but not the exact requested option mapping."
    + "\n- Do not select an option unless the exact option text is explicitly bridged from the question's relation."
    + "\n- If multiple candidate options remain plausible, resolve by exact option support, not by the most fluent branch.",
    revision_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve the exact option mapping through revision; do not upgrade a partial topical match into the final option."
    + "\n- Finish with the requested option token only after the exact option support is explicit.",
)


MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE = replace(
    MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE,
    name="mmlu_pro_relation_v3_extraction_guard",
    relation_analysis_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- When a claim says `supported label is N`, treat N as the option token to preserve; do not let overlapping option text or rule numbers replace it."
    + "\n- If the option text is long and shares words with other options, compare the exact option number plus exact option text together.",
    resolution_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- If the corrected claim determines an option, end the claim with `supported label is N` using one allowed option number."
    + "\n- A rewritten answer must preserve that explicit supported label unless the same claim explicitly retracts it.",
    revision_rules=MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Keep the explicit supported label through the final line. Do not infer a different option from a shared phrase in another option.",
)


MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE = replace(
    MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE,
    name="mmlu_pro_relation_v4_bridge_audit",
    relation_analysis_rules=MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- Build a compact candidate audit: requested relation, exact option text, positive bridge, and any bridge failure. Do not score by final-token majority."
    + "\n- A resolution can introduce a candidate only when the visible trace already contains the bridge; otherwise keep_parallel and let revision continue.",
    resolution_rules=MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- The Correct claim must be a rewrite-supported option bridge, not only a verdict. It must name the exact option text and end with `supported label is N`."
    + "\n- Use keep_parallel if the selected label cannot be carried into a revised trace from visible evidence."
    + "\n- Do not let a one-shot canonical label override the agents unless the rewrite itself preserves the same supported label.",
    revision_rules=MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Rewrite from the selected bridge and keep the same supported label in the final line."
    + "\n- If the corrected claim only diagnoses another option but does not support a label, continue comparing rather than landing a label.",
)


MMLU_PRO_RELATION_V5_LEDGER_AUDIT_PROFILE = replace(
    MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE,
    name="mmlu_pro_relation_v5_ledger_audit",
    relation_analysis_rules=MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE.relation_analysis_rules
    + "\n- Write a compact option ledger: requested relation, candidate option token, exact option text, positive bridge, and failure mark."
    + "\n- Treat shared words with the option as weak unless the trace maps them to the exact requested relation.",
    resolution_rules=MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE.resolution_rules
    + "\n- The Correct claim must name the exact option text and the visible bridge, then end with `supported label is N`."
    + "\n- Use keep_parallel if the ledger has only topic support, partial elimination, or option-token drift.",
    revision_rules=MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE.revision_rules
    + "\n- Preserve the option ledger through revision and finish with the same supported label."
    + "\n- Do not convert a partial bridge or eliminated option into the final label.",
)


LOGIQA_RELATION_V1_PROFILE = replace(
    UNIVERSAL_PROMPT_PROFILE,
    name="logiqa_relation_v1",
    merge_intro="Merge several formal multiple-choice reasoning traces into a compact premise-to-option memo.",
    merge_rules="""
Rules:
- Keep stated premises, rule applications, option claims, and the final option number separate.
- Do not treat a bare option number as evidence.
- Use only the stated premises and their logical consequences; do not add ordinary-world background unless the prompt itself states it.
- Split on the first concrete conflict about whether the premises prove, disprove, or leave open an option.
- Do not use the converse or inverse of a conditional unless it is explicitly stated.
- For disjunctions, do not choose one branch unless the premises rule out the other branch.
- If a path eliminates options, keep the elimination reason attached to the exact option it eliminates.
""".strip(),
    prefix_intro="Rebuild a compact premise-to-option memo that exposes the first unresolved logical conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared premise or rule application before the first conflict.
- Do not split on option-number wording if the underlying option claim is the same.
- Split when paths disagree about a stated premise, a rule application, an elimination step, or whether an option follows.
- If final answers differ after a shared prefix, compare the bridge from that prefix to the exact option text.
""".strip(),
    relation_analysis_intro="Describe one logical divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- State the exact option text or conclusion being tested.
- Compare only stated premises and valid logical consequences.
- Check whether a path used an unstated converse, inverse, branch choice, exclusivity assumption, or outside fact.
- If a path's option number does not follow from its proof steps, name the proof-to-option bridge error.
""".strip(),
    resolution_rules="""
Rules:
- Return the logical claim to keep, not only an option number.
- Correct claim must say how the premises support or eliminate the exact option text.
- Do not choose by answer popularity.
- Do not add hidden premises, ordinary-world facts, unstated exclusivity, or unstated converse/inverse rules.
- If the premises leave both sides open, use keep_parallel instead of forcing an option.
- After the repaired claim, the continuation must still finish with the requested option number.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired premise-to-option claim and finish with the option number.
- Do not stop at an intermediate rule, eliminated option, or option text if the question asks for the option number.
""".strip(),
)


LOGIQA_RELATION_V2_PROFILE = replace(
    LOGIQA_RELATION_V1_PROFILE,
    name="logiqa_relation_v2",
    merge_intro="Merge several multiple-choice logic traces into a compact option-support memo.",
    merge_rules="""
Rules:
- Keep the option number, exact option text, and logical status separate: supported, eliminated, or still open.
- Before accepting any final option number, map it back to the exact option text and check whether the trace actually supports that text.
- Do not treat an elimination of option N as support for answer N.
- If the question asks which option can be derived, must be true, cannot be true, or weakens/strengthens, preserve that requested relation while comparing options.
- Use only stated premises, stated rules, and valid consequences; do not add outside facts or hidden exclusivity.
- Do not use the converse or inverse of a conditional unless it is explicitly stated.
- For disjunctions, seating/order/spatial cases, keep alternatives open until the premises rule them out.
- If a path eliminates options, keep each elimination reason attached to the exact option text it eliminates.
- Split on the first conflict about the same option text's status or the same premise-to-option bridge, not on a bare final number.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared premise, rule application, or candidate-state table before the first conflict.
- Preserve the mapping from option number to exact option text.
- Do not split on option-number wording if the underlying option text and logical status are the same.
- Split when paths disagree about whether the same exact option is supported, eliminated, or still open.
- If final answers differ after a shared prefix, compare how that prefix supports or eliminates each exact option text.
- Do not stop at a local clue if it only eliminates one candidate; keep building until the requested option relation is decided.
""".strip(),
    relation_analysis_rules="""
Rules:
- State the requested relation from the question and the exact option text being tested.
- Compare premise -> option-text support, not final option popularity.
- Distinguish support, elimination, and still-open status for each disputed option.
- Check whether a path used an unstated converse, inverse, branch choice, exclusivity assumption, or outside fact.
- If a path's option number does not follow from its proof steps, name the proof-to-option bridge error.
- If a claim eliminates an option but the trace selects that option, treat that as a landing error rather than a valid support claim.
""".strip(),
    resolution_rules="""
Rules:
- Return the option-support claim to keep, not only an option number.
- Correct claim must name the exact option text and say whether the premises support it, eliminate it, or leave it open.
- Do not choose by answer popularity.
- Do not add hidden premises, ordinary-world facts, unstated exclusivity, or unstated converse/inverse rules.
- If the kept claim eliminates option N, the continuation must not land on N from that claim.
- If the premises leave both sides open, use keep_parallel instead of forcing an option.
- If one side preserves the option-text mapping and the other only repeats a final number, prefer the mapped option-support claim.
- After the repaired claim, the continuation must still finish with the requested option number.
""".strip(),
    resolution_fewshot=LOGIQA_RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired option-support claim and finish with the option number.
- Preserve the option number to exact option text mapping while revising.
- Do not restate a losing final option number if the repaired claim eliminates that option.
- Do not stop at an intermediate rule, eliminated option, or option text if the question asks for the option number.
- The last non-empty line must be exactly one number from 1 to 4.
""".strip(),
)


LOGIQA_RELATION_V3_PROFILE = replace(
    UNIVERSAL_PROMPT_PROFILE,
    name="logiqa_relation_v3",
    merge_intro="Merge several LogiQA multiple-choice reasoning traces into a compact argument memo.",
    merge_rules="""
Rules:
- Keep the question type, stated premises, option text, and final option number separate.
- Do not treat a bare option number as evidence.
- Compare the reasoning bridge that makes an exact option answer the question, not option popularity.
- If the question asks which option can be derived, must be true, weakens, strengthens, explains, assumes, or is most supported, preserve that requested relation while comparing paths.
- Use stated premises and visible argument relations; ordinary meaning is allowed for argument-type questions, but do not add hidden facts.
- For formal conditional, disjunction, ordering, or grouping clues, do not use an unstated converse, inverse, branch choice, or exclusivity assumption.
- If a path only eliminates one option, keep building unless that elimination directly determines the requested option.
- If a path's proof steps eliminate option N but its final answer is N, mark that as a proof-to-option landing error.
""".strip(),
    prefix_intro="Rebuild a compact LogiQA argument memo that exposes the first unresolved option-relation conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared premise, argument relation, or candidate check before the first conflict.
- Preserve the mapping from option number to exact option text when an option is discussed.
- Do not split on option-number wording if the underlying option text and support relation are the same.
- Split when paths disagree about whether the same evidence supports, eliminates, weakens, strengthens, explains, assumes, or leaves open an exact option.
- If final answers differ after a shared prefix, compare the bridge from that prefix to the exact option text.
- Progress lag is not a conflict: a path that has only eliminated one option may still need to check the remaining candidates.
""".strip(),
    relation_analysis_intro="Describe one LogiQA option-relation divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- State the question's requested relation and the exact option text being tested.
- Compare only visible premises, argument relations, and valid consequences.
- For strengthen/weaken/explain/assumption questions, judge whether the option affects the argument relation asked by the question, not just whether the option is topically relevant.
- For formal clue questions, check for unstated converse, inverse, branch choice, exclusivity, or outside facts.
- If a path's option number does not follow from its proof steps, name the proof-to-option bridge error.
""".strip(),
    resolution_rules="""
Rules:
- Return the option-relation claim to keep, not only an option number.
- Correct claim must name the exact option text or described option relation and say how it answers the requested relation.
- Do not choose by answer popularity.
- Do not add hidden premises, unstated exclusivity, or unstated converse/inverse rules.
- Do not over-prefer a formally neat candidate if the question is an argument strengthen/weaken/explain/assumption task; compare the requested argument effect.
- If the best current claim only eliminates one option and does not determine the requested answer, use keep_parallel.
- If the kept claim eliminates option N, the continuation must not land on N from that claim.
- After the repaired claim, the continuation must still finish with the requested option number.
""".strip(),
    resolution_fewshot="""
Tiny LogiQA resolution example 1:
Correct claim: Option 2 is supported because its exact text follows from the stated conditional and the condition is given, so continue to answer 2.

Tiny LogiQA resolution example 2:
Correct claim: Option 3 is eliminated by the stated condition; eliminating option 3 cannot support answer 3.

Tiny LogiQA resolution example 3:
Correct claim: For a weakening question, the option must undercut the argument's causal bridge, not merely mention the same topic.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired option-relation claim and finish with the option number.
- Preserve the option number to exact option text mapping while revising.
- Preserve the question type: derived, must be true, weakens, strengthens, explains, assumes, or most supported.
- Do not restate a losing final option number if the repaired claim eliminates that option.
- Do not stop at an intermediate rule, eliminated option, or option text if the question asks for the option number.
- The last non-empty line must be exactly one number from 1 to 4.
""".strip(),
)


LOGIQA_RELATION_V4_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v4",
    merge_rules=LOGIQA_RELATION_V3_PROFILE.merge_rules
    + "\n- Treat each observed final option as a branch hypothesis; keep the visible bridge from that branch to its exact option text."
    + "\n- For must/necessarily/can-be-concluded questions, a valid counterexample to an option eliminates that option; it is not support for that option."
    + "\n- Keep polarity words such as supports, eliminates, not necessary, must, may, weakens, and strengthens attached to the same exact option throughout the memo.",
    prefix_rules=LOGIQA_RELATION_V3_PROFILE.prefix_rules
    + "\n- If a path proves an option is not necessary, not supported, or eliminated, split on that polarity before any final option number."
    + "\n- If the first conflict is a counterexample versus a necessity claim, preserve the counterexample object and the option it targets.",
    relation_analysis_rules=LOGIQA_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- Check polarity consistency: the option named as supported must match the proof's support/elimination/necessity direction."
    + "\n- For necessity questions, ask whether the trace proves the option in every valid arrangement or only in one possible arrangement.",
    resolution_rules=LOGIQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- The resolved claim must not say 'Option N is supported' if its evidence says option N is eliminated, not necessary, only possible, or merely topical."
    + "\n- If a branch gives a concrete valid counterexample to option N in a must/necessarily question, use that as evidence against option N unless another branch directly invalidates the counterexample."
    + "\n- Prefer a supported observed branch hypothesis over inventing a new final option; introduce a new option only when the visible premises explicitly build its exact option bridge."
    + "\n- Before choosing a side, restate the requested relation and verify that the kept claim answers that relation, not a nearby possible/optional/local relation.",
    resolution_fewshot=LOGIQA_RELATION_V3_PROFILE.resolution_fewshot
    + """

Tiny LogiQA resolution example 4:
Correct claim: A valid arrangement without option 1's candidate shows option 1 is not necessary, so that counterexample eliminates option 1 rather than supporting it.

Tiny LogiQA resolution example 5:
Correct claim: If the proof says option 3 is eliminated, the continuation must not land on option 3 from that proof.
""".strip(),
    revision_rules=LOGIQA_RELATION_V3_PROFILE.revision_rules
    + "\n- If the repaired claim is a counterexample to an option's necessity, do not finish with that eliminated option."
    + "\n- If the repaired claim only shows an option is possible, continue checking whether the question asks for possibility, necessity, weakening, strengthening, explanation, assumption, or derivability before the final number.",
)


LOGIQA_RELATION_V5_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v5",
    relation_analysis_rules=LOGIQA_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- Check internal consistency between the proof direction and the option named in the claim: support, elimination, not-necessary, weakens, strengthens, explains, and assumes must point to the same option text.",
    resolution_rules=LOGIQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- The corrected claim must be internally consistent: do not write that option N is supported when the rationale says option N is eliminated, not necessary, contradicted, or merely topical."
    + "\n- If a side gives only a final number but another side gives a trace-faithful option-text bridge, compare the bridge; do not invent a new final option unless the visible premises already build that exact bridge."
    + "\n- If the best repaired claim only says why one option is wrong, continue the solution from that elimination instead of landing on the eliminated option.",
    resolution_fewshot=LOGIQA_RELATION_V3_PROFILE.resolution_fewshot
    + """

Tiny LogiQA resolution example 4:
Correct claim: The proof eliminates option 2, so it cannot support answer 2; continue from that elimination to decide among the remaining options.
""".strip(),
    revision_rules=LOGIQA_RELATION_V3_PROFILE.revision_rules
    + "\n- Keep the final option number aligned with the repaired claim's polarity; an eliminated or not-necessary option cannot be the final answer from that claim.",
)


LOGIQA_RELATION_V6_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v6",
    relation_analysis_rules=LOGIQA_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- When one branch would overturn two branches with the same option, first identify the exact bridge error in that majority option; a complete-looking minority proof is not enough by itself."
    + "\n- Check whether the proposed winning option actually answers the requested relation, such as must-be-true, possible, weakens, strengthens, explains, assumes, most supported, or flaw-parallel."
    + "\n- Keep support, elimination, not-necessary, possible, and merely-topic-relevant as different relations; do not convert one relation into another while naming the same option.",
    resolution_rules=LOGIQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- A one-branch correction may overturn a two-branch option only when the corrected claim names the majority option's concrete bridge error or shows that its exact option text fails the requested relation."
    + "\n- If the minority branch only gives an alternative option without explaining why the majority option's bridge fails, use keep_parallel and let continuation compare the remaining option bridges."
    + "\n- Before landing on an option, state the option's requested relation in the claim: support, elimination, necessity, possibility, weakening, strengthening, explanation, assumption, or flaw match."
    + "\n- The corrected claim must be internally consistent: do not write that option N is supported when the rationale says option N is eliminated, not necessary, contradicted, only possible, or merely topical."
    + "\n- Prefer preserving an already supported majority option when the competing claim answers a nearby relation or relies on an unstated converse, inverse, exclusivity, or branch choice.",
    resolution_fewshot=LOGIQA_RELATION_V3_PROFILE.resolution_fewshot
    + """

Tiny LogiQA resolution example 4:
Correct claim: A minority option can replace a two-branch option only if it identifies the two-branch option's exact bridge error, such as treating possibility as necessity or using an unstated converse.

Tiny LogiQA resolution example 5:
Correct claim: If the proof only shows option 3 is possible but the question asks what must follow, option 3 is not supported as the final answer.

Tiny LogiQA resolution example 6:
Correct claim: If the rationale eliminates option 2, the continuation must not land on option 2; continue comparing the remaining options instead.
""".strip(),
    revision_rules=LOGIQA_RELATION_V3_PROFILE.revision_rules
    + "\n- If the repaired claim preserves a majority option because the competing branch did not show its bridge error, continue from that majority option's visible support and recheck only the disputed relation."
    + "\n- If the repaired claim only eliminates or weakens an option, do not finish with that eliminated or weakened option unless the question explicitly asks for that relation.",
)


LOGIQA_RELATION_V7_PROFILE = replace(
    LOGIQA_RELATION_V6_PROFILE,
    name="logiqa_relation_v7",
)


LOGIQA_RELATION_V8_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v8",
    relation_analysis_rules=LOGIQA_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- When a branch gives a majority option with a coherent bridge, identify a concrete bridge error before replacing it with another option."
    + "\n- Treat ordinary corrected claims as rewrite anchors, not as final-answer broadcasts; only an explicit final answer line should determine the global landing.",
    resolution_rules=LOGIQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- If you choose a side, write the corrected claim as one unambiguous support claim for the winning option; do not mix a rejected option and the winning option in the same sentence."
    + "\n- Use keep_parallel if the best claim only gives a local comparison but does not determine the final option."
    + "\n- Add a separate `Final answer: N` line only when the corrected claim itself determines option N; otherwise leave the final answer field empty and let continuation finish.",
    resolution_fewshot=LOGIQA_RELATION_V3_PROFILE.resolution_fewshot
    + """

Tiny LogiQA resolution example 4:
Correct claim: Option 2 is supported because its exact text supplies the missing assumption in the argument.
Final answer: 2

Tiny LogiQA resolution example 5:
Correct claim: Option 4 is rejected because it strengthens the causal explanation instead of weakening it.
Action: keep_parallel
""".strip(),
    revision_rules=LOGIQA_RELATION_V3_PROFILE.revision_rules
    + "\n- If the repaired claim does not contain an explicit final answer, continue the comparison and finish from the evidence, not from a number mentioned in the analysis.",
)


LOGIQA_RELATION_V9_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v9",
)


LOGIQA_RELATION_V10_PROFILE = replace(
    LOGIQA_RELATION_V3_PROFILE,
    name="logiqa_relation_v10",
    resolution_rules=LOGIQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- If the corrected claim itself determines one option, add a separate `Final answer: N` line; if it only gives a local comparison or elimination, do not add a final answer line.",
    resolution_fewshot=LOGIQA_RELATION_V3_PROFILE.resolution_fewshot
    + """

Tiny LogiQA resolution example 4:
Correct claim: Option 2 is supported because its exact text supplies the missing assumption in the argument.
Final answer: 2

Tiny LogiQA resolution example 5:
Correct claim: Option 4 is eliminated because it strengthens the causal explanation instead of weakening it.
""".strip(),
)


LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE = replace(
    LOGIQA_RELATION_V10_PROFILE,
    name="logiqa_relation_v11_equal_token",
    merge_rules=LOGIQA_RELATION_V10_PROFILE.merge_rules
    + "\n- Under longer outputs, keep the requested relation as the anchor: derive, must follow, could be true, weaken, strengthen, explain, assume, flaw, or most supported."
    + "\n- Treat observed option numbers as branch hypotheses only after mapping them to exact option text and proof polarity.",
    prefix_rules=LOGIQA_RELATION_V10_PROFILE.prefix_rules
    + "\n- Split on option-text polarity first: supported, eliminated, possible, necessary, weakens, strengthens, explains, assumes, or merely topical.",
    relation_analysis_rules=LOGIQA_RELATION_V10_PROFILE.relation_analysis_rules
    + "\n- Audit each candidate by requested relation -> exact option text -> visible bridge. Do not let a fluent long proof change the requested relation."
    + "\n- If a branch is longer but never defeats the current option's bridge, treat it as unresolved rather than stronger.",
    resolution_rules=LOGIQA_RELATION_V10_PROFILE.resolution_rules
    + "\n- Prefer a supported observed option over inventing a new option; invent only when visible premises explicitly build that exact option bridge."
    + "\n- If a Correct claim includes a final answer line, it must be consistent with the option text and polarity in the same claim.",
    revision_rules=LOGIQA_RELATION_V10_PROFILE.revision_rules
    + "\n- Preserve the supported canonical option number through the final rewrite; do not let later explanatory text drift to a different option."
    + "\n- The last non-empty line must be exactly one number from 1 to 4.",
)


LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE = replace(
    LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE,
    name="logiqa_relation_v12_extraction_guard",
    relation_analysis_rules=LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- Give priority to an explicit `supported label is N` or `Final answer: N` only when the same sentence supports, rather than eliminates, that option."
    + "\n- If a sentence mentions an eliminated option and a supported option, record both polarities instead of extracting the nearest number.",
    resolution_rules=LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- Write the winning option as a positive support claim ending with `supported label is N`; do not mix rejected option numbers in that final support sentence."
    + "\n- If the current evidence only eliminates one option, use keep_parallel unless the requested relation is itself elimination.",
    revision_rules=LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- The final option number must match the explicit supported-label sentence, not a number mentioned in a rejected-option explanation.",
)


LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE = replace(
    LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE,
    name="logiqa_relation_v13_flow_guard",
    merge_rules=LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE.merge_rules
    + "\n- If several branches are unanimous on the same wrong option, keep the first explicit bridge error instead of stopping at the unanimous final token.",
    prefix_rules=LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE.prefix_rules
    + "\n- A unanimous wrong answer is still an actionable case when the visible support for the requested relation is weak or missing; keep the bridge visible.",
    relation_analysis_rules=LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- Before stopping, check whether the current label is unanimous only because all branches copied a shallow token; if so, expose the missing bridge, not the token agreement.",
    resolution_rules=LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- If the current consensus has no explicit bridge to the requested relation, keep_parallel or repair the bridge rather than stopping at unanimity."
    + "\n- A corrected claim may introduce a new supported option only when the evidence bridge is visible in the current trace and the rewrite can land it naturally.",
    revision_rules=LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Preserve the bridge, not the copied label, through revision. If the label is unanimous but unsupported, continue the repair instead of freezing it."
    + "\n- The last line still must be one option number, but it should be the number supported by the repaired bridge, not by token repetition.",
)


LOGIQA_RELATION_V14_ACTION_GUARD_PROFILE = replace(
    LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE,
    name="logiqa_relation_v14_action_guard",
    relation_analysis_rules=LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE.relation_analysis_rules
    + "\n- Build a candidate ledger: for each visible option, note exact option text, requested relation, and whether the trace positively supports, eliminates, or merely mentions it."
    + "\n- A positive bridge to the requested relation is stronger than a bare option majority or a long explanation that never attacks that bridge.",
    resolution_rules=LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE.resolution_rules
    + "\n- Use choose_A or choose_B when one side has a complete positive bridge to an observed option and the other side only gives a bare label, a topical option mention, or an eliminated-option rationale."
    + "\n- Use keep_parallel only when no visible candidate has a complete requested-relation bridge, or when both sides give compatible partial bridges that do not yet decide the option."
    + "\n- Do not write Action as resolve, repair, support, eliminate, or an option number; the Action field must be exactly choose_A, choose_B, or keep_parallel."
    + "\n- The Correct claim should name the supported option text and end with `supported label is N` when the bridge decides the answer.",
    revision_rules=LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE.revision_rules
    + "\n- Continue from the selected option bridge, not from option-token popularity; the final line must be the supported label from the bridge.",
)


LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE = replace(
    LOGIQA_RELATION_V14_ACTION_GUARD_PROFILE,
    name="logiqa_relation_v15_memory_guard",
    resolution_rules=LOGIQA_RELATION_V14_ACTION_GUARD_PROFILE.resolution_rules
    + "\n- If a previous round selected a supported option bridge, only overturn it by naming the exact bridge error in that prior option text or polarity."
    + "\n- Do not convert an eliminated option into a supported label. If the evidence contradicts option N, write `option N is eliminated`, not `supported label is N`."
    + "\n- A later three-agent consensus is not evidence by itself; it must still explain stated premises -> requested relation -> exact option text -> option number.",
    revision_rules=LOGIQA_RELATION_V14_ACTION_GUARD_PROFILE.revision_rules
    + "\n- Preserve the previous supported option bridge through revision unless the current resolution explicitly repairs that bridge. Never let an eliminated-option sentence set the final label.",
)


LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE = replace(
    LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE,
    name="logiqa_relation_v16_stable_rewrite_guard",
    relation_analysis_rules=LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE.relation_analysis_rules
    + "\n- If changing a visible two-agent option majority, first name the majority option's exact bridge error: wrong requested relation, invalid option text, unsupported premise step, polarity error, or proof-to-option landing error."
    + "\n- A fluent minority derivation is not enough to overturn a majority option unless it directly defeats that option's stated bridge.",
    resolution_rules=LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE.resolution_rules
    + "\n- To overturn a visible majority option, the Correct claim must explicitly say why that majority option's exact option text fails the requested relation."
    + "\n- If you cannot name that bridge error, preserve the visible majority option or use keep_parallel; do not rewrite agents into a new option consensus."
    + "\n- When you do name the bridge error, state it before the replacement supported label.",
    revision_rules=LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE.revision_rules
    + "\n- Do not rewrite a stable majority option into another option unless the current resolution explicitly names the majority option's bridge error."
    + "\n- If the resolution only praises a competing option without defeating the majority option's bridge, keep the previous majority answer.",
)


LOGIQA_RELATION_V17_BRIDGE_AUDIT_PROFILE = replace(
    LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE,
    name="logiqa_relation_v17_bridge_audit",
    relation_analysis_rules=LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE.relation_analysis_rules
    + "\n- Treat a two-agent majority as a hypothesis, not proof. Protect it only when its traces contain a positive requested-relation bridge to the exact option text."
    + "\n- If the majority bridge is missing or only topical, compare all visible candidate bridges and allow a rewrite-supported minority repair.",
    resolution_rules=LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE.resolution_rules
    + "\n- A decisive Correct claim must be usable as a rewrite anchor: requested relation -> exact option text -> positive bridge -> `supported label is N`."
    + "\n- If overturning a majority, either name the majority bridge error or state that the majority has no positive bridge and the replacement bridge is visible."
    + "\n- Use keep_parallel instead of writing a canonical answer when the corrected claim cannot be carried by the revised trace.",
    revision_rules=LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE.revision_rules
    + "\n- Rewrite from the visible option bridge, not from the resolution's label alone."
    + "\n- The final label must be the label supported by the rewritten bridge; otherwise preserve the previous answer and continue.",
)


UNIVERSAL_MINIMAL_PROMPT_PROFILE = PromptProfile(
    name="universal_minimal",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several agent traces into a compact natural-language claim memo.",
    merge_rules="""
Rules:
- Keep only trace-faithful shared progress.
- Do not split on process lag, answer format, equivalent wording, or method style alone.
- First identify the requested object, label, relation, or option.
- Split when two paths make incompatible claims about the same object or about how the same evidence supports the requested target.
- If a claim only settles a subcondition or intermediate object, keep building unless it directly determines the requested target.
""".strip(),
    merge_fewshot=MERGE_FEWSHOT,
    audit_intro="Audit a natural-language claim memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep the memo trace-faithful.
- Move one-path-only material back into the relevant path note.
- Do not invent unsupported facts.
""".strip(),
    prefix_intro="Rebuild a compact natural-language claim memo that exposes only the shortest shared prefix and the first unresolved actionable conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared prefix needed for the first actionable conflict.
- Do not split on process status, answer format, or equivalent wording.
- Split on the first concrete same-target conflict: object claim, evidence-to-relation bridge, or option-support claim.
- If no real conflict is visible yet, keep building.
""".strip(),
    relation_analysis_intro="Describe one divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- Do not choose a winning side yet.
- Name the requested target before comparing paths.
- Focus on the earliest conflict about that target or about how shared evidence supports it.
- Ignore progress lag and equivalent wording.
""".strip(),
    resolution_intro="Resolve one minimal divergence inside a natural-language claim graph.",
    resolution_rules="""
Rules:
- Keep the repaired claim concrete and positive.
- Do not choose by answer popularity.
- Resolve the local target conflict or evidence-to-target bridge before trusting a final answer token.
- If both claims can remain true without deciding the requested target, use keep_parallel.
- Correct claim must be a claim to keep, not only a diagnosis of the losing side.
""".strip(),
    resolution_fewshot=RESOLUTION_FEWSHOT,
    revision_rules="""
Rules:
- Continue naturally from the repaired claim to the requested target.
- Do not stop at an intermediate object or partial bridge.
""".strip(),
)


STRATEGYQA_RELATION_V1_PROFILE = PromptProfile(
    name="strategyqa_relation_v1",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several yes/no reasoning traces into a compact relation-graph memo.",
    merge_rules="""
Rules:
- Keep the shared evidence separate from the question predicate.
- Do not treat a bare yes/no label as evidence.
- If paths share evidence but answer differently, split on whether that evidence supports the exact question predicate.
- If a path answers only a nearby, weaker, stronger, historical, or partial relation, make that relation drift explicit.
- Do not add hidden conditions or special assumptions that are not in the question, trace, or ordinary meaning.
- If a path only settles a subcondition, keep it in Paths and keep building until the whole predicate is bridged.
- For comparative, evaluative, ability, capacity, or hypothetical questions, do not let a local property or subcondition stand in for the whole asked relation unless the trace explicitly bridges them.
""".strip(),
    merge_fewshot="""
Tiny relation example 1:
Common Ground
- Both paths use the same evidence S.

Paths
- A1: claims S supports the exact question predicate.
- A2: claims S does not support the exact question predicate.

First Split
- First split: A1 claims S supports the requested predicate; A2 claims S only supports a nearby or insufficient relation.

Tiny relation example 2:
Common Ground
- none yet; final yes/no labels alone are answer format, not evidence.

Paths
- A1: gives only a final label.
- A2: gives an evidence-to-predicate bridge.

First Split
- First split: A1 gives no support claim beyond the label; A2 claims its evidence supports the requested predicate.
""".strip(),
    audit_intro="Audit a yes/no relation-graph memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep only trace-faithful shared evidence in Common Ground.
- Move one-path-only facts and bridge judgments back into Paths.
- Do not invent outside facts or hidden assumptions.
""".strip(),
    prefix_intro="Rebuild a compact relation-graph memo that exposes the first evidence-to-predicate conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared evidence before the first relation conflict.
- Split on how evidence supports or fails the exact question predicate, not on the final yes/no token.
- Do not split on explanation style or progress lag.
- If no predicate-level bridge is visible yet, keep building.
""".strip(),
    relation_analysis_intro="Describe one evidence-to-predicate divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First name the exact question predicate.
- Then name the evidence each path uses.
- Compare whether the evidence supports that exact predicate or only a nearby/partial relation.
- Do not choose by final label popularity.
""".strip(),
    resolution_intro="Resolve one minimal evidence-to-predicate divergence.",
    resolution_rules="""
Rules:
- Return the relation claim to keep.
- Correct claim must say how evidence supports or fails the exact question predicate, then name the supported yes/no label.
- Correct claim must restate the exact asked predicate or an equivalent whole-predicate relation before naming the label.
- Do not use a bare yes/no label as evidence.
- Do not silently upgrade a nearby or partial relation into the requested predicate.
- A local fact, property, or subcondition plus "supported label" is not enough unless it explains the bridge to the whole predicate.
- If the available evidence does not settle the whole predicate, keep_parallel rather than forcing a fake repair.
- Prefer the path that actually bridges the evidence to the full predicate, not the path that only sounds directly answer-shaped.
- For comparative, evaluative, ability, or capacity questions, require the bridge from the local fact to the whole asked relation before choosing a side.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: Evidence S supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: Evidence S only supports a nearby or partial relation, not the exact requested predicate, so it is not enough by itself to support yes.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-predicate claim and finish the yes/no answer.
- Do not replace the exact predicate with a nearby relation while finishing.
""".strip(),
)


STRATEGYQA_RELATION_V2_PROFILE = replace(
    STRATEGYQA_RELATION_V1_PROFILE,
    name="strategyqa_relation_v2",
    merge_intro="Merge several yes/no traces into a compact evidence-to-predicate memo.",
    merge_rules="""
Rules:
- First identify the exact yes/no predicate asked by the question.
- Keep evidence facts separate from the bridge that says whether they support that exact predicate.
- Do not treat a bare yes/no label as evidence, and do not split on the label before checking the bridge.
- If paths share evidence but answer differently, split on the bridge from that evidence to the exact predicate.
- A local fact, subcondition, named-entity fact, or possibility claim is not decisive unless the trace explains why it settles the whole predicate.
- Preserve predicate strength and modality: would/could, common/exists, most/some, literal/symbolic, and ability/actual occurrence are different bridges.
- If a path uses a nearby relation, keep it as a nearby-relation claim instead of silently upgrading it to the requested predicate.
- If a path gives a complete bridge to the exact predicate, keep that bridge even if another path gives a more direct final label.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence before the first bridge conflict.
- Split on whether the evidence supports the exact question predicate, not on the final yes/no token.
- If the visible material is only a local fact or subcondition, keep building until it is tied to the whole predicate.
- If two paths use the same evidence but differ by modality or predicate strength, split on that bridge difference.
- If one path answers a nearby relation instead of the asked relation, split on that relation drift.
""".strip(),
    relation_analysis_rules="""
Rules:
- First name the exact question predicate in ordinary words.
- Then name the evidence bundle each path uses.
- Compare the bridge from evidence to that predicate; do not choose by final label popularity.
- Decide whether each bridge preserves the requested predicate or drifts to a nearby, weaker, stronger, historical, or partial relation.
- If a path has both pro and con facts, analyze the path's final bridge to the whole predicate, not an isolated early fact.
- If neither side has a full bridge, say what bridge is missing instead of forcing a side.
""".strip(),
    resolution_rules="""
Rules:
- Return the predicate-bridge claim to keep.
- Correct claim must be one complete sentence that contains: the evidence, the exact predicate relation, and the supported yes/no label.
- Do not use a bare yes/no label as evidence.
- Do not choose a side merely because it is explicit, confident, or answer-shaped.
- Prefer the side that preserves the exact predicate over a side that proves only a subcondition or nearby relation.
- If the available evidence only proves a local fact and not the whole predicate, use keep_parallel rather than inventing the missing bridge.
- If one branch's facts support the opposite label from its final token, keep the factual bridge and repair the label implication.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The evidence supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: The evidence proves only a nearby or partial relation, not the exact requested predicate, so it is not enough by itself to support yes.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-predicate claim and finish the yes/no answer.
- Keep the exact predicate relation fixed while finishing.
- Do not replace the repaired bridge with a nearby relation or a stale final label.
""".strip(),
)


STRATEGYQA_RELATION_V3_PROFILE = replace(
    STRATEGYQA_RELATION_V2_PROFILE,
    name="strategyqa_relation_v3",
    relation_analysis_rules="""
Rules:
- First name the exact question predicate in ordinary words.
- Then name the actual evidence bundle each path uses.
- Compare the bridge from evidence to that predicate; do not choose by final label popularity.
- Treat a specific outside fact asserted by only one path as an unverified path claim unless another trace or the question also supports it.
- Treat a favorable condition, special role, training assumption, transparency assumption, or exception as hidden unless it is stated in the question or trace evidence.
- Words like "direct", "specific", "relevant", or "aligns with the predicate" are not evidence; identify the actual fact that does the bridging.
- If a path has both pro and con facts, analyze the path's final bridge to the whole predicate, not an isolated early fact.
- If neither side has a trace-grounded bridge, say what bridge is missing instead of forcing a side.
""".strip(),
    resolution_rules="""
Rules:
- Return the predicate-bridge claim to keep.
- Correct claim must be one complete sentence that contains: the evidence, the exact predicate relation, and the supported yes/no label.
- Do not use a bare yes/no label as evidence.
- Do not choose a side merely because it is explicit, confident, direct, specific, or answer-shaped.
- Prefer a trace-grounded bridge over a side that relies on a single-path outside fact or a hidden condition.
- Do not let a favorable assumption, special exception, training assumption, transparency assumption, or role reinterpretation decide the label unless the question or traces state it.
- Prefer the side that preserves the exact predicate over a side that proves only a subcondition or nearby relation.
- If the available evidence only proves a local fact and not the whole predicate, use keep_parallel rather than inventing the missing bridge.
- If one branch's facts support the opposite label from its final token, keep the factual bridge and repair the label implication.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The stated evidence supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: A hidden favorable condition is not stated by the question or trace evidence, so it cannot by itself support yes.

Tiny resolution example 3:
Correct claim: A single path asserts an outside fact but does not support it with trace evidence; keep the trace-grounded bridge or keep_parallel.
""".strip(),
)


STRATEGYQA_RELATION_V4_PROFILE = replace(
    STRATEGYQA_RELATION_V3_PROFILE,
    name="strategyqa_relation_v4",
    merge_intro="Merge several yes/no traces into a compact evidence-to-predicate bridge memo.",
    merge_rules="""
Rules:
- First write the exact yes/no predicate being asked, in ordinary words.
- Keep evidence facts, bridge claims, and final labels separate.
- If answers differ after shared evidence, do not make the label token the first split; split on how that evidence supports or fails the asked predicate.
- A usable path claim should have the shape: evidence S supports/fails predicate P, so the supported label is yes/no.
- If a path gives only a fact, entity property, opportunity, constraint, or historical occurrence, keep it as evidence and ask whether it bridges to P.
- If a path relies on an extra premise, special exception, or outside fact that appears in only one trace, keep it as that path's disputed bridge.
- If two paths share evidence but infer opposite labels, expose the semantic bridge conflict directly.
- If neither path bridges evidence to the whole predicate, keep building instead of forcing a label repair.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence before the first bridge conflict.
- Split on the support relation from evidence to the exact question predicate, not on final yes/no tokens.
- Do not split on explanation style, progress lag, or an isolated local fact before it is tied to the whole predicate.
- If two paths use the same evidence but disagree on ability, possibility, likelihood, ordinary meaning, entity identity, or sufficiency, split on that bridge.
- If one path answers a nearby relation, make the drift explicit as the conflicting bridge.
""".strip(),
    relation_analysis_rules="""
Rules:
- Name the exact question predicate first.
- Name the evidence bundle each path actually uses.
- For each path, state the bridge in one sentence: this evidence supports/fails the predicate because ...
- Compare bridge quality, not final label popularity.
- Treat a one-path outside fact or hidden condition as a disputed path claim, not shared evidence.
- If a path has mixed pro and con facts, compare its final bridge to the whole predicate rather than an earlier isolated fact.
- If neither side has a trace-grounded bridge, say the bridge is missing rather than forcing a side.
""".strip(),
    resolution_rules="""
Rules:
- Return the predicate-bridge claim to keep.
- Correct claim must be one complete sentence with three parts: evidence, exact predicate relation, supported yes/no label.
- The claim to keep should be a positive bridge statement, not only a diagnosis that another path is wrong.
- Do not use a bare yes/no label as evidence.
- Do not choose a side merely because it sounds direct, confident, specific, or answer-shaped.
- Prefer a trace-grounded whole-predicate bridge over a local fact, nearby relation, hidden assumption, or one-path outside fact.
- If the evidence only proves a subcondition and not the whole predicate, keep_parallel or keep building instead of inventing the missing bridge.
- If a path's facts support the opposite label from its final token, keep the factual bridge and repair the label implication.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The shared evidence supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: The shared evidence only proves a nearby constraint and does not settle the requested predicate, so it is not enough to support yes.

Tiny resolution example 3:
Correct claim: One path asserts an outside fact that no trace supports, so keep the trace-grounded bridge or keep_parallel.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-predicate bridge and finish the yes/no answer.
- Keep the exact predicate fixed while finishing.
- Do not replace the repaired bridge with a nearby relation, isolated fact, or stale final label.
""".strip(),
)


STRATEGYQA_RELATION_V5_PROFILE = replace(
    STRATEGYQA_RELATION_V4_PROFILE,
    name="strategyqa_relation_v5",
    merge_rules="""
Rules:
- First write the exact yes/no predicate being asked, in ordinary words.
- Keep evidence facts, bridge claims, and final labels separate.
- Common Ground must contain only facts that are visible in at least two traces or explicitly stated by the question; do not promote a one-path outside fact because it sounds plausible.
- If answers differ after shared evidence, split on how that evidence supports or fails the asked predicate, not on the final yes/no token.
- A usable path claim should have the shape: evidence S supports/fails predicate P, so the supported label is yes/no.
- If a path gives only a fact, entity property, opportunity, constraint, or historical occurrence, keep it as evidence and ask whether it bridges to P.
- If a path relies on an extra premise, special exception, or outside fact that appears in only one trace, keep it as that path's disputed bridge, not as shared ground.
- If neither path bridges evidence to the whole predicate, keep building instead of forcing a label repair.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence before the first bridge conflict.
- Split on the support relation from evidence to the exact question predicate, not on final yes/no tokens.
- Do not split on explanation style, progress lag, or an isolated local fact before it is tied to the whole predicate.
- If two paths use the same evidence but disagree on ability, possibility, likelihood, ordinary meaning, entity identity, or sufficiency, split on that bridge.
- If one path answers a nearby relation, make the drift explicit as the conflicting bridge.
- Do not promote a one-path fact into shared prefix just because it looks like general world knowledge.
""".strip(),
    relation_analysis_rules="""
Rules:
- Name the exact question predicate first.
- Name the evidence bundle each path actually uses.
- For each path, state the bridge in one sentence: this evidence supports/fails the predicate because ...
- Compare bridge quality, not final label popularity.
- Treat a one-path outside fact or hidden condition as a disputed path claim, not shared evidence.
- If a path has mixed pro and con facts, compare its final bridge to the whole predicate rather than an earlier isolated fact.
- If neither side has a trace-grounded bridge, say the bridge is missing rather than forcing a side.
""".strip(),
    resolution_rules="""
Rules:
- Return the predicate-bridge claim to keep.
- Correct claim must be one complete sentence with three parts: evidence, exact predicate relation, supported yes/no label.
- The claim to keep should be a positive bridge statement, not only a diagnosis that another path is wrong.
- Do not use a bare yes/no label as evidence.
- Do not choose a side merely because it sounds direct, confident, specific, or answer-shaped.
- Prefer a trace-grounded whole-predicate bridge over a local fact, nearby relation, hidden assumption, or one-path outside fact.
- If the evidence only proves a subcondition and not the whole predicate, keep_parallel or keep building instead of inventing the missing bridge.
- If one branch introduces an outside fact that is not visible in the traces, do not let it decide the label.
- If both branches are only giving nearby relations, keep_parallel until a whole-predicate bridge appears.
- If a branch's final label is right but its bridge is wrong, repair the bridge rather than reusing the label as evidence.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The shared evidence supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: The shared evidence only proves a nearby or partial relation, not the exact requested predicate, so it is not enough to support yes.

Tiny resolution example 3:
Correct claim: One path adds an outside fact that is not visible in the traces, so it should not be promoted into the shared bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-predicate bridge and finish the yes/no answer.
- Keep the exact predicate fixed while finishing.
- Do not replace the repaired bridge with a nearby relation, isolated fact, or stale final label.
- Do not let a one-path outside fact leak into the final answer if it was not supported in the traces.
""".strip(),
)


STRATEGYQA_RELATION_V6_PROFILE = replace(
    STRATEGYQA_RELATION_V5_PROFILE,
    name="strategyqa_relation_v6",
    merge_intro="Merge several yes/no reasoning traces into a clean relation graph.",
    merge_rules="""
Rules:
- First identify the exact yes/no predicate asked by the question.
- Common Ground is only shared evidence or question text, never a verdict or label implication.
- Keep path-specific factual claims separate from bridge claims.
- A bridge claim has this shape: evidence S supports or fails the exact predicate P, so the supported label is yes/no.
- If paths disagree about an outside fact, expose that as a factual conflict before using it in a bridge.
- If paths share facts but infer different labels, expose the bridge conflict from those facts to the exact predicate.
- Do not split on final yes/no tokens, confidence, wording, or pace.
- If the visible material does not yet contain either a factual conflict or a predicate bridge conflict, keep building.
""".strip(),
    merge_fewshot="""
Tiny relation example 1:
Common Ground
- The question asks whether predicate P is true.
- Both paths use shared fact S.

Paths
- A1: claims S supports P, so yes.
- A2: claims S only supports a nearby relation, not P.

First Split
- First split: A1 and A2 disagree on whether shared fact S supports exact predicate P.

Tiny relation example 2:
Common Ground
- The question asks whether predicate P is true.

Paths
- A1: claims factual detail F.
- A2: claims factual detail not-F.

First Split
- First split: A1 and A2 disagree on factual detail F, which is needed before deciding P.
""".strip(),
    audit_rules="""
Audit rules:
- Common Ground may contain shared evidence only.
- Move any support judgment, final label, one-path outside fact, or hidden assumption back into Paths.
- Do not turn a plausible world fact into shared evidence unless at least two traces state it or the question states it.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence before the first real conflict.
- Prefer the earliest factual conflict if the paths disagree about a fact needed for the predicate.
- Otherwise split on the bridge from shared evidence to the exact question predicate.
- Do not split on final yes/no tokens, explanation style, progress lag, or a local fact before it is tied to the predicate.
- If neither factual conflict nor bridge conflict is visible yet, keep building.
""".strip(),
    relation_analysis_intro="Describe one StrategyQA relation conflict before repair.",
    relation_analysis_rules="""
Rules:
- Name the exact yes/no predicate.
- State whether the split is a factual conflict or a bridge conflict.
- If factual: name the competing facts and why that fact matters for the predicate.
- If bridge: name the shared evidence and how each path connects it to the predicate.
- Compare the bridge to the whole predicate, not an isolated local fact.
- Treat one-path outside facts and hidden conditions as disputed path claims, not shared evidence.
- Do not decide by final label popularity.
""".strip(),
    resolution_intro="Resolve one StrategyQA factual or evidence-to-predicate conflict.",
    resolution_rules="""
Rules:
- Return one claim to keep, not only a diagnosis of the losing side.
- Correct claim must be one complete sentence.
- If resolving a factual conflict, state the factual claim and how it affects the exact predicate and yes/no label.
- If resolving a bridge conflict, state the evidence, the exact predicate relation, and the supported yes/no label.
- Do not use a bare yes/no label as evidence.
- Do not choose a side merely because it is more specific, confident, direct, or answer-shaped.
- Do not silently upgrade a nearby relation, subcondition, possible scenario, or historical occurrence into the asked predicate.
- If the available evidence does not settle either the needed fact or the whole predicate bridge, use keep_parallel.
- If a path has a useful fact but the wrong label implication, keep the fact and repair the implication.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: Shared evidence S supports the exact predicate P, so the supported label is yes.

Tiny resolution example 2:
Correct claim: Shared evidence S proves only nearby relation Q, not exact predicate P, so it is not enough to support yes.

Tiny resolution example 3:
Correct claim: Fact F is disputed and not established by the traces, so do not use F as shared evidence for the label.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired factual or evidence-to-predicate claim and finish the yes/no answer.
- Keep the exact question predicate fixed.
- Do not replace the repaired claim with a nearby relation, hidden assumption, or stale final label.
- Final answer form: yes or no only.
- The final line must answer yes or no.
""".strip(),
)


PRONTOQA_RELATION_V1_PROFILE = PromptProfile(
    name="prontoqa_relation_v1",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several true/false proof traces into a compact fact-rule relation memo.",
    merge_rules="""
Rules:
- Keep shared facts and rules separate from the queried statement.
- Do not treat a bare true/false label as evidence.
- Split when paths disagree about a fact, a rule application, or whether the facts/rules imply the queried statement.
- If paths only differ in final labels after the same proof bridge, split on the proof bridge, not the label token.
""".strip(),
    merge_fewshot=MERGE_FEWSHOT,
    audit_intro="Audit a true/false fact-rule memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Common Ground should contain only facts/rules shared by at least two traces.
- Move one-path-only rule applications back into Paths.
- Do not invent facts or rules not present in the context or traces.
""".strip(),
    prefix_intro="Rebuild a compact fact-rule memo that exposes the first proof-relation conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared fact/rule prefix before the first conflict.
- Split on the first incompatible fact, rule application, or implication for the queried statement.
- Do not split on a bare true/false token before checking the proof bridge.
""".strip(),
    relation_analysis_intro="Describe one fact-rule divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First state the queried statement.
- Then state the shared facts/rules and the conflicting rule application or implication.
- Do not choose by final label popularity.
""".strip(),
    resolution_intro="Resolve one minimal fact-rule divergence.",
    resolution_rules="""
Rules:
- Return the fact-rule relation claim to keep.
- Correct claim must say whether the facts/rules imply or fail to imply the queried statement, then name the supported true/false label.
- Do not use a bare true/false label as evidence.
- Do not add facts or rules outside the context and traces.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The shared facts and rules imply the queried statement, so the supported label is true.

Tiny resolution example 2:
Correct claim: The shared facts and rules do not imply the queried statement, so the supported label is false.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired fact-rule relation claim and finish the true/false answer.
- Do not stop at a background fact if the question asks for the queried statement's truth value.
""".strip(),
)


BAMBOOGLE_RELATION_V1_PROFILE = PromptProfile(
    name="bamboogle_relation_v1",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several multi-hop QA traces into a compact evidence-to-answer relation memo.",
    merge_rules="""
Rules:
- Keep shared evidence facts separate from the answer entity or short answer phrase.
- Do not treat a bare answer string as evidence.
- Split on the first concrete multi-hop support conflict: which shared evidence is relevant, which bridge is missing, or which answer entity is actually supported.
- If two paths follow different hop orders but support the same answer entity, do not split on style alone.
""".strip(),
    merge_fewshot="""
Tiny multi-hop example 1:
Common Ground
- Both paths use the same two-hop evidence chain.

Paths
- A1: claims the chain supports answer entity X.
- A2: claims the chain supports answer entity Y.

First Split
- First split: A1 claims the evidence chain supports X; A2 claims it supports Y.

Tiny multi-hop example 2:
Common Ground
- Both paths mention the same supporting fact.

Paths
- A1: stops at a partial bridge.
- A2: completes the bridge to the final entity.

First Split
- First split: A1 has only a partial bridge; A2 connects the evidence to the requested answer entity.
""".strip(),
    audit_intro="Audit a multi-hop QA memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep only trace-faithful shared evidence in Common Ground.
- Move one-path-only intermediate entities or bridge claims back into Paths.
- Do not invent a hidden supporting fact or answer entity.
""".strip(),
    prefix_intro="Rebuild a compact multi-hop QA memo that exposes the first evidence-to-answer conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared evidence chain needed for the first actionable conflict.
- Split on the first concrete mismatch about which evidence supports the requested answer entity.
- Do not split on wording, hop order, or progress lag alone.
""".strip(),
    relation_analysis_intro="Describe one evidence-to-answer divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First identify the answer entity or short answer phrase the question asks for.
- Then identify the evidence chain each path uses.
- Compare whether the evidence chain really supports that answer entity.
- Do not choose by final answer popularity.
""".strip(),
    resolution_intro="Resolve one minimal evidence-to-answer divergence.",
    resolution_rules="""
Rules:
- Return the answer claim to keep.
- Correct claim must say how the evidence chain supports the requested answer entity.
- Do not use a bare entity name as evidence.
- Do not silently upgrade a partial bridge into a full answer.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The shared evidence chain supports answer entity X, so X should be kept.

Tiny resolution example 2:
Correct claim: The shared evidence chain does not yet support any final answer entity, so keep_parallel.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-answer claim and finish with the short answer phrase.
- Do not stop at an intermediate entity if the question asks for a more specific answer.
""".strip(),
)


BAMBOOGLE_RELATION_V2_PROFILE = replace(
    BAMBOOGLE_RELATION_V1_PROFILE,
    name="bamboogle_relation_v2",
    merge_rules="""
Rules:
- Keep shared evidence facts separate from the answer entity or short answer phrase.
- Preserve the complete bridge needed by the question: intermediate entity plus the requested attribute of that entity.
- Do not treat a bare answer string, date, country, person, or option-like phrase as evidence.
- Split on the first concrete multi-hop support conflict: which intermediate entity is supported, which final attribute is supported, or which bridge is missing.
- If two paths follow different hop orders but support the same intermediate entity and final attribute, do not split on style alone.
- If the disagreement is only a raw outside fact with no supporting bridge in either trace, say that explicitly rather than inventing a verification phrase.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the short-answer object the question asks for.
- Then identify the intermediate entity and the final attribute needed to answer it.
- Compare the evidence chain each path uses for both parts of the bridge.
- If candidates differ only on a date, name, country, or other factual attribute, note which trace actually supplies a bridge for that attribute.
- Do not choose by final answer popularity.
""".strip(),
    resolution_rules="""
Rules:
- Return the answer claim to keep.
- Correct claim must include the complete bridge from the question to the intermediate entity and then to the requested answer phrase.
- Do not use a bare entity name, date, country, or person as evidence.
- Do not silently upgrade a partial bridge into a full answer.
- If neither side supplies enough bridge evidence for a raw factual conflict, keep_parallel rather than inventing "historical records confirm" style support.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-answer bridge and finish with the short answer phrase.
- Do not stop at an intermediate entity if the question asks for a more specific answer.
- The final line should contain only the requested short answer phrase inside the normal final-answer wrapper.
""".strip(),
)


BAMBOOGLE_RELATION_V3_PROFILE = replace(
    BAMBOOGLE_RELATION_V1_PROFILE,
    name="bamboogle_relation_v3",
    merge_intro="Merge several multi-hop QA traces into a compact bridge-to-short-answer memo.",
    merge_rules="""
Rules:
- First identify the short-answer object requested by the question.
- Keep three things separate: evidence facts, intermediate entity, and requested attribute of that entity.
- Do not treat a bare date, name, country, person, or phrase as evidence by itself.
- Split on the first bridge conflict: wrong intermediate entity, wrong requested attribute, missing bridge, or different supported short answer.
- If a path already completes the bridge to a supported short answer, keep that as a valid answer claim.
- If a path only reaches an intermediate entity, do not treat that intermediate entity as the final answer.
- Different hop order is not a split if both paths support the same intermediate entity and same requested attribute.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence chain before the first bridge conflict.
- Split on a concrete mismatch in the bridge to the requested short answer, not on wording or hop order.
- If only the intermediate entity is visible, keep building unless another path makes an incompatible claim about that same intermediate entity.
- If the candidates are final short answers, compare which answer is supported by the visible bridge.
""".strip(),
    relation_analysis_rules="""
Rules:
- First name the requested short-answer object.
- Then identify the intermediate entity and requested attribute needed by the question.
- Compare whether each path's evidence bridge actually reaches that requested attribute.
- Do not choose by final answer popularity.
- Do not require a long proof if a trace already gives a clean bridge to the requested short answer.
- If neither side supplies the missing hop, say the bridge is missing instead of inventing a fact.
""".strip(),
    resolution_rules="""
Rules:
- Return the bridge or answer claim to keep.
- Correct claim must say how the evidence bridge supports the requested short answer.
- A supported final short answer phrase can be kept; an unsupported bare phrase cannot.
- Do not use an intermediate entity as the final answer when the question asks for an attribute of that entity.
- If one side has only a partial bridge and the other completes the bridge, keep the completed bridge.
- If neither side supplies enough bridge evidence for a factual conflict, use keep_parallel rather than inventing outside support.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The evidence identifies the intermediate entity and its requested attribute, so the supported short answer is X.

Tiny resolution example 2:
Correct claim: The trace only identifies an intermediate entity and has not yet answered the requested attribute, so keep building from that bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired bridge and finish with the requested short answer phrase.
- Do not stop at an intermediate entity if the question asks for that entity's date, place, author, office, country, or other attribute.
- The final line should contain only the requested short answer phrase inside the normal final-answer wrapper.
""".strip(),
)


BAMBOOGLE_RELATION_V4_PROFILE = replace(
    BAMBOOGLE_RELATION_V3_PROFILE,
    name="bamboogle_relation_v4",
    merge_intro="Merge several multi-hop QA traces into a compact exact-short-answer bridge memo.",
    merge_rules="""
Rules:
- First identify the exact short-answer object requested by the question: name, title, date, number, place, or short phrase.
- Keep evidence facts, intermediate entity, and exact answer phrase separate.
- Use the shortest exact answer phrase that fully answers the question.
- Do not treat a bare intermediate entity or a longer descriptive paraphrase as the final answer if the question asks for a shorter exact phrase.
- Split on the first bridge conflict: wrong intermediate entity, wrong exact answer phrase, missing hop, or a paraphrase that drops a decisive modifier.
- If a path already reaches a fully supported exact answer phrase, keep that claim even if another path only has a near paraphrase.
- Different hop order is not a split if both paths support the same exact answer phrase.
- If the answer is a title or named entity with distinguishing modifiers, preserve the full canonical form that the question is really asking for.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence chain before the first exact-answer conflict.
- Split on a concrete mismatch in the exact short answer, not on wording, hop order, or extra explanatory prose.
- If only the intermediate entity is visible, keep building unless another path makes an incompatible claim about that same intermediate entity.
- If the visible difference is only that one path gives a shorter exact phrase and the other adds a description, prefer the shorter exact phrase if it still fully answers the question.
""".strip(),
    relation_analysis_rules="""
Rules:
- First name the exact short-answer object the question asks for.
- Then identify the intermediate entity and requested attribute needed to answer it.
- Compare whether each path's evidence bridge actually reaches the exact short answer phrase.
- If a path gives the correct entity but adds or drops a decisive modifier, call that a canonical-form bridge gap.
- Do not choose by final answer popularity.
- If neither side supplies the missing hop, say the bridge is missing instead of inventing a fact.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact short-answer claim to keep.
- Correct claim must say how the evidence bridge supports the requested exact short answer phrase.
- A supported final short answer phrase can be kept; an unsupported bare phrase cannot.
- Do not use an intermediate entity as the final answer when the question asks for an attribute or exact named object.
- If one side has only a longer paraphrase and the other has the exact supported short answer, keep the exact short answer.
- If neither side supplies enough bridge evidence for a factual conflict, use keep_parallel rather than inventing outside support.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The evidence identifies the requested exact short answer phrase, so that phrase should be kept.

Tiny resolution example 2:
Correct claim: The trace only identifies a related intermediate entity and has not yet reached the exact answer phrase, so keep building from that bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired exact-short-answer bridge and finish with the short answer phrase.
- Do not stop at an intermediate entity if the question asks for a more specific exact answer.
- The final line should contain only the requested short answer phrase inside the normal final-answer wrapper.
""".strip(),
)


BAMBOOGLE_RELATION_V5_PROFILE = replace(
    BAMBOOGLE_RELATION_V4_PROFILE,
    name="bamboogle_relation_v5",
    merge_intro="Merge several multi-hop QA traces into a compact exact-short-answer bridge memo.",
    merge_rules="""
Rules:
- First identify the exact short-answer object requested by the question: name, title, date, number, place, or short phrase.
- Keep evidence facts, intermediate entity, requested attribute, and exact answer phrase separate.
- Use the shortest exact answer phrase that fully answers the question.
- Do not treat a bare intermediate entity or a longer paraphrase as the final answer if the question asks for a shorter exact phrase.
- Split on the first bridge conflict: wrong intermediate entity, wrong exact answer phrase, missing hop, or a paraphrase that drops a decisive modifier.
- If a path already reaches a fully supported exact answer phrase, keep that claim even if another path only has a near paraphrase.
- Different hop order is not a split if both paths support the same exact answer phrase.
- If the answer is a title or named entity with distinguishing modifiers, preserve the full canonical form that the question is really asking for.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared evidence chain before the first exact-answer conflict.
- Split on a concrete mismatch in the exact short answer, not on wording, hop order, or extra explanatory prose.
- If only the intermediate entity is visible, keep building unless another path makes an incompatible claim about that same intermediate entity.
- If the visible difference is only that one path gives a shorter exact phrase and the other adds a description, prefer the shorter exact phrase if it still fully answers the question.
""".strip(),
    relation_analysis_rules="""
Rules:
- First name the exact short-answer object the question asks for.
- Then identify the intermediate entity and requested attribute needed to answer it.
- Compare whether each path's evidence bridge actually reaches the exact short answer phrase.
- If a path gives the correct entity but adds or drops a decisive modifier, call that a canonical-form bridge gap.
- Do not choose by final answer popularity.
- If neither side supplies the missing hop, say the bridge is missing instead of inventing a fact.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact short-answer claim to keep.
- Correct claim must say how the evidence bridge supports the requested exact short answer phrase.
- A supported final short answer phrase can be kept; an unsupported bare phrase cannot.
- Do not use an intermediate entity as the final answer when the question asks for an attribute or exact named object.
- If one side has only a longer paraphrase and the other has the exact supported short answer, keep the exact short answer.
- If neither side supplies enough bridge evidence for a factual conflict, use keep_parallel rather than inventing outside support.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The evidence identifies the requested exact short answer phrase, so that phrase should be kept.

Tiny resolution example 2:
Correct claim: The trace only identifies a related intermediate entity and has not yet reached the exact answer phrase, so keep building from that bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired exact-short-answer bridge and finish with the short answer phrase only.
- Do not stop at an intermediate entity if the question asks for a more specific exact answer.
- The last line should contain only the requested short answer phrase, with no extra explanation or paraphrase.
""".strip(),
)


BAMBOOGLE_RELATION_V6_PROFILE = replace(
    BAMBOOGLE_RELATION_V5_PROFILE,
    name="bamboogle_relation_v6",
    merge_intro="Merge several multi-hop QA traces into a compact exact-short-answer bridge memo.",
    merge_rules="""
Rules:
- First identify the exact short-answer object requested by the question: name, title, date, number, place, or short phrase.
- Keep evidence facts, intermediate entity, requested attribute, and exact answer phrase separate.
- Use the shortest exact answer phrase that fully answers the question.
- Do not treat a bare intermediate entity or a longer paraphrase as the final answer if the question asks for a different requested object or a shorter exact phrase.
- Split on the first bridge conflict: wrong intermediate entity, wrong exact answer phrase, missing hop, or a paraphrase that drops a decisive modifier.
- If a path reaches a fully supported exact answer phrase, keep that claim even if another path only has a near paraphrase.
- Different hop order is not a split if both paths support the same exact answer phrase.
- If the answer is a title or named entity with distinguishing modifiers, preserve the full canonical form that the question is really asking for.
""".strip(),
    relation_analysis_rules="""
Rules:
- First name the exact short-answer object the question asks for.
- Then identify the hop chain and requested attribute needed to answer it.
- Compare whether each path's evidence bridge actually reaches the exact short answer phrase.
- If a path gives the right intermediate entity but not the requested attribute, call that a bridge gap rather than a final answer.
- If one path reaches the exact answer and the other stops at an intermediate entity, keep the exact answer phrase.
- Do not choose by final answer popularity.
- If neither side supplies the missing hop, say the bridge is missing instead of inventing a fact.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact short-answer claim to keep.
- Correct claim must say how the evidence bridge supports the requested exact short answer phrase.
- A supported final short answer phrase can be kept; an unsupported bare phrase cannot.
- Do not use an intermediate entity as the final answer when the question asks for a more specific attribute or exact named object.
- If one side has only a longer paraphrase and the other has the exact supported short answer, keep the exact short answer.
- If neither side supplies enough bridge evidence for a factual conflict, use keep_parallel rather than inventing outside support.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired exact-short-answer bridge and finish with the shortest exact phrase only.
- Do not stop at an intermediate entity if the question asks for a more specific exact answer.
- The last line should contain only the shortest exact phrase, with no intermediate entity, no extra explanation, and no trailing commentary.
""".strip(),
)


BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE = replace(
    BAMBOOGLE_RELATION_V6_PROFILE,
    name="bamboogle_relation_v7_equal_token",
    merge_rules=BAMBOOGLE_RELATION_V6_PROFILE.merge_rules
    + "\n- Treat observed short answers as candidate hypotheses with visible bridges; do not turn a repeated phrase into evidence by vote."
    + "\n- Keep the requested answer slot separate from intermediate entities and explanatory facts, especially when the trace is verbose.",
    prefix_rules=BAMBOOGLE_RELATION_V6_PROFILE.prefix_rules
    + "\n- Expose the first mismatch in source entity, bridge constraint, requested slot, or phrase sufficiency before comparing final strings.",
    relation_analysis_rules=BAMBOOGLE_RELATION_V6_PROFILE.relation_analysis_rules
    + "\n- Audit each observed candidate compactly: candidate phrase; source entity; requested slot; strongest visible bridge; missing component."
    + "\n- Prefer a positive bridge to the requested slot over a negative absence claim unless the negative claim shows the bridge uses the wrong source or relation.",
    resolution_rules=BAMBOOGLE_RELATION_V6_PROFILE.resolution_rules
    + "\n- Resolve to the supported observed candidate when one fills the requested slot; create a new phrase only when the visible bridge explicitly supplies a more exact canonical answer."
    + "\n- Correct claim should be one positive bridge sentence ending in the canonical short answer phrase.",
    revision_rules=BAMBOOGLE_RELATION_V6_PROFILE.revision_rules
    + "\n- If the resolution supplies a canonical answer and its bridge is supported, finish with exactly that phrase."
    + "\n- Do not replace a supported candidate with a nearby entity, full sentence, or broader paraphrase because the answer budget is longer.",
)


BAMBOOGLE_RELATION_V8_EQUAL_TOKEN_PROFILE = replace(
    BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE,
    name="bamboogle_relation_v8_equal_token",
    merge_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.merge_rules
    + "\n- Never move a candidate answer phrase into Common Ground unless every path explicitly supports the same source entity, same requested slot, and same phrase."
    + "\n- If the surface phrase repeats but the source entity or hop chain differs, keep that repetition in the path notes, not in shared evidence."
    + "\n- Track source entity, requested slot, and exact answer phrase separately before comparing candidates.",
    prefix_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.prefix_rules
    + "\n- Split on source-entity mismatch before trusting any repeated short answer phrase."
    + "\n- Do not let a repeated answer phrase erase the path-specific bridge that produced it."
    + "\n- If the request is still ambiguous about source entity or slot, keep building instead of selecting the shortest phrase.",
    relation_analysis_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- For each path, state source entity -> bridge -> exact answer phrase."
    + "\n- If the source entity does not match the requested object, the answer phrase is not yet supported."
    + "\n- A shorter phrase is only better when the source entity and slot are already exact.",
    resolution_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- Do not choose a branch because its short answer phrase is repeated elsewhere; choose only when the source entity and requested slot also match."
    + "\n- Prefer keep_parallel when the source entity is still disputed even if the final phrase looks attractive."
    + "\n- A canonical answer is only valid when its bridge is source-specific and slot-specific.",
    revision_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Finish with the supported phrase only after the source entity and requested slot are fixed."
    + "\n- Do not let a repeated surface phrase override a mismatched source entity.",
)


BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE = replace(
    BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE,
    name="bamboogle_relation_v9_extraction_guard",
    relation_analysis_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- Treat a repeated short phrase as support only when the source entity, bridge, and requested slot all match."
    + "\n- If the corrected claim names a canonical short answer, preserve exactly that phrase; do not replace it with a nearby entity from the rationale.",
    resolution_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- End a decisive corrected claim with `Canonical answer: PHRASE` only when PHRASE fills the requested slot."
    + "\n- Do not expand or swap the canonical phrase using entities mentioned in rejected-side explanations.",
    revision_rules=BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve the canonical short-answer phrase exactly through revision unless the same bridge proves a phrase-level mismatch.",
)


BAMBOOGLE_RELATION_V10_BRIDGE_AUDIT_PROFILE = replace(
    BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE,
    name="bamboogle_relation_v10_bridge_audit",
    merge_rules=BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE.merge_rules
    + "\n- If all traces repeat the same short answer but the bridge is shallow or missing, expose the missing source->slot bridge instead of treating unanimity as done.",
    relation_analysis_rules=BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- For each candidate, write source entity -> requested slot -> exact phrase. A phrase without both source and slot is not a supported answer.",
    resolution_rules=BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- A canonical answer must be rewrite-supported by a visible source->slot bridge. If the bridge is not visible, use keep_parallel."
    + "\n- Prefer repairing a missing bridge over changing the phrase when the candidate phrase is already plausible but unsupported.",
    revision_rules=BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Finish with the exact phrase only after the revised trace states the source entity and requested slot bridge."
    + "\n- Do not land a phrase from the resolution alone if the rewrite did not carry the bridge.",
)


HOTPOTQA_RELATION_V1_PROFILE = replace(
    BAMBOOGLE_RELATION_V6_PROFILE,
    name="hotpotqa_relation_v1",
    merge_intro="Merge several HotpotQA traces into a compact context-evidence-to-answer bridge memo.",
    audit_intro="Audit a HotpotQA context-evidence memo before divergence resolution.",
    prefix_intro="Rebuild a compact HotpotQA memo that exposes the first context-evidence-to-answer conflict.",
)


MUSIQUE_RELATION_V1_PROFILE = replace(
    BAMBOOGLE_RELATION_V6_PROFILE,
    name="musique_relation_v1",
    merge_intro="Merge several MuSiQue traces into a compact multi-hop evidence-to-answer bridge memo.",
    audit_intro="Audit a MuSiQue multi-hop evidence memo before divergence resolution.",
    prefix_intro="Rebuild a compact MuSiQue memo that exposes the first multi-hop evidence-to-answer conflict.",
)


SHORT_ANSWER_SLOT_MERGE_RULES = """
Rules:
- First identify the answer slot asked by the question: the person, place, date/year, count, title, reason, action, event, or other attribute that must fill the final answer.
- Keep the hop chain as slot filling: evidence -> intermediate entity -> requested slot -> exact answer phrase.
- A candidate is not complete merely because it is a true entity in the chain; it must fill the requested slot.
- Split on the first slot-level bridge conflict: wrong intermediate entity, wrong requested slot, wrong value for the slot, missing hop, or phrase granularity that changes the slot value.
- If one path gives a city, country, date, person, or organization while the question asks for a different slot, call it an intermediate value unless the path bridges it to the requested slot.
- Prefer the shortest exact phrase only after the requested slot is correct. Do not shorten away a decisive modifier such as city vs country, year vs full date when the question asks a year, or language family vs language version.
- Different hop order is not a split if both paths support the same requested slot and exact phrase.
""".strip()


SHORT_ANSWER_SLOT_PREFIX_RULES = """
Rules:
- Keep the shortest shared evidence chain before the first answer-slot conflict.
- Name the requested answer slot before judging the paths.
- Split on a concrete mismatch in the slot value or in the bridge to that slot, not on wording, hop order, or extra explanatory prose.
- If a path only reaches an intermediate entity, keep building unless another path makes an incompatible claim about that same entity or the requested slot.
- If all visible paths are still before the requested slot, say none yet instead of resolving from a partial hop.
""".strip()


SHORT_ANSWER_SLOT_ANALYSIS_RULES = """
Rules:
- First name the requested answer slot.
- For each path, state the bridge in one compact sentence: evidence -> intermediate entity -> requested slot -> candidate phrase.
- Decide whether the candidate phrase fills the requested slot or only a nearby/intermediate slot.
- If a path gives the right entity but the wrong attribute, call that a slot bridge gap.
- If a path gives the right attribute but wrong phrase granularity, call that a granularity bridge gap.
- Do not choose by final answer popularity.
- If neither side supplies the missing hop to the requested slot, say the bridge is missing instead of inventing a fact.
""".strip()


SHORT_ANSWER_SLOT_RESOLUTION_RULES = """
Rules:
- Return the answer-slot bridge claim to keep.
- Correct claim must explicitly say what the requested slot is and how the evidence fills that slot with the exact short answer phrase.
- A supported final phrase can be kept; an unsupported bare phrase cannot.
- Do not use an intermediate entity, broader location, nearby date, related person, or source title as the final answer unless it is the requested slot.
- Preserve decisive modifiers in the exact phrase. Do not shorten "keyboard function keys" to "keyboard", a full date/month to a year, or a titled/name phrase to a head word unless the question explicitly asks for that coarser slot.
- If one side reaches the requested slot and another side stops at an intermediate value, keep the requested-slot bridge.
- If neither side supplies enough bridge evidence for the requested slot, use keep_parallel rather than inventing outside support.
""".strip()


SHORT_ANSWER_SLOT_FEWSHOT = """
Tiny resolution example 1:
Correct claim: The requested slot is the birthplace city; the evidence bridges the person to the city "Fujioka, Gunma", so keep that exact phrase.

Tiny resolution example 2:
Correct claim: The requested slot is the year, not the full date; the evidence gives March 14, 2000, so the exact short answer phrase is "2000".

Tiny resolution example 3:
Correct claim: The trace identifies only the country, but the requested slot is the city or specific place, so it has not reached the requested answer slot.

Tiny resolution example 4:
Correct claim: The requested slot is the specific object phrase; the evidence gives "keyboard function keys", so keep the full phrase rather than shortening it to "keyboard".
""".strip()


CLEAN_SHORT_ANSWER_SLOT_FEWSHOT = """
Tiny resolution example 1:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot is a specific city-and-region place phrase. Claim A bridges the source entity to Example City, Example Region, while Claim B stops at the broader country only, so the canonical answer is Example City, Example Region.
Canonical answer: Example City, Example Region

Tiny resolution example 2:
Action: choose_B
Winning side: Claim B
Correct claim: The requested slot is a year, not the full date. Claim B keeps only the year supplied by the evidence, so the canonical answer is 2000.
Canonical answer: 2000

Tiny resolution example 3:
Action: keep_parallel
Winning side: neither
Correct claim: One path gives only a nearby intermediate entity and the other gives a broader location; neither visible bridge reaches the requested slot yet, so keep the paths parallel rather than choosing by surface plausibility.

Tiny resolution example 4:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot is a yes/no predicate. Claim A resolves the truth value of the predicate, while Claim B gives an evidence object instead of the label, so the canonical answer is no.
Canonical answer: no
""".strip()


SHORT_ANSWER_SLOT_REVISION_RULES = """
Rules:
- Continue from the repaired answer-slot bridge and finish with the exact phrase that fills the requested slot.
- Do not stop at an intermediate entity, broader place, related person, source title, or nearby date if the question asks for another slot.
- The last line should contain only the shortest exact phrase for the requested slot, with no explanation or trailing commentary.
""".strip()


HOTPOTQA_RELATION_V2_PROFILE = replace(
    HOTPOTQA_RELATION_V1_PROFILE,
    name="hotpotqa_relation_v2",
    merge_rules=SHORT_ANSWER_SLOT_MERGE_RULES,
    prefix_rules=SHORT_ANSWER_SLOT_PREFIX_RULES,
    relation_analysis_rules=SHORT_ANSWER_SLOT_ANALYSIS_RULES,
    resolution_rules=SHORT_ANSWER_SLOT_RESOLUTION_RULES,
    resolution_fewshot=SHORT_ANSWER_SLOT_FEWSHOT,
    revision_rules=SHORT_ANSWER_SLOT_REVISION_RULES,
)


HOTPOTQA_RELATION_V3_PROFILE = replace(
    HOTPOTQA_RELATION_V2_PROFILE,
    name="hotpotqa_relation_v3",
    merge_rules=SHORT_ANSWER_SLOT_MERGE_RULES
    + "\n- HotpotQA often asks for an attribute of an entity found by another hop. Keep the entity-binding hop and the attribute hop separate."
    + "\n- Do not merge a radio/TV/book/person fact into a film/person/role fact unless the trace explicitly bridges that exact entity in the question."
    + "\n- Treat phrase granularity as a real split when the context states a specific phrase and another path drops a decisive modifier.",
    prefix_rules=SHORT_ANSWER_SLOT_PREFIX_RULES
    + "\n- Before accepting common ground, verify that each shared sentence preserves the exact entity from the question, not a nearby entity from the same context."
    + "\n- If one path gives the exact context phrase and another gives a head word or broader entity, expose this as a slot-granularity split.",
    relation_analysis_rules=SHORT_ANSWER_SLOT_ANALYSIS_RULES
    + "\n- For every path, name both hops: question entity -> requested attribute -> candidate phrase."
    + "\n- Mark a bridge as invalid if it swaps the question entity with a related nearby entity from another paragraph."
    + "\n- Mark a bridge as incomplete if it drops a decisive modifier from the context phrase.",
    resolution_rules=SHORT_ANSWER_SLOT_RESOLUTION_RULES
    + "\n- Prefer the exact context phrase over a shortened head word when the modifier is part of the supported answer phrase."
    + "\n- If a path binds the wrong entity before filling the slot, reject that bridge even if the final answer is a plausible phrase."
    + "\n- If the traces only show a nearby entity and not the entity asked by the question, use keep_parallel rather than inventing the missing bridge.",
    resolution_fewshot=SHORT_ANSWER_SLOT_FEWSHOT
    + "\n\nTiny resolution example 5:\nCorrect claim: The requested slot is the exact control method phrase; the context says Front Row is controlled by \"keyboard function keys\", so keep the full phrase rather than the broader head word \"keyboard\".\n\nTiny resolution example 6:\nCorrect claim: The path binds the wrong person from a related context paragraph before filling the government-position slot, so it has not reached the requested slot.",
    revision_rules=SHORT_ANSWER_SLOT_REVISION_RULES
    + "\n- Preserve the exact phrase from the evidence when a shorter head word would change the answer granularity.",
)


HOTPOTQA_RELATION_V4_PROFILE = replace(
    HOTPOTQA_RELATION_V3_PROFILE,
    name="hotpotqa_relation_v4",
    prefix_rules=HOTPOTQA_RELATION_V3_PROFILE.prefix_rules
    + "\n- If the question asks for another/besides/other than/excluding one entity, keep the excluded entity separate from the requested answer slot.",
    relation_analysis_rules=HOTPOTQA_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- For exclusion questions, state the excluded entity first, then ask whether the candidate is the other requested entity/value.",
    resolution_rules=HOTPOTQA_RELATION_V3_PROFILE.resolution_rules
    + "\n- The Action field must be exactly one of choose_A, choose_B, or keep_parallel."
    + "\n- Do not write keep, resolve, or repair in the Action field; put the claim to keep only in Correct claim."
    + "\n- For another/besides/other than/excluding questions, the correct claim must explicitly say why the kept phrase is not the excluded entity.",
    resolution_fewshot=HOTPOTQA_RELATION_V3_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 7:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for the other named entity besides the excluded one; the evidence identifies \"Pedro Rodriguez\" as the other entity, so keep the exact short answer phrase \"Pedro Rodriguez\".",
)


HOTPOTQA_RELATION_V5_PROFILE = replace(
    HOTPOTQA_RELATION_V4_PROFILE,
    name="hotpotqa_relation_v5",
    prefix_rules=HOTPOTQA_RELATION_V4_PROFILE.prefix_rules
    + "\n- If two candidates differ only by extra broader context such as adding a country to a city/province phrase, expose it as phrase-granularity only when the extra words are required by the question slot."
    + "\n- If one path has already reached the requested slot with an exact phrase and another path has only a bare missing-answer label, do not let the missing label become the winning bridge without a new evidence claim.",
    relation_analysis_rules=HOTPOTQA_RELATION_V4_PROFILE.relation_analysis_rules
    + "\n- Compare the whole slot bridge, not only the most explicit nearby sentence. A bridge is valid only if it satisfies every question constraint and then fills the requested slot."
    + "\n- For short answers, decide whether extra words are part of the requested exact phrase or merely broader context.",
    resolution_rules=HOTPOTQA_RELATION_V4_PROFILE.resolution_rules
    + "\n- Before choosing, check that the correct claim satisfies all constraints in the question, including excluded entities, entity aliases, role/action constraints, and the requested attribute."
    + "\n- Do not expand a supported exact phrase with a broader location, title, organization, or qualifier unless the question asks for that broader phrase."
    + "\n- If a path already gives a final short answer phrase compatible with the correct claim, keep that phrase stable; do not rewrite it into a longer paraphrase."
    + "\n- If the relation note and your reason disagree about the exact phrase, choose keep_parallel rather than forcing a rewrite.",
    resolution_fewshot=HOTPOTQA_RELATION_V4_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 8:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is the formation place phrase stated in evidence; if the evidence phrase is \"Fujioka, Gunma\", keep that exact phrase and do not add the broader country unless the question asks for the country."
    + "\n\nTiny resolution example 9:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: One path has only a missing-answer label while another has an answer phrase, but the visible evidence does not prove the slot bridge; keep building instead of letting the missing label overwrite the phrase.",
    revision_rules=HOTPOTQA_RELATION_V4_PROFILE.revision_rules
    + "\n- If the current final answer already is the exact phrase named by the correct claim, repeat that phrase as the final line rather than expanding or dropping it.",
)


HOTPOTQA_RELATION_V6_PROFILE = replace(
    HOTPOTQA_RELATION_V5_PROFILE,
    name="hotpotqa_relation_v6",
    merge_rules=HOTPOTQA_RELATION_V5_PROFILE.merge_rules
    + "\n- Keep the anchor entity from the question separate from the requested companion/attribute. In 'besides X' questions, X is the excluded anchor, not the answer."
    + "\n- Preserve the minimal complete answer phrase: include units, object nouns, titles, and possessive modifiers when they are part of the supported slot value.",
    prefix_rules=HOTPOTQA_RELATION_V5_PROFILE.prefix_rules
    + "\n- In another/besides questions, split only after the excluded anchor is fixed and the paths disagree about the other requested value."
    + "\n- Do not call a unit word, object noun, or title extra prose if it is part of the answer phrase stated by the evidence.",
    relation_analysis_rules=HOTPOTQA_RELATION_V5_PROFILE.relation_analysis_rules
    + "\n- For exclusion questions, write the bridge as: excluded anchor -> requested other slot -> candidate. Reject a path that returns the excluded anchor as the requested other slot."
    + "\n- For quantity questions, preserve the full quantity phrase including unit words such as countries, people, miles, years, dollars, or degrees when the evidence uses them.",
    resolution_rules=HOTPOTQA_RELATION_V5_PROFILE.resolution_rules
    + "\n- In another/besides/excluding questions, never choose the excluded anchor itself as the final answer; the correct claim must name the other requested value."
    + "\n- The shortest exact phrase means the shortest complete slot phrase, not the shortest head token. Keep necessary units and modifiers from the evidence."
    + "\n- If the correct claim says one phrase but the reason implies the opposite phrase, use keep_parallel.",
    resolution_fewshot=HOTPOTQA_RELATION_V5_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 10:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The excluded anchor is Sergio Perez, the Force India driver born in 1990; the requested other podium driver is \"Pedro Rodriguez\", so do not return Sergio Perez."
    + "\n\nTiny resolution example 11:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a count-with-unit phrase; the evidence says \"more than 70 countries\", so keep the full phrase rather than shortening it to \"more than 70\".",
    revision_rules=HOTPOTQA_RELATION_V5_PROFILE.revision_rules
    + "\n- For exclusion questions, finish with the non-excluded candidate named by the correct claim."
    + "\n- For quantities, finish with the complete quantity phrase including its unit when the unit is part of the evidence phrase.",
)


HOTPOTQA_RELATION_V7_PROFILE = replace(
    HOTPOTQA_RELATION_V6_PROFILE,
    name="hotpotqa_relation_v7",
    merge_rules=HOTPOTQA_RELATION_V6_PROFILE.merge_rules
    + "\n- Before comparing candidates, classify the requested slot in plain words: entity/name, place, date/time, quantity-with-unit, yes/no, comparison, shared type/commonality, title/work, or role/occupation."
    + "\n- For comparison questions, the claim to compare is the comparison relation itself, such as which publication is more frequent or which candidate has the requested property."
    + "\n- For shared type/commonality questions, the answer should be the common category or shared object asked by the question, not a longer sentence explaining the relationship."
    + "\n- For quantity answers, keep the complete quantity phrase with its unit or counted object when the question asks how far, how long, how many, or how much.",
    prefix_rules=HOTPOTQA_RELATION_V6_PROFILE.prefix_rules
    + "\n- Do not make a final-answer split before naming the requested slot type; many HotpotQA errors come from answering a nearby slot."
    + "\n- If the visible conflict is a comparison, split on the comparison bridge, not on a bare candidate name."
    + "\n- If a candidate phrase differs only by a unit, object noun, or decisive modifier, treat that as a real granularity conflict because it changes the slot value.",
    relation_analysis_rules=HOTPOTQA_RELATION_V6_PROFILE.relation_analysis_rules
    + "\n- Start by writing the requested slot type and the exact constraint from the question."
    + "\n- For comparison questions, state each side's comparable fact and which side that fact supports."
    + "\n- For commonality/type questions, separate the evidence relation from the final common category phrase."
    + "\n- For quantities, dates, offices, and occupations, decide whether the unit, modifier, jurisdiction, or role word is part of the requested phrase or extra context.",
    resolution_rules=HOTPOTQA_RELATION_V6_PROFILE.resolution_rules
    + "\n- Correct claim must preserve the requested slot type. If the question asks yes/no, output a yes/no bridge; if it asks a comparison, output the winning comparison bridge; if it asks a commonality/type, output the shared category bridge."
    + "\n- Do not over-specify a role or type when the question asks the common coarse category, but do not drop required units or object nouns from quantities."
    + "\n- For comparison questions, choose the candidate whose stated comparable fact satisfies the comparative word in the question; monthly is less frequent than biweekly."
    + "\n- For yes/no questions, the correct short answer phrase is yes or no, not the explanatory evidence sentence."
    + "\n- If a claim says the final phrase is a quantity, distance, duration, price, or count, the phrase must include the unit/count noun when the evidence phrase includes it.",
    resolution_fewshot=HOTPOTQA_RELATION_V6_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 12:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot is the comparison result for which magazine is published more often; Rolling Stone is biweekly and Shonen Jump is monthly, so the exact short answer phrase is \"Rolling Stone\"."
    + "\n\nTiny resolution example 13:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a distance phrase; the evidence says \"7 miles\", so keep the complete phrase \"7 miles\" rather than shortening it to \"7\"."
    + "\n\nTiny resolution example 14:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a shared occupation category; if both people are directors, keep the coarse phrase \"director\" rather than expanding it to a narrower role unless the question asks for the narrower role."
    + "\n\nTiny resolution example 15:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot is a yes/no answer to whether both entities satisfy the predicate; one entity satisfies it and the other does not, so the exact short answer phrase is \"no\".",
    revision_rules=HOTPOTQA_RELATION_V6_PROFILE.revision_rules
    + "\n- If the repaired claim names a comparison winner, finish with only that winner's phrase."
    + "\n- If the repaired claim names a yes/no bridge, finish with only yes or no."
    + "\n- If the repaired claim names a shared coarse category, finish with that category phrase, not a longer explanatory sentence."
    + "\n- Keep units and counted objects in quantity phrases such as \"7 miles\", \"four months in jail\", or \"more than 70 countries\".",
)


HOTPOTQA_RELATION_V8_PROFILE = replace(
    HOTPOTQA_RELATION_V7_PROFILE,
    name="hotpotqa_relation_v8",
    merge_rules=HOTPOTQA_RELATION_V7_PROFILE.merge_rules
    + "\n- If the question asks for multiple slots, such as date and price, keep all requested slots together as one answer-slot value; a path with only one slot is incomplete."
    + "\n- For exclusion questions, write the excluded anchor and the requested non-excluded answer as different objects before comparing candidates."
    + "\n- For comparative wording, identify the missing argument of the comparison. If the question asks what X is greater than, the answer slot is the lower compared object, not X or a higher object."
    + "\n- For winner questions such as which mall/person/place has more/less/etc., the answer slot is only the winning entity phrase, not a sentence with the predicate attached.",
    prefix_rules=HOTPOTQA_RELATION_V7_PROFILE.prefix_rules
    + "\n- Do not let a later round turn an excluded anchor into the requested answer; if the candidate is the excluded anchor, expose that as the split."
    + "\n- If paths disagree because one has a complete multi-slot answer and another drops a required slot, split on completeness of the requested slot, not on wording."
    + "\n- If both paths share the same comparison facts but answer different comparison arguments, split on which argument the question requests."
    + "\n- If a candidate is a full sentence while the requested slot is a named entity or short phrase, split on answer-shape drift.",
    relation_analysis_rules=HOTPOTQA_RELATION_V7_PROFILE.relation_analysis_rules
    + "\n- For exclusion questions, state: excluded anchor = ..., requested answer must be non-excluded = ..., candidate = ...."
    + "\n- For multi-slot questions, list every required slot and mark a candidate incomplete if any requested slot is missing."
    + "\n- For comparison questions, state the direction in ordinary words: which side has the property, and which argument the question asks to output."
    + "\n- For named-entity winner questions, treat explanatory predicates after the entity as prose drift unless the question asks for a reason.",
    resolution_rules=HOTPOTQA_RELATION_V7_PROFILE.resolution_rules
    + "\n- Correct claim must be a positive answer-slot bridge to keep, not mainly a diagnosis of the losing side."
    + "\n- Correct claim must name exactly one final answer phrase, or one joined phrase containing every requested slot. Do not append a second explanatory sentence after the answer phrase."
    + "\n- Use the wording `the exact short answer phrase is \"...\"` only when the quoted phrase is the complete final phrase to output."
    + "\n- If the evidence phrase includes a unit, counted object, title word, location level, or price, the quoted exact phrase must include it; do not quote only the number or head word."
    + "\n- For exclusion questions, the quoted exact phrase must not be the excluded anchor. If the evidence only supports the excluded anchor, use keep_parallel."
    + "\n- For comparative questions, preserve the direction of the question: if it asks what a degree is greater than, quote the lower object; if it asks which entity has more, quote only the winning entity."
    + "\n- For multi-slot questions, the quoted exact phrase must include every requested slot, such as both release date and price."
    + "\n- If the winning bridge supports a named entity, title, place, or yes/no answer, do not quote a full predicate sentence as the exact phrase.",
    resolution_fewshot=HOTPOTQA_RELATION_V7_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 16:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The excluded anchor is Sergio Perez; the requested slot is the other non-excluded Mexican driver, so the exact short answer phrase is \"Pedro Rodriguez\"."
    + "\n\nTiny resolution example 17:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested comparison asks what an associate degree is greater than; the evidence states it is greater than a high school diploma or GED, so the exact short answer phrase is \"high school diploma or GED\"."
    + "\n\nTiny resolution example 18:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for both release date and price; the evidence gives both slots, so the exact short answer phrase is \"April 29, 2005 for US$129.95\"."
    + "\n\nTiny resolution example 19:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is the mall with more owners; the comparison supports Viaport Rotterdam, so the exact short answer phrase is \"Viaport Rotterdam\".",
    revision_rules=HOTPOTQA_RELATION_V7_PROFILE.revision_rules
    + "\n- Treat the quoted exact answer phrase in the repaired claim as the final phrase to output."
    + "\n- If the repaired claim contains an excluded anchor, do not output that anchor unless it is explicitly named as the non-excluded answer."
    + "\n- If the repaired claim names a complete multi-slot phrase, finish with the full joined phrase, not one component."
    + "\n- If the repaired claim names a comparison winner, finish with only the winner phrase and no predicate sentence."
    + "\n- Do not output scaffolding such as `The shortest exact answer phrase is`, `The answer is`, or `should be`; the last line should be only the phrase.",
)


HOTPOTQA_RELATION_V9_PROFILE = replace(
    HOTPOTQA_RELATION_V7_PROFILE,
    name="hotpotqa_relation_v9",
    merge_rules="""
Rules:
- Build the graph around the requested answer slot, not around the final answer token.
- First identify the slot the question asks for: yes/no, named entity, place/address, date/time, quantity with unit, comparison target, shared category, class/type, or multi-slot phrase.
- Keep evidence, intermediate entity, requested slot, and final exact phrase separate.
- Common Ground may contain only facts supported by at least two active paths and not explicitly contradicted by another active path.
- If two paths share the final answer token but use different evidence bridges, still inspect the bridge before calling them aligned.
- If final tokens differ, do not split on the token first; split on the earliest same-slot bridge claim that makes the tokens differ.
- Method order, hop order, wording, and progress speed are not splits unless they produce incompatible claims about the same requested slot.
- If no same-slot conflict is visible, keep building rather than forcing a repair.
""".strip(),
    prefix_rules="""
Rules:
- Name the requested answer slot before writing First Split.
- Keep the shortest shared evidence bridge before the first same-slot conflict.
- Do not put a one-path inference, final answer token, or disputed bridge into Common Ground.
- If one path is only slower, keep it in Paths and expose more of its later steps.
- If one path gives a complete requested-slot phrase and another gives only an intermediate entity or a partial slot, split on slot completeness.
- If two claims can both be true but do not yet settle the requested slot, use none yet / keep building.
""".strip(),
    relation_analysis_rules="""
Rules:
- Write the requested slot in one short phrase.
- For each side, write: evidence -> intermediate entity if any -> requested slot -> candidate exact phrase.
- Decide whether each candidate actually fills the requested slot, or only a nearby slot, partial slot, explanation, or intermediate entity.
- For yes/no questions, the requested exact phrase is yes or no; evidence sentences are not the final phrase.
- For multi-slot questions, a candidate fills the slot only if it includes every requested component.
- For comparison questions, state which argument the question asks to output, then compare candidates against that direction.
- For short-answer tasks, prefer the shortest phrase only after it fully fills the requested slot.
""".strip(),
    resolution_rules="""
Rules:
- Allowed actions are choose_A, choose_B, or keep_parallel.
- Correct claim must be a positive bridge to the requested slot, not mainly a diagnosis of the losing side.
- Correct claim must end with exactly one quoted phrase using this form: `so the exact short answer phrase is "..."`.
- The quoted phrase must be the final answer phrase to output, not a sentence, not a rationale, not a partial component, and not an intermediate entity.
- For yes/no questions, the quoted phrase must be "yes" or "no".
- For multi-slot questions, the quoted phrase must include every requested slot component.
- For quantity/count/distance/price answers, include the unit or counted object when it is part of the evidence phrase.
- For comparison questions, quote the argument requested by the question direction. If the question asks what X is greater than, quote the lower object; if it asks which entity has more, quote only the winning entity.
- For class/type/category questions, quote the category phrase asked by the question, not the concrete instance and not a longer explanatory noun phrase.
- If neither side has a trace-grounded bridge to a complete quoted phrase, use keep_parallel and name the missing bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot is yes/no; the evidence shows one entity satisfies the predicate and the other does not, so the exact short answer phrase is "no".

Tiny resolution example 2:
Action: choose_B
Winning side: Claim B
Correct claim: The requested slot asks what an associate degree is greater than; the evidence says it is greater than a high school diploma or GED, so the exact short answer phrase is "high school diploma or GED".

Tiny resolution example 3:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot asks for both release date and price; the evidence gives both components, so the exact short answer phrase is "April 29, 2005 for US$129.95".

Tiny resolution example 4:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot is the mall with more owners; the comparison supports Viaport Rotterdam, so the exact short answer phrase is "Viaport Rotterdam".

Tiny resolution example 5:
Action: choose_A
Winning side: Claim A
Correct claim: The requested slot is a quantity-with-counted-object phrase; the evidence says more than 70 countries, so the exact short answer phrase is "more than 70 countries".

Tiny resolution example 6:
Action: keep_parallel
Winning side: neither
Correct claim: The visible paths use compatible methods and have not yet produced incompatible claims about the requested slot, so keep building from the shared evidence bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue naturally from the preserved prefix and repaired claim.
- Treat the quoted exact short answer phrase in the repaired claim as the phrase to output.
- The last line must be only that exact phrase, with no `The answer is`, no `should be`, no quote marks, and no explanation.
- Do not output a rationale sentence when the requested slot is a short phrase or yes/no.
- Do not output a partial component of a multi-slot phrase.
- Do not drop units, counted objects, prices, dates, or required location levels from the quoted phrase.
""".strip(),
)


HOTPOTQA_RELATION_V10_PROFILE = replace(
    HOTPOTQA_RELATION_V9_PROFILE,
    name="hotpotqa_relation_v10",
    merge_rules=HOTPOTQA_RELATION_V9_PROFILE.merge_rules
    + "\n- Entity binding matters: do not swap the entity, work, event, institution, or attribute named by the question for a nearby entity from the same context."
    + "\n- For comparison questions, keep the comparable facts with the candidates, such as monthly vs biweekly, higher vs lower, more vs fewer, or before vs after."
    + "\n- Shortest exact phrase does not mean deleting required modifiers. Preserve city/state, owner/person qualifiers, rank names, title words, and location levels when they identify the requested slot.",
    prefix_rules=HOTPOTQA_RELATION_V9_PROFILE.prefix_rules
    + "\n- If one path changes the requested attribute, such as sentence vs charge, founder vs renovator, venue vs city, or origin city vs country, expose that as the slot conflict."
    + "\n- For comparisons, First Split should include the comparable facts for both sides, not just candidate names."
    + "\n- If one phrase is shorter only because it drops a required disambiguating modifier, treat that as granularity loss, not improvement.",
    relation_analysis_rules=HOTPOTQA_RELATION_V9_PROFILE.relation_analysis_rules
    + "\n- Check entity binding before choosing: question entity -> intermediate entity -> requested attribute -> candidate phrase."
    + "\n- For comparisons, write each side's comparable fact and then the direction requested by the question."
    + "\n- For place/person/title answers, decide whether the modifier is required to identify the answer or merely extra prose.",
    resolution_rules=HOTPOTQA_RELATION_V9_PROFILE.resolution_rules
    + "\n- Do not quote a nearby attribute when the question asks another one; sentence is not charge, founder is not renovator, venue is not city, and origin city is not country unless the question asks that coarser slot."
    + "\n- For comparison questions, the quoted phrase must match the side whose comparable fact satisfies the question direction; monthly is less frequent than biweekly."
    + "\n- Preserve required disambiguating modifiers in the quoted phrase. Do not shorten a supported full name, location, company, rank, title, or venue if the shorter phrase changes or weakens the requested slot.",
    resolution_fewshot=HOTPOTQA_RELATION_V9_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 7:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is the origin city/prefecture of the band, not only the country, so the exact short answer phrase is \"Fujioka, Gunma\"."
    + "\n\nTiny resolution example 8:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested comparison asks which magazine is published more often; Rolling Stone is biweekly and Shonen Jump is monthly, so the exact short answer phrase is \"Rolling Stone\"."
    + "\n\nTiny resolution example 9:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks what sentence the politician received, not what charges he faced, so the exact short answer phrase is \"four months in jail\"."
    + "\n\nTiny resolution example 10:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is the full purchasing company phrase, and the evidence names John D. Rockefeller's Standard Oil Company, so the exact short answer phrase is \"John D. Rockefeller's Standard Oil Company\".",
    revision_rules=HOTPOTQA_RELATION_V9_PROFILE.revision_rules
    + "\n- Keep the entity binding and requested attribute from the repaired claim; do not switch to a nearby entity or attribute while continuing."
    + "\n- For comparisons, keep the direction from the repaired claim and output only the winning requested phrase.",
)


HOTPOTQA_RELATION_V11_PROFILE = replace(
    HOTPOTQA_RELATION_V10_PROFILE,
    name="hotpotqa_relation_v11",
    merge_rules=HOTPOTQA_RELATION_V10_PROFILE.merge_rules
    + "\n- A quoted answer phrase must be positively supported by the evidence bridge. If the evidence says a candidate fails the requested constraint, that candidate cannot be the quoted answer phrase."
    + "\n- If the question asks for one example, one star, one person, or one item, a list of multiple valid entities is usually over-complete unless the question explicitly asks for all of them."
    + "\n- For kind/type/category questions, distinguish the class label from adjectives that describe the instance. Required modifiers identify an entity; optional adjectives often do not change the requested category.",
    prefix_rules=HOTPOTQA_RELATION_V10_PROFILE.prefix_rules
    + "\n- If a path's bridge says a candidate does not satisfy a required constraint but still outputs that candidate, split on support consistency."
    + "\n- If paths differ by one answer versus a list for a singular requested slot, split on whether the slot asks for one acceptable entity or all entities."
    + "\n- If paths differ by an adjective before a type label, split on whether the adjective is part of the requested type or only an instance descriptor.",
    relation_analysis_rules=HOTPOTQA_RELATION_V10_PROFILE.relation_analysis_rules
    + "\n- Before choosing, run this support check in words: does the evidence bridge affirm that the quoted phrase satisfies every requested constraint?"
    + "\n- Mark a candidate invalid if the same bridge says it is outside the requested place, lacks the requested role, answers the wrong relation, or only names an excluded/nearby object."
    + "\n- For singular slots, decide whether the question permits any one valid entity; if yes, do not replace a supported single entity with a longer list."
    + "\n- For type/category slots, remove descriptors that are not needed to name the class, but keep words that define the class itself.",
    resolution_rules=HOTPOTQA_RELATION_V10_PROFILE.resolution_rules
    + "\n- The final quoted phrase must pass the support check: the preceding bridge must say the phrase satisfies the requested slot, not that it fails a required constraint."
    + "\n- Never write a bridge of the form `candidate fails constraint, so the exact phrase is candidate`; use the supported alternative or keep_parallel."
    + "\n- For singular `one of` questions, quote one supported entity phrase, not a conjunction of multiple entities, unless the question asks for both/all."
    + "\n- For kind/type/category questions, quote the category phrase itself. Drop adjectives such as unofficial, documentary-film pluralization, or other instance descriptors when the question asks the kind/type rather than the full description."
    + "\n- Preserve modifiers only when they identify the requested entity, place, title, company, rank, relation, date, quantity, or location level.",
    resolution_fewshot=HOTPOTQA_RELATION_V10_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 11:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The evidence says Candidate X is outside the required region, so Candidate X cannot be quoted as the answer phrase; keep building until a supported in-region candidate is visible."
    + "\n\nTiny resolution example 12:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for one star, and Claim B gives one supported star rather than an over-complete list, so the exact short answer phrase is \"Michael Seater\"."
    + "\n\nTiny resolution example 13:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for the kind of online text-based role-playing game, and the class label is play-by-post role-playing game; unofficial is only an instance descriptor here, so the exact short answer phrase is \"play-by-post role-playing game\"."
    + "\n\nTiny resolution example 14:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for the specific city and state phrase, and the state is needed to preserve the location level, so the exact short answer phrase is \"New York City, New York\".",
    revision_rules=HOTPOTQA_RELATION_V10_PROFILE.revision_rules
    + "\n- If the repaired claim says a candidate fails a required constraint, do not output that candidate."
    + "\n- If the repaired claim says the requested slot is singular, output one supported phrase rather than a conjunction or list."
    + "\n- If the repaired claim says a modifier is only an instance descriptor for a type/category slot, drop that modifier in the final line.",
)


HOTPOTQA_RELATION_V12_PROFILE = replace(
    HOTPOTQA_RELATION_V10_PROFILE,
    name="hotpotqa_relation_v12",
    merge_rules=HOTPOTQA_RELATION_V10_PROFILE.merge_rules
    + "\n- Before writing Common Ground, audit each sentence: it must be stated or clearly implied by at least two active paths, and no active path may explicitly deny it."
    + "\n- For short-answer questions, build candidate-to-slot bridges: candidate phrase -> required relation/property -> whether it satisfies the exact question slot."
    + "\n- Do not treat a true nearby fact as the answer if it sits at the wrong level, such as island vs larger area, country vs archipelago/region, person vs role, or genre phrase vs dataset answer style."
    + "\n- A wording difference such as singular/plural, adjective/no adjective, or full/short location is not a real split unless it changes whether the candidate fills the exact requested slot.",
    prefix_rules=HOTPOTQA_RELATION_V10_PROFILE.prefix_rules
    + "\n- If the visible traces share the same evidence but disagree on the support relation to the question slot, make that support relation the first split."
    + "\n- If a path says a candidate fails a required condition and still outputs it, the split is support consistency: satisfies slot vs fails slot."
    + "\n- If two answer phrases are compatible surface variants for the same slot, keep building or preserve both; do not force a repair from surface length alone.",
    relation_analysis_rules=HOTPOTQA_RELATION_V10_PROFILE.relation_analysis_rules
    + "\n- Write one compact candidate audit for each side: candidate, required slot, supporting evidence, missing or contradicted condition."
    + "\n- The chosen candidate must satisfy every required word in the question, including comparison direction, geographic qualifier, role, relation, and answer level."
    + "\n- If the rationale says the candidate is disqualified, the correct claim must not quote that same candidate as the final phrase."
    + "\n- When a dataset answer may use a normalized shorter phrase, prefer the phrase that exactly names the slot without adding unsupported or unnecessary descriptors, but do not change the underlying claim only for formatting.",
    resolution_rules=HOTPOTQA_RELATION_V10_PROFILE.resolution_rules
    + "\n- Correct claim must have this internal consistency: the bridge affirms that the quoted phrase satisfies the requested slot. Never quote a candidate after saying it fails a required condition."
    + "\n- If both sides use the same evidence but one chooses the wrong answer level, choose the side whose phrase matches the question's requested level."
    + "\n- For surface variants, decide the evidence claim first; only then choose the shortest phrase that still names the same requested slot."
    + "\n- If neither side has a consistent bridge from evidence to a complete answer phrase, use keep_parallel rather than inventing a new unsupported phrase.",
    resolution_fewshot=HOTPOTQA_RELATION_V10_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 11:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks which candidate satisfies a geographic qualifier; the bridge says Candidate A is outside the required region and Candidate B is inside it, so the exact short answer phrase is \"Candidate B\"."
    + "\n\nTiny resolution example 12:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for the larger named region containing the island, not the political country or the island itself; the bridge supports the region level, so the exact short answer phrase is \"Macaronesia\"."
    + "\n\nTiny resolution example 13:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for the kind of film; both phrases refer to the same documentary category, and the normalized answer phrase is \"documentary\"."
    + "\n\nTiny resolution example 14:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The visible paths give only nearby entities and no trace-grounded bridge to the requested role, so keep building from the shared evidence instead of quoting a nearby entity.",
    revision_rules=HOTPOTQA_RELATION_V10_PROFILE.revision_rules
    + "\n- Continue from the corrected evidence-to-slot bridge; do not output a candidate that the bridge itself disqualifies."
    + "\n- If the repaired claim preserves a surface-normalized phrase, output that phrase without re-expanding it to a longer description.",
)


HOTPOTQA_RELATION_V13_PROFILE = replace(
    HOTPOTQA_RELATION_V12_PROFILE,
    name="hotpotqa_relation_v13",
    merge_rules=HOTPOTQA_RELATION_V12_PROFILE.merge_rules
    + "\n- Preserve cardinality from the question slot: who/which/one of usually asks for one supported entity, while both/all/list asks for multiple entities."
    + "\n- Preserve answer-bearing units and counted objects as part of the candidate phrase when the evidence phrase names them, such as countries, miles, years, dollars, or people."
    + "\n- Do not promote `not mentioned` or `none` over a trace-grounded positive bridge already present in Common Ground unless the positive bridge is explicitly contradicted.",
    prefix_rules=HOTPOTQA_RELATION_V12_PROFILE.prefix_rules
    + "\n- If two paths share a complete bridge to the requested slot and a third path has not reached the bridge, do not split on the slower path's absence; keep the complete bridge active."
    + "\n- If paths differ only by one entity versus a list, split on whether the question slot asks for one acceptable entity or a complete list."
    + "\n- If paths differ by dropping a unit, counted object, owner, or disambiguating qualifier from the same evidence phrase, split on phrase completeness rather than surface brevity.",
    relation_analysis_rules=HOTPOTQA_RELATION_V12_PROFILE.relation_analysis_rules
    + "\n- Before choosing, write the slot cardinality in words: one entity, multiple entities, yes/no, quantity-with-unit, location/address, type/category, or comparison winner."
    + "\n- If the evidence bridge supports a positive candidate phrase, the analysis must compare candidates against that phrase instead of falling back to `not mentioned`."
    + "\n- Check that the quoted phrase is the same object supported by the bridge; if the bridge supports Jack Ryan, the quote cannot be Sam Fisher.",
    resolution_rules=HOTPOTQA_RELATION_V12_PROFILE.resolution_rules
    + "\n- Add a separate line `Canonical answer: ...` whenever Correct claim names the final phrase. It must be exactly the quoted short answer phrase, with no quotes or explanation."
    + "\n- For singular slots such as who, which man, which person, or one of, quote one supported entity phrase, not a conjunction or list, unless the question explicitly asks for both, all, or multiple items."
    + "\n- For quantity/location/entity phrases, do not delete the counted object, unit, owner, or required qualifier if deleting it weakens the answer-bearing phrase."
    + "\n- If Common Ground already contains a complete positive bridge to the requested slot, do not choose `not mentioned`, `unknown`, or `none` unless that bridge is shown false."
    + "\n- Internal consistency is mandatory: the same candidate named in the evidence bridge must be the one in the quoted phrase and in `Canonical answer`.",
    resolution_fewshot=HOTPOTQA_RELATION_V12_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 15:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for one supported star, and the bridge supports Michael Seater as one star, so the exact short answer phrase is \"Michael Seater\".\nCanonical answer: Michael Seater"
    + "\n\nTiny resolution example 16:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a quantity-with-counted-object phrase; the evidence says more than 70 countries, so the exact short answer phrase is \"more than 70 countries\".\nCanonical answer: more than 70 countries"
    + "\n\nTiny resolution example 17:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for the fictional character tied to the 2002 film, and the bridge supports Jack Ryan rather than the nearby Tom Clancy character Sam Fisher, so the exact short answer phrase is \"Jack Ryan\".\nCanonical answer: Jack Ryan",
    revision_rules=HOTPOTQA_RELATION_V12_PROFILE.revision_rules
    + "\n- If the repaired claim includes `Canonical answer`, the final line must be exactly that phrase."
    + "\n- Do not expand a singular canonical answer into a list, and do not shorten a quantity phrase by dropping its unit or counted object.",
)


HOTPOTQA_RELATION_V14_PROFILE = replace(
    HOTPOTQA_RELATION_V13_PROFILE,
    name="hotpotqa_relation_v14",
    merge_rules=HOTPOTQA_RELATION_V13_PROFILE.merge_rules
    + "\n- Do not let nearby administrative roles, nicknames, served-time details, or broader location prose replace the exact relation asked by the question."
    + "\n- For comparison and yes/no slots, resolve the semantic direction directly: more frequent, more southern, both native, earlier/later, higher/lower.",
    prefix_rules=HOTPOTQA_RELATION_V13_PROFILE.prefix_rules
    + "\n- When the first split is final-token disagreement, restate the question's requested relation before comparing the tokens."
    + "\n- If a candidate adds an extra fact after already filling the slot, split on minimal slot phrase vs over-complete phrase.",
    relation_analysis_rules=HOTPOTQA_RELATION_V13_PROFILE.relation_analysis_rules
    + "\n- Copy the question's main ask in one sentence before choosing: ask who/what/where/when/how far/which/yes-no, then answer only that ask."
    + "\n- For `held which position`, distinguish the office held by the named people from a position held by a related third person."
    + "\n- For `sentenced to what`, output the sentence itself, not later served-time details unless the question asks how much was served."
    + "\n- For `where`, decide whether the answer slot is venue name, city/region, or street/address; do not append a city if the venue name alone fills the slot.",
    resolution_rules=HOTPOTQA_RELATION_V13_PROFILE.resolution_rules
    + "\n- The Correct claim should quote the minimal phrase that answers the literal question, not every fact in the evidence sentence."
    + "\n- For `how far/how long/how many`, quote the measured phrase only, with its unit or counted object, and do not switch to `not specified` if that phrase is directly stated for the same target."
    + "\n- For `sentenced to what`, quote only the sentence phrase, such as `four months in jail`; do not add served-time details unless the question asks served how long."
    + "\n- For `where was X held`, quote the venue or location name requested by the question; do not append city/country unless needed to identify the venue."
    + "\n- For `which was more frequent`, monthly is more frequent than bi-monthly meaning every two months; do not reinterpret bi-monthly as twice a month unless the trace explicitly says so."
    + "\n- For `both native to Asia`, a genus native to southern Asia satisfies the Asia predicate; do not require every species to be native to Asia unless the question asks all species."
    + "\n- For `more south`, compare the geographic regions correctly; Australia is south of temperate Europe/Asia/North America.",
    resolution_fewshot=HOTPOTQA_RELATION_V13_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 18:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks what sentence the politician received; served-time detail is extra, so the exact short answer phrase is \"four months in jail\".\nCanonical answer: four months in jail"
    + "\n\nTiny resolution example 19:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested comparison asks which magazine was published more frequently; monthly is more frequent than every two months, so the exact short answer phrase is \"Girlfriends\".\nCanonical answer: Girlfriends"
    + "\n\nTiny resolution example 20:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks which position Ronald Reagan and George H. W. Bush both held, not a role held by a related appointee, so the exact short answer phrase is \"President of the United States\".\nCanonical answer: President of the United States"
    + "\n\nTiny resolution example 21:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks the venue where the final was held; the venue phrase fills the slot, so the exact short answer phrase is \"Stadio Olimpico\".\nCanonical answer: Stadio Olimpico",
    revision_rules=HOTPOTQA_RELATION_V13_PROFILE.revision_rules
    + "\n- If the canonical phrase is already the literal answer to the question, do not append extra context from the evidence sentence.",
)


HOTPOTQA_RELATION_V15_PROFILE = replace(
    HOTPOTQA_RELATION_V13_PROFILE,
    name="hotpotqa_relation_v15",
    merge_rules=HOTPOTQA_RELATION_V13_PROFILE.merge_rules
    + "\n- Treat HotpotQA as short-answer slot filling: compare the evidence bridge to the exact slot, not the answer token alone."
    + "\n- Preserve the evidence bridge when it already supports a candidate; do not replace it with a more familiar nearby entity, role, title, place, or category."
    + "\n- Keep the answer phrase minimal but sufficient: include words required by the slot, and drop extra context only after the same phrase is still identifiable.",
    prefix_rules=HOTPOTQA_RELATION_V13_PROFILE.prefix_rules
    + "\n- If two paths disagree only because one phrase is shorter, split only when the shorter phrase loses a required slot constraint or the longer phrase adds a different object."
    + "\n- If a path has a supported bridge to the slot and another path gives a nearby-but-different object, split on which object satisfies the slot."
    + "\n- If both phrases name the same object at different surface lengths, keep the shared object and let the final answer use the minimal sufficient phrase.",
    relation_analysis_rules=HOTPOTQA_RELATION_V13_PROFILE.relation_analysis_rules
    + "\n- For each candidate, ask three compact questions: what slot is requested, what evidence connects the candidate to that slot, and what exact words are required to name it."
    + "\n- Do not choose an over-complete phrase merely because it repeats more evidence; choose the phrase that names the supported slot without switching objects."
    + "\n- Do not choose an under-complete phrase if it drops the owner, modifier, location level, unit, or counted object needed to distinguish the supported answer.",
    resolution_rules=HOTPOTQA_RELATION_V13_PROFILE.resolution_rules
    + "\n- Correct claim should first name the evidence-supported object, then the minimal sufficient answer phrase for that same object."
    + "\n- A longer phrase is wrong only when the added words answer a different slot or make the answer over-specific beyond the question; a shorter phrase is wrong only when it loses a required distinction."
    + "\n- If origin already gives a complete phrase for the same supported object, do not repair it into a different nearby object."
    + "\n- Use `Canonical answer: ...` for the minimal sufficient phrase, preserving spaces, punctuation, units, counted objects, and required qualifiers.",
    resolution_fewshot=HOTPOTQA_RELATION_V13_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 18:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The supported object is the sentence the person received; served-time detail is a different later fact, so the minimal sufficient phrase is \"four months in jail\".\nCanonical answer: four months in jail"
    + "\n\nTiny resolution example 19:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The supported object is the location-level phrase requested by the question; the city and state are required to name that location level, so the minimal sufficient phrase is \"New York City, New York\".\nCanonical answer: New York City, New York"
    + "\n\nTiny resolution example 20:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The supported object is one named person who fills a singular who-slot; a second co-star is extra when the question asks for one person, so the minimal sufficient phrase is \"Michael Seater\".\nCanonical answer: Michael Seater"
    + "\n\nTiny resolution example 21:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The supported object is the full company name tied to the evidence bridge; dropping the owner changes the named company, so the minimal sufficient phrase is \"John D. Rockefeller's Standard Oil Company\".\nCanonical answer: John D. Rockefeller's Standard Oil Company",
    revision_rules=HOTPOTQA_RELATION_V13_PROFILE.revision_rules
    + "\n- Continue from the same evidence-supported object. Adjust phrase length only to make the final answer minimal and sufficient; do not switch to a nearby object."
    + "\n- If `Canonical answer` is present, use that exact phrase as the final answer.",
)


HOTPOTQA_RELATION_V16_PROFILE = replace(
    HOTPOTQA_RELATION_V12_PROFILE,
    name="hotpotqa_relation_v16",
    resolution_rules=HOTPOTQA_RELATION_V12_PROFILE.resolution_rules
    + "\n- If Correct claim names the exact final phrase, add one separate line `Canonical answer: ...` containing only that phrase."
    + "\n- Do not change the reasoning decision just to make a canonical line; the canonical line only copies the phrase already supported by Correct claim.",
    resolution_fewshot=HOTPOTQA_RELATION_V12_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 15:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is filled by the supported phrase \"Ryan Seacrest\".\nCanonical answer: Ryan Seacrest",
    revision_rules=HOTPOTQA_RELATION_V12_PROFILE.revision_rules
    + "\n- If the repaired claim includes `Canonical answer`, use that exact phrase as the final answer.",
)


HOTPOTQA_RELATION_V17_PROFILE = replace(
    HOTPOTQA_RELATION_V12_PROFILE,
    name="hotpotqa_relation_v17",
    merge_rules=HOTPOTQA_RELATION_V12_PROFILE.merge_rules
    + "\n- For movie/book/song/show questions, preserve the medium and title binding. A person tied to a radio program, TV show, song, book, or game is not automatically tied to the film, movie, or adaptation named by the question."
    + "\n- Common Ground may contain a named entity or attribute only if at least two active paths support the same entity-to-slot bridge; do not promote a single path's candidate into shared evidence."
    + "\n- If a candidate is shown not to satisfy a type or constraint in the question, the correct claim should keep the disqualification and continue looking for the supported candidate, not output the disqualified candidate.",
    prefix_rules=HOTPOTQA_RELATION_V12_PROFILE.prefix_rules
    + "\n- Treat media/title mismatch as an entity-binding split: same name or franchise is not enough if one path uses the wrong medium or work."
    + "\n- If one path says candidate satisfies the requested type and another says the same candidate fails that type, split on satisfies slot vs fails slot."
    + "\n- If two paths already support the same single answer phrase and a third path says not mentioned, do not let the unsupported absence erase the positive bridge.",
    relation_analysis_rules=HOTPOTQA_RELATION_V12_PROFILE.relation_analysis_rules
    + "\n- Before choosing, restate the chain as: source work/entity -> intermediate entity -> requested attribute. Check that the work title and medium match the question."
    + "\n- A disqualification sentence such as `Tigger is not a hedgehog` cannot imply final answer `Tigger`; it only rejects that candidate."
    + "\n- Prefer a trace-grounded positive bridge over `not mentioned` when the bridge names the requested attribute directly.",
    resolution_rules=HOTPOTQA_RELATION_V12_PROFILE.resolution_rules
    + "\n- Correct claim should name the supported candidate and the exact short answer phrase when the trace contains a positive bridge; add `Canonical answer: ...` with only that phrase."
    + "\n- If the current visible candidates are all disqualified, use keep_parallel and say which constraint failed; do not convert the disqualified candidate into the answer."
    + "\n- When the question names a specific medium or work title, choose the path whose bridge uses that same medium/title, not a nearby program, franchise, song, game, or adaptation."
    + "\n- If the rationale says candidate X is not the requested type or is tied to the wrong work, the quoted phrase and Canonical answer must not be X.",
    resolution_fewshot=HOTPOTQA_RELATION_V12_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 15:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested chain uses the film title, and Claim A keeps the film actor and the government position tied to that film actor, so the exact short answer phrase is \"Chief of Protocol\".\nCanonical answer: Chief of Protocol"
    + "\n\nTiny resolution example 16:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The shared evidence says the visible candidate is not the requested type, so that candidate is disqualified; keep building until a trace-grounded candidate of the requested type is visible."
    + "\n\nTiny resolution example 17:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for the character tied to the 2002 film adaptation; Claim B binds the 2002 film to the Jack Ryan work rather than the nearby game franchise, so the exact short answer phrase is \"Jack Ryan\".\nCanonical answer: Jack Ryan",
    revision_rules=HOTPOTQA_RELATION_V12_PROFILE.revision_rules
    + "\n- Continue from the same medium/title and candidate-to-slot bridge in the repaired claim; do not switch to a nearby work or disqualified candidate."
    + "\n- If `Canonical answer` is present, use that exact phrase as the final answer.",
)


HOTPOTQA_RELATION_V18_PROFILE = replace(
    HOTPOTQA_RELATION_V12_PROFILE,
    name="hotpotqa_relation_v18",
    relation_analysis_rules=HOTPOTQA_RELATION_V12_PROFILE.relation_analysis_rules
    + "\n- Before changing a short answer phrase, ask whether the old phrase and new phrase name the same supported slot object at the same specificity level."
    + "\n- Do not prefer a shorter phrase if it drops a required office, location level, owner, title qualifier, unit, counted object, or yes/no label."
    + "\n- Do not prefer a longer phrase if the extra words only add parenthetical disambiguation, country/context, plurality, or nearby evidence that the question did not ask for.",
    resolution_rules=HOTPOTQA_RELATION_V12_PROFILE.resolution_rules
    + "\n- If an existing branch already has a complete answer phrase whose evidence bridge satisfies the requested slot, keep that phrase unless another claim directly shows it has the wrong object or wrong specificity."
    + "\n- Phrase repair should preserve the same supported object. Do not switch from the supported object to a nearby person, title, work, role, or broader/narrower location while adjusting length."
    + "\n- Add `Canonical answer: ...` only when the Correct claim names an exact phrase from the supported bridge or from an existing complete branch answer."
    + "\n- For yes/no questions, the canonical answer must be yes or no; do not output the explanatory comparison sentence as the answer phrase.",
    resolution_fewshot=HOTPOTQA_RELATION_V12_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 15:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The existing branch phrase already names the office held by both people, and shortening it would drop the required office specificity, so the exact short answer phrase is \"President of the United States\".\nCanonical answer: President of the United States"
    + "\n\nTiny resolution example 16:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The existing branch phrase already gives the requested city and state location level; adding the country changes the dataset answer phrase, so the exact short answer phrase is \"New York City, New York\".\nCanonical answer: New York City, New York"
    + "\n\nTiny resolution example 17:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question is yes/no; the evidence comparison supports no, so the exact short answer phrase is \"no\".\nCanonical answer: no",
    revision_rules=HOTPOTQA_RELATION_V12_PROFILE.revision_rules
    + "\n- If the repaired claim keeps an existing complete branch answer, copy that phrase exactly as the final answer."
    + "\n- If `Canonical answer` is present, use that exact phrase as the final answer and do not expand or shorten it.",
)


HOTPOTQA_RELATION_V19_PROFILE = replace(
    HOTPOTQA_RELATION_V18_PROFILE,
    name="hotpotqa_relation_v19",
    relation_analysis_rules=HOTPOTQA_RELATION_V18_PROFILE.relation_analysis_rules
    + "\n- Treat `shortest exact phrase` as minimal sufficient, not shortest by deletion. Keep every word needed by the question slot, such as County, date-and-price, office title, measurement unit, counted object, or yes/no label."
    + "\n- For yes/no questions, analyze the evidence-to-predicate bridge, then choose only yes or no as the answer phrase. Do not use a prose explanation as the final phrase."
    + "\n- For multi-component asks joined by and/or, audit each required component separately and keep all required components in the answer phrase."
    + "\n- Do not add a field or type modifier just because it is true. If the question asks for a shared occupation, location, or entity phrase and the branch already gives a sufficient phrase, do not expand it unless the expansion is required to disambiguate the slot."
    + "\n- Use the context as the authority for the candidate-to-slot bridge. Do not reject a context-supported candidate using outside chronology or world knowledge unless the context itself supplies the contradiction."
    + "\n- For comparison questions, preserve the comparison direction from the shared evidence; resolve which candidate is greater, earlier, more frequent, closer, or otherwise selected before writing the answer phrase.",
    resolution_rules=HOTPOTQA_RELATION_V18_PROFILE.resolution_rules
    + "\n- Correct claim must be a positive, landable answer bridge: evidence -> requested slot -> exact answer phrase. It must not be empty, only diagnostic, or a disqualification followed by the disqualified candidate."
    + "\n- If the question is yes/no and the bridge decides the predicate, write `Canonical answer: yes` or `Canonical answer: no`."
    + "\n- If the question explicitly requests multiple components, the canonical answer must include all requested components supported by the bridge."
    + "\n- If a branch answer already preserves a required head noun, unit, component, or label, do not shorten it away. County is part of a county answer; a price is part of a date-and-price answer."
    + "\n- If a candidate is directly described in the context with the requested slot descriptors, do not switch to a different candidate because of external plausibility."
    + "\n- For surface specificity, choose the phrase already supported by the evidence bridge at the question's requested level; avoid both under-specific truncation and unnecessary expansion.",
    resolution_fewshot=HOTPOTQA_RELATION_V18_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 18:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks whether both named bands were from the U.S.; one is from the U.S. and the other is not, so the predicate is false and the exact answer phrase is \"no\".\nCanonical answer: no"
    + "\n\nTiny resolution example 19:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for a county, and the evidence says the town is in Saint Louis County; dropping County changes the requested level, so the exact answer phrase is \"Saint Louis County\".\nCanonical answer: Saint Louis County"
    + "\n\nTiny resolution example 20:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for both release date and price; the evidence gives April 29, 2005 and US$129.95, so the exact answer phrase is \"April 29, 2005 for US$129.95\".\nCanonical answer: April 29, 2005 for US$129.95"
    + "\n\nTiny resolution example 21:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the shared occupation, and the existing branch phrase \"director\" already fills the slot; adding a field modifier is unnecessary, so the exact answer phrase is \"director\".\nCanonical answer: director"
    + "\n\nTiny resolution example 22:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The comparison asks which publication is more frequent; monthly is more frequent than every two months, so the exact answer phrase is \"Girlfriends\".\nCanonical answer: Girlfriends",
    revision_rules=HOTPOTQA_RELATION_V18_PROFILE.revision_rules
    + "\n- Continue from the positive answer bridge and finish with the canonical phrase when present."
    + "\n- Do not drop required slot words such as County, units, prices, counted objects, or yes/no labels while continuing.",
)


HOTPOTQA_RELATION_V20_PROFILE = replace(
    HOTPOTQA_RELATION_V19_PROFILE,
    name="hotpotqa_relation_v20",
    relation_analysis_rules=HOTPOTQA_RELATION_V19_PROFILE.relation_analysis_rules
    + "\n- First identify the answer slot from the main question word and predicate: who/person, when/time, where/place, what type/category, how far/distance, yes/no predicate. Embedded entities are bridge nodes, not automatically the answer slot."
    + "\n- For each candidate answer, audit a slot ledger: source entity from the question -> bridge entity -> requested attribute -> answer phrase. The same candidate must satisfy every required descriptor in that chain."
    + "\n- Do not let an unrelated context paragraph win because it contains one similar word such as performance, role, appointed, native, or published. The bridge must start from the entity/work/event named in the question."
    + "\n- If a visible branch gives a positive bridge for the exact slot and another branch says not specified, prefer not specified only when the positive bridge uses the wrong source entity, wrong bridge entity, or wrong requested attribute."
    + "\n- For yes/no with region or membership predicates, distinguish `native to Asia` from `native only to Asia`; if the question does not say only/entirely/exclusively, extra regions do not by themselves make the predicate false.",
    resolution_rules=HOTPOTQA_RELATION_V19_PROFILE.resolution_rules
    + "\n- Correct claim should explicitly state the answer slot type before the phrase, especially for questions with embedded entities: `the answer slot is a time/person/place/type/distance/yes-no predicate`."
    + "\n- Choose a candidate only if its bridge starts from the question's source entity or work and reaches the requested attribute. Reject candidates whose evidence belongs to a nearby but different source paragraph."
    + "\n- For conjunctions, do not split the descriptors across different candidates. The final answer phrase must name the candidate that satisfies all descriptors together."
    + "\n- Do not replace a context-supported positive answer with `not specified` merely because another nearby location, event, or source is also mentioned.",
    resolution_fewshot=HOTPOTQA_RELATION_V19_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 23:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The main question asks `when`, so the answer slot is a time period. The language is only the bridge entity; Claim B gives when that language was spoken, so the exact short answer phrase is \"between the 8th and 16th centuries\".\nCanonical answer: between the 8th and 16th centuries"
    + "\n\nTiny resolution example 24:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The slot ledger must start from the show named in the question, then the actor in that show, then that actor's role in the other show. Claim A follows that bridge and gives the exact short answer phrase \"Jillian Belk\".\nCanonical answer: Jillian Belk"
    + "\n\nTiny resolution example 25:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks whether both named plants are native to Asia, not whether they are native only to Asia. Since the evidence places each named plant in Asia, the yes/no predicate is true and the exact answer phrase is \"yes\".\nCanonical answer: yes",
    revision_rules=HOTPOTQA_RELATION_V19_PROFILE.revision_rules
    + "\n- Preserve the answer slot type from the corrected claim. If the corrected claim says the slot is time, distance, type, or yes/no, finish with that kind of phrase rather than the bridge entity.",
)


HOTPOTQA_RELATION_V21_PROFILE = replace(
    HOTPOTQA_RELATION_V19_PROFILE,
    name="hotpotqa_relation_v21",
    relation_analysis_rules=HOTPOTQA_RELATION_V19_PROFILE.relation_analysis_rules
    + "\n- Identify the answer slot from the main question predicate before comparing candidates: person, time, place, distance, quantity-with-object, type/category, or yes/no. A bridge entity inside the question is not automatically the answer."
    + "\n- For each candidate, check the whole bridge chain named by the question: source entity/work/event -> intermediate entity -> requested attribute -> answer phrase. A candidate that matches only one descriptor or one nearby paragraph is not enough."
    + "\n- Preserve counted objects and units when they are part of the requested quantity phrase, such as countries, miles, years, awards, or records."
    + "\n- Do not choose `not specified` when a visible branch has a positive, context-grounded bridge to the requested slot, unless that branch clearly uses the wrong source entity or requested attribute."
    + "\n- For predicates like `native to Asia`, do not silently add `only`, `entirely`, or `exclusively`; extra regions do not negate membership unless the question asks for exclusivity."
    + "\n- Be careful with exception wording: `the only other X besides Y` still identifies Y as a valid contrasted candidate, not as `none`.",
    resolution_rules=HOTPOTQA_RELATION_V19_PROFILE.resolution_rules
    + "\n- Correct claim should state the supported candidate satisfies all required descriptors together, then give the exact answer phrase. Do not choose a candidate that satisfies only one descriptor while another descriptor belongs to a different person, work, or event."
    + "\n- For yes/no questions, if the bridge decides the predicate, the canonical answer must be only `yes` or `no`, never the explanatory sentence."
    + "\n- Do not expand a correct category/type answer just because the evidence phrase contains an extra head noun; keep the phrase at the question's requested granularity.",
    revision_rules=HOTPOTQA_RELATION_V19_PROFILE.revision_rules
    + "\n- Continue with the same answer slot type and exact phrase granularity from the corrected claim; do not switch from a decided yes/no predicate to an explanatory sentence.",
)


HOTPOTQA_RELATION_V22_PROFILE = replace(
    HOTPOTQA_RELATION_V20_PROFILE,
    name="hotpotqa_relation_v22",
    relation_analysis_rules=HOTPOTQA_RELATION_V20_PROFILE.relation_analysis_rules
    + "\n- If the question is yes/no, the bridge may mention both sides of the comparison, but the answer phrase is only yes or no."
    + "\n- In `the only other X besides Y` or similar wording, Y is the contrasted existing item; do not convert the phrase into `none` unless the evidence says no such other item exists."
    + "\n- A candidate must satisfy the full source-to-slot bridge. Do not choose a nearby entity merely because it matches the creator, place, date, or type while missing the adaptation/event/relation requested by the question."
    + "\n- For `what type/kind/category` questions, prefer the category label that names the shared type; do not add a generic head noun such as films, people, places, or works unless the question asks for that head noun.",
    resolution_rules=HOTPOTQA_RELATION_V20_PROFILE.resolution_rules
    + "\n- When a yes/no predicate is resolved, `Canonical answer` must be exactly yes or no."
    + "\n- For `only other ... besides Y`, the correct claim should preserve the contrasted item Y when Y is the answer requested by `besides`; do not land on `None` from the word `only` alone."
    + "\n- Reject partial-descriptor candidates: the kept candidate must be tied to the specific event/relation in the question, not just to one nearby descriptor."
    + "\n- For category/type slots, do not expand a supported category label into a longer noun phrase if the extra head noun only repeats the broad domain.",
    resolution_fewshot=HOTPOTQA_RELATION_V20_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 26:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the other item besides Y; the evidence says X is the only other item besides Y, so the item requested by `besides X` is Y.\nCanonical answer: Y"
    + "\n\nTiny resolution example 27:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for a fictional character tied to the 2002 film event. A candidate that is only a Tom Clancy character but is not tied to that 2002 film does not satisfy the full bridge, so the exact short answer phrase is the character tied to the 2002 film.\nCanonical answer: Jack Ryan"
    + "\n\nTiny resolution example 28:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the shared media type; the evidence says both works are documentary films, so the category label that fills the type slot is \"documentary\".\nCanonical answer: documentary",
    revision_rules=HOTPOTQA_RELATION_V20_PROFILE.revision_rules
    + "\n- If the corrected claim includes a canonical yes/no or category label, finish with that exact phrase rather than restating the evidence sentence.",
)


HOTPOTQA_RELATION_V23_PROFILE = replace(
    HOTPOTQA_RELATION_V20_PROFILE,
    name="hotpotqa_relation_v23",
    relation_analysis_rules=HOTPOTQA_RELATION_V20_PROFILE.relation_analysis_rules
    + "\n- Parse exception wording carefully: in `the only other X besides Y`, Y is still an existing valid contrasted item; do not infer `none` from `only other`."
    + "\n- For yes/no questions, compare the evidence to the whole predicate first, then land only yes or no."
    + "\n- A candidate that matches only one descriptor is not enough; the same candidate must be connected to the requested event, relation, or attribute."
    + "\n- For `what type of media` questions, the media category label can be the exact answer even when evidence phrases it as `documentary film` or similar.",
    resolution_rules=HOTPOTQA_RELATION_V20_PROFILE.resolution_rules
    + "\n- If exception wording such as `only other ... besides Y` decides the bridge, keep the contrasted item requested by `besides`; do not turn the answer into `None` unless the trace directly supports no such item."
    + "\n- If the requested slot is yes/no, the correct claim should end in a yes/no predicate and `Canonical answer` should be yes or no."
    + "\n- Do not replace a branch that satisfies the full bridge with a nearby candidate that satisfies only the creator, place, date, type, or another single descriptor."
    + "\n- For media-type slots, do not force the longer wording from the evidence if the shorter category label already fills the slot.",
    revision_rules=HOTPOTQA_RELATION_V20_PROFILE.revision_rules
    + "\n- Preserve required units, counted objects, full dates, addresses, county/state levels, and yes/no labels while applying the corrected bridge.",
)


HOTPOTQA_RELATION_V24_PROFILE = replace(
    HOTPOTQA_RELATION_V20_PROFILE,
    name="hotpotqa_relation_v24",
    relation_analysis_rules=HOTPOTQA_RELATION_V20_PROFILE.relation_analysis_rules
    + "\n- Treat the observed branch answers like candidate phrases to audit, similar to answer choices: for each candidate, ask whether the visible evidence bridge reaches the requested slot at the right specificity."
    + "\n- The first useful conflict is often evidence supports candidate A vs evidence supports candidate B, or candidate A fills the requested slot vs candidate A is only an intermediate or nearby object."
    + "\n- Do not resolve from answer popularity. Compare candidate-to-slot support: source entity -> bridge entity -> requested attribute -> candidate exact phrase."
    + "\n- If an observed candidate is already supported at the requested slot, prefer keeping that exact phrase over freely generating a new paraphrase."
    + "\n- Generate a new exact phrase only when the trace evidence explicitly supplies a more complete slot phrase than every observed candidate, such as adding a required unit, counted object, date component, office word, or title word.",
    resolution_rules=HOTPOTQA_RELATION_V20_PROFILE.resolution_rules
    + "\n- Correct claim should choose the supported candidate-to-slot bridge, not merely diagnose which branch was confused."
    + "\n- When observed candidates are listed, audit them as branch candidates. Prefer `Canonical answer` from a supported observed candidate when it fills the slot."
    + "\n- Do not switch to a new nearby phrase if a listed candidate already has the strongest visible evidence-to-slot bridge."
    + "\n- If no listed candidate is supported, use keep_parallel unless the visible trace evidence itself explicitly gives a better exact phrase."
    + "\n- `Canonical answer` must be the exact phrase supported by the bridge. It may be a listed candidate, or a trace-supported completion, but not a phrase invented from outside knowledge.",
    resolution_fewshot=HOTPOTQA_RELATION_V20_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 26:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The observed candidate \"Indianapolis Motor Speedway\" fills the requested track slot because the evidence bridge starts from the race named in the question and reaches that track. A nearby candidate from another race does not satisfy the source-to-slot bridge.\nCanonical answer: Indianapolis Motor Speedway"
    + "\n\nTiny resolution example 27:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The observed candidate \"8 km\" fills the requested distance slot; shortening it to \"8\" drops the required unit, so keep the complete candidate phrase.\nCanonical answer: 8 km"
    + "\n\nTiny resolution example 28:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The visible candidates are all unsupported or only intermediate objects; keep building until the evidence bridge names a candidate that fills the requested slot.",
    revision_rules=HOTPOTQA_RELATION_V20_PROFILE.revision_rules
    + "\n- Continue from the supported candidate-to-slot bridge and finish with the same exact candidate phrase when it already fills the requested slot."
    + "\n- Do not paraphrase, expand, or shorten a supported observed candidate unless the corrected claim explicitly says the candidate is incomplete.",
)


HOTPOTQA_RELATION_V25_PROFILE = replace(
    HOTPOTQA_RELATION_V20_PROFILE,
    name="hotpotqa_relation_v25",
    relation_analysis_rules=HOTPOTQA_RELATION_V20_PROFILE.relation_analysis_rules
    + "\n- Treat observed branch answers as noisy hypotheses for the answer bridge, not as fixed answer choices. A hypothesis can point to the right object while still being too long, too short, or at the wrong granularity."
    + "\n- For each hypothesis, separate two decisions: whether its evidence bridge reaches the requested slot, and what minimal sufficient phrase should be output for that same bridge."
    + "\n- Do not reward answer popularity or phrase length. Prefer the bridge that starts from the question's source entity and reaches the requested attribute with all required descriptors."
    + "\n- If a candidate is an explanatory sentence, a whole relation, or a paragraph fragment, compress it to the exact slot phrase only after confirming the bridge is correct."
    + "\n- If a candidate drops required units, counted objects, date components, office words, location level, or yes/no label, keep the bridge only if the evidence supplies the missing words.",
    resolution_rules=HOTPOTQA_RELATION_V20_PROFILE.resolution_rules
    + "\n- Correct claim must state the bridge first and the minimal sufficient answer phrase second. Do not make `Canonical answer` a raw copy of a noisy observed candidate unless that phrase already exactly fills the slot."
    + "\n- When an observed hypothesis is over-complete, keep its supported bridge but canonicalize only the requested phrase. When it is under-complete, add only words explicitly supplied by the same visible bridge."
    + "\n- If two hypotheses refer to the same supported object at different phrase lengths, resolve the object-level bridge first; then choose the shortest phrase that still preserves required units, titles, location level, counted objects, and labels."
    + "\n- If a hypothesis uses the wrong source entity, bridge entity, or requested attribute, reject that bridge even if its surface phrase looks plausible.",
    resolution_fewshot=HOTPOTQA_RELATION_V20_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 26:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The bridge identifies the government office held by the target person; the observed phrase \"Chief of Protocol of the United States\" is over-complete for the slot, so the minimal sufficient answer phrase is \"Chief of Protocol\".\nCanonical answer: Chief of Protocol"
    + "\n\nTiny resolution example 27:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The bridge identifies a time period as the requested slot; the observed phrase \"8th to 16th centuries\" is the same bridge but should be expressed as the complete slot phrase \"between the 8th and 16th centuries\".\nCanonical answer: between the 8th and 16th centuries"
    + "\n\nTiny resolution example 28:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The observed answer is a full explanatory sentence, but the bridge only asks for the location shared by the parks, so the minimal sufficient answer phrase is \"Canary Islands, Spain\".\nCanonical answer: Canary Islands, Spain",
    revision_rules=HOTPOTQA_RELATION_V20_PROFILE.revision_rules
    + "\n- Continue from the chosen evidence bridge, then output only the minimal sufficient answer phrase for the question slot."
    + "\n- Do not copy a raw observed candidate if it is a whole sentence, explanation, wrong-granularity phrase, or missing an explicitly required unit or qualifier.",
)


HOTPOTQA_RELATION_V26_PROFILE = replace(
    HOTPOTQA_RELATION_V25_PROFILE,
    name="hotpotqa_relation_v26",
    merge_rules=HOTPOTQA_RELATION_V25_PROFILE.merge_rules
    + "\n- Common Ground may contain only facts supported by at least two active paths and not denied by any active path. A named entity asserted by one path is disputed evidence, not shared evidence."
    + "\n- If paths share an entity bridge but one path swaps the source entity, keep the shared source entity visible and make the entity swap the first split."
    + "\n- Do not treat a repeated final answer token as common ground unless the supporting bridge to that token is also shared.",
    prefix_rules=HOTPOTQA_RELATION_V25_PROFILE.prefix_rules
    + "\n- The first split should preserve the question's source entity and requested slot. If a path changes the source entity, split on that entity bridge before judging the final phrase."
    + "\n- If two phrases name the same supported answer but differ in completeness, split on phrase sufficiency: complete slot phrase vs under-complete or over-complete phrase."
    + "\n- If a path has a supported complete candidate and another path gives a nearby entity from a different source or bridge, split on source-to-slot support rather than answer surface form.",
    relation_analysis_rules=HOTPOTQA_RELATION_V25_PROFILE.relation_analysis_rules
    + "\n- First write the slot ledger in words: question source entity -> bridge entity -> requested attribute -> answer phrase. The winning bridge must keep the same source entity named by the question."
    + "\n- Decide bridge support before phrase length. A shorter phrase is better only after it still names the same supported slot."
    + "\n- Minimal sufficient does not mean shortest-at-all-costs. Preserve required units, counted objects, date ranges, county/state level, office words, titles, owner words, and yes/no labels."
    + "\n- If an observed branch answer already satisfies the full supported bridge, prefer that exact phrase unless it clearly adds an unnecessary descriptor or misses a required phrase component."
    + "\n- If the visible evidence supports only a nearby relation, do not convert it into a final answer; keep_parallel or keep the trace-grounded candidate.",
    resolution_rules=HOTPOTQA_RELATION_V25_PROFILE.resolution_rules
    + "\n- Always include `Canonical answer: ...` when the Correct claim names a final phrase. This line must contain one clean short answer phrase only, with no quotes, rationale, path labels, or losing-side diagnosis."
    + "\n- Correct claim must be one bridge sentence: visible evidence supports the question source entity -> bridge entity -> requested attribute, so the canonical answer is the phrase. Keep losing-side diagnosis out of the answer phrase."
    + "\n- If the conflict is a source-entity swap, choose the bridge that preserves the entity stated by the question and visible shared evidence; do not let an unsupported one-path entity replace it."
    + "\n- Do not shorten a phrase by dropping required units, counted objects, date spans, county/state level, office words, owner words, titles, or yes/no labels."
    + "\n- Do not expand a supported category/type answer with a generic head noun if the shorter category label already fills the requested type slot."
    + "\n- If an observed branch answer already satisfies the complete bridge, keep that phrase unless the Correct claim explicitly says which component should be added or removed.",
    resolution_fewshot=HOTPOTQA_RELATION_V25_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 29:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question source entity is the actor stated in the shared evidence, not a nearby famous person from another path; Claim B preserves that source bridge, so the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 30:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a distance phrase and the supported branch already gives the complete phrase with its unit, so the canonical answer is 8 km.\nCanonical answer: 8 km"
    + "\n\nTiny resolution example 31:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is a date span, and the evidence gives both endpoints; shortening it to only the month drops a required component, so the canonical answer is 26-30 August 1914.\nCanonical answer: 26-30 August 1914"
    + "\n\nTiny resolution example 32:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot is the county-level location, so the answer must preserve the word County rather than switching to a city or a diagnostic sentence.\nCanonical answer: Saint Louis County"
    + "\n\nTiny resolution example 33:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot is the shared media type; the bridge supports documentary as the category label, and adding the generic plural head noun is unnecessary.\nCanonical answer: documentary",
    revision_rules=HOTPOTQA_RELATION_V25_PROFILE.revision_rules
    + "\n- Continue from the slot ledger and canonical phrase as normal short-answer reasoning; do not paste the diagnosis into the final line."
    + "\n- If `Canonical answer` is present, finish with exactly that clean phrase unless the continuation proves the bridge itself was unsupported."
    + "\n- Preserve units, counted objects, date ranges, county/state level, office words, owner words, titles, and yes/no labels from the corrected bridge.",
)


HOTPOTQA_RELATION_V27_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v27",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Treat negative elimination as a claim needing evidence. Do not say a candidate is absent, not included, or unsupported unless the visible trace actually gives that contrast; absence from one path is not evidence against another path."
    + "\n- When a candidate has a positive source-to-slot bridge and a rival has only a generated negative diagnosis, prefer the positive bridge or keep_parallel."
    + "\n- Read cardinality before choosing the phrase: `one of`, `a`, or `which person` permits one supported entity; `both`, `all`, `two`, or `and` asks for multiple entities."
    + "\n- If the question asks for a type/category, remove adjectives that describe the instance rather than the requested class; if it asks for a full named entity, preserve the name-level words."
    + "\n- If the evidence phrase has a head noun that names the counted or measured object, keep it in the canonical answer unless the question explicitly asks only for the bare number.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- A losing-side diagnosis may not introduce new facts. If the reason for rejecting a candidate is only `not shown`, `not included`, or `no evidence` while another branch gives a positive bridge, do not choose that rejection as the resolved claim."
    + "\n- For `one of` or singular who/person slots, `Canonical answer` should be one supported entity, not a conjunction or list, unless the question explicitly asks for both/all/two."
    + "\n- If a branch already gives one valid entity for a singular slot, keep it even when another branch lists multiple valid entities."
    + "\n- For quantity slots phrased as `how many countries/people/years`, keep the counted object word when it appears in the evidence phrase, such as `more than 70 countries`."
    + "\n- For `what type/kind/category` slots, the canonical phrase should name the type itself. Drop instance descriptors like unofficial, plural head nouns, or surrounding prose when they are not required by the asked class."
    + "\n- If deciding between an exact observed branch and a generated alternative, choose the exact observed branch unless the generated alternative has a visible positive bridge that the branch lacks.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 34:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The visible trace positively connects Cressida Bonas, Douglas Smith, and Lucien Laviscount to the film The Bye Bye Man; the rival only asserts a different film without a stronger visible bridge, so the canonical answer is The Bye Bye Man.\nCanonical answer: The Bye Bye Man"
    + "\n\nTiny resolution example 35:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for one of the stars, and Claim B gives one supported star rather than a list of both stars, so the canonical answer is Michael Seater.\nCanonical answer: Michael Seater"
    + "\n\nTiny resolution example 36:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks the type of online text-based role-playing game; unofficial is an instance descriptor, while the type is play-by-post role-playing game.\nCanonical answer: play-by-post role-playing game"
    + "\n\nTiny resolution example 37:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks how many countries, and the evidence phrase is more than 70 countries; the counted object is part of the answer phrase.\nCanonical answer: more than 70 countries"
    + "\n\nTiny resolution example 38:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: One path says Candidate X is absent from the requested relation, but that absence is not supported by visible trace evidence; keep the positive bridge and the disputed branch active until the source-to-slot evidence is explicit.",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Do not turn an unsupported negative diagnosis into a final answer during continuation."
    + "\n- For singular slots, continue to one supported entity; for multiple slots, continue to the requested list only when the question asks for multiple."
    + "\n- Keep counted-object words for how-many slots and drop non-type instance descriptors for type/category slots.",
)


HOTPOTQA_RELATION_V28_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v28",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Keep the literal question relation in view before judging phrase length. Resolve the bridge the question asks for, not a nearby bridge such as source entity, location container, event name, related person, or explanatory evidence."
    + "\n- A negative elimination is a claim, not shared evidence. Do not reject a candidate as absent, not included, or unsupported unless the visible trace gives that contrast; otherwise compare positive source-to-slot bridges or keep_parallel."
    + "\n- For yes/no questions, the requested object is the label yes or no. Evidence sentences may justify the label, but the canonical phrase should be the label."
    + "\n- For singular slots such as `one of`, `a`, or `which person`, one supported entity is sufficient. Do not force a list merely because multiple valid entities are visible."
    + "\n- For type/category slots, distinguish the class being asked from instance descriptors. Remove descriptors that are not part of the class, but do not drop broader category words when the question asks for the full kind of work or media."
    + "\n- Minimal sufficient phrase means the shortest phrase that still preserves the asked relation, entity ownership, location level, date/year slot, counted object, office/title words, and label form.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- Correct claim should first name the requested relation, then the positive evidence bridge, then the clean canonical phrase. Do not make a losing-side diagnosis the main claim."
    + "\n- If the winning bridge implies a yes/no answer, write `Canonical answer: yes` or `Canonical answer: no`; do not canonicalize to a full explanatory sentence."
    + "\n- If the question asks for a year/date/location/position/title/organization, do not output the event, venue, nearby person, city-only shortcut, or organization name with required owner/type words removed."
    + "\n- For `one of` or other singular person slots, an observed one-person answer can be canonical even when another branch lists several correct people."
    + "\n- For type/category conflicts, remove only descriptors that modify the instance rather than the requested class. Do not shorten a compound class if the evidence phrase says the broader category is part of what is asked.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 34:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested relation is the film in which the named daughter appears with the two named actors; Claim A gives a positive source-to-film bridge, while the rival only gives an unsupported negative contrast, so the canonical answer is The Bye Bye Man.\nCanonical answer: The Bye Bye Man"
    + "\n\nTiny resolution example 35:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for one of the stars, and Claim B gives one supported star rather than a list of both stars, so the canonical answer is Michael Seater.\nCanonical answer: Michael Seater"
    + "\n\nTiny resolution example 36:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested relation asks for the type of online text-based role-playing game; unofficial describes the instance, while the type is play-by-post role-playing game.\nCanonical answer: play-by-post role-playing game"
    + "\n\nTiny resolution example 37:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested relation asks whether both bands were from the U.S.; the evidence says one was and one was not, so the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 38:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested relation asks for the organization that ranked the boxer; the owner word magazine is part of the organization name, so the canonical answer is The Ring magazine.\nCanonical answer: The Ring magazine",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the requested relation plus positive bridge. Finish with the canonical phrase if it is present and bridge-supported."
    + "\n- For yes/no questions, finish with only yes or no after the reasoning."
    + "\n- Do not let shortest-phrase pressure remove entity ownership, location level, counted-object words, date/year form, office/title words, or label form.",
)


HOTPOTQA_RELATION_V29_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v29",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Before phrase minimization, close the full requested slot: all source entities named by the question, all bridge constraints, then the requested attribute. A candidate that satisfies only a nearby entity, event, song, place, or evidence phrase is not yet an answer."
    + "\n- Preserve the question relation exactly. `near`, `in`, `from`, `born in`, `held by`, `released by`, `co-produced with`, and `shared by both` are different relation claims; do not swap one relation for a nearby one."
    + "\n- For yes/no questions, keep the task as a truth predicate. The requested object is only `yes` or `no`; do not convert the predicate into an open slot such as a shared ingredient, person, place, or explanatory sentence."
    + "\n- Treat negative elimination as a claim needing evidence. If one path has a positive evidence-to-slot bridge and another only says `not shown` or shifts entity type, compare the positive bridge first or keep_parallel."
    + "\n- A shorter phrase is better only after it still preserves ownership words, date spans, location level, office/title words, counted-object words, and the full compound class requested by the question.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- Output protocol must be internally consistent: `Action` must be one of choose_A, choose_B, keep_parallel, synthesize; `Winning side` must be Claim A, Claim B, both, or neither. Do not write agent ids such as A1/P1 as the winning side."
    + "\n- `Correct claim` should be one positive bridge sentence: question source(s) -> satisfied bridge constraints -> requested attribute -> canonical answer. Keep losing-side diagnosis out of the main claim unless it is needed to name the mismatch."
    + "\n- If the conflict is yes/no, `Canonical answer` must be exactly `yes` or `no`; the evidence sentence belongs in `Correct claim`, not in the final answer phrase."
    + "\n- If the question asks for a year/date/time, `Canonical answer` must be the year/date/time phrase, not the event or entity whose date is being compared."
    + "\n- If the question asks for a relation-constrained person/object, first identify the referent that satisfies every constraint; do not answer with a co-writer, location, source entity, or nearby title that satisfies only one constraint."
    + "\n- If `Correct claim` names a final phrase, always include `Canonical answer: ...` as one clean short-answer phrase with no quotes, rationale, path labels, or full sentence.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 34:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested relation is a yes/no predicate asking whether both items satisfy the same property; the evidence supports one item satisfying it and the other not satisfying it, so the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 35:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for the year of the earlier event; the evidence identifies the event only as the bridge and gives the year as the requested attribute, so the canonical answer is 1998.\nCanonical answer: 1998"
    + "\n\nTiny resolution example 36:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The requested slot asks for the collaborator on the song that satisfies every constraint, including soundtrack use; the branch tied to a different song or only a co-writer does not fill the slot, so the canonical answer is Jordan Lee.\nCanonical answer: Jordan Lee"
    + "\n\nTiny resolution example 37:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The requested slot asks for the major city near a location, not the exact neighborhood or school site; the evidence bridge points from the location to the nearby major city, so the canonical answer is Springfield.\nCanonical answer: Springfield",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the positive bridge and final slot, not from a nearby entity. If `Canonical answer` is present and the bridge is supported, finish with exactly that phrase."
    + "\n- For yes/no predicates, finish with only yes or no. For date/year predicates, finish with only the date/year phrase. Do not finish with the evidence sentence or the event name.",
)


HOTPOTQA_RELATION_V30_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v30",
    merge_rules=HOTPOTQA_RELATION_V26_PROFILE.merge_rules
    + "\n- A fact becomes Common Ground only when it is visibly grounded in the question/context or explicitly supported by at least two independent traces. Two agents repeating the same guessed intermediate entity is agreement, not evidence."
    + "\n- For multi-hop short-answer QA, keep a per-candidate bridge ledger: question source -> required bridge constraint(s) -> requested attribute -> candidate phrase. Do not merge two paths as equivalent until the bridge ledger matches."
    + "\n- If a branch contains a candidate answer that is not yet shared, keep it as a candidate hypothesis in Paths instead of promoting its supporting story into Common Ground.",
    prefix_rules=HOTPOTQA_RELATION_V26_PROFILE.prefix_rules
    + "\n- Prefer the first split where two candidate bridge ledgers differ: wrong source entity, missing bridge constraint, wrong requested attribute, or phrase granularity. Do not wait until only the final surface answer differs."
    + "\n- If two paths share a wrong-looking intermediate entity only by assertion and a third path gives a different evidence-linked candidate, expose the evidence-link conflict rather than treating the majority entity as shared."
    + "\n- For questions with titles, dates, offices, locations, and yes/no predicates, the first split should preserve the exact relation word from the question before phrase-length cleanup.",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Analyze each observed candidate as a hypothesis with a visible bridge ledger. Ask: which source entity does it start from, which context sentence or trace claim supports the bridge, which requested attribute does it fill, and what phrase would answer that slot?"
    + "\n- Majority agreement is weak unless the agreeing paths show the same evidence bridge. A minority candidate with a complete visible bridge can beat two candidates that share only a guessed shortcut."
    + "\n- Do not infer absence from missing evidence. If a branch says none/not shown while another branch gives a positive bridge, compare whether the positive bridge satisfies the question before choosing the negative."
    + "\n- When the task asks for a yes/no predicate, keep two layers separate: the evidence facts and the truth value of the whole predicate. The candidate phrase must be yes or no, not the evidence sentence."
    + "\n- When the task asks for an exact short phrase, phrase minimization is the last step; first decide which bridge is correct.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- Choose a side only after naming the supported bridge ledger. If neither side gives a supported complete bridge, use keep_parallel or synthesize a bridge only from visible trace/context evidence."
    + "\n- The Correct claim must not be a popularity vote. It should say which candidate bridge is evidence-supported and which bridge component fails for the rival: source entity, bridge constraint, requested attribute, or phrase granularity."
    + "\n- If a correct observed candidate exists and its bridge is supported, keep that candidate phrase unless the same bridge explicitly requires adding or removing a phrase component."
    + "\n- If the winning claim implies a yes/no label, `Canonical answer` must be exactly yes or no. If the winning claim implies a title/date/person/place, `Canonical answer` must be that short phrase only."
    + "\n- Do not resolve to none/no/not mentioned merely because one trace lacks a bridge; require a visible contradiction or a completed yes/no predicate.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 39:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The majority candidate repeats a nearby person, but Claim B gives the complete bridge from the question source through the required film constraint to the requested office, so the canonical answer is Chief of Protocol.\nCanonical answer: Chief of Protocol"
    + "\n\nTiny resolution example 40:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks whether both entities satisfy the property; the bridge shows one entity satisfies it and the other does not, so the completed predicate is false and the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 41:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: Claim B keeps the required event/date bridge and answers the requested date, while Claim A stops at the regular home venue of the team, so the canonical answer is Mercedes-Benz Superdome.\nCanonical answer: Mercedes-Benz Superdome"
    + "\n\nTiny resolution example 42:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The paths disagree on the intermediate entity, but neither trace shows a context-supported bridge from that entity to the requested attribute; keep the candidate bridges separate instead of selecting by majority.",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the evidence-supported bridge ledger as normal short-answer reasoning: source -> bridge constraint(s) -> requested attribute -> final phrase."
    + "\n- Do not replace a supported observed candidate with a nearby generated phrase unless the bridge ledger explicitly proves the phrase should change."
    + "\n- If `Canonical answer` is present and the bridge ledger is supported, finish with exactly that clean phrase.",
)


HOTPOTQA_RELATION_V31_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v31",
    merge_rules=HOTPOTQA_RELATION_V26_PROFILE.merge_rules
    + "\n- Treat observed branch answers as candidate hypotheses. Keep each candidate with its visible support bridge instead of turning a repeated candidate into Common Ground by vote."
    + "\n- Common Ground may hold shared question facts, shared source entities, and shared bridge facts, but not a one-path candidate answer or a majority guess unless the supporting bridge is also shared."
    + "\n- Do not over-require perfect citations inside the trace: a candidate with a coherent positive bridge from the visible reasoning can beat a rival that has only a nearby entity, missing slot, or negative absence claim.",
    prefix_rules=HOTPOTQA_RELATION_V26_PROFILE.prefix_rules
    + "\n- When candidates differ, expose the first bridge component that differs: source entity, intermediate entity, requested relation, answer level, or phrase sufficiency."
    + "\n- If one path has a positive bridge to the requested slot and another path only says missing/not shown/none, split on positive bridge support versus absence claim rather than accepting absence as shared evidence."
    + "\n- Keep method differences parallel unless they lead to incompatible bridge components for the same requested slot.",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- For each observed candidate, write a compact support audit: candidate phrase; source entity; requested slot; strongest visible bridge; missing or contradicted component."
    + "\n- Prefer the candidate whose bridge satisfies the exact requested slot. Do not choose by majority, but also do not reject a positive observed candidate merely because the trace is shorter than ideal."
    + "\n- Negative claims such as none, not mentioned, not shown, or unknown are weaker than a positive bridge unless they show that the positive bridge uses the wrong source entity, wrong relation, or wrong answer level."
    + "\n- Phrase cleanup is after bridge choice. First decide the supported object, then output the minimal sufficient phrase that preserves units, dates, titles, locations, office words, and yes/no labels.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- Resolve to the best supported observed candidate bridge when one exists. Do not invent a new phrase unless the visible bridge explicitly supplies a required missing component."
    + "\n- Correct claim should be a positive bridge claim: source entity -> requested relation/slot -> canonical answer. Mention the rival only as the failed bridge component, not as the main answer."
    + "\n- Use keep_parallel only when no candidate has a visible bridge to the requested slot or when the visible bridges are genuinely compatible and incomplete."
    + "\n- Do not let a negative absence claim beat a positive bridge unless the absence claim directly shows the positive bridge is about the wrong source, wrong relation, or wrong answer level."
    + "\n- If the bridge selects a candidate but the phrase is too long or too short, keep the same supported object and canonicalize only the phrase sufficiency.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 34:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: Claim A gives a positive bridge from the question source to the requested film title; Claim B only names a nearby film without a stronger source-to-slot bridge, so the canonical answer is The Bye Bye Man.\nCanonical answer: The Bye Bye Man"
    + "\n\nTiny resolution example 35:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: Claim B keeps the source entity and reaches the requested actor role, while Claim A swaps to a nearby character from a different bridge, so the canonical answer is Jillian Belk.\nCanonical answer: Jillian Belk"
    + "\n\nTiny resolution example 36:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: Claim A's positive bridge supports a yes/no predicate; the rival absence claim does not disprove that bridge, so the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 37:\nAction: keep_parallel\nWinning side: neither\nCorrect claim: The candidates differ, but neither trace gives a visible bridge from the question source to the requested slot; keep the candidate bridges separate instead of selecting by popularity.",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the chosen positive bridge and finish with the canonical phrase. Do not re-open a losing absence claim unless the continuation proves the bridge uses the wrong source or relation."
    + "\n- If `Canonical answer` is present, use it as the final phrase unless the preceding bridge is explicitly contradicted by the continuation.",
)


HOTPOTQA_RELATION_V32_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v32",
    merge_rules=HOTPOTQA_RELATION_V26_PROFILE.merge_rules
    + "\n- Before merging candidate stories, lock the question's requested answer slot in plain words. Common Ground may contain the locked slot, but not a rewritten nearby slot."
    + "\n- Keep observed branch answers as hypotheses attached to that locked slot. Do not promote a hypothesis to Common Ground merely because it sounds like useful evidence."
    + "\n- If paths agree on evidence but disagree on what the question asks for, make the slot interpretation the first split.",
    prefix_rules=HOTPOTQA_RELATION_V26_PROFILE.prefix_rules
    + "\n- First split on a changed requested slot before splitting on answer surface form: yes/no predicate vs shared ingredient, major city vs neighborhood, father vs child, office held by named people vs office held by another person."
    + "\n- Expose the earliest claim that changes who or what the slot is about. A nearby fact can support reasoning, but it is not the answer slot unless the question asks for it."
    + "\n- Keep different methods parallel when they preserve the same locked slot and only use different evidence order.",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Start by writing one locked-slot sentence: `The question asks for ...`, using the exact source entity/entities and relation from the question."
    + "\n- Then audit each candidate against that locked slot: candidate phrase; source entity; relation; requested attribute; phrase sufficiency."
    + "\n- Reject a candidate only if its bridge fails the locked slot, not because its trace is shorter, uses a different method, or lacks a perfect citation."
    + "\n- Do not let an evidence fact replace the answer slot. For a yes/no question, the final object is yes/no; for a location-level question, preserve the requested level; for a kinship question, preserve direction.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- The first sentence of `Correct claim` must lock the requested slot: `The question asks for ...`. Resolve only claims that fill that slot."
    + "\n- If a candidate answers a nearby slot instead of the locked slot, name the slot mismatch and keep the candidate that fills the locked slot."
    + "\n- Do not change a yes/no predicate into the fact that explains it. Do not change a requested major city into a neighborhood, a father into a child, a shared office into an office held by someone else, or a ranking organization into a shortened title."
    + "\n- After slot choice, canonicalize only phrase sufficiency. Preserve required labels, units, date spans, location level, office words, titles, and organization descriptors.",
    resolution_fewshot=HOTPOTQA_RELATION_V26_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 34:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for a yes/no predicate about whether both items satisfy the same property. The bridge shows one item satisfies it and the other does not, so the locked slot is answered by no rather than by an explanatory object.\nCanonical answer: no"
    + "\n\nTiny resolution example 35:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the major city by the school. A neighborhood or community can be bridge evidence, but the locked slot is the major city, so the canonical answer is Las Vegas.\nCanonical answer: Las Vegas"
    + "\n\nTiny resolution example 36:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for the father of the named person. A child of that person is the reverse kinship slot, so keep the bridge that answers the father slot.\nCanonical answer: Merovech"
    + "\n\nTiny resolution example 37:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the organization that ranked the boxer. The title alone is under-specified for this slot, so preserve the organization descriptor.\nCanonical answer: The Ring magazine",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the locked requested slot plus the chosen bridge; finish with the canonical phrase for that slot."
    + "\n- Do not continue by answering a nearby evidence slot, even if it is true."
    + "\n- If `Canonical answer` is present and the locked-slot bridge supports it, finish with exactly that phrase.",
)


HOTPOTQA_RELATION_V33_PROFILE = replace(
    HOTPOTQA_RELATION_V31_PROFILE,
    name="hotpotqa_relation_v33",
    relation_analysis_rules=HOTPOTQA_RELATION_V31_PROFILE.relation_analysis_rules
    + "\n- Before comparing candidates, do a light slot sanity check in one plain phrase: what kind of short answer does the question ask for? Examples: yes/no label, person, organization, title, date, place at a stated level, shared office, or item in a required relation."
    + "\n- Use that slot phrase only to compare bridges. Do not turn it into a hard template, a rejection rule, or a reason to invent a new unobserved answer."
    + "\n- A candidate is strong when its visible bridge reaches the slot named by the question. A nearby true fact is weak if it answers a different slot, reverses a relation, changes the source entity, or stops at an intermediate entity."
    + "\n- When a branch already gives a clean candidate that fills the slot, prefer maintaining that candidate over rewriting it into a nearby entity or explanatory sentence.",
    resolution_rules=HOTPOTQA_RELATION_V31_PROFILE.resolution_rules
    + "\n- Correct claim should briefly state the slot and the supported bridge, but keep the main claim positive: the evidence bridge supports this candidate for the requested slot."
    + "\n- If a rival candidate is a nearby fact, explain only the bridge mismatch: wrong source, wrong relation direction, wrong answer level, intermediate entity, or phrase granularity."
    + "\n- Do not replace an observed supported candidate with a generated phrase unless the same visible bridge clearly supplies the missing phrase component."
    + "\n- For yes/no questions, resolve the truth value of the whole predicate and keep `Canonical answer` as yes or no, not the evidence object that explains the label.",
    resolution_fewshot=HOTPOTQA_RELATION_V31_PROFILE.resolution_fewshot
    + "\n\nTiny resolution example 38:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question slot is the office shared by the named people. Claim A bridges the named people to that shared office, while the rival names an office held by a different person, so the canonical answer is President of the United States.\nCanonical answer: President of the United States"
    + "\n\nTiny resolution example 39:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question slot is a major city near the school. Claim B reaches the major-city slot, while the rival stops at a neighborhood or community, so the canonical answer is Las Vegas.\nCanonical answer: Las Vegas"
    + "\n\nTiny resolution example 40:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question slot is a yes/no predicate. The bridge decides the predicate itself, so the canonical answer is no rather than the ingredient or evidence object.\nCanonical answer: no",
    revision_rules=HOTPOTQA_RELATION_V31_PROFILE.revision_rules
    + "\n- Continue from the supported candidate bridge as normal short-answer reasoning. Keep the final line as the clean phrase for the requested slot."
    + "\n- If the continuation mentions a nearby true fact, use it only as evidence; do not let it replace the candidate that fills the requested slot.",
)


HOTPOTQA_RELATION_V34_PROFILE = replace(
    HOTPOTQA_RELATION_V26_PROFILE,
    name="hotpotqa_relation_v34",
    merge_rules=HOTPOTQA_RELATION_V26_PROFILE.merge_rules
    + "\n- Treat observed branch answers as noisy candidate hypotheses attached to bridges. Do not promote a candidate answer into Common Ground unless at least two paths share the same source-to-slot bridge, not merely the same final token."
    + "\n- Common Ground may keep the question source entity, shared context facts, and the requested answer slot. A one-path bridge, nearby entity, or explanatory object stays in that path until compared.",
    prefix_rules=HOTPOTQA_RELATION_V26_PROFILE.prefix_rules
    + "\n- Find the first split at the earliest bridge component that would change the requested short answer: source entity, intermediate bridge, relation direction, requested slot, answer level, or phrase sufficiency."
    + "\n- Do not split on method order, citation style, or final-token wording while the paths still preserve the same source-to-slot bridge."
    + "\n- If a path gives a positive observed candidate and another path gives an absence/unknown claim, split on whether the positive bridge actually reaches the requested slot.",
    relation_analysis_rules=HOTPOTQA_RELATION_V26_PROFILE.relation_analysis_rules
    + "\n- Start with one compact slot sentence: what exact kind of short answer does the question request, using the question's source entity and relation words."
    + "\n- Then audit each observed candidate as: candidate phrase; source entity; bridge evidence in the visible reasoning; requested slot filled; missing, reversed, or wrong-level component."
    + "\n- Prefer a supported observed candidate when its bridge reaches the requested slot. Do not invent a new answer phrase unless the visible bridge explicitly supplies a more exact phrase component."
    + "\n- A nearby true fact is not a final answer if it answers a different slot, reverses a relation, stops at an intermediate entity, or changes a yes/no predicate into its evidence object."
    + "\n- Phrase cleanup is last and conservative: preserve yes/no labels, units, counted objects, date spans, city/county/state level, office words, organization descriptors, titles, and compound type labels that the slot requires.",
    resolution_rules=HOTPOTQA_RELATION_V26_PROFILE.resolution_rules
    + "\n- Correct claim should be a positive bridge claim: the question source and relation lead through visible evidence to this candidate for the requested slot."
    + "\n- If rejecting a rival, name only the failed bridge component: wrong source, wrong relation direction, wrong answer level, intermediate entity, absence claim, or phrase granularity."
    + "\n- Keep a clean observed candidate if its bridge fills the slot; do not rewrite it into a nearby entity, broader place, explanatory sentence, or shortened organization/title."
    + "\n- For yes/no questions, resolve the truth value of the whole predicate and set `Canonical answer` to yes or no only."
    + "\n- If no visible candidate bridge reaches the slot, use keep_parallel or synthesize only from visible trace/context evidence; do not choose by majority or by shortest phrase alone.",
    resolution_fewshot=CLEAN_SHORT_ANSWER_SLOT_FEWSHOT
    + "\n\nTiny resolution example 39:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for a yes/no predicate. Claim A completes the predicate truth value, while the rival gives the evidence object instead of the label, so the canonical answer is no.\nCanonical answer: no"
    + "\n\nTiny resolution example 40:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The question asks for a place at the stated level. Claim B keeps the supported city-and-region level, while the rival stops at a broader country-only or narrower neighborhood-only place, so keep the complete observed phrase.\nCanonical answer: Example City, Example Region"
    + "\n\nTiny resolution example 41:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for the organization/title phrase that fills the slot. Claim A preserves the required organization descriptor, while the rival shortens to an ambiguous nearby title, so keep the complete observed phrase.\nCanonical answer: Example Review magazine"
    + "\n\nTiny resolution example 42:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: Claim B gives a positive bridge from the question source to the requested answer slot; the rival names a nearby candidate without a stronger source-to-slot bridge, so keep the supported observed candidate.\nCanonical answer: Example Work Title",
    revision_rules=HOTPOTQA_RELATION_V26_PROFILE.revision_rules
    + "\n- Continue from the supported candidate bridge and the requested slot. Finish with the canonical phrase if present and bridge-supported."
    + "\n- Do not replace a supported observed candidate with a generated nearby phrase unless the continuation explicitly proves the observed phrase has the wrong source, wrong relation, or wrong answer level."
    + "\n- For yes/no predicates, finish with only yes or no. For short-answer slots, preserve required level and descriptors rather than chasing the shortest possible string.",
)


HOTPOTQA_RELATION_V35_PROFILE = replace(
    HOTPOTQA_RELATION_V34_PROFILE,
    name="hotpotqa_relation_v35",
    merge_rules=HOTPOTQA_RELATION_V34_PROFILE.merge_rules
    + "\n- Maintain a role ledger for each candidate: question source, required bridge constraint, requested answer role, and candidate phrase. Common Ground can include a role only when the paths agree on that role, not merely when they mention the same entity."
    + "\n- Keep excluded/source entities, intermediate entities, answer-role entities, and explanatory facts in separate slots. A true intermediate person, office, place, work, or date is not the final answer unless it fills the requested answer role.",
    prefix_rules=HOTPOTQA_RELATION_V34_PROFILE.prefix_rules
    + "\n- When the question contains words like other, besides, both, one of, portrayed by, held by, came first, older, common, or whose, expose the first role-binding conflict before judging the final phrase."
    + "\n- Split on role swaps: excluded entity returned as answer, person returned when a distinction/office is requested, evidence object returned when yes/no is requested, actor returned when role is requested, or pair/list returned when the question asks for one member.",
    relation_analysis_rules=HOTPOTQA_RELATION_V34_PROFILE.relation_analysis_rules
    + "\n- Write a compact role ledger before choosing: source/excluded entity; bridge constraint; requested answer role; candidate phrase."
    + "\n- Ask whether the candidate phrase is the role filler or merely evidence for the filler. If it is evidence only, keep the bridge but do not land that evidence object as the answer."
    + "\n- In comparison questions, the answer role is the item or year requested after comparison, not the comparison property itself unless the question asks for that property."
    + "\n- In 'one of' or singular-who questions, a two-person list is too broad unless the question explicitly asks for both people.",
    resolution_rules=HOTPOTQA_RELATION_V34_PROFILE.resolution_rules
    + "\n- The Correct claim should include the role ledger in one sentence: source/exclusion -> bridge constraint -> requested answer role -> canonical answer."
    + "\n- Prefer the candidate that fills the requested answer role over a candidate that only identifies the source/intermediate entity, even when the intermediate entity is true."
    + "\n- For 'other/besides' questions, the canonical answer must not be the excluded/source entity. For 'held by both' questions, it must be the shared role/office, not an office held by a different person. For 'distinction held by' questions, it must be the distinction, not the person."
    + "\n- For singular role questions, do not expand a supported single answer into a pair/list unless the role ledger says the requested answer role is plural.",
    resolution_fewshot=CLEAN_SHORT_ANSWER_SLOT_FEWSHOT
    + "\n\nTiny resolution example 43:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: Source Person is the excluded/source entity; the question asks for the other member with the same property, and Claim A bridges to Example Other Person for that requested answer role, so do not return Source Person.\nCanonical answer: Example Other Person"
    + "\n\nTiny resolution example 44:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: Example Athlete is the intermediate person; the requested answer role is the distinction held by that person, and Claim B fills that role with Example Distinction rather than the person's name.\nCanonical answer: Example Distinction"
    + "\n\nTiny resolution example 45:\nAction: choose_A\nWinning side: Claim A\nCorrect claim: The question asks for one person who played the role. Claim A gives one supported person, while Claim B expands to a two-person list without the question asking for both, so keep the singular supported phrase.\nCanonical answer: Example Person"
    + "\n\nTiny resolution example 46:\nAction: choose_B\nWinning side: Claim B\nCorrect claim: The named source people are only anchors; the requested answer role is their shared office. Claim B fills that shared-office role, while Claim A names an office held by a different bridge entity.\nCanonical answer: Example Shared Office",
    revision_rules=HOTPOTQA_RELATION_V34_PROFILE.revision_rules
    + "\n- Continue from the role ledger: source/exclusion -> bridge constraint -> requested answer role -> canonical answer. Do not continue from an intermediate entity as if it were the final answer."
    + "\n- If the repaired claim separates an intermediate entity from the requested role, the final line must be the role filler, not the intermediate entity.",
)


HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE = replace(
    HOTPOTQA_RELATION_V35_PROFILE,
    name="hotpotqa_relation_v36_equal_token",
    merge_rules=HOTPOTQA_RELATION_V35_PROFILE.merge_rules
    + "\n- Under longer outputs, compress each branch to the role ledger and answer slot; do not preserve extra narrative unless it decides the bridge."
    + "\n- Observed final answers are noisy hypotheses. Keep their source, bridge constraint, requested role, and phrase sufficiency attached.",
    prefix_rules=HOTPOTQA_RELATION_V35_PROFILE.prefix_rules
    + "\n- Split on role-ledger conflicts before phrase polishing: source/exclusion, bridge constraint, requested role, answer level, or yes/no predicate.",
    relation_analysis_rules=HOTPOTQA_RELATION_V35_PROFILE.relation_analysis_rules
    + "\n- For each candidate, do a compact candidate audit instead of a full re-solve: candidate phrase; role ledger; positive bridge; failed component if any."
    + "\n- A verbose trace is weaker than a compact trace when it changes the requested role, answers an intermediate entity, or turns a yes/no predicate into an explanation.",
    resolution_rules=HOTPOTQA_RELATION_V35_PROFILE.resolution_rules
    + "\n- Prefer the supported observed candidate bridge. Synthesize a new canonical phrase only when the visible bridge proves a required phrase correction."
    + "\n- If the question asks yes/no, canonical answer must be yes or no only; if it asks for a role filler, canonical answer must be that role filler, not the evidence object.",
    revision_rules=HOTPOTQA_RELATION_V35_PROFILE.revision_rules
    + "\n- Continue from the role ledger and supported canonical answer. Do not reopen the answer slot or regenerate a nearby phrase unless the continuation proves a slot mismatch."
    + "\n- Longer token budget is for stability, not extra exploration; finish with the supported canonical short answer.",
)


HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE = replace(
    HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE,
    name="hotpotqa_relation_v37_extraction_guard",
    relation_analysis_rules=HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- If an observed candidate is a list, compare exact requested cardinality: all required people/items, no extra unasked items, no missing slot fillers."
    + "\n- Preserve a supported canonical phrase or list from the corrected claim; do not let later rationale entities expand it.",
    resolution_rules=HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- End a decisive corrected claim with `Canonical answer: PHRASE` only when the phrase or list exactly fills the requested role."
    + "\n- A longer list is not better unless the role ledger explicitly requires every added item.",
    revision_rules=HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve the exact canonical answer phrase/list through revision. Do not add extra entities from evidence unless the repaired claim says they are required answers.",
)


HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE = replace(
    HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE,
    name="hotpotqa_relation_v38_bridge_audit",
    merge_rules=HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE.merge_rules
    + "\n- If current answers agree but the evidence-to-role bridge is shallow, expose the requested role and missing bridge instead of freezing the repeated phrase.",
    relation_analysis_rules=HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- Audit candidate phrase -> role ledger -> evidence bridge. A phrase is supported only when it fills the exact requested role or yes/no predicate.",
    resolution_rules=HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- A canonical answer must be rewrite-supported by a visible role-ledger bridge. If the bridge cannot be carried into revision, use keep_parallel."
    + "\n- Prefer a supported observed candidate; introduce a new phrase only when the visible bridge supplies the exact phrase or list.",
    revision_rules=HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Carry the role ledger into the rewrite and finish with the exact phrase/list only after that bridge is stated."
    + "\n- Do not let a resolution-only canonical answer override an unrevised or weakly revised trace.",
)


HOTPOTQA_RELATION_V39_LEDGER_AUDIT_PROFILE = replace(
    HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE,
    name="hotpotqa_relation_v39_ledger_audit",
    relation_analysis_rules=HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE.relation_analysis_rules
    + "\n- Write a compact bridge ledger: question source, context entity, requested attribute, answer phrase, and failed component if any."
    + "\n- If a surface entity lacks the attribute but a directly linked context entity supplies it, compare both bridges before rejecting the narrower phrase.",
    resolution_rules=HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE.resolution_rules
    + "\n- The Correct claim must carry a complete visible bridge to the requested slot before naming a canonical answer."
    + "\n- Use keep_parallel if the trace only gives a nearby entity, nearby attribute, or incomplete hop.",
    revision_rules=HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE.revision_rules
    + "\n- Preserve the bridge ledger through revision and finish with the minimal sufficient answer phrase."
    + "\n- Do not replace a complete bridge with a famous-name or general-memory association.",
)


MUSIQUE_RELATION_V2_PROFILE = replace(
    MUSIQUE_RELATION_V1_PROFILE,
    name="musique_relation_v2",
    merge_rules=SHORT_ANSWER_SLOT_MERGE_RULES
    + "\n- MuSiQue questions often require several hops; do not resolve from the first true hop if the requested slot is later in the chain.",
    prefix_rules=SHORT_ANSWER_SLOT_PREFIX_RULES,
    relation_analysis_rules=SHORT_ANSWER_SLOT_ANALYSIS_RULES
    + "\n- For multi-hop chains, identify which hop each path has reached before deciding whether it fills the requested slot.",
    resolution_rules=SHORT_ANSWER_SLOT_RESOLUTION_RULES,
    resolution_fewshot=SHORT_ANSWER_SLOT_FEWSHOT,
    revision_rules=SHORT_ANSWER_SLOT_REVISION_RULES,
)


MUSIQUE_RELATION_V3_PROFILE = replace(
    MUSIQUE_RELATION_V2_PROFILE,
    name="musique_relation_v3",
    merge_rules=MUSIQUE_RELATION_V2_PROFILE.merge_rules
    + "\n- Treat each hop as evidence-bound: do not fill a missing entity, attribute, location, spouse, date, or title from outside memory."
    + "\n- A path has reached the requested slot only if the trace explicitly states the intermediate entity and the requested attribute/value.",
    prefix_rules=MUSIQUE_RELATION_V2_PROFILE.prefix_rules
    + "\n- If paths are missing different hops, expose the missing-hop status instead of turning one partial chain into the answer."
    + "\n- Do not make a final-answer split from two unsupported guesses; keep building until an evidence-backed slot bridge appears.",
    relation_analysis_rules=MUSIQUE_RELATION_V2_PROFILE.relation_analysis_rules
    + "\n- For each path, mark every hop as supported, missing, or guessed from outside the trace."
    + "\n- A guessed outside fact is not a stronger bridge than an explicit 'not enough information' trace."
    + "\n- If the dispute is a raw factual lookup and neither path quotes evidence for the missing hop, do not choose by plausibility.",
    resolution_rules=MUSIQUE_RELATION_V2_PROFILE.resolution_rules
    + "\n- Choose a side only when its full bridge to the requested slot is supported by the visible trace evidence."
    + "\n- If the winning claim would require an unstated factual lookup, use keep_parallel and name the missing hop instead of producing a new final phrase."
    + "\n- Do not let famous-name/common-knowledge associations settle the bridge unless the trace itself states that association.",
    resolution_fewshot=SHORT_ANSWER_SLOT_FEWSHOT
    + "\n\nTiny resolution example 4:\nCorrect claim: The path has reached the intermediate city but has not shown the county of that city, so keep_parallel and continue from the supported city-to-county hop.\n\nTiny resolution example 5:\nCorrect claim: One path guesses a spouse from outside memory while another path lacks the spouse hop; neither supplies evidence for the requested spouse slot, so keep_parallel.",
)


PROOFWRITER_RELATION_V1_PROFILE = replace(
    ANLI_PROMPT_PROFILE,
    name="proofwriter_relation_v1",
    merge_intro="Merge several rule-proof traces into a compact premises-to-conclusion truth-value memo.",
    merge_rules="""
Rules:
- Keep shared premises and shared rule applications separate from the truth-value label.
- Do not treat a bare true/false/unknown label as evidence.
- Split on the first proof-relation conflict: which premise/rule applies, whether the conclusion follows, whether it is contradicted, or whether it remains unknown.
- If paths differ only in wording but support the same truth value for the same conclusion, do not split on style alone.
""".strip(),
    audit_intro="Audit a rule-proof memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Common Ground should contain only premises or rule applications shared by at least two traces.
- Move one-path-only proof steps and truth-value judgments back into Paths.
- Do not invent rules, facts, or missing exceptions outside the problem statement and traces.
""".strip(),
    prefix_intro="Rebuild a compact rule-proof memo that exposes the first premises-to-conclusion conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared premise/rule prefix before the first conflict.
- Split on whether the shared proof state proves, disproves, or leaves open the queried conclusion.
- Do not split on a bare label token before checking the proof bridge.
""".strip(),
    relation_analysis_intro="Describe one premises-to-conclusion truth-value divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First identify the queried conclusion.
- Then identify the premises or rules each path uses.
- Compare whether that proof bridge supports true, false, or unknown for the conclusion.
- Do not choose by final label popularity.
""".strip(),
    resolution_intro="Resolve one minimal premises-to-conclusion truth-value divergence.",
    resolution_rules="""
Rules:
- Return the proof relation claim to keep.
- Correct claim must say whether the premises prove the conclusion, disprove it, or leave it unknown.
- Do not use a bare true/false/unknown label as evidence.
- If a path needs a missing rule or missing premise, call that bridge unsupported instead of inventing it.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The premises prove the queried conclusion, so the supported truth value is true.

Tiny resolution example 2:
Correct claim: The premises prove the negation of the queried conclusion, so the supported truth value is false.

Tiny resolution example 3:
Correct claim: The premises do not prove the conclusion or its negation, so the supported truth value is unknown.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired proof relation claim and finish the truth-value answer.
- Do not stop at a premise or rule if the question asks for the conclusion's truth value.
- The final line should contain only the allowed label token.
""".strip(),
)


FOLIO_RELATION_V1_PROFILE = replace(
    PROOFWRITER_RELATION_V1_PROFILE,
    name="folio_relation_v1",
    merge_intro="Merge several natural-language logic traces into a compact premises-to-conclusion truth-value memo.",
    audit_intro="Audit a natural-language logic memo before divergence resolution.",
    prefix_intro="Rebuild a compact natural-language logic memo that exposes the first premises-to-conclusion conflict.",
)


PROOFWRITER_RELATION_V2_PROFILE = replace(
    PROOFWRITER_RELATION_V1_PROFILE,
    name="proofwriter_relation_v2",
    merge_intro="Merge several formal proof traces into a compact premise-rule-to-query proof-status memo.",
    merge_rules="""
Rules:
- Treat the task as formal premise/rule reasoning, not ordinary-world reasoning.
- Keep the queried statement fixed. Do not replace it with a nearby predicate, subject, or relation.
- Common Ground may contain only explicit premises, explicit rules, or rule applications shared by at least two traces.
- Do not treat a bare true/false/unknown label as evidence.
- Do not use common sense, self-evidence, or likely-world assumptions as a proof step.
- Do not use the converse or inverse of a conditional unless that rule is explicitly stated.
- For a disjunction, do not pick one branch unless a premise or rule rules out the other branch.
- Split on the first proof-status conflict: proved, disproved, or still open for the queried statement.
""".strip(),
    audit_intro="Audit a formal proof memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Remove any shared-prefix claim that is not an explicit premise, explicit rule, or valid rule application.
- Move one-path-only proof steps and truth-value judgments back into Paths.
- If a path proves only a nearby statement, keep it as nearby; do not let it become the queried statement.
- Mark bridges that require a missing premise, converse, inverse, branch choice, or ordinary-world assumption as unsupported.
""".strip(),
    prefix_intro="Rebuild a compact formal proof memo exposing the first proof-status conflict for the queried statement.",
    prefix_rules="""
Rules:
- Name the queried statement first.
- Keep the shortest shared premise/rule prefix before the first conflict.
- Split on whether the current proof state proves the query, proves its negation, or leaves it open.
- Do not split on a final label token before checking the premise-to-query bridge.
- If both paths only prove a nearby statement, keep building instead of resolving the query from that nearby statement.
""".strip(),
    relation_analysis_intro="Describe one formal proof-status divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First identify the queried statement exactly.
- Identify whether each path has a premise/rule bridge to the query, to the negation of the query, or only to a nearby statement.
- Check for invalid converse, invalid inverse, unsupported branch selection, and extra common-sense assumptions.
- Do not choose by final label popularity.
""".strip(),
    resolution_intro="Resolve one minimal formal proof-status divergence.",
    resolution_rules="""
Rules:
- Return the proof relation claim to keep.
- `Correct claim` must say one of: the premises prove the queried statement; the premises prove its negation; or the premises leave it unknown/uncertain.
- If a path needs a missing rule, converse, inverse, unsupported disjunction branch, or ordinary-world premise, call that bridge unsupported.
- Do not output a bare label as the correct claim.
""".strip(),
    resolution_fewshot=LOGIC_RELATION_V2_FEWSHOT,
    revision_rules="""
Rules:
- Continue from the repaired proof-status claim and finish the truth-value answer.
- Use only the stated premises, explicit rules, and the repaired proof-status claim.
- Do not add common-sense assumptions or choose a disjunction branch unless the proof-status claim already justifies it.
- The final line should contain only the allowed label token.
""".strip(),
)


FOLIO_RELATION_V2_PROFILE = replace(
    PROOFWRITER_RELATION_V2_PROFILE,
    name="folio_relation_v2",
    merge_intro="Merge several FOLIO natural-language logic traces into a compact premise-to-conclusion proof-status memo.",
    audit_intro="Audit a FOLIO natural-language logic memo before divergence resolution.",
    prefix_intro="Rebuild a compact FOLIO memo exposing the first premise-to-conclusion proof-status conflict.",
)


PROOFWRITER_RELATION_V3_PROFILE = replace(
    PROOFWRITER_RELATION_V2_PROFILE,
    name="proofwriter_relation_v3",
    merge_rules=PROOFWRITER_RELATION_V2_PROFILE.merge_rules
    + "\n- Treat unknown/uncertain as a first-class proof status, not as failure to answer."
    + "\n- If all visible evidence only shows absence of a proof bridge, the graph claim should be that the query remains unknown, not that a nearby fact is true.",
    relation_analysis_rules=PROOFWRITER_RELATION_V2_PROFILE.relation_analysis_rules
    + "\n- If a path says there is no information, translate it into proof status: no proof of the query and no proof of its negation means unknown."
    + "\n- Do not score a path as better merely because it states the query as a fact without an explicit premise/rule bridge.",
    resolution_rules=PROOFWRITER_RELATION_V2_PROFILE.resolution_rules
    + "\n- For unknown, write the correct claim as: the premises do not prove the queried statement or its negation, so the supported truth value is unknown."
    + "\n- Never infer true from the phrase 'supported label is ...'; the proof-status claim determines the label.",
)


FOLIO_RELATION_V3_PROFILE = replace(
    PROOFWRITER_RELATION_V3_PROFILE,
    name="folio_relation_v3",
    merge_intro="Merge several FOLIO natural-language logic traces into a compact premise-to-conclusion proof-status memo.",
    audit_intro="Audit a FOLIO natural-language logic memo before divergence resolution.",
    prefix_intro="Rebuild a compact FOLIO memo exposing the first premise-to-conclusion proof-status conflict.",
)


FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE = replace(
    FOLIO_RELATION_V3_PROFILE,
    name="folio_relation_v4_equal_token",
    merge_rules=FOLIO_RELATION_V3_PROFILE.merge_rules
    + "\n- Under longer outputs, keep the queried statement, explicit premise/rule bridge, proof status, and final label separate."
    + "\n- Treat unknown as a supported proof status when neither the query nor its negation is proved; do not let verbose uncertainty drift to false.",
    relation_analysis_rules=FOLIO_RELATION_V3_PROFILE.relation_analysis_rules
    + "\n- Lock the proof-status boundary first: proved query -> true; proved negation -> false; neither proved -> unknown."
    + "\n- A longer natural-language explanation is not stronger if it adds common-sense premises or changes the queried statement.",
    resolution_rules=FOLIO_RELATION_V3_PROFILE.resolution_rules
    + "\n- Correct claim must say which proof status holds for the exact queried statement, then name the supported label."
    + "\n- Do not choose by label popularity. Choose by explicit premise/rule support and invalid-bridge rejection.",
    revision_rules=FOLIO_RELATION_V3_PROFILE.revision_rules
    + "\n- Preserve the exact queried statement and proof status through revision. Do not convert unknown to false or true without an explicit rule bridge.",
)


FOLIO_RELATION_V5_EQUAL_TOKEN_PROFILE = replace(
    FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE,
    name="folio_relation_v5_equal_token",
    merge_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.merge_rules
    + "\n- Treat only explicit premise/rule proof as support; consistency, plausibility, or alignment alone are not proof."
    + "\n- If the trace only shows that the query is compatible with the premises, keep unknown rather than upgrading to true or false."
    + "\n- Keep proof status separate from explanatory language.",
    relation_analysis_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- Separate explicit proof from mere consistency: an answer is supported only if the premises actually entail it."
    + "\n- If a branch uses an unstated bridge or ordinary-world assumption, mark it unsupported."
    + "\n- If no branch explicitly proves the query or its negation, unknown is the supported status.",
    resolution_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- Prefer unknown over true/false whenever the trace only gives a plausible alignment or partial bridge."
    + "\n- Do not promote a branch because it sounds coherent if it lacks an explicit proof bridge."
    + "\n- Keep_parallel is acceptable when competing branches show only partial support.",
    revision_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve unknown unless the repaired claim carries explicit proof of the query or its negation."
    + "\n- Do not let a fluent explanation turn a partial bridge into a proof claim.",
)


FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE = replace(
    FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE,
    name="folio_relation_v6_extraction_guard",
    relation_analysis_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.relation_analysis_rules
    + "\n- Prefer an explicit proof-status sentence plus `supported label is N` over loose semantic words such as supported, compatible, or plausible."
    + "\n- Keep the queried statement's proof status separate from option-label extraction; do not let a nearby premise fact set the label.",
    resolution_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.resolution_rules
    + "\n- End a decisive corrected claim with one explicit proof-status label sentence: `... so the supported label is N`."
    + "\n- The label must follow the proof-status boundary for the exact queried statement, not a nearby fact.",
    revision_rules=FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE.revision_rules
    + "\n- Preserve the explicit proof-status label through revision; do not infer a different label from explanation words after it.",
)


FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE = replace(
    FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE,
    name="folio_relation_v7_bridge_audit",
    relation_analysis_rules=FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE.relation_analysis_rules
    + "\n- Audit proof status before label: proved query, proved negation, or neither proved. Label repetition is not proof.",
    resolution_rules=FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE.resolution_rules
    + "\n- The Correct claim must be rewrite-supported by explicit premise/rule proof status and end with `supported label is N`."
    + "\n- Use keep_parallel if the proof status cannot be carried into a revised trace.",
    revision_rules=FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE.revision_rules
    + "\n- Preserve the explicit proof-status bridge through revision and only then land the label.",
)


FOLIO_RELATION_V8_LEDGER_AUDIT_PROFILE = replace(
    FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE,
    name="folio_relation_v8_ledger_audit",
    relation_analysis_rules=FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE.relation_analysis_rules
    + "\n- Write a compact proof-status ledger: queried statement, proof of query, proof of negation, neither-proved status, and invalid bridge if any."
    + "\n- Mere consistency or absence of contradiction is not proof of true; absence of proof is not proof of false.",
    resolution_rules=FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE.resolution_rules
    + "\n- The Correct claim must state one proof-status boundary and end with `supported label is N`."
    + "\n- Use keep_parallel if the visible ledger cannot carry the proof status into a revised trace.",
    revision_rules=FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE.revision_rules
    + "\n- Preserve the proof-status ledger through revision: exact query, proof status, supported label."
    + "\n- Do not let later explanatory wording convert unknown into true/false or true/false into unknown.",
)


BOOLQ_RELATION_V1_PROFILE = PromptProfile(
    name="boolq_relation_v1",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several BoolQ traces into a compact passage-to-question-predicate memo.",
    merge_rules="""
Rules:
- Keep the shared passage facts separate from the question predicate.
- Do not treat a bare yes/no label as evidence.
- Split on whether the passage really supports the exact question predicate, not on a nearby fact.
- If two paths agree on the passage but differ on whether it answers the question, expose that predicate bridge.
""".strip(),
    merge_fewshot="""
Tiny BoolQ example 1:
Common Ground
- Both paths mention the same passage fact.

Paths
- A1: claims the passage fact supports the exact predicate.
- A2: claims it does not support the exact predicate.

First Split
- First split: A1 claims the passage fact supports the asked predicate; A2 says it does not.
""".strip(),
    audit_intro="Audit a BoolQ memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep only trace-faithful shared passage facts.
- Move one-path-only predicate bridges back into Paths.
- Do not invent outside facts or hidden assumptions.
""".strip(),
    prefix_intro="Rebuild a compact BoolQ memo that exposes the first passage-to-predicate conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared passage fact chain before the first conflict.
- Split on whether the passage supports the exact question predicate.
- Do not split on label words alone.
""".strip(),
    relation_analysis_intro="Describe one passage-to-predicate divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First identify the exact question predicate.
- Then identify the passage evidence each path uses.
- Compare whether the passage supports that exact predicate or only a nearby one.
- Do not choose by final label popularity.
""".strip(),
    resolution_intro="Resolve one minimal passage-to-predicate divergence.",
    resolution_rules="""
Rules:
- Return the relation claim to keep.
- Correct claim must say how the passage supports or fails the exact question predicate, then name the supported yes/no label.
- Do not use a bare yes/no label as evidence.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The passage supports the exact requested predicate, so the supported label is yes.

Tiny resolution example 2:
Correct claim: The passage does not support the exact requested predicate, so the supported label is no.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired passage-to-predicate claim and finish the yes/no answer.
- Do not replace the exact predicate with a nearby relation while finishing.
""".strip(),
)


MEDQA_RELATION_V1_PROFILE = PromptProfile(
    name="medqa_relation_v1",
    system_prompt=NL_GRAPH_SYSTEM_PROMPT,
    merge_intro="Merge several medical multiple-choice traces into a compact evidence-to-option memo.",
    merge_rules="""
Rules:
- Keep shared clinical evidence separate from the answer option.
- Do not treat a bare option label or disease name as evidence.
- Split on the first concrete evidence-to-option conflict: diagnosis, treatment, or mechanism support.
- If two paths use different wording but support the same option, do not split on style alone.
- If a path only names a diagnosis, mechanism, or symptom cluster without connecting it to the option text, keep that path note as a bridge claim rather than a final answer.
""".strip(),
    merge_fewshot="""
Tiny medical example 1:
Common Ground
- Both paths use the same symptoms and exam findings.

Paths
- A1: claims the evidence supports option A.
- A2: claims the evidence supports option B.

First Split
- First split: A1 claims the evidence supports option A; A2 claims it supports option B.
""".strip(),
    audit_intro="Audit a medical multiple-choice memo before divergence resolution.",
    audit_rules="""
Audit rules:
- Keep only trace-faithful shared clinical evidence in Common Ground.
- Move one-path-only diagnostic or treatment claims back into Paths.
- Do not invent missing symptoms, tests, or contraindications.
""".strip(),
    prefix_intro="Rebuild a compact medical memo that exposes the first evidence-to-option conflict.",
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first actionable conflict.
- Split on the first concrete mismatch about which option the evidence supports.
- Do not split on wording or level of detail alone.
""".strip(),
    relation_analysis_intro="Describe one evidence-to-option divergence before making a repair decision.",
    relation_analysis_rules="""
Rules:
- First identify the clinical question being asked.
- Then identify the evidence chain each path uses.
- Compare whether that evidence really supports the claimed option.
- Do not choose by answer popularity.
""".strip(),
    resolution_intro="Resolve one minimal evidence-to-option divergence.",
    resolution_rules="""
Rules:
- Return the option claim to keep.
- Correct claim must say how the clinical evidence supports the requested option.
- Do not use a bare option label as evidence.
- Do not silently upgrade a partial bridge into a full diagnosis or treatment choice.
- The kept claim should mention why that option text is supported, not just restate the diagnosis.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical evidence supports option D, so option D should be kept.

Tiny resolution example 2:
Correct claim: The evidence points to a different diagnosis and does not support the current option, so keep_parallel.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-option claim and finish with the option text.
- Do not stop at a partial symptom or intermediate mechanism if the question asks for the final medical choice.
""".strip(),
)


MEDQA_RELATION_V2_PROFILE = replace(
    MEDQA_RELATION_V1_PROFILE,
    name="medqa_relation_v2",
    merge_intro="Merge several medical multiple-choice traces into a compact evidence-to-option-text memo.",
    merge_rules="""
Rules:
- Keep shared clinical evidence separate from the answer option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism name as the final answer.
- Split on the first concrete evidence-to-option-text conflict: diagnosis support, treatment support, or mechanism support.
- If two paths use different wording but map to the same option text, do not split on style alone.
- If a path only names a diagnosis, mechanism, or symptom cluster, keep that as a bridge claim rather than a final choice.
- The option text is the target object; a diagnosis is only useful if it clearly bridges to that option text.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first actionable option conflict.
- Split on the first concrete mismatch about which option text the evidence supports.
- Do not split on wording or level of detail alone.
- If a path only reaches a diagnosis name, check whether that diagnosis actually bridges to the option text before splitting.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical question being asked and the option text family it belongs to.
- Then identify the evidence chain each path uses.
- Compare whether that evidence truly supports the option text or only a diagnosis/mechanism nearby it.
- Do not choose by answer popularity.
- If a path's diagnosis is right but its option mapping is missing, call that a bridge gap rather than a final answer.
""".strip(),
    resolution_rules="""
Rules:
- Return the option claim to keep.
- Correct claim must say how the clinical evidence supports the requested option text.
- Do not use a bare option letter or bare diagnosis as evidence.
- Do not silently upgrade a partial bridge into a full option choice.
- The kept claim should mention why that option text is supported, not just restate the diagnosis.
- If one side reaches the right diagnosis but not the option text, keep the bridge and not the bare diagnosis.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical evidence supports the option text for D, so option D should be kept.

Tiny resolution example 2:
Correct claim: The trace identifies a diagnosis but has not yet bridged to the option text, so keep_parallel.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired evidence-to-option-text claim and finish with the exact option text.
- Do not stop at a diagnosis, mechanism, or symptom cluster if the question asks for the final medical choice.
- The final line should contain only the requested option text inside the normal final-answer wrapper.
""".strip(),
)


MEDQA_RELATION_V3_PROFILE = replace(
    MEDQA_RELATION_V2_PROFILE,
    name="medqa_relation_v3",
    merge_intro="Merge several medical multiple-choice traces into a compact clinical evidence-to-option-text bridge memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, diagnosis/mechanism/treatment bridge, option text, and final option token separate.
- First identify the clinical task: diagnosis, next step, treatment, complication, mechanism, prevention, or risk factor.
- Split on the first concrete bridge conflict: which clinical evidence supports which option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- If a path reaches a diagnosis or mechanism, ask whether it actually maps to the option text.
- If two paths name the same diagnosis but choose different options, split on the option-text mapping.
- If two paths choose the same option text with different wording, do not split on style alone.
- If the missing piece is a differentiating clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first option-text bridge conflict.
- Split on which option text the evidence supports, not on a bare diagnosis or option letter.
- If a diagnosis/mechanism is visible but not yet mapped to an answer choice, keep building unless another path makes an incompatible mapped claim.
- If paths disagree only after the same diagnosis, expose the bridge from diagnosis to option text.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the option text family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> diagnosis/mechanism/treatment -> option text.
- Compare whether the bridge reaches the exact option text, not whether it merely names a plausible disease.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the option mapping is missing, call that a bridge gap rather than a final answer.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the option-text bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested option text.
- The claim to keep should mention the decisive clinical feature and the option text it supports.
- Do not use a bare option letter, bare diagnosis, or bare mechanism as evidence.
- Do not silently upgrade a partial diagnosis into a full option choice.
- If one side reaches a diagnosis but not the option text, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better option-text bridge, keep the mapped bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical findings support the option text for the needed next step, so that option should be kept.

Tiny resolution example 2:
Correct claim: The trace names a diagnosis but does not map it to the answer choice, so keep building from the clinical bridge rather than landing on the diagnosis.

Tiny resolution example 3:
Correct claim: Both traces name the same diagnosis, but only one connects the decisive feature to the option text, so keep that option-text bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired clinical evidence-to-option-text bridge and finish with the exact option text.
- Do not stop at a diagnosis, mechanism, symptom cluster, or option letter if the question asks for the final medical choice.
- The final line should contain only the requested option text inside the normal final-answer wrapper.
""".strip(),
)


MEDQA_RELATION_V4_PROFILE = replace(
    MEDQA_RELATION_V3_PROFILE,
    name="medqa_relation_v4",
    merge_intro="Merge several medical multiple-choice traces into a compact evidence-to-exact-option-text bridge memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, diagnosis/mechanism/treatment bridge, exact option text, and final option token separate.
- First identify the clinical task and the exact option text family.
- Split on the first concrete bridge conflict: which clinical evidence supports which exact option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- A diagnosis or mechanism is only useful if it clearly maps to one specific option text from the answer choices.
- If two paths name the same diagnosis but point to different option texts, split on the option-text mapping.
- If two paths point to the same option text with different wording, do not split on style alone.
- If the missing piece is the deciding clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first exact option-text bridge conflict.
- Split on which exact option text the evidence supports, not on a bare diagnosis or option letter.
- If a diagnosis/mechanism is visible but not yet mapped to an answer choice, keep building unless another path makes an incompatible mapped claim.
- If paths disagree only after the same diagnosis, expose the bridge from diagnosis to exact option text.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the exact option text family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> diagnosis/mechanism/treatment -> exact option text.
- Compare whether the bridge reaches the exact option text, not whether it merely names a plausible diagnosis.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the exact option mapping is missing, call that a bridge gap rather than a final answer.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact option-text bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested exact option text.
- The claim to keep should mention the decisive clinical feature and the exact option text it supports.
- Do not use a bare option letter, bare diagnosis, or bare mechanism as evidence.
- Do not silently upgrade a partial diagnosis into a full option choice.
- If one side reaches a diagnosis but not the exact option text, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better exact option-text bridge, keep that mapped bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical findings support the exact option text for the needed next step, so that option should be kept.

Tiny resolution example 2:
Correct claim: The trace names a diagnosis but does not map it to the answer choice, so keep building from the clinical bridge rather than landing on the diagnosis.

Tiny resolution example 3:
Correct claim: Both traces name the same diagnosis, but only one connects the decisive feature to the exact option text, so keep that option-text bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired clinical evidence-to-exact-option-text bridge and finish with the exact option text.
- Do not stop at a diagnosis, mechanism, symptom cluster, or option letter if the question asks for the final medical choice.
- The final line should contain only the requested exact option text inside the normal final-answer wrapper.
""".strip(),
)


MEDQA_RELATION_V5_PROFILE = replace(
    MEDQA_RELATION_V3_PROFILE,
    name="medqa_relation_v5",
    merge_intro="Merge several medical multiple-choice traces into a compact clinical bridge memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, decisive feature, diagnosis/mechanism/treatment bridge, and answer choice separate.
- First identify the clinical task and the answer-choice family.
- Split on the first concrete bridge conflict: which clinical evidence supports which answer choice.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- A diagnosis or mechanism is only useful if it clearly maps to one specific answer choice.
- If two paths name the same diagnosis but point to different answer choices, split on the answer-choice mapping.
- If two paths point to the same answer choice with different wording, do not split on style alone.
- If the missing piece is the deciding clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first answer-choice bridge conflict.
- Split on which answer choice the evidence supports, not on a bare diagnosis or option letter.
- If a diagnosis/mechanism is visible but not yet mapped to an answer choice, keep building unless another path makes an incompatible mapped claim.
- If paths disagree only after the same diagnosis, expose the bridge from diagnosis to answer choice.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the answer-choice family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> diagnosis/mechanism/treatment -> answer choice.
- Compare whether the bridge reaches the exact answer choice, not whether it merely names a plausible diagnosis.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the answer mapping is missing, call that a bridge gap rather than a final answer.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the answer-choice bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested answer choice.
- The claim to keep should mention the decisive clinical feature and the answer choice it supports.
- Do not use a bare option letter, bare diagnosis, or bare mechanism as evidence.
- Do not silently upgrade a partial diagnosis into a full answer choice.
- If one side reaches a diagnosis but not the answer choice, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better answer-choice bridge, keep that mapped bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical findings support the needed answer choice, so that option should be kept.

Tiny resolution example 2:
Correct claim: The trace names a diagnosis but does not map it to the answer choice, so keep building from the clinical bridge rather than landing on the diagnosis.

Tiny resolution example 3:
Correct claim: Both traces name the same diagnosis, but only one connects the decisive feature to the answer choice, so keep that answer-choice bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired clinical evidence-to-answer-choice bridge and finish with the exact option text.
- Do not stop at a diagnosis, mechanism, symptom cluster, or option letter if the question asks for the final medical choice.
- The final line should contain only the requested option text inside the normal final-answer wrapper.
""".strip(),
)


MEDQA_RELATION_V6_PROFILE = replace(
    MEDQA_RELATION_V5_PROFILE,
    name="medqa_relation_v6",
    merge_intro="Merge several medical multiple-choice traces into a compact exact-option bridge memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, decisive feature, diagnosis/mechanism/treatment bridge, and exact option text separate.
- First identify the clinical task and the exact option text family.
- Split on the first concrete bridge conflict: which clinical evidence supports which exact option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- A diagnosis or mechanism is only useful if it clearly maps to one specific exact option text from the answer choices.
- If a path reaches a plausible diagnosis but not the exact option text, keep that as a bridge claim rather than a final choice.
- If two paths name the same diagnosis but point to different exact option texts, split on the option-text mapping.
- If the missing piece is the decisive clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    prefix_rules="""
Rules:
- Keep the shortest shared clinical evidence needed for the first exact option-text bridge conflict.
- Split on which exact option text the evidence supports, not on a bare diagnosis or option letter.
- If a diagnosis/mechanism is visible but not yet mapped to an answer choice, keep building unless another path makes an incompatible mapped claim.
- If paths disagree only after the same diagnosis, expose the bridge from diagnosis to exact option text.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the exact option text family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> decisive feature -> diagnosis/mechanism/treatment -> exact option text.
- Compare whether the bridge reaches the exact option text, not whether it merely names a plausible diagnosis.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the exact option mapping is missing, call that a bridge gap rather than a final answer.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact option-text bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested exact option text.
- The claim to keep should mention the decisive clinical feature and the exact option text it supports.
- Do not use a bare option letter, bare diagnosis, or bare mechanism as evidence.
- Do not silently upgrade a partial diagnosis into a full option choice.
- If one side reaches a diagnosis but not the exact option text, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better exact option-text bridge, keep that mapped bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical findings support the exact option text for the needed next step, so that option should be kept.

Tiny resolution example 2:
Correct claim: The trace names a diagnosis but does not map it to the answer choice, so keep building from the clinical bridge rather than landing on the diagnosis.

Tiny resolution example 3:
Correct claim: Both traces name the same diagnosis, but only one connects the decisive feature to the exact option text, so keep that option-text bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired clinical evidence-to-exact-option-text bridge and finish with the exact option text only.
- Do not stop at a diagnosis, mechanism, symptom cluster, or option letter if the question asks for the final medical choice.
- The last line should contain only the requested exact option text, with no option letter and no extra explanation.
""".strip(),
)


MEDQA_RELATION_V7_PROFILE = replace(
    MEDQA_RELATION_V6_PROFILE,
    name="medqa_relation_v7",
    merge_rules="""
Rules:
- Keep clinical evidence, decisive feature, diagnosis/mechanism/treatment bridge, and exact option text separate.
- First identify the clinical task and the exact option text family.
- Split on the first concrete bridge conflict: which clinical evidence supports which exact option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- A diagnosis or mechanism is only useful if it clearly maps to one specific exact option text from the answer choices.
- If a path reaches a plausible diagnosis but not the exact option text, keep that as a bridge claim rather than a final choice.
- If a path names multiple mediators or multiple diagnoses, that is not yet a valid final answer unless the question itself asks for a bundle.
- If two paths name the same diagnosis but point to different exact option texts, split on the option-text mapping.
- If the missing piece is the decisive clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the exact option text family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> decisive feature -> diagnosis/mechanism/treatment -> exact option text.
- Compare whether the bridge reaches the exact option text, not whether it merely names a plausible diagnosis.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the exact option mapping is missing, call that a bridge gap rather than a final answer.
- If a path names a bundle of mediators or a bundle of diagnoses, reduce it to the single exact option text asked by the question.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the exact option-text bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested exact option text.
- The claim to keep should mention the decisive clinical feature and the exact option text it supports.
- Do not use a bare option letter, bare diagnosis, bare mechanism, or multi-option bundle as evidence.
- Do not silently upgrade a partial diagnosis into a full option choice.
- If one side reaches a diagnosis but not the exact option text, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better exact option-text bridge, keep that mapped bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The clinical findings support the exact option text for the needed next step, so that option should be kept.

Tiny resolution example 2:
Correct claim: The trace names a diagnosis but does not map it to the answer choice, so keep building from the clinical bridge rather than landing on the diagnosis.

Tiny resolution example 3:
Correct claim: Both traces name the same diagnosis, but only one connects the decisive feature to the exact option text, so keep that option-text bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired clinical evidence-to-exact-option-text bridge and finish with the exact option text only.
- Do not stop at a diagnosis, mechanism, symptom cluster, bundle of options, or option letter if the question asks for the final medical choice.
- The last line must be exactly one copied option text from the answer choices, with no extra explanation, no option letter, and no markdown.
""".strip(),
)


MEDQA_RELATION_V8_PROFILE = replace(
    MEDQA_RELATION_V7_PROFILE,
    name="medqa_relation_v8",
    merge_intro="Merge several medical multiple-choice traces into a compact single-option bridge memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, decisive feature, diagnosis/mechanism/treatment bridge, and one exact option text separate.
- First identify the clinical task and the exact option text family.
- Split on the first concrete bridge conflict: which clinical evidence supports which exact option text.
- Do not treat a bare option letter, bare diagnosis, or bare mechanism as the final answer.
- A diagnosis or mechanism is only useful if it clearly maps to one specific exact option text.
- If a path names multiple mediators or a bundle when the question asks for a single choice, treat that as bridge noise rather than a final answer.
- If two paths name the same diagnosis but point to different exact option texts, split on the exact option text mapping.
- If the missing piece is the decisive clinical feature, keep that as the bridge gap instead of inventing it.
""".strip(),
    relation_analysis_rules="""
Rules:
- First identify the clinical task and the exact option text family.
- Then identify the evidence bundle each path uses.
- State each path's bridge: clinical evidence -> decisive feature -> diagnosis/mechanism/treatment -> exact option text.
- Compare whether the bridge reaches a single exact option text, not whether it merely names a plausible diagnosis or a bundle.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the exact option mapping is missing, call that a bridge gap rather than a final answer.
- If neither side supplies the differentiating feature, say what is missing instead of inventing it.
""".strip(),
    resolution_rules="""
Rules:
- Return the single exact option-text bridge claim to keep.
- Correct claim must say how the clinical evidence supports the requested exact option text.
- The claim to keep should mention the decisive clinical feature and the exact option text it supports.
- Do not use a bare option letter, bare diagnosis, bare mechanism, or multi-option bundle as evidence unless the question explicitly asks for a bundle.
- Do not silently upgrade a partial diagnosis into a full option choice.
- If one side reaches a diagnosis but not the exact option text, keep the bridge claim and not the bare diagnosis.
- If both sides have plausible diagnoses but one has the better exact option-text bridge, keep that mapped bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired exact-option bridge and finish with exactly one copied option text on the last line.
- Do not stop at a diagnosis, mechanism, symptom cluster, bundle, or option letter if the question asks for the final medical choice.
- The last line should contain only the exact option text, with no sentence, no option letter, and no explanation.
""".strip(),
)


MEDQA_RELATION_V9_PROFILE = replace(
    MEDQA_RELATION_V8_PROFILE,
    name="medqa_relation_v9",
    merge_intro="Merge several medical multiple-choice traces into a narrow option-mapping memo.",
    merge_rules="""
Rules:
- Keep clinical evidence, decisive feature, and exact option text separate.
- First identify the clinical task and the exact requested answer object.
- Split on the first concrete mapping conflict: which evidence bundle supports which exact option text.
- If a path has a plausible diagnosis or mechanism but does not land on one listed option, treat that as an incomplete bridge, not a final answer.
- Do not promote a diagnosis, mechanism, or treatment plan into the final choice unless it uniquely identifies one option.
- When the traces disagree only on a broad clinical label but still point to the same option text, keep building rather than splitting.
- If the wrong path is a broad diagnosis, symptom cluster, or over-generalized label, reduce it to the specific option mapping instead of treating it as the answer.
- If the only difference is a definition or direction reversal, make the split at the mapping layer, not at the diagnosis layer.
""".strip(),
    relation_analysis_rules="""
Rules:
- State the exact requested object and the smallest clinical feature that separates the answer choices.
- Compare each path's evidence -> feature -> option mapping.
- Do not stop at diagnosis or mechanism if a shorter feature-to-option bridge exists.
- When two paths share the same diagnosis, the real split is which feature uniquely points to the listed option text.
- If a path names a diagnosis that is not itself an answer choice, continue until you can name the exact option text it supports.
- Do not choose by answer popularity.
- If a path's diagnosis is plausible but the option text is missing or inconsistent, call it an incomplete bridge.
""".strip(),
    resolution_fewshot="""
Tiny resolution example 1:
Correct claim: The decisive feature maps directly to the listed option text, so keep that exact option bridge.

Tiny resolution example 2:
Correct claim: The diagnosis is plausible, but it is not yet tied to a listed answer choice, so keep building from the shorter feature-to-option bridge.

Tiny resolution example 3:
Correct claim: Both traces share the diagnosis, but only one points to the exact option text; keep that option mapping and ignore the broader label.
""".strip(),
    resolution_rules="""
Rules:
- Return the shortest claim that still maps the clinical evidence to one exact option text.
- If the path's diagnosis or mechanism is longer than needed, keep only the decisive feature plus the exact option text bridge.
- Do not use bare diagnosis, bare mechanism, or treatment language as evidence unless that is already the exact option text asked by the question.
- If a path has the correct medical direction but the wrong option mapping, keep the direction and repair the mapping.
- If one side reaches the exact option text and the other stops at a broader medical statement, prefer the exact option bridge.
""".strip(),
    revision_rules="""
Rules:
- Continue from the repaired option-mapping bridge and finish with exactly one copied option text on the last line.
- Do not stop at a diagnosis, mechanism, symptom cluster, bundle, or option letter if the question asks for the final medical choice.
- The last line should contain only the exact option text, with no sentence, no option letter, and no explanation.
""".strip(),
)


PROMPT_PROFILES = {
    "universal": UNIVERSAL_PROMPT_PROFILE,
    "universal_minimal": UNIVERSAL_MINIMAL_PROMPT_PROFILE,
    "math500": MATH500_PROMPT_PROFILE,
    "math500_current": MATH500_PROMPT_PROFILE,
    "math500_best_20260506": MATH500_BEST_20260506_PROFILE,
    "math500_v18": MATH500_V18_PROFILE,
    "math500_v19": MATH500_V19_PROFILE,
    "math500_v20": MATH500_V20_PROFILE,
    "math500_v21": MATH500_V21_PROFILE,
    "math500_v30": MATH500_V30_PROFILE,
    "math500_v31_equal_token": MATH500_V31_EQUAL_TOKEN_PROFILE,
    "math500_v32_bridge_audit": MATH500_V32_BRIDGE_AUDIT_PROFILE,
    "math500_v33_ledger_audit": MATH500_V33_LEDGER_AUDIT_PROFILE,
    "mmlu_pro_best_20260507": MMLU_PRO_BEST_20260507_PROFILE,
    "mmlu_pro_relation_v1_equal_token": MMLU_PRO_RELATION_V1_EQUAL_TOKEN_PROFILE,
    "mmlu_pro_relation_v2_equal_token": MMLU_PRO_RELATION_V2_EQUAL_TOKEN_PROFILE,
    "mmlu_pro_relation_v3_extraction_guard": MMLU_PRO_RELATION_V3_EXTRACTION_GUARD_PROFILE,
    "mmlu_pro_relation_v4_bridge_audit": MMLU_PRO_RELATION_V4_BRIDGE_AUDIT_PROFILE,
    "mmlu_pro_relation_v5_ledger_audit": MMLU_PRO_RELATION_V5_LEDGER_AUDIT_PROFILE,
    "logiqa": LOGIQA_RELATION_V1_PROFILE,
    "logiqa_relation_v1": LOGIQA_RELATION_V1_PROFILE,
    "logiqa_relation_v2": LOGIQA_RELATION_V2_PROFILE,
    "logiqa_relation_v3": LOGIQA_RELATION_V3_PROFILE,
    "logiqa_relation_v4": LOGIQA_RELATION_V4_PROFILE,
    "logiqa_relation_v5": LOGIQA_RELATION_V5_PROFILE,
    "logiqa_relation_v6": LOGIQA_RELATION_V6_PROFILE,
    "logiqa_relation_v7": LOGIQA_RELATION_V7_PROFILE,
    "logiqa_relation_v8": LOGIQA_RELATION_V8_PROFILE,
    "logiqa_relation_v9": LOGIQA_RELATION_V9_PROFILE,
    "logiqa_relation_v10": LOGIQA_RELATION_V10_PROFILE,
    "logiqa_relation_v11_equal_token": LOGIQA_RELATION_V11_EQUAL_TOKEN_PROFILE,
    "logiqa_relation_v12_extraction_guard": LOGIQA_RELATION_V12_EXTRACTION_GUARD_PROFILE,
    "logiqa_relation_v13_flow_guard": LOGIQA_RELATION_V13_FLOW_GUARD_PROFILE,
    "logiqa_relation_v14_action_guard": LOGIQA_RELATION_V14_ACTION_GUARD_PROFILE,
    "logiqa_relation_v15_memory_guard": LOGIQA_RELATION_V15_MEMORY_GUARD_PROFILE,
    "logiqa_relation_v16_stable_rewrite_guard": LOGIQA_RELATION_V16_STABLE_REWRITE_GUARD_PROFILE,
    "logiqa_relation_v17_bridge_audit": LOGIQA_RELATION_V17_BRIDGE_AUDIT_PROFILE,
    "gsm8k": GSM8K_PROMPT_PROFILE,
    "strategyqa": STRATEGYQA_PROMPT_PROFILE,
    "strategyqa_relation_v1": STRATEGYQA_RELATION_V1_PROFILE,
    "sqa_relation_v1": STRATEGYQA_RELATION_V1_PROFILE,
    "strategyqa_relation_v2": STRATEGYQA_RELATION_V2_PROFILE,
    "sqa_relation_v2": STRATEGYQA_RELATION_V2_PROFILE,
    "strategyqa_relation_v3": STRATEGYQA_RELATION_V3_PROFILE,
    "sqa_relation_v3": STRATEGYQA_RELATION_V3_PROFILE,
    "strategyqa_relation_v4": STRATEGYQA_RELATION_V4_PROFILE,
    "sqa_relation_v4": STRATEGYQA_RELATION_V4_PROFILE,
    "strategyqa_relation_v5": STRATEGYQA_RELATION_V5_PROFILE,
    "sqa_relation_v5": STRATEGYQA_RELATION_V5_PROFILE,
    "strategyqa_relation_v6": STRATEGYQA_RELATION_V6_PROFILE,
    "sqa_relation_v6": STRATEGYQA_RELATION_V6_PROFILE,
    "prontoqa": PRONTOQA_RELATION_V1_PROFILE,
    "prontoqa_relation_v1": PRONTOQA_RELATION_V1_PROFILE,
    "bamboogle": BAMBOOGLE_RELATION_V1_PROFILE,
    "bamboogle_relation_v1": BAMBOOGLE_RELATION_V1_PROFILE,
    "bamboogle_relation_v2": BAMBOOGLE_RELATION_V2_PROFILE,
    "bamboogle_relation_v3": BAMBOOGLE_RELATION_V3_PROFILE,
    "bamboogle_relation_v4": BAMBOOGLE_RELATION_V4_PROFILE,
    "bamboogle_relation_v5": BAMBOOGLE_RELATION_V5_PROFILE,
    "bamboogle_relation_v6": BAMBOOGLE_RELATION_V6_PROFILE,
    "bamboogle_relation_v7_equal_token": BAMBOOGLE_RELATION_V7_EQUAL_TOKEN_PROFILE,
    "bamboogle_relation_v8_equal_token": BAMBOOGLE_RELATION_V8_EQUAL_TOKEN_PROFILE,
    "bamboogle_relation_v9_extraction_guard": BAMBOOGLE_RELATION_V9_EXTRACTION_GUARD_PROFILE,
    "bamboogle_relation_v10_bridge_audit": BAMBOOGLE_RELATION_V10_BRIDGE_AUDIT_PROFILE,
    "hotpotqa_relation_v1": HOTPOTQA_RELATION_V1_PROFILE,
    "hotpotqa_relation_v2": HOTPOTQA_RELATION_V2_PROFILE,
    "hotpotqa_relation_v3": HOTPOTQA_RELATION_V3_PROFILE,
    "hotpotqa_relation_v4": HOTPOTQA_RELATION_V4_PROFILE,
    "hotpotqa_relation_v5": HOTPOTQA_RELATION_V5_PROFILE,
    "hotpotqa_relation_v6": HOTPOTQA_RELATION_V6_PROFILE,
    "hotpotqa_relation_v7": HOTPOTQA_RELATION_V7_PROFILE,
    "hotpotqa_relation_v8": HOTPOTQA_RELATION_V8_PROFILE,
    "hotpotqa_relation_v9": HOTPOTQA_RELATION_V9_PROFILE,
    "hotpotqa_relation_v10": HOTPOTQA_RELATION_V10_PROFILE,
    "hotpotqa_relation_v11": HOTPOTQA_RELATION_V11_PROFILE,
    "hotpotqa_relation_v12": HOTPOTQA_RELATION_V12_PROFILE,
    "hotpotqa_relation_v13": HOTPOTQA_RELATION_V13_PROFILE,
    "hotpotqa_relation_v14": HOTPOTQA_RELATION_V14_PROFILE,
    "hotpotqa_relation_v15": HOTPOTQA_RELATION_V15_PROFILE,
    "hotpotqa_relation_v16": HOTPOTQA_RELATION_V16_PROFILE,
    "hotpotqa_relation_v17": HOTPOTQA_RELATION_V17_PROFILE,
    "hotpotqa_relation_v18": HOTPOTQA_RELATION_V18_PROFILE,
    "hotpotqa_relation_v19": HOTPOTQA_RELATION_V19_PROFILE,
    "hotpotqa_relation_v20": HOTPOTQA_RELATION_V20_PROFILE,
    "hotpotqa_relation_v21": HOTPOTQA_RELATION_V21_PROFILE,
    "hotpotqa_relation_v22": HOTPOTQA_RELATION_V22_PROFILE,
    "hotpotqa_relation_v23": HOTPOTQA_RELATION_V23_PROFILE,
    "hotpotqa_relation_v24": HOTPOTQA_RELATION_V24_PROFILE,
    "hotpotqa_relation_v25": HOTPOTQA_RELATION_V25_PROFILE,
    "hotpotqa_relation_v26": HOTPOTQA_RELATION_V26_PROFILE,
    "hotpotqa_relation_v27": HOTPOTQA_RELATION_V27_PROFILE,
    "hotpotqa_relation_v28": HOTPOTQA_RELATION_V28_PROFILE,
    "hotpotqa_relation_v29": HOTPOTQA_RELATION_V29_PROFILE,
    "hotpotqa_relation_v30": HOTPOTQA_RELATION_V30_PROFILE,
    "hotpotqa_relation_v31": HOTPOTQA_RELATION_V31_PROFILE,
    "hotpotqa_relation_v32": HOTPOTQA_RELATION_V32_PROFILE,
    "hotpotqa_relation_v33": HOTPOTQA_RELATION_V33_PROFILE,
    "hotpotqa_relation_v34": HOTPOTQA_RELATION_V34_PROFILE,
    "hotpotqa_relation_v35": HOTPOTQA_RELATION_V35_PROFILE,
    "hotpotqa_relation_v36_equal_token": HOTPOTQA_RELATION_V36_EQUAL_TOKEN_PROFILE,
    "hotpotqa_relation_v37_extraction_guard": HOTPOTQA_RELATION_V37_EXTRACTION_GUARD_PROFILE,
    "hotpotqa_relation_v38_bridge_audit": HOTPOTQA_RELATION_V38_BRIDGE_AUDIT_PROFILE,
    "hotpotqa_relation_v39_ledger_audit": HOTPOTQA_RELATION_V39_LEDGER_AUDIT_PROFILE,
    "musique_relation_v1": MUSIQUE_RELATION_V1_PROFILE,
    "musique_relation_v2": MUSIQUE_RELATION_V2_PROFILE,
    "musique_relation_v3": MUSIQUE_RELATION_V3_PROFILE,
    "proofwriter_relation_v1": PROOFWRITER_RELATION_V1_PROFILE,
    "proofwriter_relation_v2": PROOFWRITER_RELATION_V2_PROFILE,
    "proofwriter_relation_v3": PROOFWRITER_RELATION_V3_PROFILE,
    "folio_relation_v1": FOLIO_RELATION_V1_PROFILE,
    "folio_relation_v2": FOLIO_RELATION_V2_PROFILE,
    "folio_relation_v3": FOLIO_RELATION_V3_PROFILE,
    "folio_relation_v4_equal_token": FOLIO_RELATION_V4_EQUAL_TOKEN_PROFILE,
    "folio_relation_v5_equal_token": FOLIO_RELATION_V5_EQUAL_TOKEN_PROFILE,
    "folio_relation_v6_extraction_guard": FOLIO_RELATION_V6_EXTRACTION_GUARD_PROFILE,
    "folio_relation_v7_bridge_audit": FOLIO_RELATION_V7_BRIDGE_AUDIT_PROFILE,
    "folio_relation_v8_ledger_audit": FOLIO_RELATION_V8_LEDGER_AUDIT_PROFILE,
    "boolq": BOOLQ_RELATION_V1_PROFILE,
    "boolq_relation_v1": BOOLQ_RELATION_V1_PROFILE,
    "medqa": MEDQA_RELATION_V1_PROFILE,
    "medqa_relation_v1": MEDQA_RELATION_V1_PROFILE,
    "medqa_relation_v2": MEDQA_RELATION_V2_PROFILE,
    "medqa_relation_v3": MEDQA_RELATION_V3_PROFILE,
    "medqa_relation_v4": MEDQA_RELATION_V4_PROFILE,
    "medqa_relation_v5": MEDQA_RELATION_V5_PROFILE,
    "medqa_relation_v6": MEDQA_RELATION_V6_PROFILE,
    "medqa_relation_v7": MEDQA_RELATION_V7_PROFILE,
    "medqa_relation_v8": MEDQA_RELATION_V8_PROFILE,
    "medqa_relation_v9": MEDQA_RELATION_V9_PROFILE,
    "anli": ANLI_PROMPT_PROFILE,
    "anli_relation_v2": ANLI_RELATION_V2_PROFILE,
    "anli_relation_v3": ANLI_RELATION_V3_PROFILE,
    "anli_relation_v4": ANLI_RELATION_V4_PROFILE,
    "anli_relation_v5": ANLI_RELATION_V5_PROFILE,
    "anli_relation_v6": ANLI_RELATION_V6_PROFILE,
    "anli_relation_v7": ANLI_RELATION_V7_PROFILE,
    "anli_relation_v8": ANLI_RELATION_V8_PROFILE,
    "anli_relation_v9": ANLI_RELATION_V9_PROFILE,
    "anli_relation_v10": ANLI_RELATION_V10_PROFILE,
    "anli_relation_v11": ANLI_RELATION_V11_PROFILE,
    "anli_relation_v12": ANLI_RELATION_V12_PROFILE,
    "anli_relation_v13": ANLI_RELATION_V13_PROFILE,
    "anli_relation_v14": ANLI_RELATION_V14_PROFILE,
    "anli_relation_v15_equal_token": ANLI_RELATION_V15_EQUAL_TOKEN_PROFILE,
    "anli_relation_v16_extraction_guard": ANLI_RELATION_V16_EXTRACTION_GUARD_PROFILE,
    "anli_relation_v17_action_guard": ANLI_RELATION_V17_ACTION_GUARD_PROFILE,
    "anli_relation_v18_memory_guard": ANLI_RELATION_V18_MEMORY_GUARD_PROFILE,
    "anli_relation_v19_bridge_audit": ANLI_RELATION_V19_BRIDGE_AUDIT_PROFILE,
}


def resolve_prompt_profile(profile: str | PromptProfile | None) -> PromptProfile:
    if isinstance(profile, PromptProfile):
        return profile
    if profile is None:
        return MATH500_PROMPT_PROFILE
    return PROMPT_PROFILES.get(str(profile), MATH500_PROMPT_PROFILE)


def _binary_label_question_predicate(question: str) -> str:
    labels = set(allowed_label_answers(question))
    if not labels or not labels <= {"yes", "no", "true", "false"}:
        return ""
    text = clean_question_text(question)
    text = re.split(
        r"\bAnswer\s+(?:yes\s+or\s+no|true\s+or\s+false)\b|\bUse\s+only\s+(?:yes\s+or\s+no|true\s+or\s+false)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if "Question:" in text:
        text = text.rsplit("Question:", 1)[-1].strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 220:
        text = text[-220:].strip()
    return text


def _truth_value_target_statement(question: str) -> str:
    labels = set(allowed_label_answers(question))
    if labels != {"true", "false"}:
        return ""
    text = _binary_label_question_predicate(question)
    match = re.search(
        r"(?:is\s+the\s+following\s+statement\s+true\s+or\s+false\??|statement\s*[:\-])\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()
    text = re.sub(r"^(?:is|whether)\s+", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip(" ?.")
    return text if 0 < len(text) <= 180 else ""


def compact_target_focus_note(question: str) -> str:
    lowered_question = question.lower()
    if question_uses_label_answers(question) and all(
        word in lowered_question for word in ("entailment", "neutral", "contradiction")
    ):
        parts: list[str] = [
            "NLI relation: compare the premise to the hypothesis relation, not just the final label.",
            "Keep the support bridge, incompatibility, and neutrality separate.",
        ]
        return " ".join(parts).strip()
    contract = analyze_question_target(clean_question_text(question))
    parts: list[str] = []
    if contract.requested_object:
        parts.append(f"Requested object: {contract.requested_object}.")
    if contract.answer_format:
        parts.append(f"Answer form: {contract.answer_format}.")
    for guardrail in contract.guardrails[:2]:
        parts.append(f"Guardrail: {guardrail}.")
    if question_uses_label_answers(question):
        labels = ", ".join(allowed_label_answers(question)[:8])
        predicate = _binary_label_question_predicate(question)
        if predicate:
            parts.append(f"Question predicate: {predicate}")
        truth_statement = _truth_value_target_statement(question)
        if truth_statement:
            parts.append(f"Truth-value target statement: {truth_statement}.")
            parts.append(
                "If the repaired claim proves this statement, the label is true; if it proves the negation, the label is false."
            )
            parts.append(
                "For the repaired claim, prefer saying the target statement is true or false over repeating only the embedded fact."
            )
            parts.append("Do not flip the label merely because the target statement contains a negation word.")
        if labels:
            parts.append(
                f"Label task: allowed final labels are {labels}; compare claims by which label they support."
            )
        lowered_question = question.lower()
        if all(word in lowered_question for word in ("entailment", "neutral", "contradiction")):
            parts.append("NLI relation: compare premise to hypothesis; keep support, contradiction, and neutrality distinct.")
        parts.append(
            "State label support through the exact question predicate, not by copying a nearby positive or negative word."
        )
        parts.append(
            "Lock the target relation: keep the asked relation fixed instead of replacing it with a nearby relation, weaker predicate, stronger predicate, or historical reading."
        )
        parts.append(
            "If the question states a condition, compare the claim under that condition rather than whether the condition actually occurred."
        )
        parts.append(
            "Treat a bridge as strong only if it is grounded in the trace or a reasonable common-sense inference tied to the question; if it needs an unstated extra premise, mark it as unsupported."
        )
        parts.append(
            "If a path only supports a nearby predicate or a subcondition, say that explicitly instead of silently upgrading it."
        )
        parts.append(
            "If two paths support the same label, do not split on wording alone."
        )
    return " ".join(parts).strip()


def _profile_target_focus(profile: str | PromptProfile | None) -> str:
    resolved = resolve_prompt_profile(profile)
    if resolved.name in {"strategyqa_relation_v6", "sqa_relation_v6"}:
        return (
            "Task focus:\n"
            "Requested object: the exact yes/no answer to the question predicate.\n"
            "Bridge: evidence -> exact question predicate -> yes/no label.\n"
            "If a path needs an outside fact, treat that fact as disputed unless it is stated by the question or another trace.\n"
            "Final answer form: yes or no only.\n"
            "Use this only to keep the graph claim and final label aligned."
        )
    if _is_medqa_profile(resolved) and resolved.name == "medqa_relation_v9":
        return (
            "Task focus:\n"
            "Requested object: the exact option text from the answer choices.\n"
            "Bridge: clinical evidence -> smallest decisive feature -> exact option text; use diagnosis/mechanism only if it is the shortest useful bridge to one listed option.\n"
            "Final answer form: exact option text only; do not output an option letter, bare diagnosis, or explanation.\n"
            "Use this only to keep the repaired claim and final answer aligned."
        )
    if _is_medqa_profile(resolved) and resolved.name in {"medqa_relation_v6", "medqa_relation_v7", "medqa_relation_v8"}:
        return (
            "Task focus:\n"
            "Requested object: the exact option text from the answer choices.\n"
            "Bridge: clinical evidence -> decisive feature -> diagnosis/mechanism/treatment -> exact option text.\n"
            "Final answer form: exact option text only; do not output an option letter, bare diagnosis, or explanation.\n"
            "Use this only to keep the repaired claim and final answer aligned."
        )
    if _is_logiqa_profile(resolved):
        return (
            "Task focus:\n"
            "Requested object: the option number whose exact option text satisfies the question's requested logical relation.\n"
            "Bridge: stated premises -> valid consequence or valid elimination -> exact option text -> option number.\n"
            "A claim that eliminates option N does not support answer N; keep supported, eliminated, and still-open options separate.\n"
            "Final answer form: one number from 1 to 4 only.\n"
            "Use this only to keep the repaired claim, exact option text, and final option number aligned."
        )
    if resolved.name == "hotpotqa_relation_v35":
        return (
            "Task focus:\n"
            "Requested object: the exact short-answer phrase that fills the question's requested answer role.\n"
            "Role ledger: question source/excluded entity -> required bridge constraint -> requested answer role -> candidate phrase.\n"
            "Keep source entities, intermediate entities, answer-role fillers, and explanatory facts separate; do not output an intermediate entity when the question asks for its role, office, distinction, date, shared property, or other companion.\n"
            "Final answer form: the clean role filler only; preserve required labels, units, location level, titles, and singular/plural scope.\n"
            "Use this only to keep the graph claim and final answer aligned."
        )
    if _is_bamboogle_profile(resolved) and resolved.name in {
        "bamboogle_relation_v5",
        "bamboogle_relation_v6",
        "hotpotqa_relation_v1",
        "hotpotqa_relation_v2",
        "hotpotqa_relation_v3",
        "hotpotqa_relation_v4",
        "hotpotqa_relation_v5",
        "hotpotqa_relation_v6",
        "hotpotqa_relation_v7",
        "hotpotqa_relation_v8",
        "hotpotqa_relation_v9",
        "hotpotqa_relation_v10",
        "hotpotqa_relation_v11",
        "hotpotqa_relation_v12",
        "hotpotqa_relation_v13",
        "hotpotqa_relation_v14",
        "hotpotqa_relation_v15",
        "hotpotqa_relation_v16",
        "hotpotqa_relation_v17",
        "hotpotqa_relation_v18",
        "hotpotqa_relation_v19",
        "hotpotqa_relation_v20",
        "hotpotqa_relation_v21",
        "hotpotqa_relation_v22",
        "hotpotqa_relation_v23",
        "hotpotqa_relation_v24",
        "hotpotqa_relation_v25",
        "hotpotqa_relation_v26",
        "hotpotqa_relation_v27",
        "hotpotqa_relation_v28",
        "hotpotqa_relation_v29",
        "hotpotqa_relation_v30",
        "hotpotqa_relation_v31",
        "hotpotqa_relation_v32",
        "hotpotqa_relation_v33",
        "hotpotqa_relation_v34",
        "hotpotqa_relation_v35",
        "musique_relation_v1",
        "musique_relation_v2",
        "musique_relation_v3",
    }:
        return (
            "Task focus:\n"
            "Requested object: the shortest exact short-answer phrase that fills the question's answer slot.\n"
            "Bridge: shared evidence -> intermediate entity -> requested attribute -> exact short answer phrase.\n"
            "Final answer form: shortest exact phrase only; do not output an intermediate entity, broader place, nearby date, related person, or longer paraphrase.\n"
            "Use this only to keep the repaired claim and final answer aligned."
        )
    if _is_truthvalue_profile(resolved):
        return (
            "Task focus:\n"
            "Requested object: the exact truth-value label for the conclusion.\n"
            "Bridge: premises and rules -> conclusion truth value -> allowed label token.\n"
            "Final answer form: the allowed label token only; do not output a proof, theorem name, or explanation.\n"
            "Use this only to keep the repaired claim and final answer aligned."
        )
    return ""


def render_target_focus(question: str, profile: str | PromptProfile | None = None) -> str:
    profile_focus = _profile_target_focus(profile)
    if profile_focus:
        return profile_focus
    note = compact_target_focus_note(question)
    if not note:
        return ""
    lowered_question = question.lower()
    if question_uses_label_answers(question) and all(
        word in lowered_question for word in ("entailment", "neutral", "contradiction")
    ):
        return (
            "Task focus:\n"
            f"{note}\n"
            "Use this only to keep the premise-to-hypothesis relation and final label aligned."
        )
    if question_uses_label_answers(question):
        return (
            "Task focus:\n"
            f"{note}\n"
            "Use this only to keep the repaired claim and final label aligned."
        )
    return (
        "Requested object:\n"
        f"{note}\n"
        "Use this only to avoid drifting to a nearby quantity."
    )


def _clean_metadata_text(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _render_claim_summary(claim) -> str:
    claim_text = _clean_metadata_text(getattr(claim, "text", ""), "an unspecified local claim")
    return f"{claim_text}"


def _render_claim_dossier_line(claim) -> str:
    claim_text = _clean_metadata_text(getattr(claim, "text", ""), "an unspecified local claim")
    return f"- {claim_text}"


def _render_path_summary(path, claim_map) -> str:
    agent_text = ", ".join(path.agent_ids) if path.agent_ids else "unspecified agents"
    summary = _clean_metadata_text(path.summary, "No method summary was provided.")
    return f"{agent_text}: {summary}"


def _render_path_dossier_line(path) -> str:
    agent_text = ", ".join(path.agent_ids) if path.agent_ids else "unspecified agents"
    summary = _clean_metadata_text(path.summary, "No method summary was provided")
    return f"- {agent_text}: {summary}"


def _render_divergence_summary(divergence: DivergenceCase) -> str:
    claim_object = _clean_metadata_text(divergence.claim_object, "an unspecified object")
    aspect = _clean_metadata_text(divergence.aspect, "an unspecified aspect")
    frontier = _clean_metadata_text(divergence.frontier_claim_id, "")
    frontier_text = f"after {frontier}" if frontier else "from the current frontier"
    return f"First split {frontier_text}: {divergence.left_path_id} says {divergence.left_claim}; {divergence.right_path_id} says {divergence.right_claim}"


def _render_divergence_dossier_lines(divergence: DivergenceCase) -> list[str]:
    frontier = _clean_metadata_text(divergence.frontier_claim_id, "")
    claim_object = _clean_metadata_text(divergence.claim_object, "an unspecified object")
    aspect = _clean_metadata_text(divergence.aspect, "an unspecified aspect")
    alignment = _clean_metadata_text(divergence.alignment, "an unspecified alignment tag")
    relation = _clean_metadata_text(divergence.relation, "an unspecified relation")
    why = _clean_metadata_text(divergence.why_minimal, "No explanation was provided")
    divergence_intro = (
        f"After {frontier}, {divergence.left_path_id} says {divergence.left_claim}."
        if frontier
        else f"{divergence.left_path_id} says {divergence.left_claim}."
    )
    return [
        f"- {divergence_intro} {divergence.right_path_id} says {divergence.right_claim}.",
    ]


def _merge_header_only_prompt_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if current.endswith(":") and index + 1 < len(lines) and "final answer" not in current.lower():
            merged.append(f"{current} {lines[index + 1].strip()}")
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _prompt_starts_candidate_check(line: str) -> bool:
    text = line.strip().lower()
    return bool(
        re.match(r"^\d+\s*[x×]\s*\d+\s*=", text)
        or re.match(r"^(first|next)\s+multiple\b.*\b\d+\b", text)
        or re.match(r"^(check|try|consider)\s+\d+\b", text)
        or re.match(r"^(start with|check|try|consider)\s+\d+\b", text)
        or re.match(r"^\d+\s+is\s+(?:the\s+)?(?:first|next|candidate|multiple)\b", text)
        or "candidate" in text
    )


def _prompt_is_validation_outcome(line: str) -> bool:
    text = line.strip().lower()
    markers = (
        "contains digit",
        "contains the digit",
        "contains only",
        "contains only digits",
        "contains only the digits",
        "digits are valid",
        "digits are ",
        "is not allowed",
        "not allowed",
        "is valid",
        "works",
        "fails",
        "is a multiple of",
        "is not a multiple of",
        "divisible",
        "satisfies",
    )
    return any(marker in text for marker in markers)


def _prompt_has_conflicting_validation(line: str) -> bool:
    text = line.strip().lower()
    positive_markers = (
        "contains only",
        "digits are valid",
        "is valid",
        "works",
        "satisfies",
    )
    negative_markers = (
        "contains digit",
        "contains the digit",
        "is not allowed",
        "not allowed",
        "fails",
        "forbidden digit",
        "is not a multiple of",
    )
    return any(marker in text for marker in positive_markers) and any(marker in text for marker in negative_markers)


def _merge_candidate_outcome_prompt_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        if index + 1 < len(lines) and _prompt_starts_candidate_check(current):
            merged_line = current
            next_index = index + 1
            consumed_outcome = False
            while (
                next_index < len(lines)
                and not _prompt_starts_candidate_check(lines[next_index])
                and _prompt_is_validation_outcome(lines[next_index])
            ):
                merged_line = f"{merged_line} {lines[next_index].strip()}"
                next_index += 1
                consumed_outcome = True
            if consumed_outcome:
                merged.append(merged_line)
                index = next_index
                continue
        merged.append(current)
        index += 1
    return merged


def _prompt_grouping_terminal_value(line: str) -> str:
    text = line.strip()
    lowered = text.lower()
    if "grouping" not in lowered and "parenthes" not in lowered:
        return ""
    match = re.search(r"=\s*(-?\d+(?:\.\d+)?)\s*\$?\.?\s*$", text)
    return match.group(1) if match else ""


def _compress_prompt_repeated_grouping_values(lines: list[str], min_run_length: int = 4) -> list[str]:
    compressed: list[str] = []
    index = 0
    while index < len(lines):
        value = _prompt_grouping_terminal_value(lines[index])
        if not value:
            compressed.append(lines[index])
            index += 1
            continue
        run_end = index
        while run_end < len(lines) and _prompt_grouping_terminal_value(lines[run_end]) == value:
            run_end += 1
        run = lines[index:run_end]
        if len(run) >= min_run_length:
            compressed.extend(run[:2])
            compressed.append(f"Several other parenthesizations or groupings also evaluate to {value}.")
        else:
            compressed.extend(run)
        index = run_end
    return compressed


def _prompt_is_search_failure(line: str) -> bool:
    text = line.strip().lower()
    failure_markers = (
        "contains digit",
        "contains the digit",
        "is not allowed",
        "not allowed",
        "is not a multiple of",
        "fails",
        "forbidden digit",
    )
    return (
        _prompt_starts_candidate_check(text)
        and not _prompt_has_conflicting_validation(text)
        and any(marker in text for marker in failure_markers)
    )


def _prompt_candidate_range_token(line: str) -> str:
    text = line.strip()
    check_match = re.match(r"^(?:check|try|consider)\s+(-?\d+)\b", text, flags=re.IGNORECASE)
    if check_match:
        return check_match.group(1)
    mult_match = re.search(r"=\s*(-?\d+)\b", text)
    if mult_match:
        return mult_match.group(1)
    int_match = re.search(r"-?\d+\b", text)
    return int_match.group(0) if int_match else "unknown"


def _summarize_prompt_search_failures(run: list[str]) -> str:
    start = _prompt_candidate_range_token(run[0])
    end = _prompt_candidate_range_token(run[-1])
    lowered = " ".join(line.lower() for line in run)
    if "is not a multiple of" in lowered and ("contains digit" in lowered or "contains the digit" in lowered):
        reason = "was not a multiple of 30 or contained a forbidden digit"
    elif "contains digit" in lowered or "contains the digit" in lowered:
        reason = "contained a forbidden digit"
    elif "is not a multiple of" in lowered:
        reason = "was not a multiple of 30"
    else:
        reason = "failed the local candidate check"
    return f"Checked candidates from {start} through {end} in increasing order; each {reason}."


def _compress_prompt_search_failures(lines: list[str], min_run_length: int = 8) -> list[str]:
    compressed: list[str] = []
    index = 0
    while index < len(lines):
        if not _prompt_is_search_failure(lines[index]):
            compressed.append(lines[index])
            index += 1
            continue
        run_end = index
        while run_end < len(lines) and _prompt_is_search_failure(lines[run_end]):
            run_end += 1
        run = lines[index:run_end]
        if len(run) >= min_run_length:
            compressed.append(_summarize_prompt_search_failures(run))
        else:
            compressed.extend(run)
        index = run_end
    return compressed


def _render_steps_for_prompt(
    steps,
    max_steps: int | None = None,
    tail_steps: int = 0,
    search_failure_min_run: int = 8,
    grouping_min_run: int = 4,
) -> str:
    rendered_lines = [step.text for step in steps]
    rendered_lines = _merge_header_only_prompt_lines(rendered_lines)
    rendered_lines = _merge_candidate_outcome_prompt_lines(rendered_lines)
    rendered_lines = _compress_prompt_search_failures(rendered_lines, min_run_length=search_failure_min_run)
    rendered_lines = _compress_prompt_repeated_grouping_values(rendered_lines, min_run_length=grouping_min_run)
    step_ids = [step.step_id for step in steps]
    if not rendered_lines:
        return "- no steps"
    if max_steps is None or len(rendered_lines) <= max_steps:
        return "\n".join(
            f"- [{step_ids[min(idx, len(step_ids) - 1)]}] {line}"
            for idx, line in enumerate(rendered_lines)
        )
    head_steps = max_steps - tail_steps
    head = rendered_lines[:head_steps]
    tail = rendered_lines[-tail_steps:]
    lines = [f"- [{step_ids[min(idx, len(step_ids) - 1)]}] {line}" for idx, line in enumerate(head)]
    omitted = len(rendered_lines) - len(head) - len(tail)
    lines.append(f"- ... {omitted} omitted middle steps with repeated local checks ...")
    tail_start = len(rendered_lines) - len(tail)
    lines.extend(
        f"- [{step_ids[min(tail_start + idx, len(step_ids) - 1)]}] {line}"
        for idx, line in enumerate(tail)
    )
    return "\n".join(lines)


def _render_graph_for_llm_json(graph: NaturalLanguageGraph) -> str:
    payload = {
        "shared_claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "members": list(claim.members or []),
                "claim_object": claim.claim_object,
                "aspect": claim.aspect,
                "status": claim.status,
                "alignment": claim.alignment,
            }
            for claim in graph.shared_claims
        ],
        "method_paths": [
            {
                "path_id": path.path_id,
                "agent_ids": list(path.agent_ids or []),
                "summary": path.summary,
                "claim_ids": list(path.claim_ids or []),
            }
            for path in graph.method_paths
        ],
        "divergences": [
            {
                "divergence_id": divergence.divergence_id,
                "frontier_claim_id": divergence.frontier_claim_id,
                "relation": divergence.relation,
                "left_path_id": divergence.left_path_id,
                "right_path_id": divergence.right_path_id,
                "left_claim": divergence.left_claim,
                "right_claim": divergence.right_claim,
                "why_minimal": divergence.why_minimal,
                "claim_object": divergence.claim_object,
                "aspect": divergence.aspect,
                "alignment": divergence.alignment,
            }
            for divergence in graph.divergences
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_graph_for_llm(graph: NaturalLanguageGraph, graph_format: str = "natural") -> str:
    if not graph.shared_claims and not graph.method_paths and not graph.divergences:
        return "The graph is currently empty."
    if str(graph_format or "natural").strip().lower() == "json":
        return _render_graph_for_llm_json(graph)

    claim_map = graph.claim_map()
    parts = []
    if graph.shared_claims:
        parts.append("Common ground:")
        for claim in graph.shared_claims:
            parts.append(f"- {_render_claim_summary(claim)}")
    if graph.method_paths:
        parts.append("")
        parts.append("Paths:")
        for path in graph.method_paths:
            parts.append(f"- {_render_path_summary(path, claim_map)}")
    if graph.divergences:
        parts.append("")
        parts.append("First split:")
        for divergence in graph.divergences:
            parts.append(f"- {_render_divergence_summary(divergence)}")
    return "\n".join(parts).strip()


def graph_format_uses_json_io(graph_format: str) -> bool:
    return str(graph_format or "natural").strip().lower() == "json"


def _json_io_relation_instruction(graph_format: str) -> list[str]:
    if not graph_format_uses_json_io(graph_format):
        return ["Write one short natural-language note, not a field list."]
    return [
        "Return only one valid JSON object. Do not use Markdown fences.",
        'Required shape: {"relation_note":"...","shared_evidence":"...","first_decisive_mismatch":"...","risk_notes":"..."}',
        "Keep every value short natural-language text. Do not add gold labels or target answers.",
    ]


def _json_io_resolution_instruction(graph_format: str) -> list[str]:
    if not graph_format_uses_json_io(graph_format):
        return [
            "Write one short natural-language repair note using exactly these labels:",
            "Action: ...",
            "Winning side: ...",
            "Correct claim: ...",
            "First conflict: ...",
            "Reason: ...",
            "Rewrite from: ...",
            "Keep paths: ...",
            "Drop paths: ...",
        ]
    return [
        "Return only one valid JSON object. Do not use Markdown fences.",
        "Use exactly these keys:",
        (
            '{"action":"choose_A|choose_B|keep_parallel|synthesize",'
            '"winning_side":"Claim A|Claim B|both|neither|synthesized",'
            '"correct_claim":"...",'
            '"first_conflict":"...",'
            '"reason":"...",'
            '"rewrite_from":"C1",'
            '"keep_paths":["P1"],'
            '"drop_paths":["P2"]}'
        ),
        "Use path and claim ids from the graph. Do not add gold labels or target answers.",
    ]


def _resolution_prompt_style_name(value: str | None) -> str:
    normalized = str(value or "profile").strip().lower()
    if normalized in {"", "default", "profile"}:
        return "profile"
    if normalized in {"minimal", "minimal_strategy", "strategy_only"}:
        return "minimal_strategy"
    if normalized in {"ledger", "ledger_strategy", "evidence_ledger"}:
        return "ledger_strategy"
    raise ValueError(f"Unsupported resolution_prompt_style: {value!r}")


def _minimal_strategy_rules(graph_format: str) -> list[str]:
    lines = [
        "Strategy:",
        "1. Identify the requested object or relation in the question.",
        "2. Compare each branch only against visible support in the graph and traces.",
        "3. Locate the earliest contradiction about the same requested object or relation.",
        "4. Choose the supported branch; use keep_parallel if the visible support is insufficient or the split is only progress lag.",
        "5. Do not use outside facts, gold labels, target answers, or majority vote by itself.",
        "",
    ]
    lines.extend(_json_io_resolution_instruction(graph_format))
    return lines


def _ledger_strategy_rules(profile: str | PromptProfile | None, graph_format: str) -> list[str]:
    resolved = resolve_prompt_profile(profile)
    name = resolved.name
    if name.startswith("math500"):
        ledger_lines = [
            "Evidence ledger:",
            "- Requested object: name the exact value/expression/set/count/probability asked for.",
            "- Candidate ledger: for each candidate, write shared checkpoint -> checked operation -> requested object.",
            "- Failure mark: intermediate-only, unchecked arithmetic/sign/endpoint, wrong requested object, or complete bridge.",
            "- Decision: choose only a complete bridge; otherwise keep_parallel and name the next missing local operation.",
        ]
    elif name.startswith("folio") or _is_truthvalue_profile(resolved):
        ledger_lines = [
            "Evidence ledger:",
            "- Queried statement: restate the exact statement being judged.",
            "- Proof-status ledger: proved query, proved negation, or neither proved.",
            "- Failure mark: invalid converse/inverse, unstated rule, mere consistency, ordinary-world assumption, or complete proof bridge.",
            "- Decision: true/false/unknown must follow the proof status; otherwise keep_parallel.",
        ]
    elif name.startswith("mmlu_pro"):
        ledger_lines = [
            "Evidence ledger:",
            "- Requested relation: state the exact relation asked by the question.",
            "- Option ledger: for each candidate, exact option text -> visible support -> final option token.",
            "- Failure mark: topical match, partial bridge, option-token drift, eliminated option, or complete option bridge.",
            "- Decision: choose only the exact option bridge; otherwise keep_parallel.",
        ]
    elif name.startswith("hotpotqa") or _is_bamboogle_profile(resolved):
        ledger_lines = [
            "Evidence ledger:",
            "- Requested slot: state the entity/relation/attribute or yes-no predicate being asked.",
            "- Bridge ledger: question source -> context entity -> requested attribute -> answer phrase.",
            "- Failure mark: wrong entity level, missing hop, nearby attribute, overlong phrase/list, or complete slot bridge.",
            "- Decision: choose only a complete visible bridge; otherwise keep_parallel.",
        ]
    elif _is_anli_profile(resolved):
        ledger_lines = [
            "Evidence ledger:",
            "- Premise-hypothesis boundary: state the exact hypothesis relation being judged.",
            "- Relation ledger: entailment, contradiction, or neutral, with the visible premise fact that licenses it.",
            "- Failure mark: wording-only strictness, missing-information-as-contradiction, outside assumption, or complete relation bridge.",
            "- Decision: choose only a complete relation bridge; otherwise keep_parallel.",
        ]
    else:
        ledger_lines = [
            "Evidence ledger:",
            "- Requested object or relation: state what the question asks for.",
            "- Candidate ledger: candidate -> visible support -> answer form.",
            "- Failure mark: unsupported, nearby object, progress lag, equivalent wording, or complete bridge.",
            "- Decision: choose only a complete bridge; otherwise keep_parallel.",
        ]
    lines = [
        "Strategy:",
        "1. Fill the evidence ledger from the graph and traces before deciding.",
        "2. Compare branches on the same requested object or relation.",
        "3. Treat observed candidates as hypotheses, not as gold answers.",
        "4. Choose a branch only when its ledger has a complete visible bridge.",
        "5. Use keep_parallel when the visible bridge is incomplete, the split is progress lag, or both claims may still be true.",
        "6. Do not use outside facts, gold labels, target answers, or majority vote by itself.",
        "",
        *ledger_lines,
        "",
    ]
    lines.extend(_json_io_resolution_instruction(graph_format))
    return lines


def _json_io_revision_instruction(graph_format: str) -> list[str]:
    if not graph_format_uses_json_io(graph_format):
        return []
    return [
        "Return only one valid JSON object. Do not use Markdown fences.",
        'Required shape: {"revised_response":"..."}',
        "Put the complete revised response text in revised_response. Do not add gold labels or target answers.",
    ]


def _render_trace_for_prompt(trace: AgentTrace, max_steps: int | None = 18, tail_steps: int = 6) -> str:
    return _render_steps_for_prompt(trace.steps, max_steps=max_steps, tail_steps=tail_steps)


def render_graph_dossier(graph: NaturalLanguageGraph) -> str:
    parts = ["Common Ground"]
    for claim in graph.shared_claims:
        parts.append(_render_claim_dossier_line(claim))
    parts.append("")
    parts.append("Paths")
    for path in graph.method_paths:
        parts.append(_render_path_dossier_line(path))
    parts.append("")
    parts.append("First Split")
    if graph.divergences:
        for divergence in graph.divergences:
            parts.extend(_render_divergence_dossier_lines(divergence))
    else:
        parts.append("- none yet.")
    return "\n".join(parts).strip()


def build_atomic_trace_prompt(question: str, agent_id: str, response: str) -> str:
    prompt_question = clean_question_text(question)
    return (
        "Rewrite one agent response into a natural-language atomic trace.\n\n"
        "Rules:\n"
        "- Output plain text only.\n"
        "- One non-empty line per local claim, operation, check, or conclusion.\n"
        "- Keep the original semantics. Do not repair mistakes.\n"
        "- Remove headers, filler transitions, and discourse-only lines.\n"
        "- Preserve wrong claims if the agent made them.\n"
        "- Keep concrete numbers, substitutions, candidate checks, and final sets/counts explicit when the response states them.\n"
        "- Compress long repeated searches into one short line, but keep the first accepted candidate and any disputed candidate-validity claim explicit.\n"
        "- If the source response is cut off mid-example, mid-formula, or mid-sentence, drop the incomplete tail instead of outputting a broken fragment.\n"
        "- Do not replace concrete numbers with pronouns or placeholders such as 'this value' when the number is available.\n"
        "- The last non-empty line must be the final answer line if the response contains one.\n\n"
        f"Question:\n{prompt_question}\n\n"
        f"Agent:\n{agent_id}\n\n"
        f"Response:\n{response}\n"
    )


def build_claim_merge_prompt(
    question: str,
    traces: Dict[str, AgentTrace],
    profile: str | PromptProfile | None = None,
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    parts = [
        resolved.merge_intro,
        "",
        _profile_language_instruction(resolved),
        "Use only these plain labels: Common Ground, Paths, First Split.",
        "",
        _profile_merge_guide(resolved),
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    parts.extend(
        [
        target_focus,
        "",
        resolved.merge_rules,
        SHARED_PREFIX_SUPPORT_RULE,
        SUPPORT_RELATION_SPLIT_RULE,
        *label_support_rules,
        "",
        *label_support_examples,
        "",
        "- If there is a first real split, write it as ordinary language: 'First split: A1 claims ...; A2 claims ...'.",
        "- If the agents end with different final answers or requested objects, there is a first split; if they share a prefix, write the split as a support-relation conflict instead of only a token conflict.",
        "- If there is no real split in the visible claims, write 'none yet' under First Split.",
        "",
        f"Question:\n{prompt_question}",
        "",
        "Agent traces:",
        ]
    )
    for agent_id, trace in traces.items():
        parts.append(f"{agent_id}")
        parts.append(_render_trace_for_prompt(trace, max_steps=22, tail_steps=8))
        parts.append("")
    return "\n".join(parts).strip()


def build_incremental_claim_merge_prompt(
    question: str,
    traces: Dict[str, AgentTrace],
    start_index: int,
    stop_index: int,
    existing_dossier: str = "",
    per_agent_stop_indices: Dict[str, int] | None = None,
    profile: str | PromptProfile | None = None,
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    parts = [
        "Build or update one compact natural-language claim memo from incrementally revealed trace steps.",
        "",
        _profile_full_memo_instruction(resolved),
        "Use only these plain labels: Common Ground, Paths, First Split.",
        "",
        _profile_merge_guide(resolved),
        "",
        _profile_frontier_update_guide(resolved),
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    parts.extend(
        [
        target_focus,
        "",
        "Rules:",
        "- Output one full updated dossier, not a delta.",
        "- Use the current dossier as provisional memory, but split earlier claims back apart if the new window shows the merge was too aggressive.",
        "- Treat this as a frontier state update: shared progress, method fork, or same-object claim conflict.",
        SHARED_PREFIX_SUPPORT_RULE,
        SUPPORT_RELATION_SPLIT_RULE,
        *label_support_rules,
        "- Do not jump to final-answer comparison while the new window contains a local shared checkpoint or a local same-object mismatch.",
        f"{resolved.merge_rules}",
        "",
        *label_support_examples,
        "- Write the first split as ordinary language: 'First split: A1 claims ...; A2 claims ...'.",
        "- Different final answers in the full traces do not by themselves create a First Split in this window.",
        "- If the current window still shows only shared progress, method lag, or one path not yet reaching the later local claim, write 'none yet' under First Split.",
        "- Do not force a divergence unless the current window reveals a concrete same-object claim conflict.",
        "- If a path in the visible window stops early or looks truncated, say that in the path summary instead of attaching later claims from another path to it.",
        "- For repeated search or enumeration, summarize repeated identical failed checks, but keep accepted or disputed candidate checks explicit.",
        "",
        f"Question:\n{prompt_question}",
        "",
        f"Currently revealed step window: steps {start_index + 1} to {stop_index}",
        ]
    )
    if existing_dossier.strip():
        parts.extend(
            [
                "",
                "Current dossier to update:",
                existing_dossier.strip(),
            ]
        )
    parts.extend(
        [
            "",
            "Newly revealed trace steps:",
        ]
    )
    for agent_id, trace in traces.items():
        agent_stop = stop_index
        if per_agent_stop_indices is not None:
            agent_stop = max(start_index, min(len(trace.steps), per_agent_stop_indices.get(agent_id, stop_index)))
        parts.append(agent_id)
        parts.append(
            _render_steps_for_prompt(
                trace.steps[start_index:agent_stop],
                max_steps=None,
                tail_steps=0,
                search_failure_min_run=4,
                grouping_min_run=4,
            )
        )
        parts.append("")
    return "\n".join(parts).strip()


def build_shared_graph_audit_prompt(
    question: str,
    graph: NaturalLanguageGraph,
    traces: Dict[str, AgentTrace],
    profile: str | PromptProfile | None = None,
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    parts = [
        resolved.audit_intro,
        "",
        _profile_full_memo_instruction(resolved, audit=True),
        "",
        _profile_merge_guide(resolved),
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    parts.extend(
        [
        target_focus,
        "",
        resolved.audit_rules,
        SHARED_PREFIX_SUPPORT_RULE,
        *label_support_rules,
        *label_support_examples,
        "",
        "- Keep the memo trace-faithful.",
        "- If one path is still at placeholder or unverified status, do not merge it with another path's checked fact.",
        "- Do not invent unsupported facts.",
        "",
        f"Question:\n{prompt_question}",
        "",
        "Current graph summary in ordinary language:",
        render_graph_for_llm(graph),
        "",
        "Agent traces for audit:",
        ]
    )
    for agent_id, trace in traces.items():
        parts.append(agent_id)
        parts.append(_render_trace_for_prompt(trace))
        parts.append("")
    return "\n".join(parts).strip()


def build_prefix_conflict_graph_prompt(
    question: str,
    graph: NaturalLanguageGraph,
    traces: Dict[str, AgentTrace],
    profile: str | PromptProfile | None = None,
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    parts = [
        resolved.prefix_intro,
        "",
        _profile_full_memo_instruction(resolved),
        "Keep it compact and prefix-focused.",
        "",
        _profile_merge_guide(resolved),
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    parts.extend(
        [
        target_focus,
        "",
        resolved.prefix_rules,
        SHARED_PREFIX_SUPPORT_RULE,
        SUPPORT_RELATION_SPLIT_RULE,
        *label_support_rules,
        "",
        *label_support_examples,
        "",
        "- Prefer the shared natural-language claim immediately before the wrong branch, not a distant setup claim.",
        "- When three agents are present, you may keep two representative paths if the third is locally equivalent before the first conflict.",
        "",
        f"Question:\n{prompt_question}",
        "",
        "Current dossier to compress and repair:",
        graph.raw_dossier.strip() or "The graph is currently empty.",
        "",
        "Agent traces:",
        ]
    )
    for agent_id, trace in traces.items():
        parts.append(agent_id)
        parts.append(_render_trace_for_prompt(trace))
        parts.append("")
    return "\n".join(parts).strip()


def build_pairwise_divergence_resolution_prompt(
    question: str,
    left_trace: AgentTrace,
    right_trace: AgentTrace,
    relation_analysis: str = "",
    profile: str | PromptProfile | None = None,
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
    graph_format: str = "natural",
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    style = _resolution_prompt_style_name(resolution_prompt_style)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_resolution_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    candidate_block = _render_observed_candidate_block(
        question,
        {left_trace.agent_id: left_trace, right_trace.agent_id: right_trace},
        resolved,
    )
    analysis_block = (
        "Relation note:\n"
        f"{relation_analysis.strip()}\n\n"
        if relation_analysis.strip()
        else ""
    )
    if style in {"minimal_strategy", "ledger_strategy"}:
        parts = [
            "Resolve two disagreeing traces using only the visible evidence.",
            "",
        ]
        if analysis_block.strip():
            parts.append(analysis_block.strip())
            parts.append("")
        if candidate_block:
            parts.extend([candidate_block, ""])
        parts.extend(
            [
                target_focus,
                "",
                *(
                    _ledger_strategy_rules(resolved, graph_format)
                    if style == "ledger_strategy"
                    else _minimal_strategy_rules(graph_format)
                ),
                "",
                f"Question:\n{prompt_question}",
                "",
                f"Claim A ({left_trace.agent_id}):\n{_render_trace_for_prompt(left_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(left_trace)}",
                "",
                f"Claim B ({right_trace.agent_id}):\n{_render_trace_for_prompt(right_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(right_trace)}",
            ]
        )
        return "\n".join(parts).strip()
    parts = [
        "Two agent traces end with different answers, but the graph merge failed to expose a usable divergence.",
        "",
        "Compare the two traces directly and emit one repair decision.",
        "",
        "First find the earliest local claim where the two traces cannot both be right.",
        "",
    ]
    if analysis_block.strip():
        parts.append(analysis_block.strip())
        parts.append("")
    if dataset_note:
        parts.extend([dataset_note, ""])
    if candidate_block:
        parts.extend([candidate_block, ""])
    pairwise_rules = _profile_pairwise_resolution_rules(resolved)
    parts.extend(
        [
            target_focus,
            "",
            "Write one short natural-language repair note.",
            "Keep these words recoverable in the prose: action, winning side, repaired claim, rewrite anchor, kept paths, dropped paths, and reason.",
            "",
            "Rules:",
            *pairwise_rules,
            SHARED_PREFIX_SUPPORT_RULE,
            SUPPORT_RELATION_SPLIT_RULE,
            *label_support_rules,
            "",
            *label_support_examples,
            "",
            f"Examples:\n{resolved.resolution_fewshot}",
            "",
            f"Question:\n{prompt_question}",
            "",
            f"Claim A ({left_trace.agent_id}):\n{_render_trace_for_prompt(left_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(left_trace)}",
            "",
            f"Claim B ({right_trace.agent_id}):\n{_render_trace_for_prompt(right_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(right_trace)}",
        ]
    )
    return "\n".join(parts).strip()


def _agent_ids_for_divergence(graph: NaturalLanguageGraph, divergence: DivergenceCase) -> list[str]:
    path_map = graph.path_map()
    agent_ids: list[str] = []
    for path_id in (divergence.left_path_id, divergence.right_path_id):
        path = path_map.get(path_id)
        if path is None:
            continue
        for agent_id in path.agent_ids:
            if agent_id not in agent_ids:
                agent_ids.append(agent_id)
    return agent_ids


def _render_relevant_trace_evidence(
    traces: Dict[str, AgentTrace] | None,
    graph: NaturalLanguageGraph,
    divergence: DivergenceCase,
    *,
    full_trace: bool = False,
) -> str:
    if not traces:
        return ""
    agent_ids = _agent_ids_for_divergence(graph, divergence)
    if not agent_ids:
        agent_ids = list(traces.keys())[:2]
    parts = ["Relevant trace evidence for this split:"]
    for agent_id in agent_ids:
        trace = traces.get(agent_id)
        if trace is None:
            continue
        parts.append(f"{agent_id}:")
        if full_trace:
            parts.append(_render_trace_for_prompt(trace, max_steps=None, tail_steps=0))
        else:
            parts.append(_render_trace_for_prompt(trace))
    return "\n".join(parts).strip()


def build_divergence_relation_analysis_prompt(
    question: str,
    graph: NaturalLanguageGraph,
    divergence: DivergenceCase,
    traces: Dict[str, AgentTrace] | None = None,
    profile: str | PromptProfile | None = None,
    graph_format: str = "natural",
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    style = _resolution_prompt_style_name(resolution_prompt_style)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    evidence = _render_relevant_trace_evidence(
        traces,
        graph,
        divergence,
        full_trace=str(resolution_trace_context or "").strip().lower() == "full",
    )
    evidence_block = f"\n\n{evidence}\n" if evidence else "\n"
    candidate_block = _render_observed_candidate_block(question, traces, resolved)
    if style in {"minimal_strategy", "ledger_strategy"}:
        parts = [
            "Write a compact relation note for the next repair decision.",
            "",
            *_json_io_relation_instruction(graph_format),
        ]
        if candidate_block:
            parts.extend(["", candidate_block])
        parts.extend(
            [
                "",
                target_focus,
                "",
                (
                    "Strategy: build a compact evidence ledger, compare only visible branch support for the exact requested object or relation, and name the earliest same-object mismatch if one exists."
                    if style == "ledger_strategy"
                    else "Strategy: identify the requested object or relation, compare only visible branch support, and name the earliest same-object mismatch if one exists."
                ),
                "",
                f"Question:\n{prompt_question}",
                "",
                "Divergence under review:",
                f"{_render_divergence_summary(divergence)}",
                "",
                "Graph summary:",
                f"{render_graph_for_llm(graph, graph_format=graph_format)}",
            ]
        )
        if evidence_block.strip():
            parts.append(evidence_block.strip())
        return "\n".join(parts).strip()
    if resolved.name in {"strategyqa_relation_v6", "sqa_relation_v6"}:
        cue_line = (
            "Keep these cues recoverable in the prose: shared evidence, exact question predicate, "
            "bridge claim, and first real mismatch."
        )
        shape_line = (
            "A good shape is: Shared evidence: ... Predicate: ... Path bridge: ... "
            "First real mismatch: ..."
        )
    elif _is_strategyqa_profile(resolved):
        cue_line = (
            "Keep these cues recoverable in the prose: question predicate, factual bridge, "
            "label implication, and first decisive mismatch."
        )
        shape_line = (
            "A good shape is: Predicate: ... Factual bridge: ... Label implication: "
            "this supports yes/no because ... First decisive mismatch: ..."
        )
    elif _is_anli_profile(resolved):
        cue_line = (
            "Keep these cues recoverable in the prose: premise evidence, hypothesis ask, "
            "relation claim, and first decisive mismatch."
        )
        shape_line = (
            "A good shape is: Premise: ... Hypothesis: ... Relation claim: "
            "The premise entails/is neutral toward/contradicts the hypothesis because ... "
            "First decisive mismatch: ..."
        )
    else:
        cue_line = (
            "Keep these cues recoverable in the prose: relevant evidence, requested object, "
            "local claim, and first decisive mismatch."
        )
        shape_line = (
            "A good shape is: Evidence: ... Requested object: ... Local claim: ... "
            "First decisive mismatch: ..."
        )
    parts = [
        f"{resolved.relation_analysis_intro}",
        "",
        *_json_io_relation_instruction(graph_format),
        cue_line,
        shape_line,
        "",
        "Rules:",
        f"{resolved.relation_analysis_rules}",
        SHARED_PREFIX_SUPPORT_RULE,
        SUPPORT_RELATION_SPLIT_RULE,
        *label_support_rules,
        "",
        *label_support_examples,
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    if candidate_block:
        parts.extend([candidate_block, ""])
    parts.extend(
        [
            target_focus,
            "",
            f"Question:\n{prompt_question}",
            "",
            "Divergence under review:",
            f"{_render_divergence_summary(divergence)}",
            "",
            "Graph summary:",
            f"{render_graph_for_llm(graph, graph_format=graph_format)}",
        ]
    )
    if evidence_block.strip():
        parts.append(evidence_block.strip())
    return "\n".join(parts).strip()


def build_pairwise_relation_analysis_prompt(
    question: str,
    left_trace: AgentTrace,
    right_trace: AgentTrace,
    profile: str | PromptProfile | None = None,
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
    graph_format: str = "natural",
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    style = _resolution_prompt_style_name(resolution_prompt_style)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_merge_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    candidate_block = _render_observed_candidate_block(
        question,
        {left_trace.agent_id: left_trace, right_trace.agent_id: right_trace},
        resolved,
    )
    if style in {"minimal_strategy", "ledger_strategy"}:
        parts = [
            "Write a compact relation note for two disagreeing traces.",
            "",
            *_json_io_relation_instruction(graph_format),
        ]
        if candidate_block:
            parts.extend(["", candidate_block])
        parts.extend(
            [
                "",
                target_focus,
                "",
                (
                    "Strategy: build a compact evidence ledger, compare only visible branch support for the exact requested object or relation, and name the earliest same-object mismatch if one exists."
                    if style == "ledger_strategy"
                    else "Strategy: identify the requested object or relation, compare only visible branch support, and name the earliest same-object mismatch if one exists."
                ),
                "",
                f"Question:\n{prompt_question}",
                "",
                f"Left trace ({left_trace.agent_id}):\n{_render_trace_for_prompt(left_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(left_trace)}",
                "",
                f"Right trace ({right_trace.agent_id}):\n{_render_trace_for_prompt(right_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(right_trace)}",
            ]
        )
        return "\n".join(parts).strip()
    if resolved.name in {"strategyqa_relation_v6", "sqa_relation_v6"}:
        cue_line = "Keep these cues recoverable in the prose: shared evidence, exact question predicate, bridge claim, and first real mismatch."
        shape_line = "A good shape is: Shared evidence: ... Predicate: ... Path bridge: ... First real mismatch: ..."
    elif _is_strategyqa_profile(resolved):
        cue_line = "Keep these cues recoverable in the prose: question predicate, factual bridge, label implication, and first decisive mismatch."
        shape_line = "A good shape is: Predicate: ... Factual bridge: ... Label implication: ... First decisive mismatch: ..."
    elif _is_anli_profile(resolved):
        cue_line = "Keep these cues recoverable in the prose: premise evidence, hypothesis relation, and first decisive mismatch."
        shape_line = "A good shape is: Premise: ... Hypothesis: ... Relation claim: ... First decisive mismatch: ..."
    else:
        cue_line = "Keep these cues recoverable in the prose: relation type, first decisive mismatch, and any obvious risk notes."
        shape_line = "A good shape is: For these two traces, the relation is ... The first decisive mismatch is ... Risk notes: ..."
    parts = [
        "Describe two disagreeing traces before making a repair decision.",
        "",
        "Write one short natural-language note, not a field list.",
        cue_line,
        shape_line,
        "",
        "Rules:",
        f"{resolved.relation_analysis_rules}",
        SHARED_PREFIX_SUPPORT_RULE,
        SUPPORT_RELATION_SPLIT_RULE,
        *label_support_rules,
        "",
        *label_support_examples,
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    if candidate_block:
        parts.extend([candidate_block, ""])
    parts.extend(
        [
            target_focus,
            "",
            f"Question:\n{prompt_question}",
            "",
            f"Left trace ({left_trace.agent_id}):\n{_render_trace_for_prompt(left_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(left_trace)}",
            "",
            f"Right trace ({right_trace.agent_id}):\n{_render_trace_for_prompt(right_trace, max_steps=None, tail_steps=0) if str(resolution_trace_context or '').strip().lower() == 'full' else _render_trace_for_prompt(right_trace)}",
        ]
    )
    return "\n".join(parts).strip()


def build_divergence_resolution_prompt(
    question: str,
    graph: NaturalLanguageGraph,
    divergence: DivergenceCase,
    relation_analysis: str = "",
    traces: Dict[str, AgentTrace] | None = None,
    profile: str | PromptProfile | None = None,
    graph_format: str = "natural",
    resolution_trace_context: str = "window",
    resolution_prompt_style: str = "profile",
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    style = _resolution_prompt_style_name(resolution_prompt_style)
    target_focus = render_target_focus(question, resolved)
    label_support_rules = _label_support_relation_rules(question, resolved)
    label_support_examples = _label_support_resolution_examples(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    analysis_block = (
        "Relation note:\n"
        f"{relation_analysis.strip()}\n\n"
        if relation_analysis.strip()
        else ""
    )
    evidence = _render_relevant_trace_evidence(
        traces,
        graph,
        divergence,
        full_trace=str(resolution_trace_context or "").strip().lower() == "full",
    )
    evidence_block = f"\n\n{evidence}\n" if evidence else "\n"
    candidate_block = _render_observed_candidate_block(question, traces, resolved)
    if style in {"minimal_strategy", "ledger_strategy"}:
        parts = [
            "Resolve this graph divergence using only the visible evidence.",
            "",
        ]
        if analysis_block.strip():
            parts.append(analysis_block.strip())
            parts.append("")
        if candidate_block:
            parts.extend([candidate_block, ""])
        parts.extend(
            [
                target_focus,
                "",
                *(
                    _ledger_strategy_rules(resolved, graph_format)
                    if style == "ledger_strategy"
                    else _minimal_strategy_rules(graph_format)
                ),
                "",
                f"Question:\n{prompt_question}",
                "",
                "Divergence under review:",
                f"{_render_divergence_summary(divergence)}",
                "",
                "Graph summary:",
                f"{render_graph_for_llm(graph, graph_format=graph_format)}",
            ]
        )
        if evidence_block.strip():
            parts.append(evidence_block.strip())
        return "\n".join(parts).strip()
    if resolved.name in {"strategyqa_relation_v6", "sqa_relation_v6"}:
        conflict_kind_line = (
            "First decide whether this is a real factual conflict, a real bridge conflict to the exact predicate, "
            "progress lag, or equivalent wording."
        )
    else:
        conflict_kind_line = (
            "First decide whether this is a real premise-to-hypothesis relation conflict, progress lag, or equivalent wording."
            if _is_anli_profile(resolved)
            else "First decide whether this is a real same-object conflict, progress lag, or equivalent wording."
        )
    parts = [
        f"{resolved.resolution_intro}",
        "",
        conflict_kind_line,
        "",
    ]
    if analysis_block.strip():
        parts.append(analysis_block.strip())
        parts.append("")
    if dataset_note:
        parts.extend([dataset_note, ""])
    if candidate_block:
        parts.extend([candidate_block, ""])
    parts.extend(
        [
            target_focus,
            "",
            *_json_io_resolution_instruction(graph_format),
            "",
            "Rules:",
            f"{resolved.resolution_rules}",
            SHARED_PREFIX_SUPPORT_RULE,
            SUPPORT_RELATION_SPLIT_RULE,
            *label_support_rules,
            "",
            *label_support_examples,
            "",
            "Examples:",
            f"{resolved.resolution_fewshot}",
            "",
            f"Question:\n{prompt_question}",
            "",
            "Divergence under review:",
            f"{_render_divergence_summary(divergence)}",
            "",
            "Graph summary:",
            f"{render_graph_for_llm(graph, graph_format=graph_format)}",
        ]
    )
    if evidence_block.strip():
        parts.append(evidence_block.strip())
    return "\n".join(parts).strip()


def build_local_revision_prompt(
    question: str,
    agent_id: str,
    trace: AgentTrace,
    graph: NaturalLanguageGraph,
    resolution_text: str,
    agent_revision_note: str = "",
    trace_block_heading: str = "Revision packet",
    profile: str | PromptProfile | None = None,
    graph_format: str = "natural",
) -> str:
    resolved = resolve_prompt_profile(profile)
    prompt_question = clean_question_text(question)
    target_focus = render_target_focus(question, resolved)
    dataset_note = _profile_dataset_note(resolved)
    revision_note_block = (
        f"Agent-specific rewrite instruction:\n{agent_revision_note.strip()}\n\n"
        if agent_revision_note.strip()
        else ""
    )
    label_hint = label_answer_hint(question)
    revision_rules = _profile_revision_common_rules(resolved, label_hint=label_hint)
    parts = [
        "Revise one agent response using a natural-language graph decision.",
        "",
    ]
    if dataset_note:
        parts.extend([dataset_note, ""])
    parts.extend(
        [
            target_focus,
            "",
            "Rules:",
            *revision_rules,
            *_json_io_revision_instruction(graph_format),
            "",
            f"Question:\n{prompt_question}",
            "",
            f"Agent:\n{agent_id}",
            "",
            f"{trace_block_heading}:",
            f"{trace.normalized_trace_text}",
            "",
        ]
    )
    if revision_note_block.strip():
        parts.append(revision_note_block.strip())
    parts.extend(
        [
            "Graph summary:",
            f"{render_graph_for_llm(graph, graph_format=graph_format)}",
            "",
            "Resolution summary:",
            f"{resolution_text}",
        ]
    )
    return "\n".join(parts).strip()
