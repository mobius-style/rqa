---
title: "Governed Memory Preserves Answer-Space Plurality: An Empirical Pilot for the Micro-Echo Chamber Framework"
author: "Taiko Toeda (MOBIUS LLC)"
version: "0.1"
date: "2026-06-14"
status: "Draft empirical companion manuscript for review before possible Zenodo deposit"
license: "CC BY-NC-SA 4.0"
doi: "to be assigned by Zenodo"
companion_to: "Toeda, T. (2026). From Filter Bubbles to Micro-Echo Chambers: Preserving Answer-Space Plurality in Personalized AI Assistants (v1.0)."
---

# Governed Memory Preserves Answer-Space Plurality
## An Empirical Pilot for the Micro-Echo Chamber Framework

Taiko Toeda
MOBIUS LLC
Version 0.1
2026-06-14
License: CC BY-NC-SA 4.0
DOI: to be assigned by Zenodo

> Companion to *From Filter Bubbles to Micro-Echo Chambers* (Toeda, 2026). That
> paper is conceptual and hypothesis-generating; this one is a small, honest
> empirical pilot of its central architectural claim.

## Abstract

The micro-echo chamber framework (Toeda, 2026) argues that personalized,
memory-enabled AI assistants can narrow not only the information users receive
but the interpretations that remain available to them, and that this narrowing
is partly **architectural**: governing context provenance before an answer is
committed should preserve answer-space plurality that ungoverned memory
destroys. That paper is explicitly conceptual and reports no new data. This
companion provides a first empirical pilot, run end-to-end on a local,
governed reflective-questioning system (a frozen Gemma-4-12B base with a
question-generation adapter, a provenance-tagged memory graph, and bounded
reflective questioning).

We measure **live-alternative retention** — the paper's plurality construct,
operationalized as retention of relevance-screened alternatives that were
**pre-registered before any model output** — across the five personalization
conditions the framework proposes (no personalization, style-only, raw memory,
governed memory, governed memory plus reflection), scored under an explicit
retention criterion by two independent LLM judges of different model families.

The keystone result is clean and judge-robust: **raw, ungoverned, frame-
confirming memory collapses live-alternative retention to roughly one-seventh
of the no-personalization baseline, and provenance governance recovers it to
baseline**, with the two judges in strong agreement on both endpoints. Style-
only personalization does not narrow. A separate generation-level measurement
shows the framework's "memory echo" mechanism (fabricated self-citation) is
input-stratified and is eliminated by output-side provenance checks. We also
report two honest negative or qualifying findings: governance does not reduce
the *volume* of self-referential memory at retrieval (its effect is provenance
labeling and output sanitization, not filtering); and the plurality benefit of
reflective *questioning* is criterion-dependent — it holds only when a frame-
reopening question is counted as retention, and that lenient criterion has poor
inter-judge reliability. We discuss why these qualifications strengthen rather
than weaken the framework, and we state the pilot's limitations plainly: a
single annotator authored the alternatives, the memory is a controlled
manipulation rather than naturalistic accumulation, n is small, and judges show
large leniency main-effects.

Keywords: micro-echo chamber; answer-space plurality; memory governance;
provenance; memory echo; personalized AI; LLM-as-judge; reflective questioning;
empirical pilot

Recommended citation. Toeda, T. (2026). *Governed Memory Preserves Answer-Space
Plurality: An Empirical Pilot for the Micro-Echo Chamber Framework* (Version 0.1).
MOBIUS LLC. Zenodo DOI to be assigned.

## 1. Introduction and Scope

The companion framework (Toeda, 2026) makes a conceptual claim with an empirical
shape: that personalization-induced narrowing of *answer space* is partly an
architectural failure, addressable by governing context before commitment rather
than only by making models less agreeable after the fact. Its sharpest mechanism
is **memory echo** — an inference-time failure in which an assistant retrieves
its own prior interpretations and treats them as independent evidence. Its
"crucial contrast" (its §8.1) is between **raw** memory personalization and
**governed** memory personalization.

This paper does not restate that framework; it tests the parts of it that a
running system can test now. We make four contributions, all modest:

1. We operationalize the framework's plurality construct as **live-alternative
   retention** with a pre-registration protocol and an explicit retention
   criterion, and we measure its inter-judge reliability rather than assume it.
2. We run the framework's full **Condition A–E ladder** on one governed
   reflective-questioning system and report the predicted tradeoff curve.
3. We measure **memory echo** directly as a fabricated-self-citation rate and
   show it is input-stratified and removed by an output-side provenance check.
4. We report two qualifying findings that bound the framework's claims, and we
   state the pilot's limitations without softening them.

The system under test is severable from the framework's truth: it is one
candidate architecture, used here only because it instruments the governance
layers the framework names. We claim a pilot, not a benchmark.

## 2. System and Method

### 2.1 System Under Test

The system is a local reflective-questioning assistant. A frozen Gemma-4-12B
base carries a small question-generation adapter trained to extract claims,
tensions, and assumptions from an input and to generate diverse deeper
questions. A memory store (a "Question Graph") records prior interactions with
**provenance tags** distinguishing user assertions from assistant-generated
interpretations. Two governance components are relevant here: a content filter
that withholds certain governance-control vocabulary from injected context, and
an output check (`sanitize_memory_refs`) that strips any memory-cross reference
to a record that was not actually injected on the current turn. Reflective
questioning is bounded to one round in conversational use.

### 2.2 The Plurality Construct: Live-Alternative Retention

Following the framework's §6, we do **not** measure semantic or embedding
diversity. For each user frame we define a set of **live alternatives** —
relevant, evidentially grounded, actionable, non-misleading, epistemically
useful interpretations or objections that the frame omits (the framework's five
liveness criteria). A response **retains** an alternative when its content is
surfaced intelligibly, appropriately qualified, not straw-manned, and connected
to the user's decision or belief. Retention rate is the coverage of these
pre-specified alternatives.

**Pre-registration.** All frames and their live alternatives were authored and
frozen **before any model output was generated**, to prevent the metric from
being fit to the outputs. We use 18 frames spanning two task-family batches
(AI-governance topics; and personal conflict, career/business, moral
self-justification, political/policy, forecasting, creative ideation,
identity/self-image, and factual-misconception advice), with four live
alternatives each (72 total).

**Two retention criteria.** Because a reflective system answers partly with
*questions*, we score under two explicit criteria. **STRICT** counts retention
only when the alternative's content is stated or directly implied; a question
that merely opens the topic does not count. **LENIENT** also counts a question
that specifically reopens the user's frame toward the alternative. We report
inter-judge reliability for each and adopt STRICT as primary.

### 2.3 Conditions (A–E)

We implement the framework's five conditions. The injected memory is a
**controlled, frame-derived manipulation** that confirms the user's frame and
never contains the live alternatives; this isolates the governance variable and
limits authoring bias, at the cost of naturalism (see Limitations).

- **A — no personalization:** a plain concise answer, no memory, no style cue.
- **B — style-only:** the same, plus a persona cue ("the user prefers concise,
  confident, validating answers"); no memory content.
- **C — raw memory:** the same, plus frame-confirming memory injected as
  undifferentiated context (no provenance, self-output treated as evidence).
- **D — governed memory:** the same memory, but provenance-labeled (user
  assertion vs. assistant interpretation) with an explicit note that assistant
  interpretations are not independent evidence; content-filtered.
- **E — governed memory + reflection:** condition D plus the system's bounded
  reflective questions appended.

### 2.4 Judges

Retention is scored by two LLM judges of different model families: a primary
judge (a 120B open model accessed through a pinned, version-controlled binding)
and an **independent** judge (a 27B model of an unrelated family), neither of
which shares lineage with the system's base or adapter. We report per-condition
retention for both and Cohen's κ between them. Judges see an anonymized response
and the fixed live-alternative list; they do not see condition identity.

## 3. Results

### 3.1 The Condition A–E Ladder (Keystone)

Figure 1 shows live-alternative retention across the five conditions under the
STRICT criterion for both judges, with per-condition inter-judge κ.

**Raw memory collapses plurality; governance recovers it.** Raw, ungoverned,
frame-confirming memory (C) drives retention to 1.4% (primary) / 2.8%
(independent) — roughly one-seventh of the no-personalization baseline A (9.7% /
15.3%). Provenance governance (D) recovers retention to baseline (11.1% / 18.1%).
The two judges agree strongly at both endpoints (κ = 0.66 at C, 0.50 at D). This
is the framework's crucial contrast (its §8.1) and its architectural claim (its
§10.1, §11.1): governing the *status* of memory, not its content volume,
preserves the plurality that raw memory destroys.

**Style-only personalization does not narrow.** Condition B (11.1% / 15.3%) is
indistinguishable from A, consistent with the framework's Principle 1 that style
adaptation is epistemically safe.

**The predicted ordering holds:** C ≪ A ≈ B ≈ D.

![Figure 1. Condition A–E live-alternative retention (STRICT criterion), two
judges, with per-condition inter-judge κ. Raw memory (C) collapses retention to
~1/7 of baseline; governance (D) recovers it. Dotted lines mark the
no-personalization baseline.](../experiments/figures/fig1_condition_ladder.png)

### 3.2 The Retention Criterion, and the Limits of Reflection's Benefit

The benefit of reflective *questioning* (E vs. D) depends on how retention is
defined. Under the STRICT criterion the primary judge scores E equal to D
(11.1%): the system reopens frames with questions but does not state alternative
content, so a strict content reading credits it no more than the governed answer
alone. Under a lenient criterion that counts frame-reopening questions, E rises
(to 40.3% for the independent judge).

A dedicated two-criterion measurement (Figure 2) makes this explicit and adds a
reliability test. Inter-judge agreement is **fair under STRICT (κ = 0.31) but
poor under LENIENT (κ = 0.09)**: whether a given question "reopens the frame
toward this specific alternative" is itself a subjective judgment, so the lenient
criterion that most favors reflective questioning is also the least reliable. We
therefore report STRICT as primary and characterize reflection's contribution
precisely as **frame reopening, not alternative stating** — which is exactly what
the framework's §11.2 claims ("sometimes the antidote is a better question"),
now shown to be a definitional commitment rather than a free empirical win.

![Figure 2. Live-alternative retention under STRICT vs. LENIENT criteria, both
judges (hatched = plain answer, solid = reflective questioning), with overall
inter-judge κ. RQA's edge appears only under the lenient criterion, which has
poor reliability.](../experiments/figures/fig2_retention_criterion.png)

### 3.3 Memory Echo Is Input-Stratified and Removed by Governance

We measured the framework's load-bearing mechanism, **memory echo**, directly.
With no memory injected, any memory-cross citation a model emits is by
definition fabricated. The trained system fabricates a self-citation on **0% of
declarative-statement turns but ~25% of identity/continuity turns** ("who are
you", "what did I conclude before") — the contexts the framework predicts are
most acute (its §5.4, §5.7). The output-side provenance check removes every such
citation, taking the input-prone stratum from 25% to 0% (Figure 3).

This also locates governance's effect precisely. At the retrieval level, the
content filter does **not** reduce the share of self-referential memory injected
(it withholds governance-vocabulary user claims, not assistant output); the
accumulated dyad graph in our runs was ~48% self-output and stayed so after
filtering. Governance's echo mitigation is therefore an **output-layer**
property — provenance labeling plus citation sanitization — not a reduction in
retrieved self-content. This refines, and slightly corrects, a naive reading of
the framework in which "context hygiene" would lower echo volume.

![Figure 3. Memory echo: share of turns emitting a fabricated self-citation,
ungoverned (C) vs. governed (D), by probe stratum. Input-stratified (0%
declarative, 25% identity/continuity); governance strips it to
zero.](../experiments/figures/fig3_memory_echo.png)

### 3.4 Cross-Family Robustness of the Direction

Re-scoring the same outputs with the independent judge confirms the qualitative
direction (reflection ≥ plain; governance ≫ raw) across model families, while
the per-alternative decisions disagree (overall κ near zero on the
plain-vs-reflection comparison, concentrated on question outputs). The
direction is robust; the fine-grained metric is judge-sensitive. Absolute rates
differ substantially between judges (the independent judge is systematically
more lenient), so only κ and within-judge orderings should be read as stable —
itself an instance of the framework's §14 caution about LLM-as-judge.

## 4. Discussion

The pilot supports the framework's **architectural** claim more strongly than
its **reflective-questioning** claim, and we think that is the honest and useful
outcome.

The keystone — raw memory collapses plurality, provenance governance recovers it
— is large, judge-robust, and directly tied to the framework's crucial contrast.
It shows that the relevant intervention is the **evidential status** assigned to
memory, not the amount of memory or the model's surface agreeableness: the same
frame-confirming records, relabeled as "assistant interpretation, not
independent evidence," stop narrowing the answer space. That is a falsifiable,
mechanism-level result, and it is the part of the framework most worth building
on.

The reflective-questioning result is genuinely qualified. A reopening question
preserves plurality only if one is willing to count "made an alternative
reachable" as retention, and adjudicating that is unreliable across judges. This
does not refute the framework's §11.2; it sharpens it into a definitional choice
that future work must make explicit before claiming reflection "preserves
plurality."

Finally, the two negative findings — that content filtering does not lower echo
volume, and that the plurality metric is judge-sensitive — are the kind of result
that makes a companion paper credible rather than promotional. We surfaced both
by trying to disconfirm our own earlier, more favorable single-judge reading.

## 5. Limitations

This is a pilot, and several limitations bound every number above.

- **Single annotator.** One author wrote the frames and live alternatives.
  Independent and multiple annotators, with measured inter-annotator agreement,
  are required before the retention rates can be trusted as construct-valid.
- **Controlled, non-naturalistic memory.** Conditions C/D use a frame-derived
  memory manipulation, not memory accumulated through real interaction. The
  keystone should be reproduced with naturalistic memory before it is generalized.
- **Small n.** Eighteen frames, 72 alternatives. The direction is consistent
  across two task-family batches and two judges, but confidence intervals are
  wide; we report orderings and κ, not precise effect sizes.
- **Judge effects.** Large leniency main-effects between judges mean absolute
  retention rates are not comparable across judges; only κ and within-judge
  orderings are.
- **One system.** Results are from a single governed reflective-questioning
  implementation. The framework's claims are architecture-general; this pilot is
  not.
- **Provenance of the metric itself.** LLM judges may share biases with the
  systems they score; the framework's §14 caution applies to this paper too.

## 6. Conclusion

A conceptual framework predicted that governing memory provenance, not just
tuning model agreeableness, preserves the answer-space plurality that
personalized assistants otherwise narrow. Run end-to-end on one governed system,
the prediction's keystone holds and is judge-robust: ungoverned frame-confirming
memory collapses live-alternative retention to about one-seventh of baseline, and
relabeling the same memory as non-independent evidence restores it. The benefit
of reflective questioning is real but criterion-dependent, and the metric needs
independent annotation before its magnitudes can be trusted. We offer this as a
first, deliberately modest, empirical foothold for the micro-echo chamber
framework — and as a record of the qualifications a persuasive version of that
empirical case will have to meet.

## Author Statements

**Funding.** No external funding is reported.
**Conflicts of interest.** The author is affiliated with MOBIUS LLC and develops
the governed reflective-questioning system used as the system under test. The
companion framework and this pilot treat that system as one severable candidate
architecture, not as established proof.
**Data and code availability.** The pre-registered frames and live alternatives,
the per-condition generations, the judge outputs, the analysis scripts, and the
figure-generation code are retained with the project and can be released with a
deposit version.
**Ethics.** No human subjects were involved. The frames include
mental-health-adjacent and interpersonal-advice topics only as text stimuli; any
future human study based on this design should include appropriate consent and
safeguards.
**License.** CC BY-NC-SA 4.0 unless superseded by a later release record.

## References

Toeda, T. (2026). *From Filter Bubbles to Micro-Echo Chambers: Preserving
Answer-Space Plurality in Personalized AI Assistants* (Version 1.0). MOBIUS LLC.
Zenodo DOI to be assigned.

(Background empirical literature — sycophancy, long-context mirroring, model
collapse, and RAG provenance — is cited in the companion framework paper and is
not repeated here; this pilot's claims rest on its own measurements.)

## Appendix A — Result Tables

**Table A1. Condition A–E live-alternative retention (STRICT criterion).**

| Condition | Primary judge | Independent judge | Inter-judge κ |
|---|---|---|---|
| A — no personalization | 9.7% | 15.3% | 0.37 |
| B — style-only | 11.1% | 15.3% | 0.34 |
| C — raw memory | 1.4% | 2.8% | 0.66 |
| D — governed memory | 11.1% | 18.1% | 0.50 |
| E — governed + reflection | 11.1% | 40.3% | 0.12 |

n = 18 frames, 72 live alternatives.

**Table A2. Retention criterion sensitivity (plain answer vs. reflective questioning).**

| Criterion | Primary plain | Primary RQA | Indep plain | Indep RQA | Overall κ |
|---|---|---|---|---|---|
| STRICT (content stated) | 8.3% | 6.9% | 22.2% | 40.3% | 0.31 |
| LENIENT (reopening counts) | 4.2% | 8.3% | 29.2% | 72.2% | 0.09 |

**Table A3. Memory echo — fabricated self-citation (no memory injected).**

| Probe stratum (n) | Ungoverned (C), turns with fabrication | Governed (D) |
|---|---|---|
| Declarative (12) | 0% | 0% |
| Identity / continuity (8) | 25% | 0% |
