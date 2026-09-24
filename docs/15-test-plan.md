# Test plan

51 cases across five files, all currently `pytest.skip("not implemented -- scaffold
only")`. Each skip is a written-down contract, not a placeholder.

**Testing philosophy for this repo.** The premise of the interview is *"we will ask you
to explain and modify your own code live."* Tests are the answer to "what happens if I
change this threshold?" — they should be readable as a specification of the policy, not
as coverage theatre. The policy ladder and the grounding validator get the deepest
coverage because they are pure Python with no LLM in them and are the parts a reviewer
will probe hardest.

---

## 1. What is tested and what is not

| Tested | Not tested |
|---|---|
| The policy ladder, exhaustively | LLM output quality — that is the eval harness's job, not pytest's |
| The grounding validator against real failure fixtures | Retrieval *relevance* — measured, not asserted |
| Every statistical function against hand-computed values | Third-party library behaviour |
| The output contract (schema + business rules) | Network behaviour — `make smoke` forbids network entirely |
| The reproduction path's offline guarantee | |

A failing eval number is a *finding*. A failing test is a *bug*. Keeping those separate
is why the judge and the classifier have no pass/fail thresholds in pytest.

---

## 2. `test_policy.py` — the ladder (18 cases)

One case per rung, plus boundary cases, plus five invariants.

**Per rung:** each of the seven rungs has at least one test that fires it and, where a
threshold is involved, one that sits just below the boundary and does *not* fire it.
`test_amount_below_ceiling_does_not_fire` uses exactly
`config.thresholds.auto_refund_ceiling_usd` — off-by-one on a money threshold is the
kind of bug that survives review.

**The five invariants** are the ones worth reading aloud in an interview:

1. `test_guardrails_never_authorize_automation` — **a property test over randomised
   contexts.** No hit on rungs 1–6 may ever yield `AUTO_SEND`. This is the single
   invariant that makes the layered design better than one LLM call, so it is tested as a
   property rather than by example.
2. `test_short_circuits_on_first_hit` — a context matching rungs 1 and 4 reports
   `safety_veto`, not `confidence_gate`. Order is semantic, not incidental.
3. `test_every_decision_records_its_layer` — `policy_layer_fired` is never null, on any
   path including `default`. Without this the audit trail has holes exactly where
   nothing interesting happened, which is where you look first when something is wrong.
4. `test_every_escalation_code_has_a_route` — every `EscalationCode` member is a key in
   `config.policy.routes`. Catches the config/code drift that produces a decision nobody
   can act on.
5. `test_escalation_emits_both_machine_and_human_reason` — the code is what dashboards
   aggregate; the text is what the human reads in three seconds. One without the other
   is half a feature.

`test_no_confidence_overrides_safety` sets `intent_confidence = 1.0` and asserts the
safety veto still wins. `test_threshold_is_read_from_config_not_hardcoded` monkeypatches
the config and asserts the behaviour changes — a hardcoded threshold passes every other
test and fails this one.

---

## 3. `test_grounding.py` — invented policy (12 cases)

Every fixture is a **real, predicted failure**, not a hypothetical.

**`test_invented_refund_window_hard_fails`** is the Air Canada case as a unit test: a
draft asserting a customer may claim retroactively within 90 days, with no supporting
thread in evidence. *Moffatt v. Air Canada* (2024 BCCRT 149) — negligent
misrepresentation, CAD $812.02, and the tribunal explicitly rejected the argument that
the chatbot was "a separate legal entity responsible for its own actions." One invented
policy sentence, one legally binding obligation. The test encodes that.

**`test_hallucinated_agent_sigil_fails`** encodes predicted failure mode #3 — the
generator emitting invented `^AB` initials, observed in a real run on this corpus. Sigils
are stripped at ingest precisely so the model never sees one; emitting one anyway means
it came from pretraining, and it must fail.

**`test_hallucinated_url_fails`** — a support URL absent from the allowlist observed in
that brand's own replies. A classic and highly visible failure on a *public* channel.

**`test_empty_slots_is_the_common_case_and_passes`** exists to prevent the validator
becoming a blanket veto. A genuinely helpful reply that promises nothing must pass
cleanly. A validator that fails everything is as useless as one that passes everything,
and only this test distinguishes them.

---

## 4. `test_schemas.py` — the output contract (6 cases)

`test_action_enum_is_closed` asserts exactly four members. This is not pedantry: the
constrained action space is the core injection defence — the model cannot "issue a
refund" because no such action exists to emit. Adding a fifth action is a security
change and should fail a test until someone means it.

The rest pin the business rules the JSON schema cannot express: non-empty
`evidence_ids`; `escalation_reason` present iff `action == "escalate"`; the 280-char cap;
intent within the frozen taxonomy; and `test_repair_retry_is_bounded_and_logged`, which
asserts both the cap **and** that every repair reaches the decision record. An unbounded
self-repair loop is a cost bomb; an unlogged one is worse, because it looks like it never
happened.

---

## 5. `test_eval.py` — the measurement apparatus (12 cases)

Statistical functions are tested against **hand-computed values from `docs/14`**, not
against themselves.

- `TestWilson` pins all three worked examples. `test_headline_slice_width` specifically
  pins **p̂=0.78, n=100 → [0.689, 0.850], ±8pp** — the correction from `docs/10 §6.3`. If
  someone later "simplifies" the headline onto all 200, this test fails, which is the
  point.
- `TestMcNemar::test_significance_boundary` pins both sides of the boundary: b=27,c=13 →
  χ²=4.23 (significant) and b=26,c=14 → 3.60 (not). A continuity correction dropped in
  refactoring moves both.
- `TestKappa::test_kappa_paradox_worked_example` pins the two 2×2 tables that share 95%
  raw agreement and differ by 0.43 in κ. This is a test *about the metric's behaviour*,
  and it exists so nobody later "fixes" the deflated κ by changing how it is computed.
- `TestHeadlineDiscipline::test_headline_over_all_200_raises` is the mechanical
  enforcement of the rule that matters most. Discipline erodes at 2 a.m. on day 5; a
  raise does not.
- `TestBootstrap::test_zero_support_convention_is_applied` asserts the declared
  convention **and** that the two conventions give demonstrably different intervals —
  proving the declaration was load-bearing rather than decorative.

---

## 6. `test_smoke.py` — the reproduction guarantee (7 cases)

This is the file that runs in CI on every commit, and the one that catches a broken
reproduction before a reviewer does.

- **`test_smoke_runs_offline`** monkeypatches `socket.socket` to raise. Asserted, not
  trusted. "It doesn't need the network" is a claim, and claims get tested.
- `test_smoke_completes_under_30s` and `test_smoke_needs_no_api_key` pin the other two
  halves of the `make smoke` contract.
- **`test_cache_miss_without_key_fails_loudly`** asserts `CacheMissWithoutKey` is raised.
  This is the most important test in the file: a silent skip on a cache miss turns a
  broken reproduction into a passing one, and it is undetectable from the outside. The
  failure mode this prevents is a reviewer seeing a green run that computed nothing.
- **`test_tier1_eval_needs_no_tweet_text`** runs `make eval` with `data/raw/` absent and
  asserts the full headline table is still produced. This is the two-tier contract from
  `docs/10 §5`, tested rather than asserted in prose.
- **`test_golden_set_content_hash_is_stable`** pins the sha256 of the frozen random
  stratum. A changed hash means the slice moved, which invalidates every number computed
  against it. Silent movement of the eval set is the single most dangerous undetected
  change in the repo.
- `test_config_thresholds_are_all_referenced` walks `config.thresholds` and asserts each
  key is read somewhere in `src/`. Dead policy is worse than no policy: it looks
  configurable and is not.

---

## 7. CI

```yaml
# .github/workflows/ci.yml
- make setup
- make test          # 51 cases, no network, no key
- make smoke         # <30s, 50 rows
- make eval          # Tier 1: regenerates artifacts/results/tables/*.md
- git diff --exit-code artifacts/results/tables/   # a changed number must be intentional
```

The last line is the interesting one: **the headline tables are committed, so a change in
any number shows up as a diff on the pull request.** A regression is visible on every
commit rather than discovered on submission day, and an *intentional* improvement arrives
with the number change in the same diff as the code change that caused it.

---

## 8. What would make this test suite dishonest

Worth stating, since the repo is partly an argument about measurement integrity:

- Asserting a **minimum accuracy** anywhere in pytest. That converts a measurement into a
  target and invites tuning against it.
- Fixtures drawn from the golden set. Test fixtures are hand-written or drawn from
  outside the 200.
- Mocking the grounding validator in policy tests. Rung 6 takes a boolean input; the
  validator's own behaviour is tested in `test_grounding.py` and nowhere else.
