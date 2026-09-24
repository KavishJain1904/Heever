# Implementation findings

What building `src/` surfaced that the specs did not anticipate. Same role `docs/10 §6`
plays for the research: the specs are the source of truth until something in the build
contradicts them, and then the contradiction is written down rather than silently
resolved in code.

Three contract defects and one verification gap. The verification gap (§4) is the one
that matters most and is the one a reviewer should read first.

---

## 1. `escalation_reason present iff escalate` did not hold as specified

`docs/11 §5` and `docs/12 §6` both give this as a Pydantic `@field_validator` on
`escalation_reason`. Implemented that way, it enforces **one direction only**.

Pydantic v2 does not run a field validator for an optional field that was never
supplied. `escalation_reason` defaults to `None`, so:

```python
Decision(action=Action.ESCALATE, ...)      # no escalation_reason key at all
```

constructs **successfully**. The validator never fires. The check catches a reason
attached to an `AUTO_SEND`, and misses an escalation with no reason at all.

That is the wrong direction to fail in. The brief asks for escalation *"with a stated
reason"*; the omitted-field case **is** how that requirement gets violated in practice,
and it was the case the specified implementation could not see. It was caught by
`tests/test_schemas.py::test_escalation_reason_present_iff_escalate`, which asserted
both directions and failed on the first.

**Correction.** The rule is a `@model_validator(mode="after")`, which runs on every
constructed `Decision` regardless of which fields were supplied. `schemas.py` carries
the reasoning inline.

**Generalisation worth carrying:** any "field A present iff field B has value X" rule in
`docs/11` has this shape. `escalation_reason` is the only one currently specified, but a
future `route_to`/`priority` invariant would need the same treatment.

---

## 2. The two bootstrap zero-support conventions cannot be compared as specified

`docs/14 §4` declares `bootstrap_zero_support: drop_class` and justifies the declaration
by asserting the alternative *"gives materially different intervals"*. `evaluate.py`
computes both so the declaration is checkable rather than merely stated.

Implemented naively, **both branches produce identical numbers**, and the comparison is
vacuous.

The reason: `macro_f1` derives its label set from the data it is handed. In a bootstrap
replicate where a rare class is absent from both `y_true` and `y_pred`, that class is not
in the derived label set at all — so it is not averaged over, which *is* `drop_class`.
The `score_zero` branch, written the same way, never sees the class either, and therefore
cannot score it `0`. Two different conventions, one behaviour.

**Correction.** `score_zero` must be evaluated against the **full label universe**, fixed
once outside the replicate loop, so an absent class is still present to be scored zero.
`bootstrap_ci_zero_support_variant` does this;
`tests/test_eval.py::test_zero_support_convention_is_applied` asserts the two intervals
**differ**, which is the only form of that test worth having.

**Why this is worth a doc entry rather than a code comment:** the failure is silent and
self-confirming. The harness would have printed two intervals, they would have agreed,
and the natural reading is *"the convention does not matter much here"* — a conclusion
about the data drawn from a bug in the harness. This is exactly the class of error
`docs/14`'s opening line is about.

---

## 3. `non_support` has no representable action

`taxonomy/intents.yaml` gives `non_support` a `default_action: filter` and describes it
as a FILTER decision — drop, do not reply. `scope_classes.filter` lists it separately
from `escalate` for that reason, and `docs/03 §2` is explicit that merging the two is a
mistake.

But `schemas.Action` is a closed four-value enum — `auto_send`, `request_info`,
`dm_handoff`, `escalate` — and `docs/12 §7` rung 7 says the default rung returns
*"`AUTO_SEND`, or `DM_HANDOFF` per the intent's playbook"*. There is no `FILTER` member,
and adding one would widen the action space that `docs/12 §5` names as the core
constrained-action-space injection defence.

So `apply_ladder` has no correct return value for a `non_support` message. Three options
were available:

| Option | Why not |
|---|---|
| Add `Action.FILTER` | Widens the action space the injection defence depends on being narrow |
| Map `filter` → `ESCALATE` | Sends every "thanks!" and every contest reply to a human queue — the predicted-failure-#2 class is *large* |
| Map `filter` → `AUTO_SEND` with an empty draft | An action that claims we replied when we did not; corrupts coverage |

**Resolution taken.** Filter-class intents are dropped in `pipeline.process_one`
**before** the ladder runs, and `policy._playbook_action` **raises** if one reaches it.
The drop still emits a `DecisionRecord` so it is auditable and counts toward coverage
denominators. The raise is deliberate: silently choosing an action for a message we
already decided not to reply to would be precisely the kind of quiet contract violation
the layered design exists to prevent.

**This is a resolution, not a ruling.** It is the one place where the implementation
chose between two defensible readings of the spec rather than implementing it, and if the
intended reading was different, `pipeline.FILTER_INTENTS` and
`policy._playbook_action` are the two places to change. Flagged here so the choice is
visible rather than buried.

---

## 4. What is implemented but **not verified**

The distinction that matters for reading any of this. `make test` passing means the
deterministic core behaves as specified. It does not mean the system works, because most
of it has never seen a tweet.

**Exercised by tests, runs today:**

| Module | Coverage |
|---|---|
| `policy.py` | one test per rung, boundary cases, 2,000-context property test on the never-authorize invariant |
| `grounding.py` | the Air Canada fixture, hallucinated URL, invented sigil, the customer-demand-as-evidence case |
| `evaluate.py` | every worked example in `docs/14` reproduced to 3 dp |
| `schemas.py`, `guards.py`, `cache.py` | contract and boundary tests |

**Written against the contracts, never executed against real data:**

`build_sample.py` · `featurize.py` · `retrieve.py` · `generate.py` · `judge.py` ·
`distill.py` · `induce_taxonomy.py` · `baselines/{tfidf,embed_lr,setfit,llm}.py`

Treat these as **untested** until `make eval-full` has run once. `build_sample.build_threads`
is the highest-risk of them: branch policy, split-reply reordering and the dual-identity
recovery in `docs/12 §1` are all the kind of logic that is wrong in ways only the real
`twcs.csv` reveals, and all three are silent when wrong. The four bugs `docs/01 §2-3`
catalogues are guarded against in code; whether the guards are correct is unknown.

**Nothing in this repo has produced a result.** No headline, no confusion matrix, no
failure analysis, no judge κ. The harness will compute all of them once
`data/golden_200.csv` exists, and not before.

---

## 5. Test count

`docs/15` plans 51 cases. The suite has **82**. The surplus is not scope creep — it is
the boundary and invariant cases the specs imply but do not enumerate: threshold
boundaries on both sides (`$50` does not fire, `$51` does), word-boundary regression on
the legal markers (`"issue"` must not match `"sue"`), `grounding_passed is None`
distinguished from `False`, cache-key forgeability under the NUL separator, and the
three findings above.

Two structural tests worth naming, because they enforce claims `docs/14` makes in prose:

- `test_config_thresholds_are_all_referenced` — every key under `config.thresholds` is
  read by some module. Config-as-policy is a stated demonstration; a threshold no code
  reads is a claim the repo does not honour.
- `test_no_module_still_raises_not_implemented` — the scaffold is fully implemented, so
  a reviewer does not discover a stub at runtime.
