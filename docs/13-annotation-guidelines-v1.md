# Annotation guidelines — v1

**Status: v1, pre-pilot.** This document is versioned deliberately. The v1 → v2 diff,
produced by the pilot round in §7, is itself evidence of rigour and ships with the
submission.

**Who this is for:** the person labelling `data/golden_200.csv`. Read it end to end
before labelling anything. If a decision is not derivable from this document, that is a
defect in the document — record it in the adjudication log (§8) and fix the document,
do not resolve it from memory.

> Research on annotation found that detailed instructions alone do not *guarantee*
> consistency. But undocumented criteria *guarantee* inconsistency and make the taxonomy
> unfalsifiable. This document exists so the labels are falsifiable.

---

## 1. Preconditions

Do not begin until all three hold:

1. `taxonomy/intents.yaml` has `status: frozen` and `frozen_at` set.
2. `config.yaml` has `taxonomy.frozen: true`.
3. The pilot round (§7) is complete and those pilot labels have been **discarded**.

Editing the taxonomy after labelling begins invalidates the golden set. If the taxonomy
must change, the labels must be redone.

---

## 2. The unit of annotation

One **first-turn inbound tweet** — the customer's opening message in a conversation.
Not the thread, not a mid-thread fragment.

You may read the rest of the thread for context, and for the
`quality_of_historical_reply` field you must. But **the intent and the
auto-handle/escalate decision are judged on the opening tweet alone**, because that is
all the system sees at inference time. Labelling with knowledge of how it turned out is
leakage, and it inflates every number downstream.

---

## 3. The precedence rule — single label, always

Real tweets carry two intents: *"you charged me twice, cancel my account."* We label a
single intent, with a documented precedence rule, rather than going multi-label. That
choice is defensible and cheaper; what makes it honest is that we **measure the
ambiguity** rather than hiding it (§9).

> **Label the action the customer is asking the company to take.**
> If two actions are requested, label the one that **determines routing**.
> If both determine the same routing, label the one **mentioned first**.

Worked examples:

| Tweet | Label | Why |
|---|---|---|
| "you charged me twice, cancel my account" | `billing_charges` | Two actions. The double charge routes to billing; the cancellation routes to retention. Billing is the one a first-line agent cannot resolve, so it determines routing. |
| "app won't open, and how do I change my email?" | `service_outage_technical` | Two actions, different routing. Neither dominates → first mentioned. |
| "cancel my subscription and refund last month" | `refund_return` | Both route to billing. Refund is the harder half and is mentioned second — **but** routing is identical, so the rule says first-mentioned: `cancel_downgrade`. ⚠ **This example is deliberately contradictory and must be resolved in the pilot.** See §10. |
| "thanks! also is there a family plan?" | `product_howto_question` | Thanks is not an action. The question is. |

**Always set `ambiguous: true`** when more than one intent genuinely applied, even
though you picked one. This field is what makes §9 possible.

---

## 4. The decision tree

Walk it in order. Stop at the first match.

```
1. Is this a support request at all?
   NO  → non_support        (praise, thanks-only, jokes, contests, marketing, bots,
                             random @-mentions, reply-guys)
   Note: "thanks!" alone is non_support. "thanks! also <question>" is the question.

2. Is the customer reporting that their account has been accessed by someone else,
   or that they cannot get in?
   YES → account_access_security
   (This rung is above everything else on purpose: a compromised account is never
    auto-handled, at any confidence, and mislabelling it downstream hides that.)

3. Is the customer expressing dissatisfaction, demanding escalation, threatening to
   leave, or invoking legal/regulatory language AS THE PRIMARY ACT?
   YES → complaint_escalation
   NO, they're annoyed but still asking for a specific fix → continue.

4. Is something the customer is entitled to use not working?
   YES → service_outage_technical

5. Is the customer asking where a paid-for thing is, or when it arrives?
   YES → order_delivery_status

6. Does the customer want money back, or to return/exchange?
   YES → refund_return

7. Does the customer dispute, question, or not recognise a charge?
   YES → billing_charges

8. Does the customer want to end or reduce a service going forward?
   YES → cancel_downgrade

9. Does the customer want to change stored details on a working account?
   YES → change_of_details

10. Is the customer asking how to do something, whether something is possible,
    or what a policy is?
    YES → product_howto_question

11. None of the above, but it IS a support request.
    → other_unclear
```

**`non_support` vs `other_unclear` are not the same and must never be merged.**
`non_support` is a **filter** decision — drop the message, no reply.
`other_unclear` is an **escalation** decision — a human should look at this.
Conflating them collapses two different downstream behaviours into one label.

---

## 5. Per-intent reference

Fill `positive` (≥2 verbatim examples) and `not_this` in `taxonomy/intents.yaml` during
freezing; this section mirrors them. Boundary cases are the whole value here.

**`order_delivery_status`** — the thing exists; the question is location or timing.
*Not this:* asking if an item is in stock (→ `product_howto_question`); asking for money
back because it never came (→ `refund_return`).

**`billing_charges`** — disputes, questions, or does not recognise a charge; pricing and
payment mechanics on an existing account.
*Not this:* asking for the money back (→ `refund_return`); asking to stop future charges
(→ `cancel_downgrade`).

**`refund_return`** — wants money back, or to return/exchange, or asks how.
*Not this:* asking hypothetically how long the return window is (→
`product_howto_question` — **this is a policy trap**, see §6).

**`account_access_security`** — cannot get in, or reports compromise.
*Not this:* wants to change an email on a working account (→ `change_of_details`).

**`cancel_downgrade`** — end or reduce a service going forward.
*Not this:* wants money back for what was already paid (→ `refund_return`).

**`service_outage_technical`** — something that should work does not. Crash, playback
failure, error code, outage. **The brand's historical answer here is a remediation step
that is still true today** — this is the crisp auto-handle class and the reason
SpotifyCares was chosen.
*Not this:* asking how to do something that does work (→ `product_howto_question`).

**`product_howto_question`** — how, whether, or what-is-the-policy. Nothing is broken.

**`change_of_details`** — change stored info on a working account.
*Not this:* cannot log in to make the change (→ `account_access_security`).

**`complaint_escalation`** — the ask is to be heard or escalated, not a transaction.
*Not this:* annoyed, but the primary ask is still a specific fix.

**`non_support`** — filter. Large class in reality and lexically obvious, which makes it
a magnet for probability mass. Over-prediction of this class is a **predicted failure
mode** and should be expected in the confusion matrix.

**`other_unclear`** — escalate. **Coverage check: if >15–20% of the random 100 lands
here, a class is missing.** That is a signal to revisit the taxonomy, not to keep
labelling.

---

## 6. Labelling `auto_handle_vs_escalate`

This is **your** judgement of what *should* happen, made independently of what the model
does and independently of what the brand historically did. It is the ground truth the
policy ladder is measured against.

Mark **`escalate`** whenever any of these hold:

- Self-harm, threats, or abuse.
- Legal or regulatory language: lawyer, sue, GDPR, chargeback, ombudsman, small claims.
- Account compromise or suspected fraud.
- A specific money amount above $50 is in play.
- PII is already exposed in the public tweet.
- You cannot tell what the customer wants (`other_unclear`).
- A correct reply would require information only the brand's private systems hold.

Mark **`auto_handle`** when a competent first-line agent could resolve it from public,
generally-true information, with no account access.

Two notes:

1. **`dm_handoff` is not an escalation.** Moving to DM is the *correct resolution path*
   whenever the next step needs an order number, an email address, or account access —
   i.e. exactly when PII would otherwise be posted publicly. Label the underlying
   decision: if a human first-liner could handle it *in DM*, that is `auto_handle`.
2. This is a binary field. The four-way action space is the model's output, not your
   label.

**Expect this field to be heavily skewed toward `auto_handle`.** That is real, and it is
why §9 reports PABAK and Gwet's AC1 alongside κ.

---

## 7. The pilot protocol — do this first

1. Draw 30 examples at random, **from outside the 200**.
2. Label them against v1 of this document.
3. Every time you hesitate, or reach for knowledge not in this document, write it down.
4. Revise this document into **v2**, resolving every recorded hesitation with an explicit
   rule.
5. **Discard all 30 pilot labels.** They were produced under a document that no longer
   exists and are contaminated by the act of writing v2.
6. Commit v1 and v2. **Ship the diff.** It is evidence.

Then label all 200 against v2, recording `guideline_version: v2` on every row.

---

## 8. The adjudication log

`reports/adjudication_log.md`. For every hard case:

```
### g0137
Tweet (paraphrased, no verbatim text committed): double charge + cancellation request
Competing readings: billing_charges / cancel_downgrade
Rule applied: precedence §3 — billing determines routing
Rule added to guidelines? yes → v2 §3 worked examples row 1
```

A rule that gets applied twice belongs in the guidelines, not the log. Fold it back.

---

## 9. Measuring label quality with exactly one annotator

You cannot compute inter-annotator agreement alone. You can produce credible evidence.

**Intra-annotator (test–retest) agreement — the solo annotator's substitute for IAA.**

Protocol:
1. Wait **≥48 hours** after finishing the 200.
2. Draw a random 40. Shuffle them.
3. Re-label **blind** — no access to the first pass.
4. Compute κ_intra for `intent` and for `auto_handle_vs_escalate` separately.
5. Write both to `reports/kappa_intra.json`.

**κ_intra is the measurement ceiling.** If κ_intra = 0.65, no downstream judge can
credibly exceed that, and **saying so is one of the strongest moves available in the
whole report**. Report it before, not after, the judge's number.

Caveat it honestly: test–retest has its own biases — fatigue, and memory of the first
pass. Memory inflates it; a second annotator would lower it. State the direction.

**Report every metric twice**: over all 200, and over the
`confidence == high AND ambiguous == false` subset. **If the gap is large, label noise
is doing real work and you must say so.**

**The ambiguity rate** — the share of rows with `ambiguous: true` — is reported as an
upper bound on achievable single-label accuracy. This is what makes the single-label
choice honest rather than merely convenient.

---

## 10. Known defects in v1

Recorded here so the pilot addresses them rather than papering over them.

- **§3 worked example 3 is self-contradictory.** "cancel my subscription and refund last
  month": the commentary argues for `refund_return`, the rule yields `cancel_downgrade`.
  This is a genuine gap — the precedence rule has no tie-break for *same routing,
  different difficulty*. The pilot must add one and pick a side. Leaving it ambiguous
  would make every multi-intent billing row unreliable.
- **`complaint_escalation` overlaps everything.** Rung 3 of the decision tree uses "as
  the primary act", which is a judgement call. Expect this class to be the largest
  contributor to κ_intra loss. If it is, the honest response is to report it, not to
  redefine the class after seeing the disagreement.
- **The `auto_handle` skew** is expected to be severe enough that κ alone will look bad
  regardless of actual agreement. §9 pre-empts this; do not "fix" it by rebalancing the
  sample, which would break the random stratum.
