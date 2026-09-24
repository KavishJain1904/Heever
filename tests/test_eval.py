"""Eval-harness correctness. Plan: docs/15 §5.

These test the MEASUREMENT APPARATUS, which on this assignment is the deliverable.
A bug here is worse than a bug in the agent: it produces a confident wrong number.
"""
import pytest

from src.evaluate import (
    HeadlineDisciplineError, bootstrap_ci, bootstrap_ci_zero_support_variant,
    cohens_kappa, holm_bonferroni, macro_f1, mcnemar, proportion,
    require_headline_slice, rule_of_three_upper, wilson_interval,
    _reply_quality_table, _judge_summary_document,
)


def _make_verdict(message_id, system, addresses_pass, grounded_pass, addresses_flip=False):
    return {
        "message_id": message_id,
        "system": system,
        "position_bias_disagreements": (
            ["addresses_stated_problem"] if addresses_flip else []
        ),
        "criteria": {
            "addresses_stated_problem": {
                "pass": addresses_pass, "order_disagreement": addresses_flip,
            },
            "grounded_in_retrieved_context": {
                "pass": grounded_pass, "order_disagreement": False,
            },
        },
    }


class TestWilson:
    def test_worked_example_a(self):
        """p=0.85, n=200 -> [0.794, 0.893]. Width ~10pp."""
        lo, hi = wilson_interval(170, 200)
        assert (round(lo, 3), round(hi, 3)) == (0.794, 0.893)

    def test_worked_example_c_perfect_slice(self):
        """p=1.0, n=25 -> [0.867, 1.0]. Wald's [1.0, 1.0] is the bug being avoided."""
        lo, hi = wilson_interval(25, 25)
        assert round(lo, 3) == 0.867
        assert hi == 1.0
        assert lo < 1.0, "a perfect slice at n=25 must not claim certainty"

    def test_headline_slice_width(self):
        """p=0.78, n=100 -> [0.689, 0.850]. The headline CI is ~+/-8pp, NOT +/-5pp:
        the +/-5pp figure is n=200 and the headline lives on the n=100 random slice."""
        lo, hi = wilson_interval(78, 100)
        assert (round(lo, 3), round(hi, 3)) == (0.689, 0.850)
        assert 15 < (hi - lo) * 100 < 17, "headline interval is ~16pp wide, i.e. +/-8pp"

        wider = wilson_interval(170, 200)
        assert (hi - lo) > (wider[1] - wider[0]), (
            "the n=100 headline slice MUST be wider than the n=200 set -- quoting "
            "the n=200 width on the headline understates our own uncertainty"
        )

    def test_per_intent_slice_is_nearly_uninformative(self):
        """p=0.84, n=25 -> ~[0.654, 0.936]. Per-class cells carry ~14pp intervals."""
        lo, hi = wilson_interval(21, 25)
        assert (hi - lo) * 100 > 25

    def test_rule_of_three_matches_wilson_at_zero(self):
        assert round(rule_of_three_upper(100), 3) == 0.030
        assert wilson_interval(0, 200)[1] < 0.03


class TestMcNemar:
    def test_significance_boundary(self):
        """b=27,c=13 -> chi2=4.23 (sig); b=26,c=14 -> chi2=3.02 (not sig).

        The boundary at 40 discordant pairs is |b-c| >= 14, NOT 13: solving
        (|b-c|-1)^2/40 > 3.841 gives |b-c| > 13.4. Minimum detectable paired
        difference is 7.0pp. docs/04 quotes 13 / 6.5pp / chi2=3.60 -- the last
        is the UNCORRECTED statistic. See docs/10 §6.4.
        """
        chi2_sig, _ = mcnemar(27, 13)
        chi2_not, _ = mcnemar(26, 14)
        assert chi2_sig == pytest.approx(4.225, abs=5e-3)
        assert chi2_not == pytest.approx(3.025, abs=5e-3)
        assert chi2_sig > 3.841 > chi2_not

    def test_boundary_is_14_not_13(self):
        """docs/04's own worked example contradicts its stated threshold of 13."""
        assert mcnemar(26, 13)[0] < 3.841   # |b-c| = 13 -> not significant
        assert mcnemar(27, 13)[0] > 3.841   # |b-c| = 14 -> significant

    def test_no_discordant_pairs_is_not_nan(self):
        chi2, p = mcnemar(0, 0)
        assert chi2 == 0.0 and p == 1.0


class TestHeadlineDiscipline:
    def test_headline_over_all_200_raises(self):
        """The harness must MECHANICALLY refuse a headline computed off anything but
        stratum=random. Enforced in code so it cannot be forgotten under deadline."""
        rows = [{"stratum": "random"}] * 100 + [{"stratum": "adversarial"}] * 25
        with pytest.raises(HeadlineDisciplineError):
            require_headline_slice(rows)

    def test_random_only_is_accepted(self):
        assert len(require_headline_slice([{"stratum": "random"}] * 3)) == 3

    def test_every_proportion_carries_a_ci(self):
        result = proportion(78, 100, "all_six_pass")
        assert {"rate", "ci_low", "ci_high", "ci_width_pp"} <= set(result)
        assert result["ci_low"] < result["rate"] < result["ci_high"]

    def test_degenerate_dm_us_baseline_in_every_reply_table(self):
        verdicts = [{
            "system": "ours",
            "criteria": {"addresses_stated_problem": {"pass": True}},
        }]
        with pytest.raises(ValueError, match="dm_us_degenerate"):
            _reply_quality_table(verdicts)


class TestReplyQualityPairing:
    """_reply_quality_table's paired McNemar extension and the judge_summary.json
    shape built from it -- the fix for the untraceable report numbers."""

    def _verdicts(self):
        verdicts = []
        # 3 message_ids, all three systems. heever always passes both criteria;
        # historical and dm_us_degenerate never do -- an extreme, easily hand-
        # checked discordant pattern.
        for mid in ("m1", "m2", "m3"):
            verdicts.append(_make_verdict(mid, "heever", True, True))
            verdicts.append(_make_verdict(mid, "historical", False, False, addresses_flip=True))
            verdicts.append(_make_verdict(mid, "dm_us_degenerate", False, False))
        return verdicts

    def test_paired_comparisons_use_message_id_pairing(self):
        table = _reply_quality_table(self._verdicts())
        paired = table["paired"]
        assert paired["n_paired"] == 3
        comp = paired["criteria"]["addresses_stated_problem"]["heever_vs_historical"]
        # heever right, historical wrong on all 3 paired ids -> b=3, c=0.
        assert comp["b"] == 3 and comp["c"] == 0

    def test_order_flip_counted_from_order_disagreement(self):
        table = _reply_quality_table(self._verdicts())
        assert table["historical"]["per_criterion"]["addresses_stated_problem"]["order_flip"] == 3
        assert table["heever"]["per_criterion"]["addresses_stated_problem"]["order_flip"] == 0

    def test_composite_has_no_order_flip_field(self):
        table = _reply_quality_table(self._verdicts())
        assert "order_flip" not in table["heever"]["all_six_pass"]

    def test_pairing_drops_ids_missing_from_any_system(self):
        verdicts = self._verdicts()
        verdicts.append(_make_verdict("m4", "heever", True, True))  # no counterpart
        table = _reply_quality_table(verdicts)
        assert table["paired"]["n_paired"] == 3

    def test_judge_summary_document_matches_committed_shape(self):
        table = _reply_quality_table(self._verdicts())
        doc = _judge_summary_document(table)
        assert doc["n_paired"] == 3
        entry = doc["criteria"]["addresses_stated_problem"]
        assert set(entry) == {
            "heever", "historical", "dm_us_degenerate",
            "heever_vs_historical", "heever_vs_dm_us_degenerate",
        }
        assert entry["heever"]["pass"] == 3
        assert entry["heever"]["wilson"][0] < entry["heever"]["rate"] <= entry["heever"]["wilson"][1]
        assert entry["heever_vs_historical"]["b"] == 3


class TestBootstrap:
    def test_zero_support_convention_is_applied(self):
        """config.evaluation.bootstrap_zero_support == drop_class, and the two
        conventions demonstrably give different intervals."""
        from src import load_config

        assert load_config().evaluation["bootstrap_zero_support"] == "drop_class"

        y_true = ["a"] * 40 + ["b"] * 8 + ["c"] * 2
        y_pred = ["a"] * 40 + ["b"] * 6 + ["a"] * 2 + ["c"] * 1 + ["a"] * 1
        variants = bootstrap_ci_zero_support_variant(y_true, y_pred, b=400, seed=42)
        assert variants["drop_class"] != variants["score_zero"], (
            "if the two conventions agree the declaration is meaningless"
        )

    def test_is_seeded_and_reproducible(self):
        y_true = ["a", "b", "c"] * 20
        y_pred = ["a", "b", "a"] * 20
        assert (
            bootstrap_ci(macro_f1, y_true, y_pred, b=300, seed=42)
            == bootstrap_ci(macro_f1, y_true, y_pred, b=300, seed=42)
        )
        assert (
            bootstrap_ci(macro_f1, y_true, y_pred, b=300, seed=42)
            != bootstrap_ci(macro_f1, y_true, y_pred, b=300, seed=7)
        )


class TestMacroF1:
    def test_formula_is_arithmetic_mean_of_per_class_f1(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        # a: P=1.0 R=0.5 F1=0.667 | b: P=0.667 R=1.0 F1=0.8 -> mean 0.7333
        assert round(macro_f1(y_true, y_pred), 4) == 0.7333

    def test_majority_predictor_has_near_zero_macro_f1(self):
        """The contrast that makes macro-F1 the headline rather than accuracy."""
        from src.baselines.trivial import majority_class

        y_true = ["a"] * 60 + ["b"] * 20 + ["c"] * 20
        y_pred = majority_class(y_true, y_true)
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        assert accuracy == 0.6
        assert macro_f1(y_true, y_pred) < 0.3

    def test_reported_twice_all_and_in_scope(self):
        y_true = ["billing_charges"] * 10 + ["other_unclear"] * 10
        y_pred = ["billing_charges"] * 10 + ["billing_charges"] * 10
        assert macro_f1(y_true, y_pred, in_scope_only=True) > macro_f1(y_true, y_pred)


class TestKappa:
    def test_kappa_paradox_worked_example(self):
        """Identical 95% raw agreement: skewed -> 0.47, balanced -> 0.90."""
        skewed = cohens_kappa(
            [True] * 10 + [False] * 190,
            [True] * 5 + [False] * 5 + [True] * 5 + [False] * 185,
        )
        balanced = cohens_kappa(
            [True] * 100 + [False] * 100,
            [True] * 95 + [False] * 5 + [True] * 5 + [False] * 95,
        )
        assert round(skewed["p0"], 2) == round(balanced["p0"], 2) == 0.95
        assert round(skewed["kappa"], 2) == 0.47
        assert round(balanced["kappa"], 2) == 0.90
        assert round(balanced["kappa"] - skewed["kappa"], 2) == 0.43

    def test_returns_p0_pe_table_pabak_and_ac1(self):
        result = cohens_kappa([True, False, True], [True, False, False])
        assert {"p0", "pe", "kappa", "table", "pabak", "ac1", "band"} <= set(result)
        assert set(result["table"]) == {"a_both_yes", "b_r1_only", "c_r2_only", "d_both_no"}

    def test_ac1_is_robust_to_the_base_rate_that_deflates_kappa(self):
        """Why all six numbers ship together, not kappa alone."""
        skewed = cohens_kappa(
            [True] * 10 + [False] * 190,
            [True] * 5 + [False] * 5 + [True] * 5 + [False] * 185,
        )
        assert skewed["ac1"] > skewed["kappa"] + 0.4
        assert round(skewed["pabak"], 2) == 0.90

    def test_bands_are_krippendorff_not_landis_koch(self):
        assert cohens_kappa([True] * 9 + [False], [True] * 9 + [False])["band"] == "reliable"
        assert "unreliable" in cohens_kappa(
            [True] * 10 + [False] * 190,
            [True] * 5 + [False] * 5 + [True] * 5 + [False] * 185,
        )["band"]


class TestHolmBonferroni:
    def test_step_down_stops_at_first_failure(self):
        """Forgetting the stop rule makes the correction anti-conservative."""
        results = holm_bonferroni([0.06, 0.001, 0.0001], alpha=0.05)
        assert results[2]["rejected"]      # 0.0001 <= 0.05/3 = 0.0167
        assert results[1]["rejected"]      # 0.001  <= 0.05/2 = 0.025
        assert not results[0]["rejected"]  # 0.06   >  0.05/1 -- and every larger
        # p after it stays retained regardless of its own value, which is the stop
        # rule. Omitting it is the common bug and makes Holm anti-conservative.

    def test_ten_slices_needs_the_smallest_p_under_0005(self):
        assert not holm_bonferroni([0.006] + [0.5] * 9)[0]["rejected"]
        assert holm_bonferroni([0.004] + [0.5] * 9)[0]["rejected"]

    def test_all_results_marked_exploratory(self):
        assert all(r["exploratory"] for r in holm_bonferroni([0.01, 0.02]))
