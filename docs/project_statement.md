# Project Statement: Synthetic Data Generation for the Stroke Risk Cohort

**Status.** Working project document. Written and self-reviewed by the project author before generation
work began. This document is structured to simulate governance specifications that are used
in real Trusted Research Environments, adapted for a solo demonstration project, without claiming
knowledge of the specific document structure. A real specification would require independent review,
named risk ownership, and formal sign-off. See Section 6.

## 1. Purpose & Scope
This specification covers the generation of synthetic versions of a stroke risk
cohort dataset for two distinct purposes, detailed in Section 3.
It does not cover any other purpose, dataset, or fidelity level.
If extending the project to a new purpose is required, this document would need to be revisited
before generation work continues.

## 2. Basis and reference material

This is a demonstration project using a public, non-sensitive dataset. There is no data
sharing agreement or research licence in force, because it isn't required for this data.
The project is conducted as though the data were sensitive, so that the governance discipline is genuine practice.

Working assumptions on anonymisation and disclosure are drawn from the ICO's
guidance on anonymisation and the three-part test it sets out (singling out,
linkability, inference). The fit-for-purpose approach to fidelity follows the
UK Synthetic Data Community Group's VSTAR framework. Both are public
reference material, cited here.

## 3. Purposes and fidelity levels

### Purpose 1: Code development and testing

Enable researchers to write and test analysis code against realistically shaped and typed data,
without access to real patient records during development.

**Fidelity level: L1 to L2 (structural to univariate).** Inter-variable relationships are not preserved.
Achieved by independent per-column sampling from the training set.

**Risk tier: low.** Disclosure risk is minimal because multivariate relationships, the primary source of
re-identification risk in this context, is not reproduced.

### Purpose 2: Exploratory analysis and model prototyping

Enable researchers to explore relationships in the cohort and prototype predictive models, with reasonable
expectation that findings will transfer in direction and approximate magnitude to analysis on the real data,
which might be subject to controlled access.

**Fidelity level: L3 (multivariate).** Inter-variable relationships, and clinical relationships identified
in exploratory analysis (age with stroke, hypertension and heart disease with stroke, and glucose levels with stroke) are preserved.

**Risk tier: elevated.** Preserving multivariate structure significantly increases the disclosure risk relative to
Purpose 1. A full disclosure risk assessment is mandatory for the dataset generated for this purpose, and is
a precondition for release. The risk assessment follows in Section 4.

## 4. Disclosure risk assessment (mandatory for Purpose 2 outputs)

Every dataset generated against Purpose 2 is assessed against the following before
it is treated as fit for its stated purpose:

| Test | Method | Acceptance threshold |
|---|---|---|
| Membership inference | Distance-based and classifier-based attack against a held-out control set | No greater than the real-to-real baseline |
| Attribute inference | Prediction of sensitive attributes from quasi-identifiers | No greater than baseline |
| Singling out | Unique combinations on quasi-identifiers, and separately on all categorical fields | No unique real individual isolated beyond baseline |
| Distance to closest record | Compared against a real-to-real baseline | No systematic excess proximity to real records |

**Risk tolerance.** The acceptance threshold for this project is the
empirical baseline risk observed between two disjoint real subsets of the
same cohort (training set against holdout set). A dataset that fails any
test above is not treated as fit for Purpose 2, regardless of its fidelity
score, and is noted as such in the evaluation report rather than released
with caveats.

## 5. Utility and fidelity requirements

For Purpose 2 outputs, in addition to the disclosure assessment above:

- The named clinical relationships are preserved within a stated tolerance of the real training data's association statistics.
- The population-level detection score (real against synthetic classifier) approaches chance.
- Predictive utility (train-on-synthetic test-on-real, or TSTR, against train-on-real test-on-real, or TRTR), shows a small gap on the
held-out real test set.

For Purpose 1 outputs, only univariate distributional fidelity is required. Multivariate fidelity is expected to be poor, and this is
not a failure condition. It means that fidelity was not maximised where the purpose didn't require it.

## 6. Declarations

This project has no data custodian, no governance board, and no sign-off process, because it is a solo demonstration, and not a commissioned
project. The structure above aims to mirror what a process would look like in practice, without assuming internal knowledge.
In a real Trusted Research Environment this kind of document would need to be reviewed by a named data custodian or information governance
lead, and generation would not proceed until it was signed-off.

## 7. Out of scope

Level 4 fidelity synthesis (augmentation) is out of the scope of this project.

Differential privacy is treated as a complementary formal safeguard, evaluated across a small sweep of privacy budgets alongside the
empirical tests in Section 4, not as a replacement of the privacy tests. If measured leakage is not found under Section 4, it is evidence
of safety under the attacks tested, not a general guarantee.
