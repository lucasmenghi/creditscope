# Model card — CreditScope

Author: Lucas Menghi. Version: 1.0. Selected candidate: lightgbm_31.

## Intended use
Educational credit-risk ranking and historical diagnostics for human review. Not credit approval, pricing, limits or automated denial.

## Target and data
Competition TARGET=1: payment difficulties, more than X days late in at least one of first Y installments. X/Y undisclosed. 307,511 labeled applications from the 2018 Home Credit Default Risk release. Full provenance/hash in data_manifest.json. Unlabeled application_test excluded from evaluation. No raw datasets or individual predictions published.

## Availability and features
26 features; borrower identifier, target and gender excluded. Age and anonymized external scores remain. Relative-day historical cutoffs documented in prepare.py. Installment counts refer to payment-event rows, not guaranteed unique installments. Retrospective snapshots do not prove operational point-in-time availability. Train-only imputation/scaling; employment sentinel handled explicitly.

## Evaluation design
Independent stratified customers, 60/10/10/5/15 train/development/calibration/policy/test. No global calendar, no temporal claim. Four fixed candidates, best development AP. Separate sigmoid calibration and frozen policy threshold, no test-driven retraining. One split, no confidence intervals. Test n=46128; prevalence 0.0807; ROC AUC 0.7642; AP 0.2547; Brier 0.0673; KS 0.3957.

## Limitations and impact
Gender exclusion does not demonstrate fairness: proxies and age require review. Descriptive subgroup metrics in reports/subgroups.json. Anonymized external scores limit explanation. No demonstrated temporal generalization, ROI, loss reduction, legal compliance or causal effect. Validate availability, drift, calibration, cohort impacts and credit policy before a human-supervised pilot. This repository contains no production deployment or individual credit decisions.
