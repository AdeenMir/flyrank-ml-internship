# FlyRank ML Capstone Report

**Author:** AdeenMir  
**Lane:** CTR / Engagement Opportunity Scoring  
**Repo:** `AdeenMir/flyrank-ml-internship`  
**Date:** 2026-08-31

## 0. Abstract

This study asks whether a learned ranking model can prioritize content pages that are observed as declining better than a transparent CTR baseline. The analysis uses the public-safe anonymized FlyRank internship dataset, with `trend_direction == "down"` as an independent proxy label and a client-grouped 70/30 holdout to prevent client overlap between training and test. Three models were compared with the Week-4 position-tier CTR baseline: logistic regression, a depth-4 decision tree, and a random forest. On the committed test run, the random forest had the strongest discrimination (ROC AUC 0.623, average precision 0.656, precision@50 0.760) versus 0.562, 0.604, and 0.720 for the refit baseline. The resulting output is best used as a human-review queue for prioritizing pages with meaningful exposure, not as an automated SEO decision or a causal prediction of search-engine behavior.

## 1. Problem framing

The supported decision is: **which content pages should an editor review first for possible engagement/CTR improvement?**

- **Unit of analysis:** content page.
- **Output:** a ranked score for review priority.
- **Human action:** inspect the page, its search context, and editorial evidence before deciding whether to change anything.
- **Cost of a wrong call:** false positives consume reviewer time; false negatives can leave a potential opportunity unreviewed.
- **Why ML:** the page-level feature set contains multiple interacting exposure, position, content-age, and engagement signals. A learned ranker can test whether those signals add useful discrimination beyond a simple transparent rule.

The target is deliberately not the Week-4 CTR-gap rule. Re-learning a hand-written rule would make the model comparison circular.

## 2. Data safety

The analysis uses the repository's anonymized content-refresh dataset. The source is a single trailing-90-day snapshot per page.

The following were excluded:

- `trend_direction` and `trend_pct`, because they define the target.
- `impressions_last_30d`, `impressions_prev_30d`, `clicks_last_30d`, `clicks_prev_30d`, `sessions_last_30d`, and `sessions_prev_30d`, because they directly feed the observed trend calculation and could reconstruct the label.
- `client_id`, because it is used only to create grouped train/test partitions.
- `content_id`, because it is an identifier, not a predictive feature.

The model uses 90-day totals and derived rates, plus explicit missingness flags and categorical tiers. Missing numeric values are imputed after feature selection; missing categoricals become `unknown`.

No client-identifying details are used in the paper.

## 3. Baseline

The transparent Week-4 baseline compares a page's CTR with the median CTR expected for its position tier. It requires a volume floor of **500 impressions over 90 days** and combines the positive CTR gap with an exposure rank.

For this capstone comparison, the position-tier medians are refit on training rows only and then applied to the held-out test rows. This prevents the baseline from using test-set information while the learned models are kept train-only.

On the same grouped test split, the baseline achieved:

- **ROC AUC:** 0.562
- **Average precision:** 0.604
- **Precision@50:** 0.720

The test-set positive base rate was **0.559**, so precision@50 is interpreted alongside that base rate rather than as a standalone accuracy claim.

## 4. Model / analysis

The target is:

> `is_declining_label = 1` when `trend_direction == "down"`, otherwise 0.

The feature set includes search demand, competition, content size, 90-day exposure and traffic totals, days with activity, content age, freshness, CTR, average position, engagement/scroll rates, AI-traffic share, missingness indicators, and categorical tiers.

Three models were tested:

1. Logistic regression with standardized inputs.
2. Decision tree with maximum depth 4.
3. Random forest with 300 trees and random seed 42.

The random forest was retained because it produced the strongest held-out ranking results. The depth-4 tree was retained as a useful negative result: its simplicity produced many tied probabilities, which hurt top-50 ranking.

## 5. Evaluation

The split is a **70/30 `GroupShuffleSplit` by `client_id`, random seed 42**.

The committed run produced:

| Method | ROC AUC | Average precision | Precision@50 |
|---|---:|---:|---:|
| Week-4 baseline, refit on train | 0.562 | 0.604 | 0.720 |
| Logistic regression | 0.620 | 0.653 | 0.700 |
| Decision tree, depth 4 | 0.582 | 0.609 | 0.680 |
| **Random forest** | **0.623** | **0.656** | **0.760** |

The training set contained 19,166 rows and the test set 10,834 rows. There were 22 distinct clients in train and 10 in test, with **zero clients appearing in both**.

The random forest's clearest false negatives were pages with only 1–2 impressions over 90 days. These pages have too little exposure evidence for reliable ranking. This supports the Week-4 volume floor as a useful abstention mechanism.

The strongest false positives included pages with very low CTR despite reasonable position and moderate exposure. Those cases illustrate a genuine distinction between "structurally weak CTR" and "currently labeled as declining": they are related signals, but not identical outcomes.

## 6. Interpretation

The random forest's top features were:

| Feature | Importance |
|---|---:|
| `avg_position` | 0.1096 |
| `log_impressions_90d` | 0.1057 |
| `days_with_impressions` | 0.0928 |
| `content_age_days` | 0.0757 |
| `log_sessions_90d` | 0.0521 |
| `word_count` | 0.0498 |
| `char_count` | 0.0483 |
| `days_with_sessions` | 0.0473 |
| `ctr` | 0.0470 |
| `scroll_rate` | 0.0453 |

No single feature dominates: the top feature contributes about 0.110 of the forest's importance, while the second contributes about 0.106. That is consistent with a multi-signal pattern rather than an obvious leaked label column.

The decision tree's negative result is also informative. At depth 4 it had 16 leaves and only 14 unique predicted probabilities across the 10,834 test rows. That makes the top-50 ranking among tied scores close to arbitrary. The model is readable, but its simplicity does not earn enough performance to replace the baseline.

## 7. Recommendation

### 1. Use the random-forest score as a review queue

The random forest is the strongest tested ranker on the same client-grouped holdout. Its output should determine **review order**, not automatically trigger an editorial change.

### 2. Keep the 500-impression volume floor

Near-zero-volume pages are where the strongest model failures appear. If there is not enough exposure data, abstaining is more defensible than manufacturing a confident action.

### 3. Prioritize high-score pages with meaningful exposure

Pages combining a high model score with substantial exposure are more useful review candidates than pages with almost no observations. Average position, impressions, days with impressions, content age, and sessions are among the strongest signals.

### 4. Require a human reason code before action

A reviewer should see a plain-language cue such as "high decline score + adequate exposure + weak CTR" and then inspect the actual page and search context. The score is evidence for prioritization, not a prescribed rewrite.

**Confidence:** moderate for ranking within this dataset and grouped split; low for causal or future-time claims.

## 8. Reproducibility

From a fresh clone:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb --output /tmp/capstone.executed.ipynb
```

Alternatively, open `work/notebooks/capstone.ipynb` in Colab and use **Runtime → Run all**.

The notebook uses:

- `GroupShuffleSplit(test_size=0.30, random_state=42)`
- Logistic regression `max_iter=2000`, seed 42
- Decision tree `max_depth=4`, seed 42
- Random forest `n_estimators=300`, seed 42

The notebook writes the following artifacts under `work/outputs/`:

- `capstone_model_comparison.csv`
- `capstone_feature_importance.csv`
- `capstone_auc.png`

The authoritative evaluation is the executed notebook, because it rebuilds the feature matrix, split, baseline, models, and metrics from the source data rather than relying on pasted numbers.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset: [FlyRank](https://flyrank.ai).

