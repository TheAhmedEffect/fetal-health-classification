# Fetal Health Classification

Three-class classification of fetal health (normal, suspect, pathological) from
cardiotocography measurements, across roughly 2,100 records and 21 features. Python,
pandas, scikit-learn, imbalanced-learn.

The dataset is **not included** in this repository. The notebook expects a
`fetal_health.csv` with a `fetal_health` target column.

## What the notebook does

- **Missing values** — `IterativeImputer` with a `RandomForestRegressor` estimator, rather
  than mean filling, so each missing value is modelled from the other features.
- **Split** — 80/20, fixed `random_state`.
- **Feature screening** — one-way ANOVA (`scipy.stats.f_oneway`) per feature against the
  three classes, **computed on the training split only**, to drop features with no
  separation between classes.
- **Class imbalance** — `SMOTEENN` from imbalanced-learn, which over-samples the minority
  classes and then cleans the result with edited nearest neighbours, applied to the
  training set only. Class distributions printed before and after.
- **Feature selection** — `SelectKBest` with a chi-squared score, k = 7, fitted on the
  resampled training data and then applied unchanged to the test set.
- **Model and tuning** — random forest tuned by `RandomizedSearchCV` over the number of
  trees, depth, `max_features`, leaf and split minimums, and bootstrap.
- **Evaluation** — classification report and confusion matrix on the held-out set.

## Results

426 held-out records.

Best hyperparameters found: 253 trees, max depth 34, `max_features='sqrt'`,
`min_samples_leaf=2`, `min_samples_split=2`, `bootstrap=False`.

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.77 | 0.94 | 0.85 | 35 |
| Suspect | 0.99 | 0.89 | 0.94 | 332 |
| Pathological | 0.63 | 0.92 | 0.74 | 59 |
| **Accuracy** | | | **0.90** | 426 |
| Macro average | 0.80 | 0.92 | 0.84 | 426 |

## What the results actually say

Cross-validated score on the resampled training data was 0.9915 against 0.90 on the
held-out set. That gap is the number worth noting: SMOTEENN cleans the training
distribution, which makes cross-validation on it easier than the real test set, so the
0.9915 should not be quoted as model performance.

The recall-heavy profile is the intended trade. Macro recall (0.92) sits above macro
precision (0.80), and both minority classes are recalled above 0.89 — for a clinical
screening target, missing a pathological case costs more than a false alarm. The price is
precision of 0.63 on the pathological class: roughly a third of the cases flagged
pathological are not.

## Known limitations

- The `IterativeImputer` is fitted on the full dataset before the train/test split, so
  imputed values carry information from the test rows. It belongs after the split.
- A single 80/20 split with one random seed, so the per-class figures on the 35-record
  normal class carry wide error bars.
- Five of twenty-five `RandomizedSearchCV` fits failed and were scored as `nan`, from
  parameter combinations the estimator rejected.
