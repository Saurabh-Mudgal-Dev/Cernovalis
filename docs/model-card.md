# Cernovalis Fake-News Classifier — Model Card

**Status:** Experimental research prototype; not validated for operational fact-checking.

## Summary

Cernovalis classifies an English-language article as **fake (0)** or **real (1)** using TF–IDF classical models, a GloVe-initialized BiLSTM, and probability-level ensemble methods. The current saved evaluation uses a held-out test partition from the ISOT Fake News dataset. Results below are dataset-specific and should not be interpreted as real-world fact verification.

## Intended use

- Compare text-classification baselines and inspect their errors on the supplied dataset.
- Demonstrate model evaluation, inference, and explainability workflows.

**Not for:** determining truth, moderation or enforcement decisions, publishing claims about a person or source, or other high-impact use without independent verification and human review. A predicted class is a learned label association, not evidence that an article is factually true or false.

## Data and preprocessing

- **Source:** local ISOT `True.csv` and `Fake.csv` files; 21,417 real and 23,481 fake rows (44,898 total).
- **Labels:** `True.csv` → real (1); `Fake.csv` → fake (0).
- **Partition:** stratified, fixed-seed 70/15/15 split: 31,428 train, 6,735 validation, 6,735 test. The stored index manifest is `backend/ml/data/processed/split_indices.json`.
- **Test class counts:** 3,522 fake and 3,213 real. No resampling is reported.
- **Text:** title and body are cleaned separately, then combined for TF–IDF. The current training path applies regex normalization, contraction expansion, lowercasing and English stopword removal; it does not enable spaCy lemmatization. TF–IDF uses up to 10,000 features, 1–3-grams and sublinear term frequency.
- **Deep-learning tokens:** lowercase alphanumeric/apostrophe tokens, top 50,000 vocabulary entries, sequences truncated/padded to 300 tokens; 100-dimensional pretrained GloVe embeddings, frozen during training.

The dataset split is random and stratified, not chronological or source-grouped. Duplicate, syndicated or source-specific content may therefore appear across partitions and inflate apparent generalization. The dataset's labeling process and source provenance are not independently audited here.

## Models and selection

- **Logistic Regression:** `C` selected from 0.1, 1, 10; best validation macro-F1 at `C=10`.
- **Multinomial Naive Bayes:** `alpha` selected from 0.1, 0.5, 1; best at `alpha=0.1`.
- **Random Forest:** search over 100/300 trees, unlimited/20 max depth, and minimum split sizes 2/5; best validation macro-F1 at 300 trees, unlimited depth, split size 2.
- **XGBoost:** search over 200/500 trees, depths 3/6, learning rates 0.1/0.3; best at 500 trees, depth 6, learning rate 0.1.
- **BiLSTM:** validation search over 64/128 LSTM units and dropout 0.3/0.5; best at 64 units, dropout 0.3. Training uses Adam (learning rate 0.001), batch size 64, maximum 20 epochs and early stopping (patience 3).
- **Soft vote:** equal weights and validation-macro-F1-derived weights are evaluated.
- **Stacked meta-classifier:** logistic regression over the five base-model probabilities; `C` selected by 5-fold stratified cross-validation on validation predictions.

Classical hyperparameters are selected on validation macro-F1. The saved ensemble report additionally compares ensemble candidates on the test set and declares a test-set winner. That makes the reported meta-classifier result susceptible to selection bias; it is not an untouched final estimate after model selection. Use a new external test set or nested validation for an unbiased final comparison.

## Test-set results

Metrics are from the saved reports in `backend/ml/models/saved/`, each evaluated on the same 6,735 test samples. Confusion matrices use rows = true class and columns = predicted class, in order **fake, real**.

| Model | Accuracy | Macro-F1 | Fake F1 | Real F1 | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.9918 | 0.9918 | 0.9922 | 0.9915 | `[[3490, 32], [23, 3190]]` |
| Multinomial Naive Bayes | 0.9547 | 0.9546 | 0.9566 | 0.9526 | `[[3365, 157], [148, 3065]]` |
| Random Forest | 0.9846 | 0.9845 | 0.9852 | 0.9839 | `[[3455, 67], [37, 3176]]` |
| XGBoost | 0.9893 | 0.9893 | 0.9897 | 0.9888 | `[[3476, 46], [26, 3187]]` |
| BiLSTM | 0.9941 | 0.9940 | 0.9943 | 0.9938 | `[[3500, 22], [18, 3195]]` |
| Soft vote (validation-scored weights) | 0.9942 | 0.9942 | 0.9945 | 0.9939 | `[[3497, 25], [14, 3199]]` |
| Stacked meta-classifier | 0.9964 | 0.9964 | 0.9966 | 0.9963 | `[[3505, 17], [7, 3206]]` |

These are observed scores on one random split, not confidence intervals or estimates across independent datasets. The high scores may reflect dataset artifacts, overlap, or source/topic cues; they do not establish performance on contemporary or unseen publishers. The evaluation report (`evaluation_report.json`) also contains ROC-AUC, average precision, curves and feature rankings when generated; metrics are not calibration guarantees.

## Error analysis

Run `backend/ml/scripts/analyze_errors.py` with the project environment to regenerate `backend/ml/models/saved/error_analysis.json`. It lists held-out errors with dataset indices, actual/predicted classes, predicted confidence, and up to 500 characters of article text, plus counts of false positives/negatives and errors shared across models. Treat confidence as model probability, not evidential certainty. The report contains article excerpts from the local dataset; review and share it accordingly.

## Limitations and risks

- **Domain and time shift:** evaluation is confined to the ISOT collection; no external, chronological, multilingual, social-media, or adversarial evaluation is reported.
- **Label validity and leakage:** source, topic, writing-style and duplicated/syndicated content may act as shortcuts. Random article-level splitting does not control these factors. Dataset labels are inherited, not independently fact-checked.
- **False claims risk:** the classifier scores linguistic patterns, not claim veracity, citations, evidence quality, or source credibility. Either error type can be consequential.
- **Bias:** demographic, political, geographic, language-variety and publisher-specific performance has not been measured. Per-class test metrics do not substitute for subgroup evaluation.
- **Interpretability:** linear coefficients, tree importances and Naive Bayes log-odds indicate model associations, not causal explanations. BiLSTM explanations are not provided by these feature rankings.
- **Ensembling:** component errors are correlated; a higher score on this split does not guarantee broader robustness. Test-set-based winner selection further biases the stacked-model comparison.
- **Reproducibility:** serialized scikit-learn artifacts require a compatible environment. Use `backend/.venv/Scripts/python.exe` on Windows (or the environment's equivalent); reports record split indices and training metrics, but the raw CSVs and model binaries are local artifacts and may not be present in a clone.

## Evaluation and reproduction

From the repository root, with raw ISOT CSVs and trained artifacts available:

```powershell
backend\.venv\Scripts\python.exe backend/ml/scripts/evaluate_models.py
backend\.venv\Scripts\python.exe backend/ml/scripts/analyze_errors.py
```

Evaluation reports are generated from the saved stratified test split. Do not tune thresholds, select models, or make repeated design decisions using that test partition if it will be presented as an unbiased final evaluation. Keep the test set for final reporting and use train/validation data for iteration.

## Version and artifact provenance

The checked environment used for the saved reports was Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6, XGBoost 3.4.1 and Keras 3.15.1. The authoritative recorded metrics are the JSON training reports under `backend/ml/models/saved/`; regenerate reports after changing data, preprocessing, splits, dependencies or model artifacts.
