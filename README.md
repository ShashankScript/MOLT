# MOLT

A small benchmark for one question: when fraud patterns change, how should you decide to retrain the model?

It replays real credit-card transactions (the ULB/Worldline Kaggle dataset) as a stream of 4,000-row windows, injects controlled drift into some runs, and compares five ways of keeping the model current. There is a Streamlit dashboard on top of the saved results.

This is a research toy. The drift is injected, not observed, and nothing here is connected to real payments.

## What I compared

| Strategy | Idea |
|---|---|
| Static | train once, never update |
| Periodic | retrain every 3 windows |
| Drift-triggered (MOLT) | retrain only when a fraud-miss ADWIN detector fires, promote the new model only if it beats the old one on a held-out window |
| River online | incremental logistic regression |
| Zero-delay reference | retrain on every window, labels arrive right after scoring |

Labels arrive one window late, and a window is always scored before its own labels are released.

## Result

Harmful abrupt drift, 5 seeds, recall at 1% false-positive rate:

| Strategy | Recall |
|---|---:|
| Static | 0.29 |
| Drift-triggered | 0.50 |
| Periodic | 0.76 |
| River online | 0.79 |
| Zero-delay reference | 0.80 |

Drift-triggered retraining beats doing nothing but loses clearly to plain periodic retraining. The detector needs about 3.4 windows to fire, and then labels are delayed on top of that, so the fix arrives late. On the no-drift and benign-shift controls it never retrains, which is the good part.

Only five seeds, and they resample legitimate rows rather than giving independent datasets, so treat the intervals as rough.

![headline](results/headline_performance.png)

## Run it

Python 3.10.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
streamlit run app.py
```

The dashboard only needs the files already in the repo. To regenerate results, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/raw/` and run:

```bash
python threshold_analysis.py                 # batch model + threshold
python -m experiments.run_all --quick        # one-seed smoke test
python -m experiments.run_all                # full suite, takes a while
python -m experiments.diagnose_drift_triggered
```

## Layout

- `molt/` – data splits, drift injection, detectors, metrics, the window simulator
- `experiments/` – scripts that produce everything in `results/`
- `app.py`, `style.css` – dashboard
- `tests/` – leakage and simulation checks
- `docs/protocol.md` – the full protocol, detector details and result tables
- `NOTES.md` – design decisions and loose ends



## Data

Dataset: [Kaggle credit card fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ODbL 1.0 / DbCL 1.0). The raw CSV is not included. See `DATA_LICENSE.md`.
