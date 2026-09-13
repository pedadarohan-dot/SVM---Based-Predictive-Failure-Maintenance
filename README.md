# svm-predictive-maintenance

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)
![status](https://img.shields.io/badge/status-course%20project-orange)

Can a Support Vector Machine tell a healthy machine from a failing one, using nothing but five sensor readings?

That's the question this repo answers, end to end: load sensor data, pick honest features, deal with a 96/4 class imbalance, compare three SVM kernels, tune the winner, and be upfront about what got worse in the process.

## the pipeline

```
raw data → inspect → select features → stratified split → standardize
   → train {linear, poly, rbf} → evaluate → grid-search the rbf kernel
   → re-evaluate → compare
```

## the data

[AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 machine readings, 14 columns, 3.39% labeled as failures.

Five continuous operating parameters go in as features:

```
Air temperature [K]  Process temperature [K]  Rotational speed [rpm]
Torque [Nm]           Tool wear [min]
```

Two columns are dropped for being identifiers (`UDI`, `Product ID`), and five more (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are dropped because they're derived from the failure label itself — training on them would be leakage, not prediction.

## results

Accuracy is nearly meaningless here — always predicting "normal" already gets you 96.6%. Precision/recall/F1 tell the real story:

| model            | accuracy | precision | recall  | f1     |
|------------------|:--------:|:---------:|:-------:|:------:|
| Linear SVM       | 82.8%    | 14.25%    | 80.88%  | 24.23% |
| Polynomial SVM   | 90.3%    | 25.20%    | 94.12%  | 39.75% |
| RBF SVM          | 91.3%    | 27.16%    | 92.65%  | 42.00% |
| **Optimized RBF**| **95.2%**| **40.14%**| 83.82%  | **54.29%** |

*(Optimized RBF: `C=100, gamma="scale"`, found by 5-fold grid search on F1.)*

Tuning is not a free lunch. The optimized model raises accuracy, precision, and F1 by a wide margin, but recall drops from 92.65% → 83.82% — it now misses 11 failures instead of 5, out of 68. In a real deployment that trade-off would need a human decision, not just a leaderboard number.

## quickstart

```bash
git clone https://github.com/pedadarohan-dot/SVM---Based-Predictive-Failure-Maintenance
cd SVM---Based-Predictive-Failure-Maintenance
pip install -r requirements.txt

# drop ai4i2020.csv in this directory, then:
python train.py --data ai4i2020.csv
```

Useful flags:

```bash
python train.py --data ai4i2020.csv --no-plots          # just the numbers
python train.py --data ai4i2020.csv --save-plots out/   # write PNGs instead of showing windows
```

## repo layout

```
.
├── train.py           # the whole pipeline: data, models, tuning, plots
├── requirements.txt
└── README.md
```

## what this project is (and isn't)

This started as a Soft Computing course assignment on SVM theory — the numerical margin/hyperplane derivation was done by hand before any code was written. The five features are a domain-informed baseline, not a claim that they're provably optimal, and the results above are exactly what the code produces, no cherry-picking. The same pipeline was also reproduced in MATLAB to check the methodology wasn't a scikit-learn artifact.

It's a learning project, not a production failure-detection system — treat the numbers accordingly.
