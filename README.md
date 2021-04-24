# UNISEL
UNsupervised Instance SELection (UNISEL)

This repository contains the code and data necessary to reproduce the results reported in our manuscript under review:\
**Unsupervised Instance Selection with Low-Label Supervised Learning for Outlier Detection**
Trent J. Bradberry, Christopher H. Hase, LeAnna Kent, and Joel A. Góngora (2021)

## Abstract
The laborious process of labeling data often bottlenecks projects that aim to leverage the power of supervised machine learning. Active Learning (AL) has been established as
a technique to ameliorate this condition through an iterative framework that queries a human annotator for labels of instances with the most uncertain class assignment. Via
this mechanism, AL produces a binary classifier trained on less labeled data but with little, if any, loss in predictive performance. Despite its advantages, AL can have difficulty with class-imbalanced datasets and results in an inefficient labeling process. To address these drawbacks, we investigate our unsupervised instance selection (UNISEL) technique followed by a Random Forest (RF) classifer on 10 outlier detection datasets under low-label conditions. These results are compared to AL performed on the same datasets. Further, we investigate the combination of UNISEL and AL. Results indicate that UNISEL followed by an RF performs comparably to AL with an RF and that the combination of UNISEL and AL demonstrates superior performance. The practical implications of these findings in terms of time savings and generalizability afforded by UNISEL are discussed.

## Files
- `run_experiment.py` executes the full experiment, writing results to the `output` directory (this can take 1-2 days to complete)
- `process_results.ipynb` reads the results and produces figures and tables
