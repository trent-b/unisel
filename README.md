# UNISEL
UNsupervised Instance SELection (UNISEL)

This repository contains the code and data necessary to reproduce the results reported in our paper:\
Trent J. Bradberry, Christopher H. Hase, LeAnna Kent, and Joel A. Góngora (2021) **Unsupervised Instance Selection with Low-Label Supervised Learning for Outlier Detection**

## Abstract
The laborious process of labeling data often bottlenecks projects that aim to leverage the power of supervised machine learning. Active Learning (AL) has established itself as a technique to ameliorate this condition through an iterative framework that queries an annotator for labels of instances with uncertain class assignment. Via this mechanism, AL produces a binary classifier trained on less labeled data but with little, if any, loss in predictive performance. Despite its advantages, AL can have difficulty with class-imbalanced datasets such as those inherent in outlier detection problems. In this work, we investigate our unsupervised instance selection (UNISEL) technique followed by a Random Forest (RF) classifier on 10 outlier detection datasets under low-label conditions. Further, we investigate the combination of UNISEL and AL. Experimental results indicate that UNISEL followed by an RF performs comparably to AL with an RF and that the combination of UNISEL and AL demonstrates superior performance. The practical implications of these findings in terms of time savings afforded by UNISEL for low-label supervised learning are then discussed.

## Files
- `run_experiment.py` executes the full experiment, writing results to the `output` directory (this can take 1-2 days complete)
- `process_results.ipynb` reads the results and produces figures and tables
