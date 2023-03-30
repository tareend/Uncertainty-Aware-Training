# Uncertainty Aware Training - Model Calibration for Classification of Cardiac MR Images #
Code for various uncertainty-aware methods used to improve calibration of cardiac DL applications.
TBR on date of publication.

This repository contains the following files used to implement the uncertainty aware strategies presented in the journal:

1. Pre-processing files to allow .npy files to be created and utilised for each strategy.
2. Python file (' ') with code on how the nested cross-validation was implemented, to show the optimal hyper param search.
3. Python file (' ') with the final process of the retrained optimal model.
4. Six seperate python files for each UCA strategy - i.e each new method (List them here)
5. Evaluation files for all Calibration Metrics
6. Two Evaluation Files for Epistemic and Aleatoric UC
