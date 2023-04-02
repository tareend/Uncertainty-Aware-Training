# Uncertainty Aware Training - Model Calibration for Classification of Cardiac MR Images #
Code for various uncertainty-aware methods used to improve calibration of cardiac DL applications.
TBR on date of publication.

This repository contains the following files used to implement the uncertainty aware strategies presented in the journal:

1. Pre-processing files to allow .npy files to be created and utilised for each strategy.

2. Python file ('Find_Optimal_Params.py') with code on how the nested cross-validation was implemented, to show the optimal hyper param search.

3. Six seperate python files for each UCA strategy - i.e each new method (and used as the final training file after choosing optimal validation accuracy):
  3.1 Baseline - ('Baseline.py')
  3.2 Paired Confidence Loss - ('Paired_Confidence_Loss.py')
  3.3 Probability_Loss - ('Probability_Loss.py')
  3.4 Confidence_Weight - ('Confidence_Weight.py')
  3.5 AvUC_Loss - ('AvUC_Loss.py') Krishnan, R., Tickoo, O., 2020. Improving model calibration with accuracy versus uncertainty optimization. Advances in Neural Information Processing Systems.
  3.6 Soft_ECE_Loss - ('Soft_ECE_Loss.py') Karandikar, A., Cain, N., Tran, D., Lakshminarayanan, B., Shlens, J., Mozer, M.C., Roelofs, R., 2021. Soft calibration ] objectives for neural networks. Advances in Neural Information Processing Systems 34.
  3.7 MMCE_Loss - ('MMCE_Loss') Kumar, A., Sarawagi, S., Jain, U., 2018. Trainable calibration measures for neural networks from kernel mean embeddings, in: International Conference on Machine Learning, PMLR. pp. 2805–2814.
  
4. Evaluation files for all Calibration Metrics. ('Calibration_Evaluation_Metrics.py')

5. Two Evaluation Files for Epistemic and Aleatoric UC.('Epistemic_UC_Eval.py' and 'Aleatoric_UC_Eval.py' respectively.)
