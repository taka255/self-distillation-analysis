# self-distillation-analysis

This repository provides everything needed to replicate main findings in the paper: "The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model" by [Takanami et al.](https://arxiv.org/abs/2501.16226).

- **Replica theory**  
  - Solve the saddle-point equations for the noisy Gaussian mixture model  
  - Compute the optimal generalization error analytically  

- **Synthetic experiments**  
  - Simulate data from the noisy Gaussian mixture  
  - Compare empirical test error to the theoretical predictions  

- **CIFAR-10 linear probe experiments**  
  - Extract ResNet-18 and ResNet-50 features  
  - Train and evaluate logistic self-distillation (0-SD vs. 1-SD, hard vs. soft labels)  

Detailed setup and usage instructions are in the Appendix of the paper.  
