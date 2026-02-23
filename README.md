Hardware Aware QAOA for Portfolio Optimization

#Overview 
This repository implements a utility-scale, hybrid quantum-classical optimizer for solving the constrained portfolio optimization problem. It leverages a Hardware-Aware Linear XY-Mixer Quantum Approximate Optimization Algorithm (QAOA) and combines it with advanced error mitigation and Bayesian Optimization.Designed with NISQ (Noisy Intermediate-Scale Quantum) devices in mind, the architecture robustly handles simulated hardware noise—including thermal relaxation, depolarizing, and readout errors.+1This project serves as a comprehensive demonstration of full-stack quantum algorithm development, integrating Qiskit for quantum circuit execution and BoTorch/PyTorch for classical optimization.

Key Features & Methodology:

1.Hardware Aware Ansatz:Utilizes a Linear XY-Mixer to strictly preserve the Hamming weight of the quantum state, naturally enforcing the portfolio's cardinality (budget) constraints. 

2.Fourier Parameterization: Reduces the high-dimensional parameter search space by mapping a small number of Fourier modes to the physical $\gamma$ and $\beta$ circuit layers.

3.CVaR Objective Function: Implements Conditional Value at Risk (CVaR) aggregation to compute the energy expectation, focusing only on the tail-end of the highest-performing measurement shots rather than the raw average.

4.Advanced Error Mitigation Zero-Noise Extrapolation (ZNE): Employs 2nd-Order Quadratic Richardson Extrapolation to mitigate gate errors. The circuits are globally folded at scale factors of 1, 3, and 5 to extrapolate the zero-noise expectation value.

5.Readout Error Mitigation (TREX / Pseudo-M3): Corrects systematic measurement bias by constructing and efficiently inverting a tensored measurement confusion matrix based on $P(0|1)$ and $P(1|0)$ transition probabilities.

6.(BoTorch)Bayesian Optimization: Replaces standard gradient-free optimizers (like COBYLA) with a SingleTaskGP Gaussian Process model.

7.Smart Initialization & Acquisition: Initializes the search space using a Quasi-random Sobol sequence , followed by optimizing the Expected Improvement(EI) acquisition function.Trust Region Constraints: Bounds the acquisition function optimization within a dynamically updated Trust Region to ensure stable convergence.

8.Problem FormulationPortfolio Optimization: Balances expected returns ($\mu$) against covariance risk ($\Sigma$) mapped to a Quadratic Unconstrained Binary Optimization (QUBO) formulation.
$$Q = \Sigma - \lambda \cdot \text{diag}(\mu)$$

9.Probability Redistributing Map (PRM): Classical post-processing technique that probabilistically remaps invalid measured bitstrings to the nearest valid portfolio states based on expected asset returns. 

Tech StackQuantum Backend: qiskit, qiskit_aer Classical Optimizer: botorch, gpytorch, torch Data & Visualization: numpy, matplotlib 

Infrastructure: Python dataclasses, logging, Json Usage & ConfigurationThe experiment is highly configurable via the Config dataclass.

Default Configuration Parameters:Assets: 5 (Budget: 2) Ansatz Depth: 4 Fourier Modes: 2 Optimization Iterations: 30 (10 Sobol Init + 20 BO) 

Simulated Noise: $T_1 = 50\mu s$, $T_2 = 70\mu s$, Gate Time = 100ns, Depolarizing Error = 0.5% 

