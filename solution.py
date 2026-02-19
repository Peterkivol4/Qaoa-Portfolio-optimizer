import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
import time
import logging
import json
import itertools
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, List, Any, Union
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    thermal_relaxation_error,
    ReadoutError
)
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

def setup_logging(experiment_name: str = "advanced_qaoa") -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(f"{experiment_name}_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def set_reproducibility(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data = {"metadata": {"name": name, "timestamp": self.timestamp}, "config": {}, "history": []}
        
    def log_config(self, cfg: 'Config'):
        self.data["config"] = asdict(cfg)

    def log_step(self, iteration: int, metrics: Dict[str, Any]):
        self.data["history"].append({"iter": iteration, **metrics})

    def save(self):
        filename = f"{self.name}_{self.timestamp}_results.json"
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4)
        return filename
@dataclass
class Config:
    seed: int = 42
    n_assets: int = 5         
    budget: int = 2      
    risk_aversion: float = 0.5 
    depth: int = 4            
    fourier_modes: int = 2    
    base_shots: int = 512     
    cvar_alpha: float = 0.25  
    bo_iters: int = 30
    sobol_init_iters: int = 10 
    use_noise: bool = True    
    use_zne: bool = True      
    use_rem: bool = True
    readout_p10: float = 0.01  
    readout_p01: float = 0.03  
    t1_time: float = 50e3     
    t2_time: float = 70e3     
    gate_time: float = 100    
    depol_error: float = 0.005
    @property
    def dynamic_shots(self) -> int:
        return int(self.base_shots / self.cvar_alpha)


class PortfolioProblem:
    def __init__(self, cfg: Config):
        self.n = cfg.n_assets
        self.budget = cfg.budget
        self.lam = cfg.risk_aversion
        
        rng = np.random.default_rng(cfg.seed)
        self.mu = rng.uniform(0.05, 0.20, self.n)
        A = rng.standard_normal((self.n, self.n))
        self.sigma = (A @ A.T) * 0.01
        self.Q = self.sigma - np.diag(self.lam * self.mu)

    def evaluate_energy(self, bitstring: str) -> float:
        try:
            x = np.array([int(b) for b in bitstring])
            return float(x @ self.Q @ x)
        except Exception:
            return 100.0

    def is_valid(self, bitstring: str) -> bool:
        return bitstring.count('1') == self.budget

    def remap_to_valid(self, bitstring: str) -> str:
        """Probability Redistributing Map (PRM)"""
        if self.is_valid(bitstring):
            return bitstring
            
        bits = [int(b) for b in bitstring]
        current_ones = sum(bits)
        
        if current_ones > self.budget:
            one_indices = [i for i, b in enumerate(bits) if b == 1]
            one_indices.sort(key=lambda i: self.mu[i])
            for i in range(current_ones - self.budget):
                bits[one_indices[i]] = 0
        else:
            zero_indices = [i for i, b in enumerate(bits) if b == 0]
            zero_indices.sort(key=lambda i: self.mu[i], reverse=True)
            for i in range(self.budget - current_ones):
                bits[zero_indices[i]] = 1
                
        return "".join(str(b) for b in bits)

class NoiseModelFactory:
    @staticmethod
    def build(cfg: Config) -> NoiseModel:
        noise_model = NoiseModel()
        
        error_thermal = thermal_relaxation_error(cfg.t1_time, cfg.t2_time, cfg.gate_time)
        error_depol = depolarizing_error(cfg.depol_error, 1)
        combined_error = error_depol.compose(error_thermal)
        noise_model.add_all_qubit_quantum_error(combined_error, ['rz', 'rxx', 'ryy', 'x'])
        
        error_cx = depolarizing_error(cfg.depol_error * 10, 2)
        noise_model.add_all_qubit_quantum_error(error_cx, ['rxx', 'ryy'])
        if cfg.use_noise:
            p00 = 1.0 - cfg.readout_p10
            p11 = 1.0 - cfg.readout_p01
            ro_error = ReadoutError([[p00, cfg.readout_p10], [cfg.readout_p01, p11]])
            noise_model.add_all_qubit_quantum_error(ro_error, "measure")
            
        return noise_model
class ReadoutMitigator:
    """
    Tensor-based Readout Error Mitigation (Pseudo-M3).
    Inverts the measurement confusion matrix efficiently to correct systematic bias.
    """
    def __init__(self, n_qubits: int, p10: float, p01: float):
        self.n_qubits = n_qubits
        M = np.array([
            [1.0 - p10, p01],
            [p10, 1.0 - p01]
        ])
        self.inv_M = np.linalg.inv(M)

    def mitigate(self, counts: Dict[str, Union[int, float]]) -> Dict[str, float]:
        total_shots = sum(counts.values())
        if total_shots == 0: return {}
        vec = np.zeros(2**self.n_qubits)
        for bitstring, count in counts.items():
            vec[int(bitstring, 2)] = count / total_shots
        for q in range(self.n_qubits):
            shape = (2**(self.n_qubits - 1 - q), 2, 2**q)
            vec = vec.reshape(shape)
            vec = np.einsum('ij,kjl->kil', self.inv_M, vec)
            
        vec = vec.flatten()
        vec = np.clip(vec, 0, None)
        norm = np.sum(vec)
        if norm > 0: vec /= norm
        mitigated_counts = {}
        for i in range(2**self.n_qubits):
            if vec[i] > 1e-6:
                bitstring = format(i, f'0{self.n_qubits}b')
                mitigated_counts[bitstring] = vec[i] * total_shots
                
        return mitigated_counts

class QuantumRunner:
    def __init__(self, cfg: Config, problem: PortfolioProblem):
        self.cfg = cfg
        self.problem = problem
        self.mitigator = ReadoutMitigator(cfg.n_assets, cfg.readout_p10, cfg.readout_p01)
        
        self._build_ansatz()
        
        if cfg.use_noise:
            noise_model = NoiseModelFactory.build(cfg)
            self.backend = AerSimulator(noise_model=noise_model, seed_simulator=cfg.seed)
        else:
            self.backend = AerSimulator(seed_simulator=cfg.seed)
            
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        self.isa_circuit = pm.run(self.full_qc)
        
        if self.cfg.use_zne:
            self.isa_folded_3 = pm.run(self.folded_qc_3)
            self.isa_folded_5 = pm.run(self.folded_qc_5)
            
        self.sampler = BackendSamplerV2(backend=self.backend)

    def _build_ansatz(self):
        """Hardware-Aware Linear XY-Mixer."""
        self.gamma_params = ParameterVector('γ', self.cfg.depth)
        self.beta_params = ParameterVector('β', self.cfg.depth)
        
        self.init_qc = QuantumCircuit(self.cfg.n_assets)
        for i in range(self.cfg.budget):
            self.init_qc.x(i)
            
        self.var_qc = QuantumCircuit(self.cfg.n_assets)
        for d in range(self.cfg.depth):
            for i in range(self.cfg.n_assets):
                self.var_qc.rz(self.gamma_params[d], i)
                
            for i in range(self.cfg.n_assets - 1):
                self.var_qc.rxx(self.beta_params[d], i, i + 1)
                self.var_qc.ryy(self.beta_params[d], i, i + 1)
                
        self.full_qc = self.init_qc.compose(self.var_qc)
        self.full_qc.measure_all()
        
        if self.cfg.use_zne:
            var_inv = self.var_qc.inverse()
            folded_var_3 = self.var_qc.compose(var_inv).compose(self.var_qc)
            self.folded_qc_3 = self.init_qc.compose(folded_var_3)
            self.folded_qc_3.measure_all()
            
            folded_var_5 = folded_var_3.compose(var_inv).compose(self.var_qc)
            self.folded_qc_5 = self.init_qc.compose(folded_var_5)
            self.folded_qc_5.measure_all()

    def get_fourier_bindings(self, fourier_params: np.ndarray) -> np.ndarray:
        q = self.cfg.fourier_modes
        p = self.cfg.depth
        A = fourier_params[:q]
        B = fourier_params[q:] 

        gamma = np.zeros(p)
        beta = np.zeros(p)
        
        for d in range(p):
            layer_idx = d + 1
            for k in range(q):
                freq = (k + 0.5) * layer_idx * np.pi / p
                gamma[d] += A[k] * np.sin(freq)
                beta[d] += B[k] * np.sin(freq)

        return np.concatenate([gamma, beta])

    def _execute_circuit(self, isa_qc: QuantumCircuit, bound_array: List[float]) -> Tuple[float, float]:
        pub = (isa_qc, bound_array, self.cfg.dynamic_shots)
        job = self.sampler.run([pub])
        counts = job.result()[0].data.meas.get_counts()
        if self.cfg.use_rem and self.cfg.use_noise:
            counts = self.mitigator.mitigate(counts)
        
        weighted_energies = []
        valid_shots, total_shots = 0.0, 0.0
        for bitstring, count in counts.items():
            b_str = bitstring[::-1]
            total_shots += count
            
            if self.problem.is_valid(b_str):
                valid_shots += count
                
            remapped_str = self.problem.remap_to_valid(b_str)
            energy = self.problem.evaluate_energy(remapped_str)
            weighted_energies.append((energy, count))
            
        raw_valid_ratio = valid_shots / total_shots if total_shots > 0 else 0.0
        if not weighted_energies: return 5.0, 0.0
        weighted_energies.sort(key=lambda x: x[0]) 
        
        target_weight = total_shots * self.cfg.cvar_alpha
        cumulative_weight = 0.0
        cvar_sum = 0.0
        
        for energy, weight in weighted_energies:
            if cumulative_weight + weight <= target_weight:
                cvar_sum += energy * weight
                cumulative_weight += weight
            else:
                remaining = target_weight - cumulative_weight
                cvar_sum += energy * remaining
                cumulative_weight += remaining
                break
                
        cvar_value = cvar_sum / target_weight if target_weight > 0 else 5.0
        return float(cvar_value), raw_valid_ratio

    def run(self, fourier_params: np.ndarray) -> Tuple[float, float]:
        physical_params = self.get_fourier_bindings(fourier_params)
        
        bind_dict = {}
        for i in range(self.cfg.depth):
            bind_dict[self.gamma_params[i]] = physical_params[i]
            bind_dict[self.beta_params[i]] = physical_params[self.cfg.depth + i]
            
        base_vals = [bind_dict[p] for p in self.isa_circuit.parameters]
        cvar_s1, vr_s1 = self._execute_circuit(self.isa_circuit, base_vals)
        
        if self.cfg.use_zne:
            fold_vals_3 = [bind_dict[p] for p in self.isa_folded_3.parameters]
            cvar_s3, _ = self._execute_circuit(self.isa_folded_3, fold_vals_3)
            
            fold_vals_5 = [bind_dict[p] for p in self.isa_folded_5.parameters]
            cvar_s5, _ = self._execute_circuit(self.isa_folded_5, fold_vals_5)
            
class BoTorchOptimizer:
    def __init__(self, n_fourier_params: int, n_init: int, bounds: Tuple[float, float] = (-np.pi, np.pi)):
        self.n_params = n_fourier_params
        self.n_init = n_init
        self.bounds = torch.tensor([bounds] * n_fourier_params).T
        
        self.train_x = torch.empty((0, n_fourier_params))
        self.train_y = torch.empty((0, 1))
        
        sobol = SobolEngine(dimension=self.n_params, scramble=True, seed=42)
        self.sobol_samples = sobol.draw(self.n_init).numpy() * (bounds[1] - bounds[0]) + bounds[0]
        
    def suggest(self) -> np.ndarray:
        current_len = len(self.train_x)
        
        if current_len < self.n_init:
            return self.sobol_samples[current_len]
            
        try:
            gp = SingleTaskGP(
                self.train_x, 
                self.train_y, 
                input_transform=Normalize(d=self.n_params),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            acq = ExpectedImprovement(gp, best_f=self.train_y.min(), maximize=False)
            best_x = self.train_x[self.train_y.argmin()]
            tr_radius = np.pi / 2.0 
            tr_bounds = torch.stack([
                torch.maximum(self.bounds[0], best_x - tr_radius),
                torch.minimum(self.bounds[1], best_x + tr_radius)
            ])

            candidate, _ = optimize_acqf(
                acq, bounds=tr_bounds, q=1, num_restarts=5, raw_samples=256
            )
            return candidate.detach().numpy().flatten()
            
        except Exception as e:
            logging.error(f"BoTorch failed, falling back to random: {e}")
            return (torch.rand(self.n_params) * (self.bounds[0,1]-self.bounds[0,0]) + self.bounds[0,0]).numpy()

    def observe(self, x: np.ndarray, y: float):
        self.train_x = torch.cat([self.train_x, torch.tensor(x).unsqueeze(0)])
        self.train_y = torch.cat([self.train_y, torch.tensor([[y]])])
def main():
    logger = setup_logging("Advanced_QAOA")
    cfg = Config()
    set_reproducibility(cfg.seed)
    tracker = ExperimentTracker("Advanced_QAOA")
    tracker.log_config(cfg)
    
    logger.info("Initializing Final Utility-Scale Hybrid Optimizer...")
    logger.info(f"Ansatz: Hardware-Aware Linear XY-Mixer | Depth: {cfg.depth}")
    logger.info(f"Error Mitigation: 2nd-Order ZNE (Gate) + Tensored M3 Inversion (Readout)")
    logger.info(f"Fourier Space: {cfg.fourier_modes*2} params. CVaR Alpha: {cfg.cvar_alpha}")
    
    problem = PortfolioProblem(cfg)
    runner = QuantumRunner(cfg, problem)
    
    n_search_params = 2 * cfg.fourier_modes 
    optimizer = BoTorchOptimizer(n_search_params, n_init=cfg.sobol_init_iters)
    
    best_cvar = float("inf")
    history_cvar = []
    
    logger.info(f"Starting Loop ({cfg.bo_iters} total iters. {cfg.sobol_init_iters} Sobol)")
    start_time = time.time()
    
    try:
         for i in range(cfg.bo_iters):
            iter_start = time.time()
            
            x_next = optimizer.suggest()
            cvar_e, raw_valid_ratio = runner.run(x_next)
            optimizer.observe(x_next, cvar_e)
            
            if cvar_e < best_cvar:
                best_cvar = cvar_e
                
            history_cvar.append(best_cvar)
            elapsed = time.time() - iter_start
            
            phase = "SOBOL" if i < cfg.sobol_init_iters else "BO-TR"
            logger.info(f"[{phase}] Iter {i+1:02d} | CVaR: {cvar_e: .4f} | Best: {best_cvar: .4f} | Raw Valid: {raw_valid_ratio:.0%} | {elapsed:.1f}s")
            tracker.log_step(i, {"cvar": cvar_e, "raw_valid": raw_valid_ratio})
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:
        logger.info(f"Finished in {time.time() - start_time:.2f}s. Global Best CVaR: {best_cvar:.4f}")
        tracker.save()
        
        plt.figure(figsize=(10, 5))
        plt.plot(history_cvar, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.axvline(x=cfg.sobol_init_iters-1, color='r', linestyle='--', label='End of Sobol Init')
        plt.title(f"Full-Stack Hardware-Aware QAOA (Linear XY + Fourier + ZNE + M3/TREX)")
        plt.xlabel("Iteration")
        plt.ylabel("Best CVaR Objective")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.argv.pop()
    else:
        main()