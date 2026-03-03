import json
import logging
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

ZNE_C1 =  1.875
ZNE_C3 = -1.250
ZNE_C5 =  0.375

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
        self.data = {
            "metadata": {"name": name, "timestamp": self.timestamp},
            "config": {},
            "history": [],
        }

    def log_config(self, cfg: 'Config'):
        self.data["config"] = asdict(cfg)

    def log_step(self, iteration: int, metrics: Dict[str, Any]):
        self.data["history"].append({"iter": iteration, **metrics})

    def save(self) -> str:
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

    t1_time: float = 50e3
    t2_time: float = 70e3
    gate_time: float = 100.0
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
        """
        Probability Redistributing Map (PRM).
        Snaps noise-corrupted bitstrings back to the valid budget subspace
        using expected returns as a tiebreaker.
        """
        if self.is_valid(bitstring):
            return bitstring

        bits = [int(b) for b in bitstring]
        current_ones = sum(bits)

        if current_ones > self.budget:
            one_indices = sorted(
                [i for i, b in enumerate(bits) if b == 1],
                key=lambda i: self.mu[i]
            )
            for i in range(current_ones - self.budget):
                bits[one_indices[i]] = 0
        else:
            zero_indices = sorted(
                [i for i, b in enumerate(bits) if b == 0],
                key=lambda i: self.mu[i],
                reverse=True
            )
            for i in range(self.budget - current_ones):
                bits[zero_indices[i]] = 1

        return "".join(str(b) for b in bits)


class NoiseModelFactory:
    @staticmethod
    def build(cfg: Config) -> NoiseModel:
        noise_model = NoiseModel()
        combined = depolarizing_error(cfg.depol_error, 1).compose(
            thermal_relaxation_error(cfg.t1_time, cfg.t2_time, cfg.gate_time)
        )
        noise_model.add_all_qubit_quantum_error(combined, ['rz', 'rxx', 'ryy', 'x'])
        cx_error = depolarizing_error(cfg.depol_error * 10, 2)
        noise_model.add_all_qubit_quantum_error(cx_error, ['rxx', 'ryy'])
        return noise_model

class QuantumRunner:
    def __init__(self, cfg: Config, problem: PortfolioProblem):
        self.cfg = cfg
        self.problem = problem

        self._build_ansatz()

        if cfg.use_noise:
            self.backend = AerSimulator(
                noise_model=NoiseModelFactory.build(cfg),
                seed_simulator=cfg.seed,
            )
        else:
            self.backend = AerSimulator(seed_simulator=cfg.seed)

        pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        self.isa_circuit = pm.run(self.full_qc)

        if cfg.use_zne:
            self.isa_folded_3 = pm.run(self.folded_qc_3)
            self.isa_folded_5 = pm.run(self.folded_qc_5)

        self.sampler = BackendSamplerV2(backend=self.backend)

    def _build_ansatz(self):
        """Hardware-aware linear XY-mixer ansatz with digital ZNE folding."""
        self.gamma_params = ParameterVector('γ', self.cfg.depth)
        self.beta_params  = ParameterVector('β', self.cfg.depth)

        init_qc = QuantumCircuit(self.cfg.n_assets)
        for i in range(self.cfg.budget):
            init_qc.x(i)

        var_qc = QuantumCircuit(self.cfg.n_assets)
        for d in range(self.cfg.depth):
            for i in range(self.cfg.n_assets):
                var_qc.rz(self.gamma_params[d], i)
            for i in range(self.cfg.n_assets - 1):
                var_qc.rxx(self.beta_params[d], i, i + 1)
                var_qc.ryy(self.beta_params[d], i, i + 1)

        self.full_qc = init_qc.copy()
        self.full_qc.compose(var_qc, inplace=True)
        self.full_qc.measure_all()

        if self.cfg.use_zne:
            var_inv = var_qc.inverse()

            folded_var_3 = var_qc.copy()
            folded_var_3.compose(var_inv, inplace=True)
            folded_var_3.compose(var_qc, inplace=True)

            folded_var_5 = folded_var_3.copy()
            folded_var_5.compose(var_inv, inplace=True)
            folded_var_5.compose(var_qc, inplace=True)

            self.folded_qc_3 = init_qc.copy()
            self.folded_qc_3.compose(folded_var_3, inplace=True)
            self.folded_qc_3.measure_all()

            self.folded_qc_5 = init_qc.copy()
            self.folded_qc_5.compose(folded_var_5, inplace=True)
            self.folded_qc_5.measure_all()

    def get_fourier_bindings(self, fourier_params: np.ndarray) -> np.ndarray:
        q = self.cfg.fourier_modes
        p = self.cfg.depth
        A, B = fourier_params[:q], fourier_params[q:]
        gamma, beta = np.zeros(p), np.zeros(p)

        for d in range(p):
            for k in range(q):
                freq = (k + 0.5) * (d + 1) * np.pi / p
                gamma[d] += A[k] * np.sin(freq)
                beta[d]  += B[k] * np.sin(freq)

        return np.concatenate([gamma, beta])

    def _execute_circuit(self, isa_qc: QuantumCircuit, bound_array: List[float]) -> Tuple[float, float]:
        self.sampler.options.default_shots = self.cfg.dynamic_shots
        bindings = {p: [v] for p, v in zip(isa_qc.parameters, bound_array)}
        pub    = (isa_qc, bindings)
        result = self.sampler.run([pub]).result()
        counts = result[0].data.meas.get_counts() 

        valid_shots = total_shots = 0
        all_energies: List[float] = []

        for bitstring, count in counts.items():
            b_str = bitstring[::-1]
            total_shots += count
            if self.problem.is_valid(b_str):
                valid_shots += count
            remapped = self.problem.remap_to_valid(b_str)
            all_energies.extend([self.problem.evaluate_energy(remapped)] * count)

        raw_valid_ratio = valid_shots / total_shots if total_shots > 0 else 0.0

        sorted_e = np.sort(all_energies)
        k = max(1, int(len(sorted_e) * self.cfg.cvar_alpha))
        return float(np.mean(sorted_e[:k])), raw_valid_ratio

    def run(self, fourier_params: np.ndarray) -> Tuple[float, float]:
        physical = self.get_fourier_bindings(fourier_params)
        bind_dict = {
            **{self.gamma_params[i]: physical[i] for i in range(self.cfg.depth)},
            **{self.beta_params[i]:  physical[self.cfg.depth + i] for i in range(self.cfg.depth)},
        }

        base_vals  = [bind_dict[p] for p in self.isa_circuit.parameters]
        cvar_s1, vr = self._execute_circuit(self.isa_circuit, base_vals)

        if self.cfg.use_zne:
            vals_3  = [bind_dict[p] for p in self.isa_folded_3.parameters]
            cvar_s3, _ = self._execute_circuit(self.isa_folded_3, vals_3)
            vals_5  = [bind_dict[p] for p in self.isa_folded_5.parameters]
            cvar_s5, _ = self._execute_circuit(self.isa_folded_5, vals_5)
            return ZNE_C1 * cvar_s1 + ZNE_C3 * cvar_s3 + ZNE_C5 * cvar_s5, vr

        return cvar_s1, vr
        
class BoTorchOptimizer:
    def __init__(
        self,
        n_fourier_params: int,
        n_init: int,
        bounds: Tuple[float, float] = (-np.pi, np.pi),
        logger: Optional[logging.Logger] = None,
    ):
        self.n_params = n_fourier_params
        self.n_init   = n_init
        self.logger   = logger or logging.getLogger(__name__)

        lo, hi = bounds
        self.bounds = torch.tensor([[lo] * n_fourier_params,
                                    [hi] * n_fourier_params])

        self.train_x = torch.empty((0, n_fourier_params))
        self.train_y = torch.empty((0, 1))

        sobol = SobolEngine(dimension=n_fourier_params, scramble=True, seed=42)
        self.sobol_samples = sobol.draw(n_init).numpy() * (hi - lo) + lo

    def suggest(self) -> np.ndarray:
        n = len(self.train_x)
        if n < self.n_init:
            return self.sobol_samples[n]

        try:
            gp = SingleTaskGP(
                self.train_x,
                self.train_y,
                input_transform=Normalize(d=self.n_params),
                outcome_transform=Standardize(m=1),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

            best_x = self.train_x[self.train_y.argmin()]
            radius = np.pi / 2.0
            tr_bounds = torch.stack([
                torch.maximum(self.bounds[0], best_x - radius),
                torch.minimum(self.bounds[1], best_x + radius),
            ])

            acq = LogExpectedImprovement(gp, best_f=self.train_y.min(), maximize=False)
            candidate, _ = optimize_acqf(
                acq, bounds=tr_bounds, q=1, num_restarts=5, raw_samples=256
            )
            return candidate.detach().numpy().flatten()

        except Exception as e:
            self.logger.error(f"BoTorch GP failed, falling back to random: {e}")
            lo, hi = self.bounds[0], self.bounds[1]   
            return (torch.rand(self.n_params) * (hi - lo) + lo).numpy()

    def observe(self, x: np.ndarray, y: float):
        self.train_x = torch.cat([self.train_x, torch.tensor(x).unsqueeze(0)])
        self.train_y = torch.cat([self.train_y, torch.tensor([[y]])])
        
def main():
    cfg    = Config()
    logger = setup_logging("Advanced_QAOA")
    set_reproducibility(cfg.seed)

    tracker = ExperimentTracker("Advanced_QAOA")
    tracker.log_config(cfg)

    logger.info(f"Ansatz: Linear XY-Mixer, depth={cfg.depth} | ZNE={cfg.use_zne}")
    logger.info(f"Fourier modes: {cfg.fourier_modes * 2} params → {cfg.depth}-layer circuit")
    logger.info(f"Shots: {cfg.dynamic_shots}  (base={cfg.base_shots}, CVaR α={cfg.cvar_alpha})")

    problem   = PortfolioProblem(cfg)
    runner    = QuantumRunner(cfg, problem)
    optimizer = BoTorchOptimizer(
        n_fourier_params=2 * cfg.fourier_modes,
        n_init=cfg.sobol_init_iters,
        logger=logger,
    )

    best_cvar    = float("inf")
    history_cvar: List[float] = []

    logger.info(f"Starting loop: {cfg.bo_iters} iters ({cfg.sobol_init_iters} Sobol init)")
    start_time = time.time()

    try:
        for i in range(cfg.bo_iters): 
            iter_start = time.time()

            x_next     = optimizer.suggest()
            cvar_e, vr = runner.run(x_next)
            optimizer.observe(x_next, cvar_e)

            if cvar_e < best_cvar:
                best_cvar = cvar_e

            history_cvar.append(best_cvar)
            phase = "SOBOL" if i < cfg.sobol_init_iters else "BO-TR"
            logger.info(
                f"[{phase}] Iter {i+1:02d} | CVaR: {cvar_e:.4f} | "
                f"Best: {best_cvar:.4f} | Valid: {vr:.0%} | {time.time()-iter_start:.1f}s"
            )
            tracker.log_step(i, {"cvar": cvar_e, "raw_valid": vr})

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
    finally:
        logger.info(f"Done in {time.time()-start_time:.2f}s. Best CVaR: {best_cvar:.4f}")
        tracker.save()

        plt.figure(figsize=(10, 5))
        plt.plot(history_cvar, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.axvline(x=cfg.sobol_init_iters - 1, color='r', linestyle='--', label='Sobol → BO-TR')
        plt.title(f"Hardware-Aware QAOA  |  Linear XY + Fourier + ZNE  |  depth={cfg.depth}")
        plt.xlabel("Iteration")
        plt.ylabel("Best CVaR")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.argv.pop()
    else:
        main()
