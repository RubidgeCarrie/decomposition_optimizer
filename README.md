# Decomposition Optimizer for a Simple Noisy Quanutm Circuit

Decomposition optimizer is Python code to optimize a circuit decomposition into a target unitary using two user defined two-qubit gates and 6 general two qubit gates that are optimized over to minimize the final fidelity between it and our target gate.

## Installation

Requires numpy and scipy.

```bash
pip install numpy
pip install scipy
```

## Usage

Example use to plot decompositon of iSWAP gate into CNOT and 6 unitaries over various levels of dephasing assuming noise on two qubits after each two-qubit gate.

```python
form gates import *
from target_decompositon_comparison import *

# Create function to plot results
def rand_opt_general_plot(idx_min, idx_max, string_title, result_dict, epsilon_ls):
    plt.figure(figsize=(14, 5))
    plt.title(string_title, fontsize=15, y=1)
    opt_inf_ls = result_dict['optimization_w_error']
    opt_inf__rand_ls = result_dict['optimization_w_error_rand']
    noop_inf_ls = result_dict['pre_error_exact'] 
    cut_eps_ls = epsilon_ls[idx_min:idx_max]
    plt.plot(cut_eps_ls, opt_inf_ls[idx_min:idx_max], color = 'indianred', label = 'BFGS optimized opt start')
    plt.plot(cut_eps_ls, opt_inf__rand_ls[idx_min:idx_max], color = 'lightgreen', label = 'BFGS optimized random start')
    plt.plot(cut_eps_ls, noop_inf_ls[idx_min:idx_max], color = 'darkcyan', label = 'General decomposition')
    plt.xlabel('Dephasing amount $\epsilon$', fontsize=12)
    plt.ylabel('Unitary infidelity', fontsize=12)
    plt.legend()
    plt.grid()
    return 
   
# Setting up inputs
noise_gate_ls = ['single', 'none']
theta = 8*(np.pi/180)
target_gate = iSWAPGate(theta).matrix
two_qubit_gate = cnot()
eps_ls = np.arange(0, 1, 0.001)
one_single_iswap_cnot_dict, unitary_results_ls = dephasing_optimizer_any_target(target_gate, two_qubit_gate, noise_gate_ls, eps_ls, phi=0)

# Setting graph parameters
idx_min = 800
idx_max = None
string_title = "Optimized BFGS decomposition vs general with dephasing error $\epsilon$ for target iSWAP(8 $\degree$) using CNOT"
rand_opt_general_plot(idx_min, idx_max, string_title, one_single_iswap_cnot_dict, eps_ls)

```
