from scipy.optimize import minimize
from cost_function import *


def approx_error_free_decomp(target_gate, two_qubit_gate):
    """Find 18 parameters that give the target gate with three layers of U3.

    Args:
        target_gate (ndarray): Matrix of target gate.
        two_qubit_gate (ndarray): Two qubit gate used in decomposition.
        eps (float): Amount of dephasing.

    Returns:
        Optimal parameters found using scipy.

    """
    # Set dephasing and cphase error to 0 to find decomposition before errors
    phi = 0
    initial_params = [2*np.pi*np.random.rand() for i in range(18)]

    res = minimize(cphase_dephasing_cost_function,
                   initial_params,
                   args=(target_gate, two_qubit_gate, [np.eye(16, 16), np.eye(16, 16)], phi),
                   method='BFGS',
                   options={'maxiter': 1000 * 30})
    print(res.fun)
    circuit = circuit_unitary(two_qubit_gate, res.x)

    print(1 - average_fidelity(circuit, target_gate))
    return res.x


def dephasing_optimizer_any_target(target_gate, two_qubit_gate, noise_ls, eps_ls, phi=0):

    """Perform scipy optimization to minimize infidelity for different cephase errors (phi)
    and different amounts of dephasing (eps).

    Args:
        target_gate (ndarray): Gate targetted by the decomposition of set noisy 2 qubit
        gates and variable unitaries.
        two_qubit_gate (ndarray): Two qubit gate used in gate decomposition.
        noise_ls (ls): 'double', 'single', 'none'
        eps_ls (ls): List of amount of dephasing 0-1 where 1 is maximum.
        phi (float): Degree for CPhase error in radian.

    Output:
        Dictionary of results
        """

    # Find approximate values for U3 parameters without errors
    initial_params_ls = approx_error_free_decomp(target_gate, two_qubit_gate)

    # Create empty dictionary for results
    result_dict = {'pre_error_exact': [], 'optimization_w_error': [], 'optimization_w_error_rand': []}

    # Set count for number of unitaries wanted
    count = 0
    max_count = 3
    unitary_type_ls = []
    unitary_ls = []
    unitary_eps_ls = []
    fidelity_ls = []

    for eps in eps_ls:
        # Create a set of random parameters
        rand_params = [2 * np.pi * np.random.rand() for i in range(18)]

        # Infidelity if we keep optimal parameters for no errors:
        noise_mapping_dict = {'double': DephasingNoiseSuperop(eps).matrix,
                              'single': SingleDephasingNoiseSuperop(eps).matrix, 'none': np.eye(16, 16)}

        noise_superop_ls = [noise_mapping_dict[i] for i in noise_ls]
        pre_error_infid = cphase_dephasing_cost_function(initial_params_ls, target_gate, two_qubit_gate,
                                                         noise_superop_ls, phi)

        # Optimize for error decomposition:

        res_opt = minimize(cphase_dephasing_cost_function,
                       initial_params_ls,
                       args=(target_gate, two_qubit_gate, noise_superop_ls, phi),
                       method='BFGS',
                       options={'maxiter': 1000 * 30})

        res_rand = minimize(cphase_dephasing_cost_function,
                       rand_params,
                       args=(target_gate, two_qubit_gate, noise_superop_ls, phi),
                       method='BFGS',
                       options={'maxiter': 1000 * 30})

        result_dict['pre_error_exact'].append(pre_error_infid)
        result_dict['optimization_w_error'].append(res_opt.fun)
        result_dict['optimization_w_error_rand'].append(res_rand.fun)

        # Save resulting unitaries for comparison
        if count < max_count:
            if ((pre_error_infid - res_opt.fun) > 0.01) | ((pre_error_infid - res_rand.fun) > 0.01):
                count += 1
                unitary_eps_ls.append(str(eps))
                if res_rand.fun < res_opt.fun:
                    unitary_type_ls.append('rand')
                    circuit = circuit_unitary(two_qubit_gate, res_rand.x)
                    unitary_ls.append(circuit)
                    fidelity_ls.append(average_fidelity(circuit, target_gate))
                else:
                    unitary_type_ls.append('optimization')
                    circuit = circuit_unitary(two_qubit_gate, res_opt.x)
                    unitary_ls.append(circuit)
                    fidelity_ls.append(average_fidelity(circuit, target_gate))

    unitary_results_ls = list(zip(unitary_type_ls, unitary_eps_ls, unitary_ls, fidelity_ls))

    return result_dict, unitary_results_ls


def circuit_unitary(two_qubit_gate, param_ls):
    """Calculates the unitary assuming 3 layers of two U3 + two-qubit gates"""

    # Compute first layers.
    params_gate_1 = U3Gate(param_ls[0], param_ls[1], param_ls[2]).matrix
    params_gate_2 = U3Gate(param_ls[3], param_ls[4], param_ls[5]).matrix
    u3_layer = np.kron(params_gate_1, params_gate_2)

    no_layers = 2
    no_qubits = 2

    unitary = u3_layer

    # Computing the superoperator for the rest of the n layers.
    for layer in range(no_layers):
        layer = layer + 1
        params_gate_1 = param_ls[(no_qubits * 3 * (layer)):(no_qubits * 3 * (layer) + 3)]
        params_gate_2 = param_ls[(no_qubits * 3 * (layer) + 3):(no_qubits * 3 * (layer + 1))]
        u3_gate_1 = U3Gate(*params_gate_1).matrix
        u3_gate_2 = U3Gate(*params_gate_2).matrix
        u3_layer = np.kron(u3_gate_1, u3_gate_2)
        unitary = unitary @ two_qubit_gate @ u3_layer

    return unitary

