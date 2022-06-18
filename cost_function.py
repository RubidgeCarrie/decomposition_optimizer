from gates import *


def cphase_dephasing_cost_function(params_ls, target_gate, two_qubit_gate, noise_superop_ls, phi):
    """Assumes simple circuit with two layers, CPhase error and dephasing weighted by eps.
    Args:
        params_ls (ls): 18 parameters for the 6 unitary gates.
        target_gate (ndarray): Target gates matrix.
        two_qubit_gate (ndarray): Two qubit gate used in decomposition.
        noise_superop_ls (ls): List of noise superoperators in circuit.
        phi (float): CPhase error in radians.
        """

    # Cphase error on sqrt iswap conjugate gate
    error_gate = FermiSimGate(0, phi, 0, 0, 0).matrix
    error_two_qubit_gate = two_qubit_gate @ error_gate

    # Computing dephasing noise superoperator for 'degree' of dephasing eps
    infidelity = 1 - cphase_dephasing_infidelity(error_two_qubit_gate, params_ls,
                                                 target_gate, noise_superop_ls)

    return infidelity


def cphase_dephasing_infidelity(two_qubit_gate, param_ls, target_gate,
                                noise_superop_ls):
    """Computes the average fidelity using the average entanglement fidelity given
    in terms of Superoperator representation.
    Args:
        two_qubit_gate (ndarray): Two qubit gate used in gate decomposition.
        param_ls (ls): 18 float values for the six unitary gates.
        target_gate (ndarray): Target gate being decomposed in terms of noisy gates.
        noise_superop_ls (ndarray): Superoperator for noise model.
    Returns:
        """

    no_qubits = 2
    no_layers = 2

    # Computing the superoperator for the first set of u3 gates
    params_gate_1 = U3Gate(param_ls[0], param_ls[1], param_ls[2]).matrix
    params_gate_2 = U3Gate(param_ls[3], param_ls[4], param_ls[5]).matrix
    u3_layer = np.kron(params_gate_1, params_gate_2)
    u3_layer_superop = np.kron(u3_layer, np.conjugate(u3_layer))

    # Computing the superoperator for the two qubit gate and noise channel (given superoperator)
    s_n = u3_layer_superop
    two_qubit_superop = np.kron(two_qubit_gate, np.conjugate(two_qubit_gate))

    # Computing the superoperator for the rest of the n layers
    for layer in range(no_layers):
        layer = layer + 1
        params_gate_1 = param_ls[(no_qubits * 3 * (layer)):(no_qubits * 3 * (layer) + 3)]
        params_gate_2 = param_ls[(no_qubits * 3 * (layer) + 3):(no_qubits * 3 * (layer + 1))]
        u3_gate_1 = U3Gate(*params_gate_1).matrix
        u3_gate_2 = U3Gate(*params_gate_2).matrix
        u3_layer = np.kron(u3_gate_1, u3_gate_2)
        u3_layer_superop = np.kron(u3_layer, np.conjugate(u3_layer))

        s_n = s_n@two_qubit_superop@noise_superop_ls[::-1][layer-1]@u3_layer_superop

    # Computing the superoperator for the target gate
    s_u = np.kron(target_gate, np.conjugate(target_gate))
    d = no_qubits**2
    fidelity = (np.trace(np.conjugate(s_u).T @ s_n)/d + 1) / (d + 1)

    return fidelity


def average_fidelity(U, V):
    """Calculates the average fidelity of two unitaries U and V"""

    tr = np.trace(U.conj().T @ V)
    d = U.shape[0]

    return (abs(tr)**2/d + 1) / (d + 1)