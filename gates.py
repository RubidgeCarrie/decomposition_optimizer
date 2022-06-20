import numpy as np

class Circuit:
    """Generic circuit defined using dictionary of all gates per layer and
        resulting unitary."""

    def __init__(self, num_qubits, gate_dict={}, unitary=None):
        """
        Args:
            num_qubits (int): Number of qubit in circuit.
            gate_dict (dict): Key is the "layer" number and value is a list of gates.
            unitary (ndarray): Total unitary of circuit.
         
        """
        self.num_qubits = num_qubits
        self.unitary = unitary
        self.gate_dict = gate_dict
        if not self.unitary:
            self.unitary = np.eye(2**num_qubits)
        else:
            self.unitary = unitary
        self.no_layers = len(self.gate_dict)

    def add_layer(self, gate_ls):
        """ Adds gates in gate_ls to circuit assuming they are in the form of unitaries acting on all qubits."""

        unitary_layer = gate_ls[0]

        if len(gate_ls) > 1:
            for gate in gate_ls[1:]:
                unitary_layer = np.kron(unitary_layer, gate)

        self.unitary = self.unitary@unitary_layer
        self.no_layers = self.no_layers + 1
        self.gate_dict[self.no_layers] = gate_ls


class U3Gate:
    """One qubit unitary in SU(2)."""
    def __init__(self, theta, phi, lam):
        self.theta = theta
        self.phi = phi
        self.lam = lam

    @property
    def matrix(self):
        return np.array([
                [np.cos(self.theta/2),
                -np.exp(1j*self.lam)*np.sin(self.theta/2)
                ],
                [np.exp(1j*self.phi)*np.sin(self.theta/2),
                np.exp(1j*self.lam + 1j*self.phi)*np.cos(self.theta/2)
                ]])


class FermiSimGate:

    def __init__(self, theta, phi, zeta, chi, eta):
        """ Fermionic simulation gate.

            Args:
            theta: iSWAP angle.
            phi: CPhase angle.
            zeta, chi, eta: single-qubit phase angles.
            """
        self.theta = theta
        self.phi = phi
        self.zeta = zeta
        self.chi = chi
        self.eta = eta
        
    @property
    def matrix(self):

        return np.array([
            [1, 0, 0, 0],
            [0, np.exp(-1j*(self.eta + self.zeta))*np.cos(self.theta), -1j*np.exp(-1j*(self.eta - self.chi))*np.sin(self.theta), 0 ],
            [0, -1j*np.exp(-1j*(self.eta + self.chi))*np.sin(self.theta), np.exp(-1j*(self.eta - self.zeta))*np.cos(self.theta), 0],
            [0, 0, 0, np.exp(-1j*(2*self.eta + self.phi))]
            ])

class DephasingNoiseSuperop():

    def __init__(self, eps):
        """ Superoperator for dephasing channels on two qubits."""
        self.eps = eps

    @property
    def matrix(self):

        noise_superop = np.zeros((16, 16))
        noise_superop[0, 0] = 1
        noise_superop[1, 1] = 1 - self.eps
        noise_superop[2, 2] = 1 - self.eps
        noise_superop[3, 3] = (1 - self.eps) ** 2
        noise_superop[4, 4] = 1 - self.eps
        noise_superop[5, 5] = 1
        noise_superop[6, 6] = (1 - self.eps) ** 2
        noise_superop[7, 7] = 1 - self.eps
        noise_superop[8, 8] = 1 - self.eps
        noise_superop[9, 9] = (1 - self.eps) ** 2
        noise_superop[10, 10] = 1
        noise_superop[11, 11] = 1 - self.eps
        noise_superop[12, 12] = (1 - self.eps) ** 2
        noise_superop[13, 13] = 1 - self.eps
        noise_superop[14, 14] = 1 - self.eps
        noise_superop[15, 15] = 1

        return noise_superop


class SingleDephasingNoiseSuperop():
    """Superoperator for dephasing angle on single qubit"""

    def __init__(self, eps):
        self.eps = eps

    @property
    def matrix(self):

        noise_superop = np.zeros((16, 16))
        noise_superop[0, 0] = 1
        noise_superop[1, 1] = 1
        noise_superop[2, 2] = 1 - self.eps
        noise_superop[3, 3] = 1 - self.eps
        noise_superop[4, 4] = 1
        noise_superop[5, 5] = 1
        noise_superop[6, 6] = 1 - self.eps
        noise_superop[7, 7] = 1 - self.eps
        noise_superop[8, 8] = 1 - self.eps
        noise_superop[9, 9] = 1 - self.eps
        noise_superop[10, 10] = 1
        noise_superop[11, 11] = 1
        noise_superop[12, 12] = 1 - self.eps
        noise_superop[13, 13] = 1 - self.eps
        noise_superop[14, 14] = 1
        noise_superop[15, 15] = 1

        return noise_superop

class iSWAPGate:
    def __init__(self, theta):
        self.theta = theta

    @property
    def matrix(self):

        return np.array([
               [1, 0, 0, 0],
               [0, np.cos(self.theta), -1j*np.sin(self.theta), 0],
               [0, -1j*np.sin(self.theta), np.cos(self.theta), 0],
               [0, 0, 0, 1]
               ])

class RZ:
    def __init__(self, theta):
        self.theta = theta

    @property
    def matrix(self):

        return np.array([
               [np.exp(-1j*self.theta/2) , 0],
               [0 , np.exp(1j*self.theta/2)]
               ])


Z = np.array([[1, 0], [0, -1]])


def cnot():

    return np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])


def cz():

    return np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, -1]])


