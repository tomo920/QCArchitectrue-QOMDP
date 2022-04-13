import numpy as np


def apply_operator(X, U):
    """
    Calculate X† U X
    """

    return np.dot(np.dot(X.conj().T, U), X)


def get_observation_probability(s, a, A, observation_set):
    p_o = []
    for o in observation_set:
        A_a_o_s = np.dot(A[a][o], s)
        p_o.append(np.dot(A_a_o_s.conj().T, A_a_o_s).real[0][0])
    return p_o


def state_update(s, a, o, A):
    """
    Perform state update by transition operators A
    when state is s, action the agent executed is a and observation the agent observed is o.
    |s_> = A[a][o]|s>/sqrt(p(o||s>, a))
    p(o||s>, a) = <s|A[a][o]†A[a][o]|s>

    Args
    ----------
    s: list
        Quantum state before update.
    a: int
        Action agent executed.
    o: int
        Observation agent observed.
    A: dictionary
        Set of transition operators.
        A[a][o] is transition operator matrix of executing action a and
        getting observation o.

    Returns
    ----------
    s_: list
        Quantum state after update.
    """

    A_a_o_s = np.dot(A[a][o], s)
    p_o_s_a = np.dot(A_a_o_s.conj().T, A_a_o_s).real[0][0]

    s_ = A_a_o_s / np.sqrt(p_o_s_a)

    return s_
