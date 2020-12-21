from scipy.spatial.distance import pdist, squareform
import numpy as np


def velocity_coordination(
    Sx_sz: np.ndarray, delta_S: np.ndarray, n_genes: int = 20
) -> np.ndarray:
    """Calculate velocity coordination

    Arguments
    ---------
    Sx_sz: np.ndarray, shape=(genes, cells)
        Imputed size normalized expression matrix
        (e.g. VelocytoLoom.Sx_sz)

    delta_S: np.ndarray, shape=(genes, cells)
        The velocity used for the extrapulation (not multiplied for deltat)
        (e.g. VelocytoLoom.delta_S)

    n_genes: int
        The number of gene neighbours to use to calculate the score

    Returns
    -------
    np.ndarray (genes,)
        A vector containing the estimated velocity coordination per each gene.
    """

    R = 1 - squareform(pdist(Sx_sz, "correlation"))
    vR = 1 - squareform(pdist(delta_S, "correlation"))

    ixsort = np.argsort(R, 1)[:, ::-1][:, 1 : n_genes + 1]  # nearest neighbours indexes

    # Estiamte of the correlation of velocities of different genes
    vR_topavg = vR[np.arange(R.shape[0])[:, None], ixsort].mean(
        1
    )  # estimates from neigherest neigbours
    vR_bg = (
        vR.mean(1) - 1 / vR.shape[1]
    )  # background exprectation (correction for the diagonal value that is 1)

    coordination = vR_topavg - vR_bg

    return coordination
