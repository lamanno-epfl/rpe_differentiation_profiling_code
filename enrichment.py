import numpy as np
import numpy_groupies


def enrichment_score(
    matrix: np.ndarray, labels: np.ndarray, psc1: float = 0.1, psc2: float = 0.01
) -> np.ndarray:
    """Find features enriched in each cluster of samples
    
    Arguments
    ---------
    matrix: array, shape=(features, samples)
        Array containing the raw counts
        
    labels: array, shape=(samples,)
        The cluster names or ixs
        
    psc1: float, default=0.1
        pseudo count to use in the score
        
    psc2: float, default=0.01
        pseudo count to use in the score
    
    Returns
    -------
    enrichment_score: np.array, shape=(features, clusters))
        Enrichment score for each cluster.
        Note that column i corresponds to cluster np.unique(labels)[i]
        clusters = len(np.unique(labels))
        
    """
    uq_labels, uq_ix = np.unique(labels, return_inverse=True)
    n_labels = len(uq_labels)
    n_genes, n_cells = matrix.shape

    # Number of cells per cluster
    sizes = np.bincount(uq_ix, minlength=n_labels)
    # Number of nonzero values per cluster
    nnz = numpy_groupies.aggregate(uq_ix, matrix > 0, axis=1)
    # Mean value per cluster
    means = numpy_groupies.aggregate(uq_ix, matrix, func="mean", axis=1)
    # Non-zeros and means over all cells
    (nnz_overall, means_overall) = (matrix > 0).sum(1), matrix.mean(1)
    # Scale by number of cells
    f_nnz = nnz / sizes
    f_nnz_overall = nnz_overall / n_cells

    # Means and fraction non-zero values in other clusters (per cluster)
    means_other = ((means_overall * n_cells)[None].T - (means * sizes)) / (n_cells - sizes)
    f_nnz_other = ((f_nnz_overall * n_cells)[None].T - (f_nnz * sizes)) / (n_cells - sizes)
    enrichment = (f_nnz + psc1) / (f_nnz_other + psc1) * (means + psc2) / (means_other + psc2)

    return enrichment


def extract_markers(
    escore: np.ndarray, names: np.ndarray = None, n_markers: int = 20
) -> np.ndarray:
    """Determine unique markers for each cluster using the enrichment score
    
    Arguments
    ---------
    escore: np.ndarray, shape=(features, clusters)
        The matrix of enrichment scores as returned by `enrichment_score`
    
    names: np.ndarray, (usually dtype=str)
        If this is provided the results will be given in term of names
        Otherwise indexes are returned
        
    n_markers: int
        Number of markers to extract per cluster
    
    Returns
    -------
    markers:
        If `n_markers` are available for each cluster returns an array
        If less than `n_markers` are available for each cluster returns a list
        Depending on whether names is passed names or indexes are returned
    
    Note
    ----
    The `escore` can be calculated by the function enrichment_score
    """
    markers = [list() for i in range(escore.shape[1])]
    clust_where_max = np.argmax(escore, 1)
    order_genes = np.argsort(escore[range(escore.shape[0]), clust_where_max])[::-1]
    for i in order_genes:
        cwm = clust_where_max[i]
        if len(markers[cwm]) < n_markers:
            markers[cwm].append(i)
    if all([len(i) == len(markers[0]) for i in markers]):
        # Can make an array
        out = np.zeros((len(markers[0]), escore.shape[1]), dtype=np.intp)
        for i in range(len(markers)):
            out[:, i] = np.array(markers[i], dtype=np.intp)
        if names is not None:
            return names[out]
        else:
            return out
    else:
        print(f"Less than {n_markers} markes were found returning a list")
        for i in range(len(markers)):
            markers[i] = np.array(markers[i], dtype=np.intp)
        if names is not None:
            return [names[i] for i in markers]
        else:
            return markers


def extract_enriched(
    escore: np.ndarray, names: np.ndarray = None, n_enriched: int = 20
) -> np.ndarray:
    """Determine enriched features (non-unique) for each cluster using the enrichment score
    
    Arguments
    ---------
    escore: np.ndarray, shape=(features, clusters)
        The matrix of enrichment scores as returned by `enrichment_score`
    
    names: np.ndarray, (usually dtype=str)
        If this is provided the results will be given in term of names
        Otherwise indexes are returned
        
    n_enriched: int
        Number of erniched genes to extract
    
    Returns
    -------
    enriched: np.ndarray, shape=(n_markers, clusters)
        Depending on whether names is passed names or indexes are returned
    
    Note
    ----
    The `escore` can be calculated by the function enrichment_score
    """
    scores_ixs = np.argsort(escore, 0)[::-1, :][:n_enriched, :]
    if names is not None:
        return names[scores_ixs]
    else:
        return scores_ixs
