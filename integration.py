import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSSVD
from scipy import linalg
from scipy.sparse.linalg import svds
from scipy import sparse
from typing import Iterable, Tuple, Union, Any, List, Dict
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage
import logging
import scanpy, anndata

from typing import Tuple

def preprocess_object_list(object_list):
    """
    Checks that input object list (count tables) are in appropriate dictionary format
    object_list: list of dictionary or scanpy anndata objects

    """
    types = [type(i) for i in object_list]
    
    if (len(set(types)) != 1):
        print("Error: objects are of multiple different datatypes")
        return
    
    data_type = types[0]
    if (data_type == dict):
        return object_list # already in proper format
    elif (data_type == anndata._core.anndata.AnnData):
        new_object_list = []
        for obj in object_list:
            counts = np.array(obj.X)
            
            if (type(counts) != np.ndarray):
                counts = obj.X.toarray()
            
            genes = np.array(obj.var.index)
            new_object_list.append({"normalized_array": counts, "GeneID": genes})
    return new_object_list

def random_cellsplit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the counts in a matrix in 2 randomly
    
    Parameters
    ----------
    X: np.ndarray, shape=(cells, genes) or (genes, cells)
        The data matrix. The data should be integers
        
    Returns
    -------
    Tuple of the two matrix
    """
    Xa = np.random.binomial(X.astype(np.int), 0.5, X.shape)
    Xb = X - Xa
    return Xa, Xb

def random_pseudo_cellsplit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split the counts in a matrix in 2 randomly even if the matrix has float
    
    Parameters
    ----------
    X: np.ndarray, shape=(cells, genes) or (genes, cells)
        The data matrix. The data can be floats
        
    Returns
    -------
    Tuple of the two matrix
    """
    eps = 1e-8
    Xa = np.random.normal(0.5 * X, np.sqrt(0.25 * X) + eps, X.shape)
    Xa = np.clip(Xa, 0, X)
    Xb = X - Xa
    return Xa, Xb

def svd(
    X: np.ndarray, n_components: int, solver_type: str = "truncated", rseed: int = 19900715
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD as computed inside scikit learn

    Arguments
    ---------
    X: np.ndarray
        If shape=(samples, features) then U * S is the sample in principal component space
        X = U * S * Vt
        X_pca = X * V = U * S * V^T * V = U * S

    n_components: int
        The number of components to retain.
        If 'truncated' the rest are never calculated

    solver_type: str, default='truncated'
        The kind of method to compute svd.
        It can be 'full' or 'truncated'

    rseed: int
        Seed for random number generator used if `solver_type` = 'truncated'
    
    See:
    https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/decomposition/_pca.py#L104
    """
    # Center data
    assert np.allclose(0, np.mean(X, axis=0), rtol=1e-2, atol=1e-03), "Matrix was not centered"

    if solver_type == "full":
        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]
    elif solver_type == "truncated":
        # use arpack
        np.random.seed(rseed)
        U, S, Vt = svds(X, k=n_components, v0=np.random.uniform(-1, 1, size=min(X.shape)))
        # arpack output is ordered differently
        S = S[::-1]
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U[:, ::-1], Vt[::-1])
    else:
        raise NotImplementedError(f"solver_type = {solver_type} is not implemented")
    return U, S, Vt


def expand_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xG, yG = np.meshgrid(x, y)  # create the actual grid
    ixs = np.triu_indices(xG.shape[0], 1)
    return np.column_stack([xG.flat[ixs], yG.flat[ixs]])


def svd_flip(
    u: np.ndarray, v: np.ndarray, u_based_decision: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : np.ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : np.ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def kneighbors(
    data: np.ndarray,
    query: np.ndarray = None,
    k: int = 10,
    method: str = "deterministic",
    return_distance: bool = False,
    metric: str = "euclidean",
    n_jobs: int = 32,
    out_type: str = "indexes",
) -> Union[sparse.csr_matrix, np.ndarray]:
    """Function that wraps different kind of knn searches

    Arguments
    ---------
    data: np.ndarray
        The data to use as a reference

    query: np.ndarray = None
        Optional, the points whose neighbors will be found
        If not provided query = data

    k: int = 10
        Number of neighbors
    
    method: str = "deterministic",
        One of "deterministic", "approximated"

    metric: str = "euclidean"
        One accepted by sklearn.neighbors.NearestNeighbors 
    
    n_jobs: int = 16
        Number of parallel jobs

    out_type: str, default='indexes'
        'sparse' or 'indexes'
        Whether returning the neighborhood as a graph or as indexes

    Returns
    -------
    If out_type == "sparse"
        neighbour_graph: sparse.csr_matrix, shape shape=(query.shape[0], data.shape[0])
    If out_type == "indexes"
        neigh_ind: np.ndarray, shape=(query.shape[0], k)

    """
    if method == "deterministic":
        nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=n_jobs, leaf_size=30)
        nn.fit(data)
        if query is None:
            if out_type == "indexes":
                return nn.kneighbors(return_distance=return_distance)
            elif out_type == "sparse":
                return nn.kneighbors_graph(
                    mode="connectivity" if not return_distance else "distance"
                )
            else:
                raise ValueError("out_type {out_type} is not supported")
        else:
            if out_type == "indexes":
                return nn.kneighbors(query, return_distance=return_distance)
            elif out_type == "sparse":
                return nn.kneighbors_graph(
                    query, mode="connectivity" if not return_distance else "distance"
                )
            else:
                raise ValueError("out_type {out_type} is not supported")
    elif method == "approximated":
        raise NotImplementedError("Running with knn with annoy was not implemented")
        # NOTE: see https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html#sphx-glr-auto-examples-neighbors-approximate-nearest-neighbors-py
    else:
        raise ValueError(f"method={method} is not supported by kneighbors")


def find_nn(
    self_spaces: Tuple[np.ndarray, np.ndarray] = None,
    joint_space: np.ndarray = None,
    reciprocal_spaces: Tuple[np.ndarray, np.ndarray] = None,
    internal_neighbors: np.ndarray = None,
    n_cells_i: int = None,
    n_cells_j: int = None,
    k: int = 20,
    nn_method: str = "deterministic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find nearest neighbors between two datasets

    Arguments
    ---------
    self_spaces: Tuple[np.ndarray, np.ndarray],
    
    joint_space: np.ndarray,
    
    reciprocal_spaces: Tuple[np.ndarray, np.ndarray],
    
    internal_neighbors: np.ndarray,
    
    n_cells_i: int
    
    n_cells_j: int
    
    k: int, default = 20,
    
    nn_method: str, default='deterministic'
    

    Returns
    -------
    (nnaa, nnab, nnba, nnbb)
    The neighborhoods in different spaces
    """
    if bool(reciprocal_spaces is not None) and bool(joint_space is not None):
        raise ValueError("joint_space and reciprocal_spaces are non compatible arguments")

    if bool(self_spaces is not None) and bool(internal_neighbors is not None):
        raise ValueError("self_spaces and internal_neighbors are non compatible arguments")

    if internal_neighbors is not None:
        # if neighborhoods were precomputed
        nnaa = internal_neighbors[0]
        nnbb = internal_neighbors[1]
    elif self_spaces is not None:
        # self_spaces[0] contains `n_cells_i` points of sample i in their preferred space to calculate their neighborhood
        nnaa = kneighbors(data=self_spaces[0], k=k, method=nn_method)
        # self_spaces[1] contains `n_cells_j` points of sample j in their preferred space to calculate their neighborhood
        nnbb = kneighbors(data=self_spaces[1], k=k, method=nn_method)
    else:
        raise ValueError("This cannot be! Check your input")

    if reciprocal_spaces is not None:
        # This if two spaces are provided
        # In j-space find nearest neighbors of points of sample i among points of sample j
        # reciprocal_spaces[1] = object_pair["pca_j"]
        # nnab = kneighbors(data=reciprocal_spaces[1][:n_cells_j, :], query=reciprocal_spaces[1][n_cells_j:, :], k=k, method=nn_method)
        nnba = kneighbors(
            data=reciprocal_spaces[1][1], query=reciprocal_spaces[1][0], k=k, method=nn_method
        )
        # In i-space find nearest neighbors of points of sample j among points of sample i
        # reciprocal_spaces[0] = object_pair["pca_i"]
        # nnba = kneighbors(data=reciprocal_spaces[0][:n_cells_i, :], query=reciprocal_spaces[0][n_cells_i:, :], k=k, method=nn_method)
        nnab = kneighbors(
            data=reciprocal_spaces[0][1], query=reciprocal_spaces[0][0], k=k, method=nn_method
        )
    elif joint_space is not None:
        # This if one joint space is provided
        # joint_space = object_pair["cca"]
        # Find nearest neighbors of points of sample i among points of sample j
        nnab = kneighbors(data=joint_space[1], query=joint_space[0], k=k, method=nn_method)
        # Find nearest neighbors of points of sample j among points of sample i
        nnba = kneighbors(data=joint_space[0], query=joint_space[1], k=k, method=nn_method)
    else:
        raise ValueError("This cannot be! Check your input")
    return nnaa, nnab, nnba, nnbb


def find_anchor_pairs(
    nnab: sparse.csr_matrix, nnba: sparse.csr_matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """Find anchors as mutual nearest neighbors between the graphs

    Arguments
    ---------
    nnab: sparse.csr_matrix, shape=(points_in_i, points_in_j)
        Neighborhoods calculated in j-space or joint space
    
    nnba: sparse.csr_matrix, shape=(points_in_j, points_in_i)
        Neighborhood calculated in i-space or joint space

    Returns
    -------
    anchor_pairs: np.ndarray, shape=(paired_points, w)
        first column contains the indices of points of the pair from dataset i (the points of nnab.shape[0])
        seconds column contains the indices of points of the pair from dataset j (the points of nnab.shape[1])
    anchor_scores: np.ndarray, shape=(paired_points,)
        Contain the scores (for now is all ones
    """
    assert isinstance(
        nnab, sparse.csr_matrix
    ), "find_anchor_pairs only accept graph representation"
    # Find mutual nearest neighbors
    # print(f"{nnab.shape}, {nnba.shape}")
    points_i, points_j = nnab.multiply(nnba.T).nonzero()
    anchor_pairs = np.column_stack([points_i, points_j])
    # Note better store them in different arrays becouse of differnet data type
    anchor_scores = np.ones(points_i.shape)
    return anchor_pairs, anchor_scores


def top_features(
    space_loadings: np.ndarray, dim: int = 0, nfeatures: int = 20, balanced: bool = False
) -> np.ndarray:
    """Find the features that have highest loadings

    Arguments
    ---------
    space_loadings: np.ndarray
        Corresponds to the values of V (i.e. Vt.T) the right matrix basis in the SVD 

    dim: int = 0
        The index of the singular vector (pc) whose top feature are to be extracted
    
    nfeatures: int = 20
        Number of features to extract
    
    balanced: bool = False
        Equally extract from negative and positive loadings
        (in most cases this is the right thing to do)

    Return
    ------
    np.ndarray
        The top features with higher loadings
    """
    # space_loadings = Vit.T
    loadings = space_loadings[:, dim]
    if balanced:
        nfeatures = int(nfeatures / 2)
        ixs = np.argsort(loadings)
        positive = ixs[::-1][:nfeatures]
        negative = ixs[:nfeatures]
        top = np.concatenate([positive, negative])
    else:
        ixs = np.argsort(np.abs(loadings))[::-1]
        top = ixs[:nfeatures]
    return top


def top_dim_features(
    space_loadings: np.ndarray,
    features_per_dim: int = 100,
    dims: Union[int, np.ndarray] = 10,
    max_features: int = 200,
) -> np.ndarray:
    """Get top n features across given set of dimensions

    Arguments
    ---------
    space_loadings: np.ndarray, shape=(n_features, n_dims)
        Corresponds to V (i.e. Vt.T) in the svd 

    features_per_dim: int, default = 100
        How many features to consider per dimension

    dims: int, default = 10
        Which dimensions to use

    max_features: int, default=200
        Number of features to return at most

    Returns
    -------
    np.ndarray
        indexes of the top features extracted from a set of pcs
    """
    f: np.ndarray
    if isinstance(dims, int):
        max_features = max(dims * 2, max_features)
        range_dims = range(dims)
    else:
        max_features = max(len(dims) * 2, max_features)
        range_dims = dims
    num_features = np.zeros(features_per_dim)
    for y in range(features_per_dim):
        feats: List = []
        for x in range_dims:
            f = top_features(space_loadings, dim=x, nfeatures=y + 1, balanced=True)
            feats.extend(f)
        num_features[y] = len(set(feats))
    max_per_pc = np.argmax(num_features[num_features < max_features])
    features: List = []
    for x in range_dims:
        f = top_features(space_loadings, dim=x, nfeatures=max_per_pc + 1, balanced=True)
        features.extend(f)
    return np.array(features)


def norm2_normalize(X: np.ndarray, axis: int = 1, norm2_clip: float = None) -> np.ndarray:
    norm_factor = np.sqrt(np.sum(X ** 2, axis))
    if norm2_clip is None:
        # Just avoid division by zero
        small_norm = norm_factor < 1e-8
        norm_factor[~small_norm] = 1.0 / norm_factor[~small_norm]  # avoid division by 0
        norm_factor[small_norm] = np.min(norm_factor[~small_norm])
    else:
        norm_factor = 1.0 / np.maximum(norm_factor, norm2_clip)
    if axis == 1:
        return X * norm_factor[:, None]
    elif axis == 0:
        return X * norm_factor
    else:
        raise ValueError(f"axis = {axis} is not supported")


def filter_anchors(
    anchor_pairs: np.ndarray,
    anchor_scores: np.ndarray,
    features: np.ndarray,
    data_tables: Tuple[np.ndarray, np.ndarray],
    k_filter: int = 200,
    nn_method: str = "deterministic",
) -> np.ndarray:
    logging.info("Filtering Anchors")
    Xi, Xj = data_tables

    nn = kneighbors(
        data=norm2_normalize(Xj[:, features]),
        query=norm2_normalize(Xi[:, features]),
        k=np.minimum(k_filter, Xj.shape[0]),
        method=nn_method,
    )
    # print(f"nn.shape{nn.shape}, anchor_pairs.shape={anchor_pairs.shape}")
    # print(f"np.max(anchor_pairs[:, 0])={np.max(anchor_pairs[:, 0])}")
    bool_f = np.any(
        nn[anchor_pairs[:, 0], :] == anchor_pairs[:, 1][:, None], 1
    )  # NOTE Can be made much faster with numba !!!!
    return anchor_pairs[bool_f, :], anchor_scores[bool_f]


def score_anchors(
    anchor_pairs: np.ndarray,
    anchor_scores: np.ndarray,
    nnaa: sparse.csr_matrix,
    nnab: sparse.csr_matrix,
    nnba: sparse.csr_matrix,
    nnbb: sparse.csr_matrix,
    lower_pc: float = 1.0,
    upper_pc: float = 90.0,
) -> np.ndarray:
    """Score anchors by examining the consistency of edges between cells in the same local neighborhood
    like one would do in shared nearest neighbours.

    For each anchor correspondence we compute the shared neighbor overlap between the anchor and query cells
    and assign this value as the anchor score. To dampen the potential effect of outlier scores, we use the 0.01 and 0.90 quantiles.

    Arguments
    ---------
    anchor_pairs: np.ndarray, shape=(n_anchors, 3)
        The anchor pairs array.
        anchor_pairs[:, 0] indexes of cells in sample i
        anchor_pairs[:, 1] indexes of cells in sample j
        anchor_pairs[:, 2] score of the pair
    
    nnaa: sparse.csr_matrix
        Sparse matrix representing neighbourhood between cells in sample i searched in sample i.

    nnab: sparse.csr_matrix
        Sparse matrix representing neighbourhood between cells in sample i searched in sample j.
    
    nnba: sparse.csr_matrix
        Sparse matrix representing neighbourhood between cells in sample j searched in sample i.
    
    nnbb: sparse.csr_matrix
        Sparse matrix representing neighbourhood between cells in sample j searched in sample j.

    lower_pc: float, default= 1.0
        Percentile below wich score is equal to 0.
    
    upper_pc: float, dafault = 90.0
        Percentile below wich score is equal to 1.

    Returns
    -------
    anchor_pairs: np.ndarray, shape=(n_anchors, 3)

    NOTE: It modifies anchor_pairs inplace and (redundantly) returns it.
    """
    # NOTE only the first k_score elements
    ixa = anchor_pairs[:, 0]
    ixb = anchor_pairs[:, 1]
    # The line below is the fast version of the following
    # nnAB = sparse.bmat([[nnaa, nnab],
    #                     [nnba, nnbb]], format="csr")
    # scores = nnAB.dot(nnAB.T)[ixa, ixb]
    # or scores = nnAB[ixa, :].multiply(nnAB[ixb, :]).sum(1)
    # NOTE! Actually input is not used here
    anchor_scores = (
        nnaa[ixa, :].multiply(nnba[ixb, :]).sum(1).A.flat[:]
        + nnab[ixa, :].multiply(nnbb[ixb, :]).sum(1).A.flat[:]
    )
    max_score = np.percentile(anchor_scores, upper_pc)
    min_score = np.percentile(anchor_scores, lower_pc)
    anchor_scores = np.clip((anchor_scores - min_score) / (max_score - min_score), 0, 1)
    return anchor_pairs, anchor_scores


def from_ind_to_graph(
    ind: np.ndarray, M: int, weights: np.ndarray = None
) -> sparse.csr_matrix:
    """Transforms a kneighbors to a kneighbors_graph output

    Arguments
    ---------
    ind: np.ndarray, shape=(N_points_i, k_neighbors)
        The values are the indices of points_j
    M: int
        The number of the points_j
    weights: np.ndarray (Optional), shape=ind.shape
        The entries that the matrix will contain if None
        A connectivity matrix with only ones as entries will be provided
    
    Returns
    -------
    kneighbors_graph: sparse.csr_matrix, shape=(N_points_i, M_points_j)
    """
    N, k = ind.shape
    indices = np.ravel(ind)  # this does not create copies
    if weights is None:
        weights = np.ones_like(indices)
    else:
        weights = weights.ravel()
    indptr = np.arange(0, N * k + 1, k)  # this for some reason is faster than range
    return sparse.csr_matrix((weights, indices, indptr), (N, M))


def find_anchors(
    n_cells_i: int,
    n_cells_j: int,
    data_tables: Tuple[np.ndarray, np.ndarray],
    self_spaces: Tuple[np.ndarray, np.ndarray],
    joint_space: np.ndarray = None,
    reciprocal_spaces: Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ] = None,
    space_loadings: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]] = None,
    k_anchor: int = 5,
    k_filter: int = 200,
    k_score: int = 30,
    max_dims: int = 10,
    max_features: int = 200,
    internal_neighbors: np.ndarray = None,
    nn_method: str = "deterministic",
    verbose: bool = False,
) -> Any:
    """

    Arguments
    ---------
    n_cells_i: int
    
    n_cells_j: int

    self_spaces: Tuple[np.ndarray, np.ndarray] = None

    joint_space: np.ndarray = None

    reciprocal_spaces: Tuple[np.ndarray, np.ndarray] = None

    space_loadings: np.ndarray = None
    
    k_anchor: int = 5
        The number of neighbours used for finding anchors

    k_filter: int = 200
        The number of neighbours used in low dimensional confirmation of the anchors
        To deactivate filtering set this to None

    k_score: int = 30,
        The number of neighbours used in scoring the anchors
        To deactivate scoring set this to None
    
    max_features: int = 200

    max_dims: int = 10
        The number of dims for which corresponding top feature will be computed
        The actual number of dim is the minimum between the dimensionality of the space passed and `dims`

    internal_neighbors: np.ndarray = None
    
    nn_method: str = "deterministic"

    verbose: bool= False

    Returns
    -------
    """
    # compute local neighborhoods, use max of k.anchor and k.score if also scoring to avoid
    # recomputing neighborhoods

    if k_score is not None:
        k_neighbors = max(k_anchor, k_score)
    else:
        k_neighbors = k_anchor
    # NOTE: returning the neighbourhoods in index format is easier as we can select the number of neighbours
    innaa, innab, innba, innbb = find_nn(
        self_spaces,
        joint_space,
        reciprocal_spaces,
        internal_neighbors,
        n_cells_i,
        n_cells_j,
        k=k_neighbors,
        nn_method=nn_method,
    )
    logging.debug("Neighborhoods found")
    # (f"innaa.shape {innaa.shape}, innab.shape {innab.shape}, innba.shape {innba.shape}, innbb.shape {innbb.shape}")
    anchor_pairs, anchor_scores = find_anchor_pairs(
        from_ind_to_graph(innab[:, :k_anchor], n_cells_j),
        from_ind_to_graph(innba[:, :k_anchor], n_cells_i),
    )

    if k_filter is not None:
        if joint_space is not None:
            logging.debug("Using joint space to find top dim features")
            top_features = top_dim_features(
                space_loadings[0], dims=int(min(space_loadings[0].shape[1], max_dims))
            )  # NOTE: Not sure this is the right thing
        elif reciprocal_spaces is not None:
            top_features = top_dim_features(
                space_loadings[0], dims=int(min(space_loadings[0].shape[1], max_dims))
            )
            # NOTE Maybe I should calculate the set union of the top features for both the datasets
            # top_features0 = top_dim_features(space_loadings[0])
            # top_features1 = top_dim_features(space_loadings[1])
            # top_features = np.union1d(top_features0, top_features1)
        else:
            raise ValueError("This situation should not occur")
        anchor_pairs, anchor_scores = filter_anchors(
            anchor_pairs=anchor_pairs,
            anchor_scores=anchor_scores,
            features=top_features,
            data_tables=data_tables,
            k_filter=k_filter,
        )

    if k_score is not None:
        # NOTE: I could have concatenated the indexes and build a single graph
        anchor_pairs, anchor_scores = score_anchors(
            anchor_pairs,
            anchor_scores,
            from_ind_to_graph(innaa[:, :k_score], n_cells_i),
            from_ind_to_graph(innab[:, :k_score], n_cells_j),
            from_ind_to_graph(innba[:, :k_score], n_cells_i),
            from_ind_to_graph(innbb[:, :k_score], n_cells_j),
        )

    return anchor_pairs, anchor_scores


def find_integration_anchors(
    object_list: List[Dict[str, np.ndarray]] = None,
    dims: int = 30,
    k_anchor: int = 5,
    k_filter: int = 200,
    k_score: int = 30,
    max_features: int = 200,
    reference: Union[int, np.ndarray] = None,
    anchor_features: Union[int, np.ndarray] = 2000,
    normalization_method: str = None,
    dim_reduction: str = "rpca",
    l2_norm=True,
    sct_clip_range=None,
    nn_method="deterministic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find integration anchors (Main Function)


    Arguments
    ---------

    object_list: List[Dict[str, np.ndarray]] = None
        A list of dictionaries between which to find anchors for downstream integration.
        They should contain the entries
            Either:
                object_list[i]["counts"] : np.ndarray, shape = (samples, features)
                    The raw data matrices matrices.
                object_list[i]["normalized_array"] : np.ndarray, shape = (samples, features)
                    The data matrices matrices (already log normalized)
            object_list[i]["GeneID"] : np.ndarray, shape = (features,)
                Unique ids for the genes
                if not object_list[i]["GeneID"] == object_list[j]["GeneID"] for all i, j
                then object_list[i]["GeneID"] must be sorted!

    dims: int = 30
        Which dimensions to use from the CCA to specify the neighbor search space

    k_anchor: int = 5
        How many neighbors (k) to use when picking anchors
    
    k_filter: int = 200
        How many neighbors (k) to use when filtering anchors
    
    k_score: int = 30
        How many neighbors (k) to use when scoring anchors
    
    max_features: int = 200
        The maximum number of features to use when specifying the
        neighborhood search space in the anchor filtering
    
    reference: Union[int, np.ndarray], default = None
        A vector specifying the object/s to be used as a reference
        during integration. If None (default), all pairwise anchors are found (no
        reference/s). If not None, the corresponding objects in objects_list
        will be used as references. When using a set of specified references, anchors
        are first found between each query and each reference. The references are
        then integrated through pairwise integration. Each query is then mapped to
        the integrated reference.
    
    anchor_features: Union[int, np.ndarray] = 2000
    
    normalization_method: str = None
        If None the data is assumed normalized.
        Name of normalization method used: "log", "size_log" or "sct"
    
    dim_reduction: str = "rpca"
        Dimensional reduction to perform when finding anchors. Can be one of:
        rpca: Reciprocal PCA
        crossSVD: called CCA in seurat, this is the SVD of the center standardized cross product matrix (crosscovariance) 
        cca: Canonical correlation analysis
    
    l2_norm=True
        Perform L2 normalization on the cell embeddings after dimensional reduction
    
    sct_clip_range=None
        Numeric of length two specifying the min and max values the Pearson residual will be clipped to
    
    nn_method="deterministic"
        Method for nearest neighbor finding. Options include:
        'deterministic' and 'annoy'

    Returns
    -------
    anchor_pairs: np.ndarray, shape = (n_anchors, 2)
        The indexes corresponding to each anchor pair
        Note: tindexes are relative to a particular dataset that is sepcified by the `indicators_pairs` array
        So this array needs to be used to identify which datasets the indexes refer to
    
    anchor_scores: np.ndarray, shape = (n_anchors,)
        The score associated to each anchor pair (how confidence is there that it is a good anchor)
    
    indicators_pairs: np.ndarray, shape = (n_anchors, 2)
        An integer array corresponding to each ancor pair that identifies from whihc sample each ancor is taken
        indicators_pairs[i, 0] indicates the sample from which anchor_pairs[i, 0] needs to be taken
    """

    NotImplementedError("Not finished implementing the function. Please wait")

    # BOILERPLATE START #####################
    if normalization_method is None:
        pass
    elif normalization_method.lower() == "sct":
        if isinstance(anchor_features, int):
            raise ValueError(
                "If normalization_method is SCT then anchor_features should be provided"
            )
        for obj in object_list:
            if "prep" not in obj:
                raise ValueError("For SCT integration, samples should be preprocessed first")
    elif normalization_method.lower() == "log":
        for obj in object_list:
            obj["normalized_array"] = np.log2(obj["counts"]+1)
    elif normalization_method.lower() == "size_log":
        for obj in object_list:
            tmp = obj["counts"]
            tot = tmp.sum(0)
            tmp = tmp / tot * np.mean(tot)
            obj["normalized_array"] = np.log2(tmp+1)
    else:
        raise ValueError(f"{normalization_method} is not a valid normalization method")
    
    # Check format of input files and preprocess to appropriate dictionary format
    object_list = preprocess_object_list(object_list)

    # DO SelectIntegrationFeatures
    # DO Scale

    if isinstance(anchor_features, int) and normalization_method == "SCT":
        logging.info(f"Computing {anchor_features} integration features")
        raise NotImplementedError("SelectIntegrationFeatures not implemented yet")
    # BOILERPLATE END #####################

    # nn.reduction <- reduction??
    internal_neighbors = []
    loadings = []
    pcas = []
    
    # This below was used if one wanted to speed up calculations in a one to many implementations
    # if dim_reduction == "rpca":
    #     k_neighbors = np.maximum(k_anchor, k_score) if k_score is not None else k_anchor
    #     for i in range(len(object_list)):
    #         # NOTE: Not sure if this should be computed only on the common features. But porbably not.
    #         logging.info(f"Computing PCA and neighborhoods within dataset {i+1}")
    #         X = object_list[i]["normalized_array"]
    #         object_list[i]["mean"] = X.mean(0)
    #         U, S, Vt = svd(X - object_list[i]["mean"], n_components=dims)
    #         object_list[i]["pcas"] = U * S
    #         object_list[i]["loadings"] = Vt
    #         # object_list[i]["weights"] = S
    #         inn = kneighbors(U, k=k_neighbors, method=nn_method)
    #         object_list[i]["internal_neighbors"] = inn

    if reference is None:

        def condition(si: int, sj: int) -> bool:
            return si < sj

    else:

        def condition(si: int, sj: int) -> bool:
            return (si < sj) and (reference in (si, sj))

    if len(object_list) == 2:
        logging.info("Finding anchors, 2 datasets")
    else:
        logging.info(
            f"Finding all pairwise anchors. {(len(object_list)**2 - len(object_list)) / 2} comparisons to be computed!"
        )
    all_anchors = []
    all_anchors_scores = []
    all_indicators = []
    for si in range(len(object_list)):
        for sj in range(len(object_list)):
            if condition(si, sj):  # NOTE Check this
                object_i = object_list[si]
                object_j = object_list[sj]
                Xi = object_i["normalized_array"]  # NOTE everything would be faster is these Xs were C-contiguos
                Xj = object_j["normalized_array"]
                # NOTE: Maybe they should be centered
                n_cells_i = Xi.shape[0]
                n_cells_j = Xj.shape[0]

                # NOTE Here it is assumed inputs are given with uniques alphabetically sorted GenesIDs
                # TODO For accepting more general input I should have used ixs_that_sort_a2b
                common_features = np.intersect1d(object_i["GeneID"], object_j["GeneID"])
                ixs_i = np.where(np.in1d(object_i["GeneID"], common_features))[0]
                ixs_j = np.where(np.in1d(object_j["GeneID"], common_features))[0]
                assert np.alltrue(
                    object_i["GeneID"][ixs_i] == object_j["GeneID"][ixs_j]
                ), "Features were not sorted"

                if dim_reduction == "cca":
                    # CCV project the two datasets into a correlated low-dimensional space
                    # but global differences in scale can still preclude comparisons
                    # so we perform L2-normalization of the cell embeddings
                    cca = CCA(n_components=dims, scale=True)
                    cca.fit(Xi.T, Xj.T)
                    # Note since those are almost never used in the other orientation maybe passing the trnasposed as input is a better design choice
                    Ui_jointspace = cca.x_loadings_
                    Uj_jointspace = cca.y_loadings_
                    Vit = cca.x_scores_
                    Vjt = cca.y_scores_

                    if l2_norm:
                        Ui_jointspace = norm2_normalize(Ui_jointspace, axis=1)
                        Uj_jointspace = norm2_normalize(Uj_jointspace, axis=1)

                    anchor_pairs, anchor_scores = find_anchors(
                        n_cells_i=n_cells_i,
                        n_cells_j=n_cells_j,
                        data_tables=(
                            Xi[:, ixs_i],
                            Xj[:, ixs_j],
                        ),  # NOTE Should this really be the normalized? I think it should be the raw
                        self_spaces=(Ui_jointspace, Uj_jointspace),
                        joint_space=(Ui_jointspace, Uj_jointspace),
                        reciprocal_spaces=None,
                        space_loadings=(Vit, Vjt),
                        k_anchor=k_anchor,
                        k_filter=k_filter,
                        k_score=k_score,
                        max_features=max_features,
                        internal_neighbors=None,
                        nn_method=nn_method,
                    )
                elif dim_reduction == "crossSVD":
                    logging.info("Dim reduction with crossSVD")
                    # CCV project the two datasets into a correlated low-dimensional space
                    # but global differences in scale can still preclude comparisons
                    # so we perform L2-normalization of the cell embeddings
                    crossSVD = PLSSVD(n_components=dims, scale=True)
                    crossSVD.fit(Xi.T, Xj.T)
                    # Note since those are almost never used in the other orientation maybe passing the trnasposed as input is a better design choice
                    Ui_jointspace = crossSVD.x_weights_
                    Uj_jointspace = crossSVD.y_weights_
                    Vit = crossSVD.x_scores_
                    Vjt = crossSVD.y_scores_

                    # In alternative the same thing done manually
                    # mu_i = Xi.mean(1)
                    # mu_j = Xj.mean(1)
                    # std_i = Xi.std(1)
                    # std_j = Xj.std(1)
                    # X = (Xi - mu_i) / std_i[:, None]
                    # Y = (Xj - mu_j) / std_j[:, None]
                    # C = X.dot(Y.T)
                    # Ui_jointspace, _, Uj_jointspace = svds(C, k=self.n_components)
                    # Vit = np.dot(X, Ui_jointspace)
                    # Vjt = np.dot(Y, Uj_jointspace)
                    # del C, X, Y  # optinally garbage collect this matrixes

                    if l2_norm:
                        Ui_jointspace = norm2_normalize(Ui_jointspace, axis=1)
                        Uj_jointspace = norm2_normalize(Uj_jointspace, axis=1)

                    anchor_pairs, anchor_scores = find_anchors(
                        n_cells_i=n_cells_i,
                        n_cells_j=n_cells_j,
                        data_tables=(
                            Xi[:, ixs_i],
                            Xj[:, ixs_j],
                        ),  # NOTE Should this really be the normalized? I think it should be the raw
                        self_spaces=(Ui_jointspace, Uj_jointspace),
                        joint_space=(Ui_jointspace, Uj_jointspace),
                        reciprocal_spaces=None,
                        space_loadings=(Vit, Vjt),
                        k_anchor=k_anchor,
                        k_filter=k_filter,
                        k_score=k_score,
                        max_features=max_features,
                        internal_neighbors=None,
                        nn_method=nn_method,
                    )
                    # Calculate nn on the l2_norm data
                elif dim_reduction == "rpca":
                    k_neighbors = np.maximum(k_anchor, k_score) if k_score is not None else k_anchor
                    
                    # object_pair["array"] = np.column_stack([object_i["array"], object_j["array"]])
                    # NOTE: Better not allocate further memory
                    mui = Xi.mean(0)
                    muj = Xj.mean(0)

                    UiSi, Si, Vit = svd(Xi[:, ixs_i] - mui[ixs_i], n_components=dims)
                    UjSj, Sj, Vjt = svd(Xj[:, ixs_j] - muj[ixs_j], n_components=dims)

                    UiSi = UiSi*Si
                    UjSj = UjSj*Sj

                    Ui_jspace = (Xi[:, ixs_i] - muj[ixs_j]) @ Vjt.T  # / Sj[ixs_j] # if you want to divide check concordance below
                    Uj_ispace = (Xj[:, ixs_j] - mui[ixs_i]) @ Vit.T  # / Si[ixs_i] # at self_spaces=(Ui * Si, Uj * Sj),

                    if l2_norm:
                        # This is the same operation that would happen if I were to stack the matrices,
                        # I avoid explicitelly concatenating that because it would be memory intensive if datsests are big
                        # projected_j = np.concatenate((Uj_ispace,
                        #                               Ui), axis=0)
                        # projected_i = np.concatenate((Ui_jspace,
                        #                               Uj), axis=0)
                        # cell_embeddings_i = projected_i / projected_i.std(0)
                        # cell_embeddings_j = projected_j / projected_j.std(0)
                        # # cell_embeddings_i = cell_embeddings_i / np.sqrt(np.sum(cell_embeddings_i**2, 1))[:, None]
                        # # cell_embeddings_j = cell_embeddings_j / np.sqrt(np.sum(cell_embeddings_j**2, 1))[:, None]

                        # NOTE Because these calculations are meant to happen in the concatenated matrix
                        # They should not affect the origina Ui, Uj so I do not do caluclations in place and
                        # I pass to the function below the original PCA
                        # This makes the difference in particular afeter the L2 normalization

                        # Standardize the concateneted matrix (each dims has std = 1)
                        mu_proj_j = (Uj_ispace.sum(0) + UiSi.sum(0)) / (Uj_ispace.shape[0] + UiSi.shape[0])
                        mu_proj_i = (Ui_jspace.sum(0) + UjSj.sum(0)) / (Ui_jspace.shape[0] + UjSj.shape[0])

                        std_projected_j = np.sqrt(
                            (np.sum((mu_proj_j - Uj_ispace) ** 2, 0)
                            + np.sum((mu_proj_j - UiSi) ** 2, 0)) / (Uj_ispace.shape[0] + UiSi.shape[0])
                        )
                        std_projected_i = np.sqrt((
                            np.sum((mu_proj_i - Ui_jspace) ** 2, 0)
                            + np.sum((mu_proj_i - UjSj) ** 2, 0)) / (Ui_jspace.shape[0] + UjSj.shape[0])
                        )
                        Uj_ispace = Uj_ispace / std_projected_j
                        UiSi = UiSi / std_projected_j
                        Ui_jspace = Ui_jspace / std_projected_i
                        UjSj = UjSj / std_projected_i

                        # Normalize the cells so that the norm of each cell is equal to 1
                        Uj_ispace = norm2_normalize(Uj_ispace, axis=1)
                        UiSi = norm2_normalize(UiSi, axis=1)  # same as Ui / np.sqrt(np.sum(Ui**2, 1))[:, None]
                        Ui_jspace = norm2_normalize(Ui_jspace, axis=1)
                        UjSj = norm2_normalize(UjSj, axis=1)

                    anchor_pairs, anchor_scores = find_anchors(
                        n_cells_i=n_cells_i,
                        n_cells_j=n_cells_j,
                        data_tables=(
                            Xi[:, ixs_i],
                            Xj[:, ixs_j],
                        ),  # NOTE Should this really be the normalized? I think it should be the raw
                        self_spaces=(UiSi, UjSj),
                        joint_space=None,
                        reciprocal_spaces=(
                            (UiSi, Uj_ispace),
                            (UjSj, Ui_jspace),
                        ),  # Note in R is ((Ui_jspace, Uj), (Uj_ispace, Ui))
                        space_loadings=(Vit, Vjt),
                        k_anchor=k_anchor,
                        k_filter=k_filter,
                        k_score=k_score,
                        max_features=max_features,
                        internal_neighbors=None,
                        nn_method=nn_method
                    )
                else:
                    raise NotImplementedError(
                        f"Reduction method '{dim_reduction}' is not implemented"
                    )
                all_anchors.append(anchor_pairs)
                # In Seurat implementation they do all_anchors.append(anchor_pairs[:, ::-1])
                all_anchors_scores.append(anchor_scores)
                all_indicators.append(
                    np.ones((anchor_pairs.shape[0], 2), dtype=int)
                    * np.array([si, sj], dtype=int)
                )
    indicators_pairs = np.row_stack(all_indicators)
    anchor_pairs = np.row_stack(all_anchors)
    anchor_scores = np.concatenate(all_anchors_scores)
    # NOTE in case the anchors are recoded with indexes offset one can also add the reciprocal
    # anchor_pairs = np.row_stack([anchor_pairs, anchor_pairs[:, [1, 0, 2]]])  # symmetrical
    return anchor_pairs, anchor_scores, indicators_pairs


def condensed_similarity(similarity_matrix: np.ndarray) -> np.ndarray:
    """Transform a square to a condensed similarity matrix. Maybe redundant with squareform
    """
    return similarity_matrix.flat[np.triu_indices(similarity_matrix.shape[0], k=1)]


def count_anchors(indicators_pairs: np.ndarray, n_cells_per_obj: List[int]) -> np.ndarray:
    """Computer a similarity matrix by counting how many anchors were found between each couple of datasets.
       It normalizes the count by number of point in the sample, to buffer for unequally sized datasets.
    """
    similarity_matrix = np.full((len(n_cells_per_obj), len(n_cells_per_obj)), np.NaN)
    for i in range(len(n_cells_per_obj)):
        for j in range(len(n_cells_per_obj)):
            if i < j:
                anchors_ij_bool = np.in1d(indicators_pairs[:, 0], (i, j)) & np.in1d(
                    indicators_pairs[:, 1], (i, j)
                )
                similarity_matrix[i, j] = np.sum(anchors_ij_bool) / np.minimum(
                    n_cells_per_obj[i], n_cells_per_obj[j]
                )
    return similarity_matrix


def _find_weights(
    data_matrices: Tuple[np.ndarray, np.ndarray],
    anchor_pairs: np.ndarray,
    k_weight: int,
    which: int = 0,
    sd: float = 1.0,
    kernel: str = "Seurat",
    squared_dist: bool = False,
) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    """ Find weights that linking each point to the anchors

    Arguments
    ---------
    data_matrices

    anchor_pairs

    k_weight: int
    
    which: int = 0,
    
    sd: float = 1.0
    
    kernel: str = "Seurat"
    
    squared_dist: bool = False

    Returns
    -------
    W: sparse.csr_matrix, shape=(points, anchors)
    """
    X = data_matrices[0]
    Y = data_matrices[1]
    ii, jj, scores = anchor_pairs.T

    if which == 1 or which == -1:
        Ax = X[ii, :]
        neigh_dist_x, neigh_ind_x = kneighbors(data=Ax, query=X, k=k_weight)
        if squared_dist:
            Wx = (1 - neigh_dist_x / neigh_dist_x[:, -1]) ** 2 * scores[neigh_ind_x]
        else:
            Wx = (1 - neigh_dist_x / neigh_dist_x[:, -1]) * scores[neigh_ind_x]
        if kernel == "corrected":
            Wx = 1 - np.exp(-Wx / (2.0 * sd ** 2))
        elif kernel == "Seurat":
            Wx = 1 - np.exp(-Wx / ((2.0 / sd) ** 2))
        Wx /= Wx.sum(1)[:, None]
        Wx = from_ind_to_graph(neigh_ind_x, Ax.shape[0], Wx)

    if which == 0 or which == -1:
        Ay = Y[jj, :]
        neigh_dist_y, neigh_ind_y = kneighbors(data=Ay, query=Y, k=k_weight)
        if squared_dist:
            Wy = (1 - neigh_dist_y / neigh_dist_y[:, -1]) ** 2 * scores[neigh_ind_y]
        else:
            Wy = (1 - neigh_dist_y / neigh_dist_y[:, -1]) * scores[neigh_ind_y]
        if kernel == "corrected":
            Wy = 1 - np.exp(-Wy / (2.0 * sd ** 2))
        elif kernel == "Seurat":
            Wy = 1 - np.exp(-Wy / ((2.0 / sd) ** 2))
        Wy /= Wy.sum(1)[:, None]
        Wy = from_ind_to_graph(neigh_ind_y, Ay.shape[0], Wy)

    if which == 1:
        return Wx
    elif which == 0:
        return Wy
    else:
        return Wx, Wy


def find_weight(
    data_matrix: np.ndarray,
    anchor_index: np.ndarray,
    anchor_score: np.ndarray,
    k_weight: int,
    sd: float = 1.0,
    kernel: str = "Seurat",
    squared_dist: bool = False,
) -> sparse.csr_matrix:
    """ Find weights that linking each point to the anchors

    Arguments
    ---------
    data_matrices

    anchor_pairs

    k_weight: int
    
    which: int = 0,
    
    sd: float = 1.0
    
    kernel: str = "Seurat"
    
    squared_dist: bool = False

    Returns
    -------
    W: sparse.csr_matrix, shape=(points, anchors)
    """
    Y = data_matrix
    scores = anchor_score
    jj = anchor_index

    Ay = Y[jj, :]
    neigh_dist_y, neigh_ind_y = kneighbors(data=Ay, query=Y, k=k_weight, return_distance=True)
    if squared_dist:
        Wy = (1 - neigh_dist_y / neigh_dist_y[:, -1][:, None]) ** 2 * scores[neigh_ind_y]
    else:
        Wy = (1 - neigh_dist_y / neigh_dist_y[:, -1][:, None]) * scores[neigh_ind_y]
    if kernel == "corrected":
        Wy = 1 - np.exp(-Wy / (2.0 * sd ** 2))
    elif kernel == "Seurat":
        Wy = 1 - np.exp(-Wy / ((2.0 / sd) ** 2))
    Wy /= Wy.sum(1)[:, None]
    Wy = from_ind_to_graph(neigh_ind_y, Ay.shape[0], Wy)

    return Wy


def transform_data_matrix():
    return None


def find_integration_matrix(
    data_matrices: Tuple[np.ndarray, np.ndarray],
    anchor_pairs: np.ndarray,
    which_reference: int,
) -> np.ndarray:
    """Returns a matrix containing the pariwise differences between anchor points

    Arguments
    ---------
    data_matrices: Tuple[np.ndarray, np.ndarray]
    
    anchor_pairs: np.ndarray
    
    merge_pair: Tuple[int, int]
    
    reference: int

    Returns
    -------
    B: np.darray, shape=(anchors, features)
        integration matrix
    """
    if which_reference == 0:
        X, Y = data_matrices
        # X is the reference
        ax, ay = anchor_pairs[:, 0], anchor_pairs[:, 1]
    elif which_reference == 1:
        Y, X = data_matrices
        # X is the reference
        ay, ax = anchor_pairs[:, 0], anchor_pairs[:, 1]
    else:
        raise ValueError(f"which_reference={which_reference} is not a valid input")
    B = Y[ay, :] - X[ax, :]
    return B


def run_integration(
    data_matrices: Tuple[np.ndarray, np.ndarray],
    anchor_pairs: np.ndarray,
    anchor_score: np.ndarray,
    which_reference: int = 0,
    k_weight: int = 100,
    sd: float = 1.0,
    kernel: str = "Seurat",
    squared_dist: bool = False,
) -> np.ndarray:
    """ Integrates two data matrices given a set of anchors

    Arguments
    ---------
    data_matrices: Tuple[np.ndarray, np.ndarray]
        The 2 data matrices to be integrated. 
        They are usually object_list[i]["normalized_array"] and object_list[j]["normalized_array"]
        But it is not important that the matrix is that any matrix can do. (It might have to be centered correctly)
    
    anchor_pairs: np.ndarray, shape=(n_anchors, 2)
        The indexes corresponding to each anchor pair between - two samples -
        Normally it is one of the output `find_integration_anchors`
        However is samples where more than 2 the output of `find_integration_anchors` contains all the pairs of datasets
        Remember that you need to subset:
        anchor_pairs[(indicator_pairs[:, 0] == i) & (indicator_pairs[:, 1] == j), :]
    
    anchor_scores: np.ndarray, shape = (n_anchors,)
        The score associated to each anchor pair (how confidence is there that it is a good anchor)
        Normally it is one of the output `find_integration_anchors`
        However is samples where more than 2 the output of `find_integration_anchors` contains all the pairs of datasets
        Remember that you need to subset:
        anchor_scores[(indicator_pairs[:, 0] == i) & (indicator_pairs[:, 1] == j)]
    
    which_reference: int = 0
        Whether the first or second array should be used as a reference
    
    k_weight: int = 100
        The number of anchor neighbours to consider when wheighting the transormation
    
    sd: float = 1.0,
        The standard deviation to use in the kernel for the weighting
    
    kernel: str = "Seurat"
        Which kind of weight funtion to use
    
    squared_dist: bool = False
        Whether to square the distances or not in the kernel used for weighting


    Returns
    -------
    A tuple like the `data_matrices` input but after the transformation
    (note that no copies are made so that one entry is the pointer to the data_matrices that was not changed)
    """
    # NOTE For the future it will be better to split anchor_pairs in 3 arrays so that each anchor set can be stored in a dictionary with its own matrix
    # NOTE Right now I am not alowing to transform two matrices in the same space into another space (but maybe I should just concatenate and remove the old!)
    B = find_integration_matrix(data_matrices, anchor_pairs, which_reference)
    which_transformed = 1 if which_reference == 0 else 0
    if k_weight > anchor_pairs.shape[0]:
        k_weight = anchor_pairs.shape[0] - 1
        logging.info(
            f"k_weight is larger than the number of anchors ({anchor_pairs.shape[0]}) setting it to {k_weight}"
        )
    W = find_weight(
        data_matrix=data_matrices[which_transformed],
        anchor_index=anchor_pairs[:, which_transformed],
        anchor_score=anchor_score,
        k_weight=k_weight,
        sd=sd,
        kernel=kernel,
        squared_dist=squared_dist,
    )
    #  transform_data_matrix()
    # NOTE, Maybe it should be an affine function not a linear function?
    C = W.dot(
        B
    )  # NOTE: maybe this can be faster if in the other orientation maybe this sparse.csr_matrix.dot(B.T, W.T).T or this sparse.csr_matrix.dot(W, B)
    Y = data_matrices[which_transformed]
    Y_hat = Y - C  # It was minus but y inspection made no sense!
    if which_reference == 0:
        return (data_matrices[which_reference], Y_hat)
    else:
        return (Y_hat, data_matrices[which_reference])


def build_sample_tree(similarity_matrix: np.ndarray) -> np.ndarray:
    dist_matrix_condensed = 1 / condensed_similarity(similarity_matrix)
    Z = linkage(
        dist_matrix_condensed, method="complete", metric="precomputed", optimal_ordering=False
    )
    # Z = Z[:, :3]
    # Z[:, 2] = Z[:, 0]  # the therd column contain the reference
    return Z[:, :2]


def transverse(sample_tree):
    n = sample_tree.shape[0] + 1
    roots = sample_tree.copy()
    for i in range(n - 1):
        roots[i, :] = parse(sample_tree, i)
    return roots


# def parse(x, i):
#     n = x.shape[0] + 1
#     ixs = x[i, :].copy()
#     if ixs[0] >= n:
#         ixs[0] = parse(x, ixs[0] - n)[0]
#     if ixs[1] >= n:
#         ixs[1] = parse(x, ixs[1] - n)[1]
#     return ixs


def sizeof(x, i):
    n = x.shape[0] + 1
    if i < n:
        return 1
    else:
        return x[i - n, 2]


def parse(x, i):
    n = x.shape[0] + 1
    ixs = x[i, :].copy()
    if ixs[0] >= n:
        left, right, _ = parse(x, ixs[0] - n)
        i = np.argmax([sizeof(x, left), sizeof(x, right)])
        ixs[0] = (left, right)[i]
    if ixs[1] >= n:
        left, right, _ = parse(x, ixs[1] - n)
        # you could swap both
        i = np.argmax([sizeof(x, left), sizeof(x, right)])
        ixs[1] = (left, right)[i]
    return ixs


def adjust_sample_tree(sample_tree, reference):
    x = sample_tree.copy()
    n = x.shape[0] + 1
    x[x < n] = reference[x[x < n]]
    return x


def integrate_data(
    object_list: List[Dict[str, np.ndarray]],
    anchor_pairs: np.ndarray,
    indicators_pairs: np.ndarray,
    k_weight: int = 100,
    reference_objects: np.ndarray = None,
    sample_tree: Any = None,
    sd_weight: float = 1.0,
    kernel_weight: str = "Seurat",
    squared_dist_weight: bool = False,
    features_to_integrate: np.ndarray = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    raise NotImplementedError("This implementation is not finished yet")
    if reference_objects is None:
        reference_objects = np.arange(len(object_list))
    if sample_tree is None:
        n_cells_per_obj = [obj["normalized_array"].shape[0] for obj in object_list]
        similarity_matrix = count_anchors(indicators_pairs, n_cells_per_obj)
        similarity_matrix = similarity_matrix[
            reference_objects, reference_objects
        ]  # NOTE: What happens here if I pass a reference_objects array is not clear
        sample_tree = build_sample_tree(similarity_matrix)
        sample_tree = adjust_sample_tree(sample_tree, reference_objects)
    object_dict = {i: obj for i, obj in enumerate(object_list)}
    for ii in range(sample_tree.shape[0]):
        si, sj, reference = sample_tree[ii, :3]
        # NOTE The code below assumes that indicators_pairs does not include the same copule of anchors in the opposite order
        # So it asjust for the right orientation to match the indexes saved in anchor array
        if si >= len(object_list) or sj >= len(object_list):
            # look up what is the reference of this merged pair!
            object_dict[si]
        select_anchors = (indicators_pairs[:, 0] == si) & (indicators_pairs[:, 1] == sj)
        if np.sum(select_anchors) == 0:
            si, sj = sj, si
        select_anchors = (indicators_pairs[:, 0] == si) & (indicators_pairs[:, 1] == sj)
        if np.sum(select_anchors) == 0:
            raise ValueError(f"The pair {si}, {sj} has no anchors")
        data_matrices = (
            object_dict[si]["normalized_array"],
            object_dict[sj]["normalized_array"],
        )  # NOTE Important! Decide in which orientation the data matrixes come and remove transpose accordingly
        Y_hat = run_integration(
            data_matrices=data_matrices,
            anchor_pairs=anchor_pairs[select_anchors, :],
            merge_pair=(si, sj),
            reference=reference,
            k_weight=k_weight,
            sd=sd_weight,
            kernel=kernel_weight,
            squared_dist=squared_dist_weight,
        )
        # NOTE: I need to figure out a right way to resolve the tree
        # object_dict[ii + sample_tree.shape[0]] = integrated_dict
        # if merged_matrix is None:
        #     merged_matrix = integrated_matrix
        # else:
        #     merged_matrix = np.row_stack([merged_matrix, integrated_matrix])

    return object_dict, sample_tree
