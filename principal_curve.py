import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture


class PrincipalCurve:
    """Subspace Constrained Mean Shift algorithm based on the paper 
    "Locally Defined Principal Curves and Surfaces" from Ozertem et al..
    """
    
    def __init__(self, tolerance: float = 0.0001, maxiter: int = 600, d: int = 1, h: int = 3) -> None:
        """Initiate the class.
        
        Parameters
        ----------
        tolerance: float
            Used to check convergence of the projection. 0.0001 by default. 
        maxiter: int
            Maximal number of iterations for the projection of a point onto the principal curve. 600 by default.
        d: int
            Principal curve dimension. 1 by default.
        h: int
            Gaussian kernel bandwidth used if density=='KDE' (see below). 3 by default.
        """
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.d = d
        self.h = h
        
    def fit(self, data: np.ndarray, points: np.ndarray, density: str=None, localcov: str=None, nbneighbors: int=None, nbcomponents: int=None, plot: bool=False) -> np.ndarray:
        """Find the principal curve of a dataset given a mesh of points.
        
        Parameters
        ----------
        data: array_like 
            Must has size (points, dimension).
        points: array_like 
            In the most common use, data and points are the same array. More generally points can be any grid of points. 
            Must has size (points, dimension).
        density: str 
            Density estimation method used. Can be 'KDE' or 'GMM'.
        
        If density=='KDE'
            localcov: str
                Name of the local covariance matrix used for the eigenvalue decomposition. 
                Can be 'hess', 'inversecov', 'localcov1' or 'localcov2'.
            nbneighbors: int
                Number of nearest neighbors used to compute 'localcov1' or 'localcov2'.
            
        If density=='GMM'
             nbcomponents: int
                 Number of mixture components. Optional.
             plot: bool
                 If True, the GMM pdf is plotted (only works in 2 dimensions). False by default.  
        
        Returns
        -------
        points: array_like
            Points projected onto the principal curve. Has size (points, dimension).
        """
        # Transpose and copy the input data and points
        data, points = data.T, points.T
        data = np.copy(np.atleast_2d(data)) # view inputs as arrays with at least two dimensions
        n, N = data.shape
        points = np.copy(np.atleast_2d(points))
        m, M = points.shape
        if m == 1 and M == n:  # row vector
            points = np.reshape(points, (n, 1)) # transform it in column vector
            m, M = points.shape
            
        # Check the parameters
        if density != 'KDE' and density != 'GMM':
            sys.exit("Density estimation method should be specified. Can be 'KDE' or 'GMM'.")
        if density == 'KDE':
            if localcov != 'hess' and localcov != 'inversecov' and localcov != 'localcov1' and localcov != 'localcov2':
                sys.exit("Local covariance matrix used for the eigenvalue decomposition should be specified. Can be 'hess', 'inversecov', 'localcov1' or 'localcov2'.")
            if localcov == 'localcov1' or localcov == 'localcov2':
                if nbneighbors is None:
                    sys.exit("The number of nearest neighbors used to compute 'localcov1' or 'localcov2' should be specified.")
        elif density == 'GMM':
            if nbcomponents is None:
                # Find the best number of mixture components using a BIC criterion
                nbcomponents = self._find_nbcomponents(data)
                print("The best number of mixture components found is " + str(nbcomponents) + ".")
            
        if density == 'KDE' and localcov == 'localcov1': 
            tree = KDTree(points.T, leaf_size=2)
            
        alphas, means, mcovs = None, None, None
        if density == 'GMM':
            # Fit a gaussian mixture model to the data        
            gmm = GaussianMixture(n_components=nbcomponents)
            gmm.fit(data.T)
            alphas, means, mcovs = gmm.weights_, gmm.means_.T, gmm.covariances_.T
            # Plot the probability density function
            if plot: 
                self._plotpdf(data[0].min(axis=0), data[0].max(axis=0), data[1].min(axis=0), data[1].max(axis=0), 10, alphas, means, mcovs)
        
        # For every point
        for k in range(M):
            if k == int(N / 4): print("25% complete")
            if k == int(N / 2): print("50% complete")
            if k == int(3 * N / 4): print("75% complete")
            
            # Find the nearest neighbors of the point in case matrix is 'localcov1' or 'localcov2'
            neighbors = None
            if density == 'KDE':
                if localcov == 'localcov1':
                    _, neighborsID = tree.query(np.reshape(points[:, k], (1, -1)), k=nbneighbors)
                    neighbors = np.take(data, neighborsID, axis=1) # neighbors are found from the original data points
                elif localcov == 'localcov2':
                    tree = KDTree(points.T, leaf_size=2)
                    _, neighborsID = tree.query(np.reshape(points[:, k], (1, -1)), k=nbneighbors)
                    neighbors = np.take(points, neighborsID, axis=1) # neighbors are found from the output points
                # Replace input point by its projection on the principal curve    
                points[:, k] = self._projectpoint_KDE(data, points[:, k], localcov, neighbors)
            else:
                points[:, k] = self._projectpoint_GMM(data, points[:, k], alphas, means, mcovs)          
        return points.T
    
  
    # =============================================================================
    # # Methods for the KDE-based algorithm
    # =============================================================================
    def _projectpoint_KDE(self, data: np.ndarray, point: np.ndarray, localcov: str=None, neighbors: np.ndarray=None) -> np.ndarray:
        """Project a point onto the principal curve, using a KDE"""
        n, N = data.shape
        converged = False
        citer = 0
        while not(converged):
            # Calculate the local covariance matrix
            if localcov == 'hess': cov = self._hess(data, point)
            elif localcov == 'inversecov': cov = self._inversecov_KDE(data, point)
            elif localcov == 'localcov1' or localcov == 'localcov2': cov = self._localcov(neighbors)
            w, v = np.linalg.eigh(cov) # get the eigenvalues in ascending order and the corresponding normalized eigenvectors
            index = np.argsort(w)  # arguments that sort from small to large
            V = np.reshape(v[:, index[:(n - self.d)]], (n, n - self.d)) # take the (n-d) smallest eigenvectors
            ospace = np.dot(V, V.T) # projection matrix
            proj = np.reshape(self._ms_KDE(data, point), (n, 1)) - np.reshape(point, (n, 1)) # evaluate the mean shift update
            proj = np.dot(ospace, proj) + np.reshape(point, (n, 1))
            diff = np.linalg.norm(np.reshape(point, (n, 1)) - proj)
            point = np.reshape(proj, (n, ))
            citer = citer + 1
            if diff < self.tolerance: # stopping condition based on distance
                converged = True
            if citer > self.maxiter: # stopping condition based on the number of iterations
                converged = True
                print("maximum iterations exceeded")
        return point
    
    def _hess(self, data: np.ndarray, x: np.ndarray) -> float:
        """Calculate the hessian (numpy-broadcasting speedup calculation)"""
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = x.T
        Sigmainv = np.identity(n) * (1 / (self.h**2))
        cs = self._c(data, x)
        us = self._u(data, x)
        Hx = np.sum(cs * ((us[:, None, :] * us) - Sigmainv[:, :, None]), 2) / N
        return Hx
    
    def _inversecov_KDE(self, data: np.ndarray, x: np.ndarray) -> float:
        """Calculate the inverse covariance matrix
        cov(x) = H(x)/p(x) - g(x)g(x).T/p(x)^2"""
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = x.T
        H = self._hess(data, x)
        g = self._grad2(data, x)
        p = self._p_KDE(data, x)
        cov = H / p - g.dot(g.T) / p**2
        return cov
    
    def _localcov(self, neighbors: np.ndarray) -> float:
        """Calculate the local covariance matrix
        as described in the paper "On Some Convergence Properties of the Subspace
        Constrained Mean Shift" from Aliyari Ghassabeh et al.
        
        if matrix='localcov1', neighbors are found from the original data points
        if matrix='localcov2', neighbors are found from the output points"""
        mean = np.mean(neighbors, axis=2)
        diff = neighbors[:, 0, :] - mean
        cov = diff.dot(diff.T) / (neighbors.shape[2]-1)
        return cov

    def _kern(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Kernel Profile"""
        return np.exp(-x / 2.0)

    def _p_KDE(self, data: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Evaluate KDE on a set of points based on data"""
        data = np.atleast_2d(data)
        n, N = data.shape
        points = np.atleast_2d(points)
        m, M = points.shape
        if m == 1 and M == n:  # row vector
            points = np.reshape(points, (n, 1))
            m, M = points.shape
        const = (1.0 / N) * ((self.h)**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        probs = np.zeros((M,), dtype=np.float)
        for i in range(M):
            diff = (data - points[:, i, None]) / self.h
            x = np.sum(diff * diff, axis=0)
            probs[i] = np.sum(self._kern(x), axis=0) * const
        return probs

    def _u(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        us = (data - x) / (self.h**2)
        return us

    def _c(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = x.T
        us = self._u(data, x)
        const = (self.h**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        u2 = np.sum(us * (us * (self.h**2)), axis=0)
        cs = self._kern(u2) * const
        return cs

    def _ms_KDE(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate the mean-shift at point x"""
        data = np.atleast_2d(data)
        n, N = data.shape
        const = (1.0 / N) * ((self.h)**(-n)) * (2.0 * np.pi)**(-n / 2.0)
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        unprobs = self._p_KDE(data, x) / const
        diff = (data - x) / self.h
        diff2 = np.sum(diff * diff, axis=0)
        mx = np.sum(self._kern(diff2) * data, axis=1) / unprobs
        mx = np.reshape(mx, (n, 1))
        return mx

    def _grad1(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate the local gradient of the kernel density
        g(x) = (p(x)/h^2)*[mean-shift(x) - x]"""
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        probs = self._p_KDE(data, x)
        mx = np.reshape(self._ms_KDE(data, x), (n, 1))
        gx = (probs / (self.h**2)) * (mx - x)
        return gx 

    def _grad2(self, data: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate the local gradient of the kernel density
        g(x) = (-1/N)sum(ci*ui)
        (faster than _grad1)"""
        data = np.atleast_2d(data)
        n, N = data.shape
        x = np.atleast_2d(x)
        m, M = x.shape
        if M != 1:
            x = np.reshape(x, (n, 1))
        cs = self._c(data, x)
        us = self._u(data, x)
        gx = -np.sum(cs * us, axis=1) / N
        gx = np.reshape(gx, (n, 1))
        return gx
    
    # =============================================================================
    # # Methods for the GMM-based algorithm
    # =============================================================================
    def _projectpoint_GMM(self, data: np.ndarray, point: np.ndarray, alphas: np.ndarray=None, means: np.ndarray=None, mcovs: np.ndarray=None) -> np.ndarray:
        """Project a point onto the principal curve, using a GMM"""
        n, N = data.shape
        converged = False
        citer = 0
        while not(converged):
            cov, G = self._inversecov_GMM(point, alphas, means, mcovs) # calculate the local covariance matrix
            w, v = np.linalg.eigh(cov) # get the eigenvalues in ascending order and the corresponding normalized eigenvectors
            index = np.argsort(w)  # arguments that sort from small to large
            V = v[:, index[:(n - self.d)]] # take the (n-d) smallest eigenvectors
            ospace = np.dot(V, V.T) # projection matrix
            proj = self._ms_GMM(n, cov, G, alphas, means, mcovs) - point # evaluate the mean shift update
            proj = np.dot(ospace, proj) + point
            diff = np.linalg.norm(point - proj)
            point = proj
            citer = citer + 1
            if diff < self.tolerance: # stopping condition based on distance
                converged = True
            if citer > self.maxiter: # stopping condition based on the number of iterations
                converged = True
                print("maximum iterations exceeded")
        return point
    
    def _find_nbcomponents(self, data: np.ndarray, min_nb: int=1, max_nb: int=40) -> np.ndarray:
        """Find the best number of mixture components using the BIC criterion"""
        nb_clusters = np.arange(min_nb, max_nb)
        Bics = []
        for n in nb_clusters:
            bics = []
            for _ in range(20):
                gmm = GaussianMixture(n_components=n)
                gmm.fit(data.T)
                bics.append(gmm.bic(data.T))
            Bics.append(np.mean(bics))
        nbcomponents = np.argmin(Bics) + min_nb
        return nbcomponents
    
    def _plotpdf(self, xmin: float, xmax: float, ymin: float, ymax: float, density: int, alpha: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> None:
        """Plot the GMM pdf (only works in 2 dimensions)"""
        x = np.linspace(xmin, xmax, 10*density+1)
        y = np.linspace(ymin, ymax, 10*density+1)
        X, Y = np.meshgrid(x, y)        
        p = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                p[i,j] = self._pGMM(np.vstack((x[i],y[j])), alpha, mu, Sigma)
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, p.T, cmap='viridis', edgecolor='none')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("pdf")
        
    def _p_GMM(self, x: np.ndarray, alpha: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        ndim, ncomp = mu.shape
        t1 = np.zeros((ndim, ncomp))
        G = np.zeros((ncomp))
        for m in range(ncomp):
            invSigma = np.linalg.inv(Sigma[:, :, m])
            C = 1 / ((2*np.pi) * np.sqrt(np.linalg.det(invSigma)))
            t1[:, m] = invSigma.dot(x.ravel() - mu[:, m])
            G[m] = C * np.exp(-0.5 * (x.ravel() - mu[:, m]).T.dot(t1[:, m]))    
        p = alpha.T.dot(G)
        return p
        
    def _inversecov_GMM(self, x: np.ndarray, alpha: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> (np.ndarray, np.ndarray):
        """Calculate the inverse covariance matrix 
        cov(x) = H(x)/p(x) - g(x)g(x).T/p(x)^2 
        and G(x), another vector used in the projection.
        """
        ndim, ncomp = mu.shape
        t1 = np.zeros((ndim, ncomp))
        G = np.zeros((ncomp))
        t2 = np.zeros((ndim, ndim, ncomp))
        for i in range(ncomp):
            invSigma = np.linalg.inv(Sigma[:, :, i])
            C = 1 / ((2*np.pi) * np.sqrt(np.linalg.det(invSigma)))
            t1[:, i] = invSigma.dot(x.ravel() - mu[:, i])
            G[i] = C * np.exp(-0.5 * (x.ravel() - mu[:, i]).T.dot(t1[:, i]))
            t2[:, :, i] = alpha[i] * G[i] * (t1[:, i].dot(t1[:, i].T) - invSigma)
              
        p = alpha.T.dot(G)
        g = -t1.dot(G * alpha)
        H = t2.sum(axis=2)
        cov = H / p - g.dot(g.T) / p**2
        return cov, G 
    
    def _ms_GMM(self, n: int, cov: np.ndarray, G: np.ndarray, alpha: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Calculate the mean-shift at point x"""        
        a = np.linalg.inv(G.T.dot(alpha) * cov)
        b = cov.dot(mu).dot(G*alpha)
        ms = a.dot(b)
        return ms
