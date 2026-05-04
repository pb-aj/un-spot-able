import numpy as np
import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
import sys

def pca(arr, method='full_pca', ncomp=None):
    """
    Runs principle component analysis on the input array.

    Arguments
    ---------
    arr: 2D array
        (m, n) array, where n is the size of the dataset (e.g., times
        in an observation) and m is the number of vectors

    Returns
    -------
    evalues: 1D array
        array of eigenvalues of size n

    evectors: 2D array
        (m, m) array of sorted eigenvectors

    proj: 2D array
        (m, n) array of data projected in the new space

    """

    nt = arr.shape[1]

    if method == "full_pca":
        pca = PCA(n_components=ncomp, svd_solver = "full")
        pca.fit(arr.T)
        evalues  = pca.explained_variance_
        evectors = pca.components_
        evectors = evectors.T

        arr = arr.T
        # Subtract the mean
        m = (arr - np.mean(arr.T, axis=1)).T
        
        proj = np.dot(m.T,evectors).T
    
    elif method == 'pca':
        
        arr = arr.T
        # Subtract the mean
        m = (arr - np.mean(arr.T, axis=1)).T
        #m = arr
        # Compute eigenvalues
        evalues, evectors = np.linalg.eig(np.cov(m))

        # Sort descending
        idx = np.argsort(evalues)[::-1]
        evalues  = evalues[idx]
        evectors = evectors[:,idx]
        # Calculate projection of the data in the new space
        # Need to do it this way for numpy==1.21 (see below)
        # print(np.array_equal(np.cov(m).T,np.linalg.inv(np.cov(m))))

        proj = np.dot(m.T,evectors).T

        """
        *NOTE* on proj
        The initial version had:
        proj = np.dot(evectors.T, m)
        This version will cause numpy 1.21 to crash!
        Need above version, which is equivalent:
        (m.T * evectors).T = evectors.T * m.T.T = evectors.T * m.T
        """
    
    elif method == 'tsvd':
        # np.random.seed(26)
        tpca = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
        tpca.fit(arr.T)
        evalues  = tpca.explained_variance_
        evectors = tpca.components_
        proj = np.zeros((ncomp, nt))

        for i in range(ncomp):
            proj[i] = np.sum(evectors[i] * arr.T, axis=1)

        evectors = evectors.T

    else:
        print("----------------------------------------------------------------------------")
        print("\033[31mNo valid PCA present, please check the confirguration file.\033[0m")
        print("----------------------------------------------------------------------------")
        sys.exit()
            
        
    return evalues, evectors, proj
