'''
Created on 8 Sep 2017

Constrained ALS (NMF) implementation (paper)

@author: Miguel Molina Romero
@contact: miguel.molina@tum.de
@license: LPGL
'''

from dwybss.bss.bss import BSS, BssException
from dwybss.bss.utils import t2_from_slope
import numpy as np

class BssCals(BSS):
    """Blind source separation implementation for Constrained ALS (NMF)
    """
    
    def _check_method_params_consistency(self, params):
        if 'max_iters' not in params:
            raise BssException('Missing max_iters parameter.')
        
        if 'tolx' not in params:
            raise BssException('Missing tolx parameter.')
        
        if 'tolfun' not in params:
            raise BssException('Missing tolfun parameter.')
        
        if params['tolx'] <= 0:
            raise BssException('Parameter tolx must be >= 0.')

        if params['tolfun'] <= 0:
            raise BssException('Parameter tolfun must be >= 0.')

    def __compute_rel_fac_error(self, X, s0, A, S):
        """ Calculate the factorization relative error e = |X - S0*A*S|/|X|
        """
        return np.linalg.norm(X - s0*np.dot(A,S)) / np.linalg.norm(X)
        
    def _method_factorization(self, data, tes, mask, params, results, r, c, s, s_prior = None, t2_bounds = None):
        """Reference:  M.W. Berry et al. (2007), Algorithms and Applications for Approximate Nonnegative Matrix
        Factorization, Computational Statistics and Data Analysis, vol. 52, no. 1, pp. 155-173.
        """
        
        X = np.squeeze(data)
        mask = np.squeeze(mask)
        
        if not mask:
            return
        
        A = np.ones([len(tes),params['max_sources']])
        A0 = self._bound_A(A,t2_bounds, tes)
        S0 = X
        dnorm0 = np.linalg.norm(X)*1e6
        
        eps = 1e-8        
        X[X <= 0] = eps
                
        for _ in range(params['max_iters']):
            S = np.linalg.solve(A0, X)         
            S[S <= 0] = eps
            S = self._fix_S(S, s_prior)
            
            SST = np.dot(S,S.T)
            SXT = np.dot(S,X.T)
            
            A = np.linalg.solve(SST, SXT).T
            A [A <= 0] = eps
            A = self._bound_A(A,t2_bounds, tes)
            
            d = X - np.dot(A,S)
            dnorm = np.linalg.norm(d)
            da = np.max(np.abs(A-A0) / (eps + np.max(np.abs(A0))))
            dh = np.max(np.abs(S-S0) / (eps + np.max(np.abs(S0))))
            delta  = max((da, dh))
             
            if delta < params['tolx']:
                break
            if dnorm0-dnorm <= params['tolfun']*max((1,dnorm0)):
                break
             
            dnorm0 = dnorm
            A0 = A
            S0 = S
        
            
        A = A / np.linalg.norm(A, None, 0)
        result = self._compute_actual_A(X, A, tes, params)
        
        results.T2[r,c,s,:] = result['t2']
        results.f[r,c,s,:] = result['f']
        results.pd[r,c,s] = result['s0']
        results.nsources[r,c,s] = result['nsources']
        results.sources[r,c,s,:,:] = result['sources']
        results.rel_error[r,c,s] = self.__compute_rel_fac_error(data, result['s0'], result['A'], result['sources'])
        
    def _bound_A(self, A, t2_bounds, tes):
        
        if t2_bounds == None:
            return A
        
        dims = np.shape(A)
        
        for c in range(dims[1]):
            col = str(c+1)
            if col in t2_bounds:
                bounds = t2_bounds[col]           
                t2 = t2_from_slope(A[:,c], tes)
                if t2 < np.min(bounds) or t2 > np.max(bounds):
                    t2 = np.mean(bounds)
                    A[:,c] = np.squeeze(np.exp( -tes * (1./t2)))
        
        return A
    
    def _fix_S(self, S, s_priors):
        if s_priors == None:
            return S
        
        dims = np.shape(S)    
            
        for r in range(dims[0]):
            row = str(r+1)
            if row in s_priors:
                S[r,:] = s_priors[row] 
        
        return S / np.max(S, 1)[:, None]
        