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

    def __compute_rel_fac_error(self, X, s0, A, S):
        """ Calculate the factorization relative error e = |X - S0*A*S|/|X|
        """
        return np.linalg.norm(X - s0*np.dot(A,S)) / np.linalg.norm(X)
        
    def _method_factorization(self, data, tes, mask, params, results, r, c, s, s_prior = None, t2_bounds = None):
        """TO BE IMPLEMENTED BY THE SPECIFIC METHOD"""
        
        X = np.squeeze(data)
        mask = np.squeeze(mask)
        
        if not mask:
            return
        
        A = np.ones([len(tes),params['max_sources']])
        A = self._bound_A(A,t2_bounds, tes)
        
        eps_H =  1e-6 * np.ones(np.shape(X))
        eps_A =  1e-6 * np.ones(np.shape(A))
        
        for _ in range(params['max_iters']):
            H = np.maximum(eps_H, np.linalg.lstsq(A,X)[0])
            H = self._fix_H(H, s_prior)
            A = np.maximum(eps_A, np.linalg.lstsq(H.T, X.T)[0].T)
            A = self._bound_A(A,t2_bounds, tes)
            
            
        result = self._compute_actual_A(X, A, tes, params)
        
        results.T2[r,c,s,:] = result['t2']
        results.f[r,c,s,:] = result['f']
        results.pd[r,c,s] = result['s0']
        results.sources[r,c,s] = result['nsources']
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
        
        A = np.divide(A, np.linalg.norm(A,None,0))
        return A
    
    def _fix_H(self, H, s_priors):
        if s_priors == None:
            return H
        
        dims = np.shape(H)    
            
        for r in range(dims[0]):
            row = str(r+1)
            if row in s_priors:
                mod = np.linalg.norm(H[r,:])
                H[r,:] = mod * s_priors[row] / np.linalg.norm(s_priors[row])
        
        return H
        