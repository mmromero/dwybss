'''
Created on 26 Sep 2017

@author: Miguel Molina Romero
@contact: miguel.molina@tum.de
@license: LPGL
'''
import numpy as np

def t2_from_slope(a,tes):
    (lte, ste) = generate_indexes(len(tes))
    
    # filter out nan or infs 
    alte = a[lte]
    aste = a[ste]
    idxnan = np.squeeze(np.argwhere(np.logical_or(np.isnan(alte), np.isnan(aste))))
    idxinf = np.squeeze(np.argwhere(np.logical_or(np.isinf(alte), np.isinf(aste))))
    idxzer = np.squeeze(np.argwhere(np.logical_and(alte == 0, aste == 0)))
    alte = np.delete(alte, idxnan)
    aste = np.delete(aste, idxnan)
    alte = np.delete(alte, idxinf)
    aste = np.delete(aste, idxinf)
    alte = np.delete(alte, idxzer)
    aste = np.delete(aste, idxzer)
    
    # If no valid array return 0
    if len(alte) == 0 and len(aste) == 0:
        return 0
    
    # Calculate m
    m = np.divide(alte, aste)
    m[m <= 0] = 1e-8
    m[m >= 1] = 1 - 1e-8
    
    # Calculate T2
    t2 = np.divide((tes[ste] - tes[lte]).transpose()[0],np.log(m))
    
    # Remove incorrect values
    t2[np.argwhere(np.isnan(t2))] = 0
    t2[np.argwhere(np.isinf(t2))] = 0
    t2[t2 < 0] = 0
    return np.mean(t2[t2 > 0])


def generate_indexes(N):
    """ Creates the indexes to calculate T2 in for all the combinations
    in the mixing matrix, A
    
    :param N: Number of TE measured
    """
    
    lte = list()
    ste = list()
    
    idx = range(N)    
        
    for l in idx[1:]:
        for s in idx:
            if s == l:
                break
            else:
                ste.append(s)
                lte.append(l)
                
    return (lte, ste)
        
def build_A(t2, tes, f):
    A = np.exp(-tes * (1./t2) )
    A = A * f
    Anorm = np.linalg.norm(A, None, 0)
    Anorm[Anorm == 0] = 1e-8
    Anorm = np.divide(A, Anorm)
    return (A, Anorm)