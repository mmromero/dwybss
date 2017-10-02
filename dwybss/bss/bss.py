'''
Created on 8 Sep 2017

Root class for all the bss methods (scalability)

@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''

import nibabel as nib
import numpy as np
import dwybss.bss.utils as ut
import os
from joblib import Parallel, delayed


class BSS:
    """Blind Source separation parent class
    
    Functions: 
    factorize: for general purpose applications of BSS.
    
    fwe: specifically designed for Free Water Elimination. 
    """
    
    
    def __check_data_consistency(self,data):
        if data == None:
            raise BssException('Missing data.')

        if 'dwi' not in data:
            raise BssException('Missing diffusion data.')

        if 'te' not in data:
            raise BssException('Missing TE data.')
        
        if data['dwi'] is None:
            raise BssException('Missing diffusion data.')
        
        if data['te'] is None:
            raise BssException('Missing TE data.')
        
        if len(data['dwi']) != len(data['te']):
            raise BssException('Number of TE values and diffusion files do not match')
        
    def __check_mask_consistency(self, mask):
        if mask is None:
            raise BssException('Missing mask.')
        
        if mask == []:
            raise BssException('Missing mask.')
        
    def __check_out_path_consistency(self, out_path):
        if out_path is None:
            raise BssException('Missing out_path.')
        
        if not os.path.exists(out_path):
            raise BssException('Non existing out_path.')
        
    def __check_params_cosnsistency(self, params):
        if params is None:
            raise BssException('Missing parameters.')
        
        if 'max_sources' not in params:
            raise BssException('Missing max_sources parameter.')
        
        if params['max_sources'] < 1:
            raise BssException('The parameter max_sources must be >= 1.')
         
        self._check_method_params_consistency(params)
    
    def _check_method_params_consistency(self, params):
        """TO BE IMPLEMENTED BY THE SPECIFIC METHOD"""
    
    def factorize(self, data, mask, params, out_path, s_prior = None, t2_bounds = None, run_parallel = True ):
        """Factorizes the data X, into the mixing matrix A and the sources S.
        
        This is a general function that accepts any kind of prior knowledge and constraints.         
        
        Usage::
        
        :param data: Dictionary containing the fields 'dwi' and 'TE'. 'dwi' contains a list of nifti files 
                     with the diffusion data to build X. 'TE' is the echo time value at which each dwi was
                     acquired, the order and number of elements of 'dwi' and 'TE' must match.
        :param mask: Path to the mask file.
        :param params: Dictionary containing the parameters for the factorization method.
        :param out_path: Path where the output files will be created.
        :param s_prior: Dictionary with prior knowledge for one or more sources. 'source' is the index of 
                        the column matrix of A associated with the source. 'data' is an array containing 
                        the actual information. ``s_prior = {'1':[1, 0.2, 0.4, ...], '3': [1, 0.45, 0.90, ....]}
        :param t2_bounds: Dictionary containing the bounds for the T2 value of the columns of A: 
                        ``t2_bounds = {'1': [0, 0.04], '3': [2000, 2000]}`
        :param run_parallel: True to use all the CPUs. False to do not parallelize execution.
        
        :raise BssExcpetion: When there is an error in the input parameters.
        
        :rtype: Dictionary
        :return: Path to the Nifti files containing the results: 'sources', 't2s', 'fs', 's0', 'nsources' and 'rel_error'. 
        """
        
        self.__check_data_consistency(data)
        self.__check_mask_consistency(mask)
        self.__check_params_cosnsistency(params)
        self.__check_out_path_consistency(out_path)
       
        results = self._volume_factorization(data, mask, params, out_path, s_prior, t2_bounds, run_parallel)
        
        return results
    
    def _volume_factorization(self, data, mask, params, out_path, s_prior, t2_bounds, run_parallel):
        """ Iterates over all the voxels in the volume and performs BSS in each of them.
        """
        
        # Load the mask
        msk = nib.load(mask).get_data()
        
        # Load the data
        nii = nib.load(data['dwi'][0])
        res = np.shape(nii.get_data())
        ntes = len(data['te'])
        X = np.zeros([res[0], res[1], res[2], ntes, res[3]])
        for i  in range(ntes):
            nii = nib.load(data['dwi'][i])
            niidata = nii.get_data()
            X[:,:,:,i,:] = niidata
            
        # Lunch BSS over the volume
        if run_parallel:
            num_cpus = -1
        else:
            num_cpus = 1
            
        result =  OutputResults(res,params['max_sources'], nii.header, out_path)    
        Parallel(n_jobs=num_cpus, backend='threading')(delayed(self._method_factorization)(X[r,c,s,:,:], data['te'], msk[r,c,s], params, result, r, c, s, s_prior, t2_bounds) for r in range(res[0]) for c in range(res[1]) for s in range(res[2]))

        # Save results
        return result.save()
        
    
    def _method_factorization(self, data, tes, mask, params, results, r, c, s, s_prior = None, t2_bounds = None):
        """TO BE IMPLEMENTED BY THE SPECIFIC METHOD"""
        
    def _compute_actual_A(self, X, A, tes, params):
        """Computes T2, f, proton density and sources from factorized A and the measurements X.
        """
        max_sources = params['max_sources']
        dims = np.shape(A)
        t2 = list()
        
        for c in range(dims[1]):        
            t2.append(ut.t2_from_slope(A[:,c], tes))
        
        t2 = np.round(t2,3)
        
        A = np.exp(-tes * (1./(t2 + 1e-8)))
        nsources = np.linalg.matrix_rank(A)
        
        if nsources == 0:
            return {'t2': np.zeros(max_sources), 'f': np.zeros(max_sources), 's0': 0, 'nsources': nsources, 
                    'sources': np.zeros(np.shape(X)), 'A': np.zeros(dims)}
        else:
            Xmax = np.max(X, 1)
            f = np.linalg.lstsq(A,Xmax[:, None])[0]
            f[f < 0] = 0
            s0 = np.round(np.sum(f),3)
            f = f / s0
            f = f.transpose()[0]
            f = np.round(f,2)
            A = A * f
            nsources = np.linalg.matrix_rank(A)
            S = np.linalg.lstsq(A,X)[0] / s0
            t2[f == 0] = 0
            return {'t2': t2, 'f': f, 's0': s0, 'nsources': nsources, 'sources': S, 'A': A}
        
    def fwe(self, data, mask, b_values, out_path):
        """Free-water elimination method.
        
        All the necessary assumptions on tissue relaxation and diffusivity are encoded in this function. 
        Notice that it always tries to run in parallel.
        
        Usage::
        
        :param data: Dictionary containing the fields 'dwi' and 'TE'. 'dwi' contains a list of nifti files 
                     with the diffusion data to build X. 'TE' is the echo time value at which each dwi was
                     acquired, the order and number of elements of 'dwi' and 'TE' must match.
        :param mask: Path to the mask file.
        :param b_values: Path to the *.bval file. Units must be mm^2/s.
        :param out_path: Path where the output files will be created.
        
        :raise BssExcpetion: When there is an error in the input parameters.
        
        :rtype: Dictionary
        :return: Path to the Nifti files containing the results: 'sources', 't2s', 'fs', 's0', 'nsources' and 'rel_error'. 
        """
        
        # Check b values consistency
        if b_values == None:
            raise BssException("b_values is a mandatory parameter")
    
        if not os.path.exists(b_values):
            raise BssException('Wrong path to b_values')
                
        # Read bo
        bval = np.fromfile(b_values, float, -1, sep=' ')
        if 0 not in bval:
            raise BssException('At least one b0 is required')
        
        # Define priors on CSF
        Dcsf = 3e-3; # mm^2/s
        Scsf = np.exp(-bval * Dcsf)
        t2_bounds = {'1': [0, 0.3], '2': [1.5, 2.5]}
        s_prior = {'2': Scsf}
        params = {'max_sources': 2,'max_iters': 100, 'tolx': 1e-12, 'tolfun': 1e-12}
        
        return self.factorize(data, mask, params, out_path, s_prior, t2_bounds, True)
    
    
class BssException(Exception):
    def __init__(self, message, errors=0):

        # Call the base class constructor with the parameters it needs
        super(BssException, self).__init__(message)

        # Now for your custom code...
        self.errors = errors      
        
        
class OutputResults():
    """Storage object to keep and save the factorization results.  
    """
    def __init__(self, res, max_sources, nii_header, out_path):
        """An OutputResult object is defined by the image resolution to be stored and saved,
        the maximum number of sources, the voxel dimensions included in the Nifti header,
        and the output path for the factorization results that will be stored in Nifti format.
                
        :param res: Four elements list containing the number of rows, columns, slices, and diffusion directions in this order.
        :param max_sources: Maximum number of sources as defined in the `param` dictionary
        :param nii_header: Nifti header of the one of the `dwi` files containing the voxel dimensions and affine.
        :param out_path: Existing directory to save the output Nifti files.
        
        :ivar T2: Matrix of shape (res[0], res[1], res[2], max_sources) that stores the resulting T2 values.
        :ivar f: Matrix of shape (res[0], res[1], res[2], max_sources) that stores the resulting f values.
        :ivar pd: Matrix of shape (res[0], res[1], res[2]) that stores the resulting proton density value.
        :ivar nsources: Matrix of shape (res[0], res[1], res[2]) that stores the resulting number of sources per voxel.
        :ivar sources: Matrix of shape (res[0], res[1], res[2], max_sources, diff_directions ) that stores the resulting sources.
        :ivar rel_error: Matrix of shape (res[0], res[1], res[2]) that stores the resulting factorization relative error. 
        """
        self.__max_sources = max_sources
        self.__nii_header = nii_header
        self.__out_path = out_path
        self.T2 = np.zeros([res[0], res[1], res[2], max_sources])
        self.f =  np.zeros([res[0], res[1], res[2], max_sources])
        self.pd = np.zeros([res[0], res[1], res[2]])
        self.nsources = np.zeros([res[0], res[1], res[2]])
        self.sources = np.zeros([res[0], res[1], res[2], max_sources, res[3]])
        self.rel_error = np.zeros([res[0], res[1], res[2]])

    
    def __save_file(self, data, name):
        data = np.squeeze(data)
        
        # Check dimensions of the header
        dims = np.shape(data)
        self.__nii_header.set_data_shape(dims)
                
        # Build NIFTI
        nii = nib.Nifti1Image(data.astype(np.float32), self.__nii_header.get_best_affine(), self.__nii_header)
    
        # Save it
        fpath = os.path.join(self.__out_path, name + '.nii.gz')
        nib.save(nii, fpath)
        return fpath
    
    def save(self):
        """Once the instance matrix variables have been filled up. Call save to create the Nifti files in the 
        `out_path` folder.
        
        :return: Dictionary with the paths and names for the output Nifti files.
        """
        T2 = list()
        f = list()
        sources = list()
                
        S = self.pd[:,:,:, None, None] * (self.f[:,:,:,:, None] * self.sources)
                
        for i in range(self.__max_sources):
            T2.append(self.__save_file(self.T2[:,:,:,i],'T2_{}'.format(i)))
            f.append(self.__save_file(self.f[:,:,:,i],  'f_{}'.format(i)))
            sources.append(self.__save_file(S[:,:,:,i,:],'source_{}'.format(i)))
            
        files = {'T2': T2, 'f': f, 'sources': sources}
        files['pd'] = self.__save_file(self.pd, 'pd')
        files['nsources'] = self.__save_file(self.nsources, 'nsources')
        files['rel_error'] = self.__save_file(self.rel_error, 'rel_error')
        return files        