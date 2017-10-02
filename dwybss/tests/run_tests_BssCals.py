'''
Created on 8 Sep 2017

@author: Miguel Molina Romero, Techical University of Munich
@contact: miguel.molina@tum.de
@license: LPGL
'''
import unittest
import nibabel as nib
import numpy as np
import os
import shutil
from dwybss.bss import bss, bss_cals, utils


class TestBssCals(unittest.TestCase):
    def setUp(self):
        self.out_path = os.path.join(os.getcwd(), 'test_files')
        shutil.rmtree(self.out_path, True)
        os.makedirs(self.out_path)
        
    def tearDown(self):
        shutil.rmtree(self.out_path)
        
    
    def testNoMaxIters(self):
        obss = bss_cals.BssCals()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing max_iters parameter.' in context.exception)
   
    def testNoTolx(self):
        obss = bss_cals.BssCals()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_iters': 100, 'max_sources': 2, 'tolfun': 1e-6}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing tolx parameter.' in context.exception)
            
    def testNegativeTolx(self):
        obss = bss_cals.BssCals()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_iters': 100, 'max_sources': 2, 'tolfun': 1e-6, 'tolx': -1}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Parameter tolx must be >= 0.' in context.exception)
        
    def testNoTolFun(self):
        obss = bss_cals.BssCals()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_iters': 100, 'max_sources': 2, 'tolx': 1e-6}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing tolfun parameter.' in context.exception)
          
    def testNegativeTolfun(self):
        obss = bss_cals.BssCals()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_iters': 100, 'max_sources': 2, 'tolfun': -1, 'tolx': 1e-6}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Parameter tolfun must be >= 0.' in context.exception)     
              
    def testBoundANoBounds(self):
        t2_bounds = None
        tes = np.array([0.07, 0.13])
        tes = tes[:, None]
        Aref = np.ones((2, 2))
        A = np.ones((2, 2))
        obss = bss_cals.BssCals()
        A = obss._bound_A(A, t2_bounds, tes)
        self.assertIsNotNone(A, 'Missing bounded A')
        np.testing.assert_array_almost_equal(A, Aref, 6, 'Wrong A')
        
    def testBoundAOverlap(self):
        t2_bounds = {'1':[0, 0.15], '2':[0, 0.3]}
        
        t2  = np.array([0.075, 0.15])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([0.7, 0.3])
        (_, Aref) = utils.build_A(t2,tes,f)
        
        A = np.ones((2, 2))
        obss = bss_cals.BssCals()
        A = obss._bound_A(A, t2_bounds, tes)
        A = A / np.linalg.norm(A, None, 0)
        self.assertIsNotNone(A, 'Missing bounded A')
        np.testing.assert_array_almost_equal(A, Aref, 6, 'Wrong A')
        
    def testBoundAFourTes(self):
        t2_bounds = {'1':[0, 0.15], '2':[0, 0.3], '3':[0, 0.5], '4':[0, 2]}
        
        t2  = np.array([0.075, 0.15, 0.25, 1])
        tes = np.array([0.04, 0.07, 0.13, 0.2])
        tes = tes[:, np.newaxis]
        f = np.array([0.25, 0.25, 0.25, 0.25])
        (_, Aref) = utils.build_A(t2,tes,f)
        
        A = np.ones((4, 4))
        obss = bss_cals.BssCals()
        A = obss._bound_A(A, t2_bounds, tes)
        A = A / np.linalg.norm(A, None, 0)
        self.assertIsNotNone(A, 'Missing bounded A')
        np.testing.assert_array_almost_equal(A, Aref, 6,'Wrong A')

    def testFixSNoPrior(self):
        nsamples = 30
        Sref = np.random.random((2, nsamples))
        s_priors = None
        obss = bss_cals.BssCals()
        S =  obss._fix_S(Sref, s_priors)
        np.testing.assert_array_almost_equal(S, Sref, 6,'Wrong H')
    
    def testFixSPrior(self):
        nsamples = 30
        S = np.random.random((2, nsamples))
        s_prior = np.random.random(nsamples)
        s_priors = {'2': s_prior}
        obss = bss_cals.BssCals()
        
        s_prior = np.linalg.norm(S[1,:]) * s_prior / np.linalg.norm(s_prior)
        Sref = np.stack((S[0,:], s_prior))
        Sref = Sref / np.max(Sref,1)[:, None]
        
        S =  obss._fix_S(S, s_priors)
        np.testing.assert_array_almost_equal(S, Sref, 6,'Wrong H')
        
    def testMethodFactorizationMask0(self):
        tes = np.array([0.06, 0.1])
        tes = tes[:, np.newaxis]
        X = np.random.random((2, 30))  
        mask = 0
        params = {'max_iters': 100, 'max_sources': 2 }
        r = c = s = 1
        result = self.__create_result_object()
        
        obss = bss_cals.BssCals()
        obss._method_factorization(X, tes, mask, params, result, r, c, s)
        self.assertEqual(np.sum(result.T2[r,c,s,:]),0, 'Unexpected T2 values')
        self.assertEqual(np.sum(result.f[r,c,s,:]),0, 'Unexpected f values')
        self.assertEqual(result.pd[r,c,s], 0, 'Unexpected S0 values')
        self.assertEqual(np.sum(np.sum(result.sources)), 0, 'Unexpected sources')
        self.assertEqual(result.rel_error[r,c,s], 0, 'Unexpected S0 values')
    
    def testMethodFatorizationPriorBound(self):
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        t2 = np.array([0.1, 2])
        f = np.array([0.7, 0.3])
        (A, _) = utils.build_A(t2, tes, f)
        b = 1000*np.ones(31)
        b[0] = 0
        Scsf = np.exp(np.dot(-b, 3e-3))
        S = np.random.random(31)
        S[0] = 1
        S = np.stack((S, Scsf))
        X = np.dot(A,S)
        
        mask = 1
        params = {'max_iters': 10, 'max_sources': 2, 'tolfun': 1e-8, 'tolx': 1e-8 }
        t2_bounds = {'1': [0, 0.2], '2': [1.5, 2.5]}
        s_prior = {'2': Scsf}
        
        r = c = s = 1
        result = self.__create_result_object()
        
        obss = bss_cals.BssCals()
        obss._method_factorization(X, tes, mask, params, result, r, c, s, s_prior, t2_bounds)
        np.testing.assert_array_almost_equal(result.T2[r,c,s,:], t2, 6, 'Unexpected T2 values')
        np.testing.assert_array_almost_equal(result.f[r,c,s,:], f, 6, 'Unexpected f values')
        self.assertEqual(result.pd[r,c,s], 1, 'Unexpected S0 values')
        np.testing.assert_array_almost_equal(result.sources[r,c,s,:,:], S, 3, 'Unexpected sources')
        self.assertAlmostEqual(result.rel_error[r,c,s], 0, 6, 'Unexpected S0 values')
       

    def __create_result_object(self):
        res = [96, 96, 25, 31]
        max_sources = 2
        nii_header = nib.Nifti1Header()
        result = bss.OutputResults(res, max_sources, nii_header, self.out_path)
        return result

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()