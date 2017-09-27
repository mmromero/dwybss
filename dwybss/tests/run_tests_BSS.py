'''
Created on 26 Sep 2017

@author: Miguel Molina Romero
@contact: miguel.molina@tum.de
@license: LPGL
'''
import unittest
import numpy as np
import os
import shutil
from dwybss.bss import bss, utils

class TestBss(unittest.TestCase):

    def setUp(self):
        self.out_path = os.path.join(os.getcwd(), 'test_files')
        shutil.rmtree(self.out_path, True)
        os.makedirs(self.out_path)
        pass
        
    def tearDown(self):
        shutil.rmtree(self.out_path)
        pass

    def testEmptyData(self):
        obss = bss.BSS()
        data = None
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing data' in context.exception)
            
    def testEmptyDwiData(self):
        obss = bss.BSS()
        dwi = None
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}        
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing diffusion data' in context.exception)

    def testEmptyTEData(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = None
        data = {'dwi': dwi, 'te': TE}        
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing TE data' in context.exception)

    def testNoDwiData(self):
        obss = bss.BSS()
        TE = [60, 120]
        data = {'te': TE}        
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing diffusion data' in context.exception)
    
    def testNoTEData(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        data = {'dwi': dwi}        
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing TE data' in context.exception)
            
    def testEmptyMask(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = None
        params = {'max_sources': 2}
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing mask' in context.exception)   
            
    def testNoMask(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = []
        params = {'max_sources': 2}
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing mask' in context.exception)   
            
            
    def testEmptyParams(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = None
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing mask' in context.exception)   
                
    def testMissmatchingTeData(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Number of TE values and diffusion files do not match' in context.exception)    
            
    def testMissingOutpath(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
        out_path = None
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params, out_path)
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing out_path.' in context.exception)    
            
    def testInvalidOutpath(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        TE = [60, 120]
        data = {'dwi': dwi, 'te': TE}
        mask = 'mask.nii.gz'
        params = {'max_sources': 2}
        out_path = '/bss_test_doesnt_exist'
        
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params, out_path)
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Non existing out_path.' in context.exception) 
            
            
    def test__volume_factorization_ParallelTrue(self):
        obss = bss.BSS()
        max_sources = 2
        params = {'max_sources' : max_sources}
        cwd = os.path.join(os.getcwd(), '..', 'examples', 'data')
        dwi = [os.path.join(cwd,'data_TE1.nii.gz'), os.path.join(cwd,'data_TE2.nii.gz')]
        TE = [0.0751, 0.1351]
        data = {'dwi': dwi, 'te': TE}
        mask = os.path.join(cwd,'mask.nii.gz')
        
        niiout = obss._volume_factorization(data, mask, params, None, self.out_path, None, True)
        
        self.assertIsNotNone(niiout, 'Missing Result')
        self.assertEqual(len(niiout['T2']), max_sources, 'Incorrect number of T2 files');
        self.assertEqual(len(niiout['f']), max_sources, 'Incorrect number of volume fraction files');
        self.assertEqual(len(niiout['sources']), max_sources, 'Incorrect number of source files');
        self.assertIsNotNone(niiout['pd'], 'No Proton Density file');
        self.assertIsNotNone(niiout['rel_error'], 'No Relative Error file');     
        self.assertTrue(os.path.isfile(niiout['pd']), 'Non existing proton density file') 
        self.assertTrue(os.path.isfile(niiout['rel_error']), 'Non existing relative error file')  
        for i in range(max_sources):
            self.assertTrue(os.path.isfile(niiout['T2'][i]), 'Non existing T2 file') 
            self.assertTrue('T2_' in niiout['T2'][i],'Wrong Name for T2 file')
            self.assertTrue(os.path.isfile(niiout['f'][i]), 'Non existing volume fraction file')  
            self.assertTrue('f_' in niiout['f'][i],'Wrong Name for volume fraction file')
            self.assertTrue(os.path.isfile(niiout['sources'][i]), 'Non existing source file')   
            self.assertTrue('source_' in niiout['sources'][i],'Wrong Name for source file')

    def test__volume_factorization_ParallelFalse(self):
        obss = bss.BSS()
        max_sources = 2
        params = {'max_sources' : max_sources}
        cwd = os.path.join(os.getcwd(), '..', 'examples', 'data')
        dwi = [os.path.join(cwd,'data_TE1.nii.gz'), os.path.join(cwd,'data_TE2.nii.gz')]
        TE = [0.0751, 0.1351]
        data = {'dwi': dwi, 'te': TE}
        mask = os.path.join(cwd,'mask.nii.gz')
        
        niiout = obss._volume_factorization(data, mask, params, None, self.out_path, None, False)
        
        self.assertIsNotNone(niiout, 'Missing Result')
        self.assertEqual(len(niiout['T2']), max_sources, 'Incorrect number of T2 files');
        self.assertEqual(len(niiout['f']), max_sources, 'Incorrect number of volume fraction files');
        self.assertEqual(len(niiout['sources']), max_sources, 'Incorrect number of source files');
        self.assertIsNotNone(niiout['pd'], 'No Proton Density file');
        self.assertIsNotNone(niiout['rel_error'], 'No Relative Error file');     
        self.assertTrue(os.path.isfile(niiout['pd']), 'Non existing proton density file') 
        self.assertTrue(os.path.isfile(niiout['rel_error']), 'Non existing relative error file')  
        for i in range(max_sources):
            self.assertTrue(os.path.isfile(niiout['T2'][i]), 'Non existing T2 file') 
            self.assertTrue('T2_' in niiout['T2'][i],'Wrong Name for T2 file')
            self.assertTrue(os.path.isfile(niiout['f'][i]), 'Non existing volume fraction file')  
            self.assertTrue('f_' in niiout['f'][i],'Wrong Name for volume fraction file')
            self.assertTrue(os.path.isfile(niiout['sources'][i]), 'Non existing source file')   
            self.assertTrue('source_' in niiout['sources'][i],'Wrong Name for source file')
        
    def testComputeActualAwithNaN(self):
        t2  = np.array([0.075, 2])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([0.7, 0.3])
        (A, _) = utils.build_A(t2,tes,f)
                
        S = np.random.random((2, 30))
        S[:,0] = 1
        X = np.dot(A,S)
        
        A[1,1] = np.nan
        
        params = {'max_sources': 2}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], [0.075, 0], 6, 'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], [1, 0], 6, 'Wrong f values')
        
        
    def testComputeActualAwithInf(self):
        t2  = np.array([0.075, 2])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([0.7, 0.3])
        (A, _) = utils.build_A(t2,tes,f)
                
        S = np.random.random((2, 30))
        S[:,0] = 1
        X = np.dot(A,S)
        
        A[1,1] = np.inf
        
        params = {'max_sources': 2}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], [0.075, 0], 6, 'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], [1, 0], 6, 'Wrong f values')
    
    def testComputeActualAwithZero(self):
        t2  = np.array([0.075, 2])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([1, 0])
        (A, _) = utils.build_A(t2,tes,f)
                
        S = np.random.random((2, 30))
        S[:,0] = 1
        X = np.dot(A,S)
                
        params = {'max_sources': 2}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], [0.075, 0], 6, 'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], [1, 0], 6, 'Wrong f values')
        np.testing.assert_array_almost_equal(result['s0'], 1, 6, 'Wrong S0 value')    
    
    def testComputeActualANonExisitngSource(self):
        t2  = np.array([0.075, 2])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([1, 0])
        A = np.exp(-tes * (1./t2) )
                
        S = np.random.random(30)
        S[0] = 1
        S = np.stack((S, np.zeros(30)))
        X = np.dot(A,S)
                
        params = {'max_sources': 2}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], [0.075, 0], 6,'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], [1, 0], 6, 'Wrong f values')
        np.testing.assert_array_almost_equal(result['s0'], 1, 6, 'Wrong S0 value')

    def testComputeActualA2ExisitingSources(self):
        t2  = np.array([0.075, 2])
        tes = np.array([0.07, 0.13])
        tes = tes[:, np.newaxis]
        f = np.array([0.7, 0.3])
        (A, _) = utils.build_A(t2,tes,f)
                
        S = np.random.random((2, 30))
        S[:,0] = 1
        X = np.dot(A,S)
        
        params = {'max_sources': 2}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], t2, 6, 'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], f, 6, 'Wrong f values')
        np.testing.assert_array_almost_equal(result['s0'], 1, 6, 'Wrong S0 value')

    def testComputeActualA3ExisitingSources(self):
        t2  = np.array([0.075, 0.15, 2])
        tes = np.array([0.07, 0.13, 0.2])
        tes = tes[:, np.newaxis]
        f = np.array([0.5, 0.3, 0.2])
        (A, _) = utils.build_A(t2,tes,f)
                
        S = np.random.random((3, 30))
        S[:,0] = 1
        X = np.dot(A,S)
                
        params = {'max_sources': 3}
        obss = bss.BSS()
        result = obss._compute_actual_A(X, A, tes, params)
        np.testing.assert_array_almost_equal(result['t2'], t2, 6, 'Wrong T2 values')
        np.testing.assert_array_almost_equal(result['f'], f, 6, 'Wrong f values')
        self.assertEqual(result['s0'], 1, 'Wrong S0 value')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()