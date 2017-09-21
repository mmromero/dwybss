'''
Created on 8 Sep 2017

@author: miguel
'''
import unittest
import nibabel as nib
import os
import shutil
from dwybss.bss import bss

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
        params = {'maxiters': 100, 's_priors': 0, 'max_sources': 2}
                
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
        params = {'maxiters': 100, 's_priors': 0, 'max_sources': 2}
                
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
        params = {'maxiters': 100, 's_priors': 0, 'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing TE data' in context.exception)

    def testNoDwiData(self):
        obss = bss.BSS()
        TE = [60, 120]
        data = {'te': TE}        
        mask = 'mask.nii.gz'
        params = {'maxiters': 100, 's_priors': 0, 'max_sources': 2}
                
        with self.assertRaises(bss.BssException) as context:
            results = obss.factorize(data, mask, params,'/')
            self.assertIsNone(results, 'Unexpected result')
            self.assertTrue('Missing diffusion data' in context.exception)
    
    def testNoTEData(self):
        obss = bss.BSS()
        dwi = ['data_TE1.nii.gz', 'data_TE2.nii.gz']
        data = {'dwi': dwi}        
        mask = 'mask.nii.gz'
        params = {'maxiters': 100, 's_priors': 0, 'max_sources': 2}
                
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
        params = {'maxiters': 100, 's_priors': 1, 'max_sources': 2}
        
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
        params = {'maxiters': 100, 's_priors': 1, 'max_sources': 2}
        
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
        params = {'maxiters': 100, 's_priors': 1, 'max_sources': 2}
        
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
        params = {'maxiters': 100, 's_priors': 1, 'max_sources': 2}
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
        params = {'maxiters': 100, 's_priors': 1, 'max_sources': 2}
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
        

class TestOutputResult(unittest.TestCase):
    
    def setUp(self):
        self.out_path = os.path.join(os.getcwd(), 'test_files')
        shutil.rmtree(self.out_path, True)
        os.makedirs(self.out_path)
        pass
        
    def tearDown(self):
        shutil.rmtree(self.out_path)
        pass
    
    def testCreateResultObject(self):
        res = [96, 96, 25, 31]
        max_sources = 2
        nii_header = nib.Nifti1Header()
        result = bss.OutputResults(res, max_sources, nii_header, self.out_path)
        self.assertIsNotNone(result, 'Null result not expected')

    def testSaveResultObject(self):
        res = [96, 96, 25, 31]
        max_sources = 2
        nii_header = nib.Nifti1Header()
        
        result = bss.OutputResults(res, max_sources, nii_header, self.out_path)
        self.assertIsNotNone(result, 'Null result not expected')        
        
        niiout = result.save()
        
        self.assertEqual(len(niiout['T2']), max_sources, 'Incorrect number of T2 files');
        self.assertEqual(len(niiout['f']), max_sources, 'Incorrect number of volume fraction files');
        self.assertEqual(len(niiout['sources']), max_sources, 'Incorrect number of source files');
        self.assertIsNotNone(niiout['pd'], 'No Proton Density file');
        self.assertIsNotNone(niiout['rel_error'], 'No Relative Error file');     
        
        for i in range(max_sources):
            self.assertTrue(os.path.isfile(niiout['T2'][i]), 'Non existing T2 file') 
            self.assertTrue('T2_' in niiout['T2'][i],'Wrong Name for T2 file')
            self.assertTrue(os.path.isfile(niiout['f'][i]), 'Non existing volume fraction file')  
            self.assertTrue('f_' in niiout['f'][i],'Wrong Name for volume fraction file')
            self.assertTrue(os.path.isfile(niiout['sources'][i]), 'Non existing source file')   
            self.assertTrue('source_' in niiout['sources'][i],'Wrong Name for source file')
            
        
        self.assertTrue(os.path.isfile(niiout['pd']), 'Non existing proton density file') 
        self.assertTrue(os.path.isfile(niiout['rel_error']), 'Non existing relative error file')     
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()