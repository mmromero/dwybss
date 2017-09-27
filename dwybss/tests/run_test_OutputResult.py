'''
Created on 26 Sep 2017

@author: Miguel Molina Romero
@contact: miguel.molina@tum.de
@license: LPGL
'''
import unittest
import nibabel as nib
import os
import shutil
from dwybss.bss import bss
       
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
        self.assertIsNotNone(niiout['nsources'], 'No number of sources file');
        self.assertIsNotNone(niiout['rel_error'], 'No Relative Error file');     
        
        for i in range(max_sources):
            self.assertTrue(os.path.isfile(niiout['T2'][i]), 'Non existing T2 file') 
            self.assertTrue('T2_' in niiout['T2'][i],'Wrong Name for T2 file')
            self.assertTrue(os.path.isfile(niiout['f'][i]), 'Non existing volume fraction file')  
            self.assertTrue('f_' in niiout['f'][i],'Wrong Name for volume fraction file')
            self.assertTrue(os.path.isfile(niiout['sources'][i]), 'Non existing source file')   
            self.assertTrue('source_' in niiout['sources'][i],'Wrong Name for source file')
            
        
        self.assertTrue(os.path.isfile(niiout['pd']), 'Non existing proton density file') 
        self.assertTrue(os.path.isfile(niiout['nsources']), 'Non existing number of sources file') 
        self.assertTrue(os.path.isfile(niiout['rel_error']), 'Non existing relative error file')     
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()