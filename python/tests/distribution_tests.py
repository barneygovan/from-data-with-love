import unittest

import numpy as np

from stats.distributions import *
from stats import SingularMatrixError

class DmvnormTest(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_standard_norm(self):
        testData = [[-0.98678747, -0.60957898],
                    [ 0.03695369,  0.30576259],
                    [-0.20501798, -0.45028401],
                    [ 0.48431871,  0.83691930],
                    [ 0.99531565,  0.87849261],
                    [-0.93168255, -0.13436839],
                    [ 0.12957123,  0.24968244],
                    [ 0.91785772,  0.72786262],
                    [ 0.47459646,  0.31605869],
                    [-0.90350400, -0.88214619]]
                    
        expectedResults = [0.08122395, 0.15178272, 0.14082024, 0.09972127, 
                       0.06593550, 0.10219008, 0.15298092, 0.08013837, 
                       0.13527514, 0.07170981]
                       
        testData = np.array(testData)
        expectedResults = np.array(expectedResults)
        
        self.assertEquals(testData.shape[0],expectedResults.shape[0])
        
        results = dmvnorm(testData)
        
        self.assertEquals(results.shape,expectedResults.shape)
        self.assertTrue(np.allclose(results, expectedResults))
        
        
    def test_location_scale(self):
        mean = np.array([4.0,12.0])
        cov = np.array([[2.0,-0.5],[-0.5,3.2]])
        
        testData = np.array([[3.151483, 12.86639],
                            [3.471154, 11.63464],
                            [3.165634, 10.19458],
                            [4.118668, 13.94394],
                            [4.754333, 11.01473],
                            [3.231982, 11.65272],
                            [3.670970, 10.60487],
                            [3.375193, 11.89571],
                            [4.412968, 12.55705],
                            [3.569596, 13.29352]])

        expectedResults = np.array([0.05000157, 0.05748222, 0.02788413, 
                                    0.03394635, 0.05020890, 0.05282051, 
                                    0.04380215, 0.05757120, 0.05728992, 0.04874755])
                                    
        self.assertEquals(testData.shape[0], expectedResults.shape[0])
        
        results = dmvnorm(testData,mu=mean,sigma=cov)

        self.assertEquals(results.shape,expectedResults.shape)
        self.assertTrue(np.allclose(results, expectedResults))

        
    def test_singular_covariance(self):
        testData = np.array([[3.151483, 12.86639],
                            [3.471154, 11.63464],
                            [3.165634, 10.19458],
                            [4.118668, 13.94394],
                            [4.754333, 11.01473],
                            [3.231982, 11.65272],
                            [3.670970, 10.60487],
                            [3.375193, 11.89571],
                            [4.412968, 12.55705],
                            [3.569596, 13.29352]])
                            
        mean = np.array([4.0,12.0])
        cov = np.array([[0.5,-0.5],[-0.5,0.5]])
        
        with self.assertRaises(SingularMatrixError):
            dmvnorm(testData,mu=mean,sigma=cov)
    
    
class RwishTest(unittest.TestCase):
    
    def test_rwish(self):
        shape = 5.2
        scale = np.array([[2.0,-0.5],[-0.5,3.2]])
        number_of_samples = 1
        
        result = rwish(shape, scale, number_of_samples)
        
        self.assertEquals(len(result.shape), 3)
        self.assertEquals(result.shape[0], number_of_samples)
        self.assertEquals(result.shape[1], scale.shape[0])
        self.assertEquals(result.shape[2], scale.shape[1])
        
    def test_rwish_multiple(self):
        shape = 5.2
        scale = np.array([[2.0,-0.5],[-0.5,3.2]])
        number_of_samples = 10
        
        result = rwish(shape, scale, number_of_samples)
        
        self.assertEquals(len(result.shape), 3)
        self.assertEquals(result.shape[0], number_of_samples)
        self.assertEquals(result.shape[1], scale.shape[0])
        self.assertEquals(result.shape[2], scale.shape[1])
        
        