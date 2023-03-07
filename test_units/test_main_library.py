import numpy as np
import unittest
import mock
import os
import sys
import platform
from matplotlib import pyplot as plt
from copy import deepcopy as COPY
from scipy.stats import norm
from tqdm import tqdm
import pdb 

#path finding code start
next_dir, next_folder = os.path.split(os.getcwd())
main_folder = "tables_9_10_11_reproduction"
count = 1
paths = [os.getcwd(), next_dir]
folders = ["", next_folder]
while next_folder != main_folder and count < 4:
    next_dir, next_folder = os.path.split(next_dir)
    paths.append(next_dir)
    folders.append(next_folder)
    count += 1
if count >= 4:
    message = "error: important paths have been renamed or reorganized. "
    message += "If this was intentional, then change the path "
    message += "finding code in test_main_library.py"
    print(message)
os.chdir(paths[count - 1])
sys.path.insert(1, os.getcwd())
#path finding code end

import recreate_tables_9_10_11_library as main_lib

class test_main_library(unittest.TestCase):
    np.random.seed(0)

    def test_asd_cutoff(self):
        tests = []
        norm_vals = norm.ppf(np.linspace(1E-6, 1 - 1E-6, 999999))
        # ensures the standard deviation is almost exactly 1
        norm_vals = (norm_vals - np.mean(norm_vals))/np.std(norm_vals)
        for a in [1, 2, 3, 4]:
            inds = main_lib.asd_cutoff(norm_vals, a)
            inliers = norm_vals[inds]
            outliers = norm_vals[inds == False]
            tests.append(np.min(inliers) >= -a)
            tests.append(np.max(inliers) <= a)
            tests.append(np.min(np.abs(outliers)) > a)
        is_correct = np.all(np.array(tests) == True)
        message = "mean_shift_process may have a math error, "
        message = "or (more likely) there is a rounding error in test_asd_cutoff"
        self.assertTrue(is_correct, message)

    def test_mad_cutoff(self):
        tests = []
        norm_vals = norm.ppf(np.linspace(1E-6, 1 - 1E-6, 999999))
        # ensures the median is almost exactly 0
        # thereby ensureing the implicit standard deviation estimate is almost exactly 1
        norm_vals = norm_vals - np.median(norm_vals)
        med = np.median(norm_vals)
        MAD = (1/norm.ppf(3/4))*np.median(np.abs(med - norm_vals))
        for a in [1, 2, 3, 4]:
            inds = main_lib.mad_cutoff(norm_vals, a)
            inliers = norm_vals[inds]
            outliers = norm_vals[inds == False]
            tests.append(np.min(inliers) >= -a)
            tests.append(np.max(inliers) <= a)
            tests.append(np.min(np.abs(outliers)) > a)
        is_correct = np.all(np.array(tests) == True)
        message = "mean_shift_process may have a math error, "
        message = "or (more likely) there is a rounding error in test_mad_cutoff"
        self.assertTrue(is_correct, message)

    def test_iqr_cutoff(self):
        tests = []
        norm_vals = norm.ppf(np.linspace(1E-6, 1 - 1E-6, 999999))
        
        # these are the theoretical IQR test lower and upper outlier bounds.
        Q1, Q3 = norm.ppf([0.25, 0.75])
        IQR = (Q3 - Q1)
        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        
        tests = []
        inds = main_lib.iqr_cutoff(norm_vals)
        inliers = norm_vals[inds]
        outliers = norm_vals[inds == False]
        tests.append(np.min(inliers) >= lb)
        tests.append(np.max(inliers) <= ub)

        is_correct = np.all(np.array(tests) == True)
        message = "mean_shift_process may have a math error, "
        message = "or (more likely) there is a rounding error in test_iqr_cutoff"
        self.assertTrue(is_correct, message)

    def test_T2_thresh_data(self):
        tests = []
        norm_vals = norm.ppf(np.linspace(1E-10, 1 - 1E-10, 10000000))
        def casd(x): return(main_lib.asd_cutoff(x, 3))
        
        theoretical_std = np.array([0.98658, 0.98505, 0.98487, 0.98485])
        for T in range(4):
            asd_inds = main_lib.T2_thresh_data(norm_vals, T + 1, casd)
            std_actual = np.round(np.std(norm_vals[asd_inds == False]), 5)
            tests.append(std_actual == np.round(theoretical_std[T], 5))

        message = "The thresh data function or the asd cutoff function "
        message += "may have an error. Alternatively, there may be a "
        message += "rounding error in the test_T2_thresh_data function." 
        self.assertTrue(np.all(tests), message)

    def test_mean_shift_process(self):
        # notice that X is a length 2 cube centered at the origin
        # the high degree of symmetry creates
        # many easily predictable mean shift process results
        s = 2
        X = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        X += [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
        X = s*np.array(X, dtype = float)

        output3 = np.array([main_lib.mean_shift_process(X, 3, i) for i in range(1,11)])
        predicted_output3 = (np.array(10*[1/3])**range(1,11))
        predicted_output3 = np.array([X*i for i in predicted_output3])
        test1_passes = np.all(np.round(output3, 8) == np.round(predicted_output3, 8))

        output6 = np.array([main_lib.mean_shift_process(X, 6, i) for i in range(1,11)])
        predicted_output6 = np.zeros((10, len(X), len(X[0])))
        test2_passes = np.all(np.round(output6, 8) == np.round(predicted_output6, 8))

        output7 = np.array([main_lib.mean_shift_process(X, 7, i) for i in range(1,11)])
        predicted_output7 = (np.array(10*[-1/7])**range(1,11))
        predicted_output7 = np.array([X*i for i in predicted_output7])
        test3_passes = np.all(np.round(output7, 8) == np.round(predicted_output7, 8))

        is_correct = np.all([test1_passes, test2_passes, test3_passes])
        self.assertTrue(is_correct, "mean_shift_process may have a math error")
        
    f_name1 = "recreate_tables_9_10_11_library.mean_shift_process"
    def mean_shift_process_mock(X, k, l):
        if k == 3:
            return(((1/3)**l)*X)
        elif k == 6:
            return(np.zeros(X.shape))
        elif k == 7:
            return(((-1/7)**l)*X)
        else:
            return(X)
    @mock.patch(f_name1, side_effect = mean_shift_process_mock)
    def test_MOD(self, mock1):
        s = 2
        X = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        X += [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
        X = s*np.array(X, dtype = float)
        
        output3 = np.array([main_lib.MOD(X, 3, i) for i in range(1, 11)])
        def solution(s, l, X):
            output = np.sqrt(3)*(1 - (1/(3**l)))*s
            return(np.repeat(output, len(X)))
        predicted_output3 = np.array([solution(s, i, X) for i in range(1, 11)])
        test1_passes = np.all(np.round(output3, 8) == np.round(predicted_output3, 8))

        output6 = np.array([main_lib.MOD(X, 6, i) for i in range(1, 11)])
        def solution(s, l, X):
            output = np.sqrt(3)*s
            return(np.repeat(output, len(X)))
        predicted_output6 = np.array([solution(s, i, X) for i in range(1, 11)])
        test2_passes = np.all(np.round(output6, 8) == np.round(predicted_output6, 8))

        output7 = np.array([main_lib.MOD(X, 7, i) for i in range(1, 11)])
        def solution(s, l, X):
            output = np.sqrt(3)*(1 - (1/((-7)**l)))*s
            return(np.repeat(output, len(X)))
        predicted_output7 = np.array([solution(s, i, X) for i in range(1, 11)])
        test3_passes = np.all(np.round(output7, 8) == np.round(predicted_output7, 8))

        is_correct = np.all([test1_passes, test2_passes, test3_passes])
        self.assertTrue(is_correct, "MOD function may have a math error")
       
if __name__ == '__main__':
    unittest.main()
