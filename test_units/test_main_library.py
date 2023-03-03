import numpy as np
import unittest
import mock
import os
import sys
import platform
from matplotlib import pyplot as plt
from copy import deepcopy as COPY
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
# pdb.set_trace()

class test_main_library(unittest.TestCase):
    np.random.seed(0)

    def test_mean_shift_process(self):
        # notice that X is a length 2 cube centered at the origin
        # the high degree of symmetry creates
        # many easily predictable mean shift process results
        X = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        X += [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
        X = np.array(X, dtype = float)

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
        X = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        X += [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]
        X = np.array(X, dtype = float)
        print(main_lib.MOD(X, 3, 1))
        print(main_lib.MOD(X, 6, 1))
        print(main_lib.MOD(X, 7, 1))
        print(main_lib.MOD(X, 5, 1))
       
if __name__ == '__main__':
    unittest.main()
