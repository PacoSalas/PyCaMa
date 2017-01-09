# Tests for PyCaMa. Python for cash management

from PyCaMa import *
import numpy as np
import unittest

# Input data for testing
# Planning horizon
h = 5

# Transactions
trans = [1,2,3,4,5,6]

# Bank accounts
banks = [1,2,3]

# Trans fixed costs
g0 = {1:50, 2:50, 3:100, 4:50, 5:100, 6:50}

# Trans variable costs
g1 = {1:0.0, 2:0.0, 3:0.0001, 4:0.00001, 5:0.0001, 6:0.00001}

# Initial balance
b0 = [0, 0, 10000000]

# Minimum balances
bmin = [0, 0, 0]

# Holding costs per bank account
v = {1:0.0001, 2:0.0001, 3:0}

# Allowed transactions between accounts
A = np.array([[1, -1, 0, 0, 1, -1],[-1,  1, 1, -1,  0,  0],[ 0,  0, -1,  1, -1, 1]]).T

# Creates an instance of the problem

test_problem = multibank(banks, trans, A, g0, g1, v, bmin)

# Random forecast
size = h*len(banks)
fcast = np.random.randint(low=-1000,high=1000,size=size).reshape((h,len(banks)))

test_problem.h = fcast.shape[0]


class TestPyCaMa(unittest.TestCase):

    def test_int(self):
        self.assertEqual(type(1), type(test_problem.h))

    def test_dict(self):
        self.assertEqual(type({1:2}), type(test_problem.gzero))
        self.assertEqual(type({1:2}), type(test_problem.gone))
        self.assertEqual(type({1:2}), type(test_problem.v))

    def test_list(self):
        self.assertEqual(type([1]), type(test_problem.trans))
        self.assertEqual(type([1]), type(test_problem.banks))
        self.assertEqual(type([1]), type(test_problem.bmin))

    def test_matrix(self):
        self.assertEqual(type(np.array([1])), type(fcast))
        self.assertEqual(type(np.array([1])), type(A))

    def test_dimensions(self):
        self.assertEqual(A.shape, (len(test_problem.trans),len(test_problem.banks)))
        self.assertEqual(fcast.shape, (test_problem.h,len(test_problem.banks)))
		self.assertEqual(len(b0), len(test_problem.banks))