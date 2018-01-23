#=======================================================================
#
# Author: Oscar Javier Hernandez
#
#=======================================================================
import numpy as np
from scipy.optimize import minimize
import random
from scipy.integrate import quad

class arbitrary_distribution:
	
	#--------------------------------------------------------------
	# The constructor for the arbitrary distribution sample generator
	# Three arguments:
	# 1. the desired distribution profile
	# 2. the minimum sample value,
	# 3. the maximum sample value
	#--------------------------------------------------------------
	def __init__(self,func,a,b):
		self.func = func
		self.a = a
		self.b = b
		self.Norm = quad(self.func, a, b)[0]
		
	#--------------------------------------------------------------
	# This function defines the cummulative distribution of a function
	# Int[F(y),{y,-Infinity,x}]
	#--------------------------------------------------------------
	def cumulative_distribution(self,x):
		s = quad(self.func, self.a, x)[0]/self.Norm
		return s
	
	#--------------------------------------------------------------
	# This function inverts, by finding the value of x which
	# minimizes the value (y-func(x))**2
	#--------------------------------------------------------------
	def invert_function(self,yi):
		
		def residual(x):
			return (yi-self.cumulative_distribution(x))**2
		
		# Minimize the residual using Nelder-Mead method
		x0 = yi
		res = minimize(residual, x0, method='nelder-mead')
		res = minimize(residual, res.x, method='nelder-mead')
		xi=res.x
		
		return xi
	
	#--------------------------------------
	# Evaluate the Normalized input function
	#--------------------------------------
	def eval_func(self,x):
		return self.func(x)/self.Norm
	
	#-------------------------------------------------------------------
	# Compute the moment of the target PDF distribution
	#-------------------------------------------------------------------
	def eval_PDF_moment(self,k):
		
		def kernel(x):
			return self.eval_func(x)*(x**k)
		
		s = quad(kernel, self.a, self.b)[0]
		
		return s
	
	# Compute the moment of a sample
	def eval_sample_moment(self,v,k):
		'''
		Arguments:
		k = the order of the moment
		v = the vector containing the sample values
		'''
		
		return s
	
	#-------------------------------------------------------------------
	# This function generates a single sample with target distribution.
	# The method employed here is known as the: 
	#-------------------------------------------------------------------
	def draw_sample(self):
		
		u = random.uniform(0, 1)
		yi = self.invert_function(u)[0]
		
		return yi
	
	#-------------------------------------------------------------------
	# Draw a large number of samples
	#-------------------------------------------------------------------
	def draw_samples(self,N):
		
		# Initialize the vector of samples
		v = np.zeros(N)
		
		# Draw N samples from the target distrobution
		v = [self.draw_sample() for i in range(0,N)]
		
		v = np.asarray(v)
		
		return v
		
	
	
