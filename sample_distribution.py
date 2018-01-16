#=======================================================================
#
# Author: Oscar Javier Hernandez
#
#=======================================================================
import numpy as np
from scipy.optimize import minimize
import random

class arbitrary_distribution:
	
	def __init__(self,func):
		self.func = func
	
	#--------------------------------------------------------------
	# This function inverts, by finding the value of x which
	# minimizes the value abs(y-func(x))
	#--------------------------------------------------------------
	def invert_function(self,yi):
		
		def residual(x):
			return (yi-self.func(x))**2
		
		# Minimize the residual using Nelder-Mead method
		x0 = yi
		res = minimize(residual, x0, method='nelder-mead')
		res = minimize(residual, res.x, method='nelder-mead')
		xi=res.x
		
		return xi
	
	#--------------------------------------
	# Evaluate the function
	#--------------------------------------
	def eval_func(self,x):
		return self.func(x)
	
	#--------------------------------------
	#
	#--------------------------------------
	def draw_sample(self):
		
		u = random.uniform(0, 1)
		yi = self.invert_function(u)[0]
		
		return yi
		
		
	
	
