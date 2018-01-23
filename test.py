import numpy as np
import sample_distribution as sd
import matplotlib.pyplot as plt


def func(x):
	return x*np.exp(-50*(x-0.2)**2)+np.exp(-60*(x-0.6)**2)
	#return (x-0.4)**2
	
a0 = 0.0
b0 = 1.0
dist1 = sd.arbitrary_distribution(func,a0,b0)

x = np.arange(a0,b0,0.01)
a = dist1.draw_samples(80000)

for k in range(1,50):
	print 'm('+str(k)+')', np.mean(a**k), dist1.eval_moment(k)

exit()
plt.hist(a, bins='auto',normed=True,alpha=0.6)  # arguments are passed to np.histogram
plt.plot(x,dist1.eval_func(x),'-', linewidth=2)
plt.title("Histogram with 'auto' bins")
plt.show()
