import numpy as np
import sample_distribution as sd
import matplotlib.pyplot as plt


def func(x):
	return np.exp(-x)

dist1 = sd.arbitrary_distribution(func)

a = [dist1.draw_sample() for i in range(500)]

plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
#plt.show()
