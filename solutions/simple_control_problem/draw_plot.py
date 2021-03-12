import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

naf = np.load('./test/naf.npy')
bounded = np.load('./test/bounded.npy')
bounded_r = np.load('./test/bounded_r.npy')
bounded_r_g = np.load('./test/bounded_r_g.npy')
x = np.arange(200)
plt.plot(x, naf)
plt.plot(x, bounded)
plt.plot(x, bounded_r)
plt.plot(x, bounded_r_g)
plt.legend(['NAF', 'BOUNDED NAF', 'BOUNDED R NAF', 'BOUNDED R G NAF'])
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
