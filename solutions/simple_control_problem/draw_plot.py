import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

naf = np.load('./test/naf.npy')
bounded = np.load('./test/bounded.npy')
x = np.arange(200)
plt.plot(x, naf)
plt.plot(x, bounded)
plt.legend(['NAF', 'BOUNDED NAF'])
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
