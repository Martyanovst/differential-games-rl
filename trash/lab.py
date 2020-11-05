import math

from numpy import double

count = 0
res = 0
n = 2020
for k1 in range(30):
    for k2 in range(30):
        for k3 in range(30):
            dx = 2 * k1 + 3 * k2 + 5 * k3
            if dx == 24:
                count += 1
                k0 = n - k1 - k2 - k3
                res += (math.factorial(n) /
                        math.factorial(k0)) / (math.factorial(k1) * math.factorial(k2) * math.factorial(k3))
print(count)
print(int(res))
