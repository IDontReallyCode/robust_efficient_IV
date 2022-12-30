import bsiv
import numpy as np
import matplotlib.pyplot as plt

S = 40.0
K = np.linspace(30, 60, 2, dtype=float)
# T = np.linspace(7/365, 700/365, 4, dtype=float)
T = np.array([1])
v = 3.35
r = 0.045

Km, Tm = np.meshgrid(K,T)
K = np.reshape(Km, Km.shape[0]*Km.shape[1])
T = np.reshape(Tm, Tm.shape[0]*Tm.shape[1])

c = bsiv.black_scholesv(v, S, K, r, T, 1)
p = bsiv.black_scholesv(v, S, K, r, T, 0)

print(c)
print(p)

otmcp = c[S>=K]
otmcK = K[S>=K]
otmcT = T[S>=K]
otmpp = p[S<=K]
otmpK = K[S<=K]
otmpT = T[S<=K]

# ivc = [bsiv.implied_volatility(otmcp[x], S, otmcK[x], r, otmcT[x], 1) for x in range(len(otmcp))]
# ivp = [bsiv.implied_volatility(otmpp[x], S, otmpK[x], r, otmpT[x], 0) for x in range(len(otmpp))]

# ivc = [bsiv.find_vol(otmcp[x], S, otmcK[x], r, otmcT[x], 1) for x in range(len(otmcp))]
# ivp = [bsiv.find_vol(otmpp[x], S, otmpK[x], r, otmpT[x], 0) for x in range(len(otmpp))]
minv = 0.1
maxv = 5
nvs = 100
vvec = np.linspace(minv, maxv, nvs)

objective = bsiv.grid_search(otmcp[0], S, otmcK[0], r, otmcT[0], 1, bsiv.objfunc_square, vvec)
plt.plot(vvec, objective)
plt.show()

ivc = [bsiv.find_vol(otmcp[x], S, otmcK[x], r, otmcT[x], 1) for x in range(len(otmcp))]

pause=1


