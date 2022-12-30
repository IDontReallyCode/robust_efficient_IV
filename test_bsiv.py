import reiv
import numpy as np
# import 

S = 40.0
K = np.linspace(20, 100, 10, dtype=float)
T = np.linspace(7/365, 700/365, 10, dtype=float)
# T = np.array([1])
v = np.linspace(0.15, 6.5, 10, dtype=float)
r = 0.045

Km, Tm, Vm = np.meshgrid(K,T,v)
K = np.reshape(Km, Km.shape[0]*Km.shape[1]*Km.shape[2])
T = np.reshape(Tm, Tm.shape[0]*Tm.shape[1]*Km.shape[2])
v = np.reshape(Vm, Vm.shape[0]*Vm.shape[1]*Km.shape[2])

c = reiv.black_scholesv(v, S, K, r, T, 1)
p = reiv.black_scholesv(v, S, K, r, T, 0)

# print(c)
# print(p)

otmcp = c[S>=K]
otmcK = K[S>=K]
otmcT = T[S>=K]
otmcv = v[S>=K]
otmpp = p[S<=K]
otmpK = K[S<=K]
otmpT = T[S<=K]
otmpv = v[S<=K]

ivc = [reiv.implied_volatility(otmcp[x], S, otmcK[x], r, otmcT[x], 1, guessiv=otmcv[x]) for x in range(len(otmcp))]
ivp = [reiv.implied_volatility(otmpp[x], S, otmpK[x], r, otmpT[x], 0, guessiv=otmpv[x]) for x in range(len(otmpp))]

# ivc = [bsiv.find_vol(otmcp[x], S, otmcK[x], r, otmcT[x], 1) for x in range(len(otmcp))]
# ivp = [bsiv.find_vol(otmpp[x], S, otmpK[x], r, otmpT[x], 0) for x in range(len(otmpp))]

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print((ivc-otmcv))
print((ivp-otmpv))
