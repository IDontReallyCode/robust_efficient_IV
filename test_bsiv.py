import reiv
import numpy as np
import time
from scipy.stats import norm

# Generate an array of options with different moneyness, maturity, and IV.
S = 40.0
K = np.linspace(20, 100, 20, dtype=float)
T = np.linspace(7/365, 700/365, 20, dtype=float)
# T = np.array([1])
v = np.linspace(0.15, 6.5, 20, dtype=float)
r = 0.045

Km, Tm, Vm = np.meshgrid(K,T,v)
K = np.reshape(Km, Km.shape[0]*Km.shape[1]*Km.shape[2])
T = np.reshape(Tm, Tm.shape[0]*Tm.shape[1]*Km.shape[2])
v = np.reshape(Vm, Vm.shape[0]*Vm.shape[1]*Km.shape[2])

# Calculate the option prices for those
c = reiv.black_scholesv(v, S, K, r, T, 1)
p = reiv.black_scholesv(v, S, K, r, T, 0)

# print(c)
# print(p)

# Get OTM options only.
otmcp = c[S>=K]
otmcK = K[S>=K]
otmcT = T[S>=K]
otmcv = v[S>=K]
otmpp = p[S<=K]
otmpK = K[S<=K]
otmpT = T[S<=K]
otmpv = v[S<=K]

# Solve for the IV
# ivc = [reiv.implied_volatility(otmcp[x], S, otmcK[x], r, otmcT[x], 1, guessiv=otmcv[x]*1.1) for x in range(len(otmcp))]
# ivp = [reiv.implied_volatility(otmpp[x], S, otmpK[x], r, otmpT[x], 0, guessiv=otmpv[x]*1.1) for x in range(len(otmpp))]
# pdf1 = norm.pdf(0)
# pdf2 = reiv.npdf_numba(0)

# x = np.linspace(-5, 5, 1000000)
# start = time.time()
# pdf1 = norm.pdf(x)
# print(time.time()-start)
# start = time.time()
# pdf2 = reiv.npdf_numba(x)
# print(time.time()-start)

# print(pdf1-pdf2)

start = time.time()
ivc = reiv.getallimpliedvolatilities(otmcp, otmcK, otmcT, otmcv, S, r, 1, reiv.LBFGSB)
ivp = reiv.getallimpliedvolatilities(otmpp, otmpK, otmpT, otmpv, S, r, 0, reiv.LBFGSB)
print(time.time()-start)


start = time.time()
ivc = reiv.getallimpliedvolatilities(otmcp, otmcK, otmcT, otmcv, S, r, 1, reiv.NEWTONCG)
ivp = reiv.getallimpliedvolatilities(otmpp, otmpK, otmpT, otmpv, S, r, 0, reiv.NEWTONCG)
print(time.time()-start)


# ivc = [bsiv.find_vol(otmcp[x], S, otmcK[x], r, otmcT[x], 1) for x in range(len(otmcp))]
# ivp = [bsiv.find_vol(otmpp[x], S, otmpK[x], r, otmpT[x], 0) for x in range(len(otmpp))]

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# print((ivc-otmcv))
# # print((iv2-otmcv))
# print((ivp-otmpv))
