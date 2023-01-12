import numba
import numpy as np
# from corncdf import normCdf, ndtr_numba
from scipy import optimize
from scipy.stats import norm
from numba import njit
from math import fabs, erf, erfc #, exp

NEWTON = 0
LBFGSB = 1
NEWTONCG = 2

NPY_SQRT1_2 = 1.0/ np.sqrt(2)
NPY_SQRT1_2pi = 1.0/ np.sqrt(2*np.pi)
     
@njit(cache=True, fastmath=True)
def ndtr_numba(a):
    # (This is from: https://github.com/cuemacro/teaching/blob/master/pythoncourse/notebooks/numba_example.ipynb)

    if (np.isnan(a)):
        return np.nan

    x = a * NPY_SQRT1_2
    z = fabs(x)

    if (z < NPY_SQRT1_2):
        y = 0.5 + 0.5 * erf(x)

    else:
        y = 0.5 * erfc(z)

        if (x > 0):
            y = 1.0 - y

    return y



@njit(cache=True, fastmath=True)
def npdf_numba(a):
    # 
    return NPY_SQRT1_2pi*np.exp(-0.5*(a)**2)


@numba.jit
def baw_implied_volatility(option_price, S, K, r, t, option_type):
    # Calculate the intrinsic value of the option
    intrinsic_value = np.maximum(0, option_type * (S - K))
    
    # Calculate the forward price of the underlying asset
    F = S * np.exp(r * t)
    
    # Calculate the moneyness of the option
    m = F / K
    
    # Calculate the BAW approximation for the Black-Scholes implied volatility
    if option_type == 1:  # Call option
        sigma = np.where(m <= 0.92,
                         (1.13 - (2.24 * m) + (3.17 * m ** 2)) * np.sqrt(t),
                         (0.63 - (0.91 * m) + (1.01 * m ** 2)) * np.sqrt(t))
    else:  # Put option
        sigma = np.where(m <= 0.83,
                         (1.32 - (3.22 * m) + (3.54 * m ** 2)) * np.sqrt(t),
                         (1.45 - (4.17 * m) + (4.97 * m ** 2)) * np.sqrt(t))
    
    return sigma

# Vectorize the baw_implied_volatility function
v_baw_implied_volatility = np.vectorize(baw_implied_volatility)


@njit
def black_scholes(v:float, S:float, K:float, r:float, T:float, optiontype) -> float:
    # This function returns the price of a put

    # requires import numpy as np

    # S is the current stock price
    # T is the maturity in years
    # r is the risk-free rate
    # d is the continuous time dividend rate
    # v is the implied volatility
    # https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
    SoverK = np.divide(S,K)
    B = np.multiply(v,v)
    # B2 = np.divide(B,2)
    B2 = B*0.5
    B3 = np.multiply(r + B2,T)
    denum = np.multiply(v,np.sqrt(T))
    Num = np.add(np.log(SoverK),B3)
    d1 = np.divide(Num,denum)
    d2 = d1-denum
    
    if optiontype==0:     #put
        # if len(d2.shape)==0:
        nminusd2 = ndtr_numba(-d2)
        nminusd1 = ndtr_numba(-d1)
        # else:
        #     nminusd2 = np.array([normCdf(-thatd2) for thatd2 in d2])
        #     nminusd1 = np.array([normCdf(-thatd1) for thatd1 in d1])
    
        return K * np.exp(-r * T) * nminusd2 - S * nminusd1
    else:           #call
        # if len(d2.shape)==0:
        nd2 = ndtr_numba(d2)
        nd1 = ndtr_numba(d1)
        # else:
        #     nd2 = np.array([normCdf(thatd2) for thatd2 in d2])
        #     nd1 = np.array([normCdf(thatd1) for thatd1 in d1])
    
        return S * nd1 - K * np.exp(-r * T) * nd2


@njit
def black_scholesv(v:np.ndarray, S:np.ndarray, K:np.ndarray, r:np.ndarray, T:np.ndarray, optiontype) -> np.ndarray:
    # This function returns the price of a put

    # requires import numpy as np

    # S is the current stock price
    # T is the maturity in years
    # r is the risk-free rate
    # d is the continuous time dividend rate
    # v is the implied volatility
    # https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
    SoverK = np.divide(S,K)
    B = np.multiply(v,v)
    # B2 = np.divide(B,2)
    B2 = B*0.5
    B3 = np.multiply(r + B2,T)
    denum = np.multiply(v,np.sqrt(T))
    Num = np.add(np.log(SoverK),B3)
    d1 = np.divide(Num,denum)
    d2 = d1-denum
    
    if optiontype==0:     #put
        # if len(d2.shape)==0:
        # nminusd2 = normCdf(-d2)
        # nminusd1 = normCdf(-d1)
        # else:
        # nminusd2 = np.array([normCdf(-thatd2) for thatd2 in d2])
        # nminusd1 = np.array([normCdf(-thatd1) for thatd1 in d1])
        nminusd2 = np.array([ndtr_numba(-thatd2) for thatd2 in d2])
        nminusd1 = np.array([ndtr_numba(-thatd1) for thatd1 in d1])
        # nminusd2 = np.array([norm.cdf(-thatd2) for thatd2 in d2])
        # nminusd1 = np.array([norm.cdf(-thatd1) for thatd1 in d1])
    
        return K * np.exp(-r * T) * nminusd2 - S * nminusd1
    else:           #call
        # if len(d2.shape)==0:
        # nd2 = normCdf(d2)
        # nd1 = normCdf(d1)
        # else:
        # nd2 = np.array([normCdf(thatd2) for thatd2 in d2])
        # nd1 = np.array([normCdf(thatd1) for thatd1 in d1])
        nd2 = np.array([ndtr_numba(thatd2) for thatd2 in d2])
        nd1 = np.array([ndtr_numba(thatd1) for thatd1 in d1])
        # nd2 = np.array([norm.cdf(thatd2) for thatd2 in d2])
        # nd1 = np.array([norm.cdf(thatd1) for thatd1 in d1])
    
        return S * nd1 - K * np.exp(-r * T) * nd2

@njit
def vegav(v:np.ndarray, S:np.ndarray, K:np.ndarray, r:np.ndarray, T:np.ndarray) -> np.ndarray:
    SoverK = np.divide(S,K)
    B = np.multiply(v,v)
    # B2 = np.divide(B,2)
    B2 = B*0.5
    B3 = np.multiply(r + B2,T)
    denum = np.multiply(v,np.sqrt(T))
    Num = np.add(np.log(SoverK),B3)
    d1 = np.divide(Num,denum)
    # TODO get numbad norm PDF
    return S * npdf_numba(d1) * np.sqrt(T)


def jac_square(sigma, option_price, S, K, r, t, option_type, chain):
    return 

def objfunc_square(sigma, option_price, S, K, r, t, option_type):
    return (option_price - black_scholesv(sigma, S, K, r, t, option_type))**2

# @njit
def objfuncjac_square(sigma, option_price, S, K, r, T, option_type):
    price = black_scholes(float(sigma), S, K, r, T, option_type)
    vega = vegav(sigma, S, K, r, T)
    objective = (option_price - black_scholesv(sigma, S, K, r, T, option_type))**2
    jac = -2*(option_price-price)*vega
    return objective, jac


def objfunc_abs(sigma, option_price, S, K, r, t, option_type):
    return abs(option_price - black_scholesv(sigma, S, K, r, t, option_type))


def objfunc_sqrt_abs(sigma, option_price, S, K, r, t, option_type):
    return np.sqrt(abs(option_price - black_scholesv(sigma, S, K, r, t, option_type)))


def implied_volatility(option_price, S, K, r, t, option_type, guessiv=0.75, ivbounds=[(0.05, 20)], method=LBFGSB):#, optimizer, objective_function):
    # if optimizer==NEWTON
    # Use root_scalar to solve for sigma that minimizes min_func
    # bounds = optimize.Bounds(0.05, 20)
    # result = optimize.minimize(objfuncjac_square, 5, args=(option_price, S, K, r, t, option_type), method='L-BFGS-B', bounds=[(0.05, 20)], tol=1.0E-22, jac=True,
    #                            options={'ftol':1.0E-22, 'gtol':1.0E-22, 'iprint':-1})
    # result = optimize.minimize(objfunc_square, guessiv, args=(option_price, S, K, r, t, option_type), method='L-BFGS-B', bounds=ivbounds, tol=1.0E-22, 
    #                            options={'ftol':1.0E-22, 'gtol':1.0E-22, 'iprint':-1})
    if method==LBFGSB:
        result = optimize.minimize(objfuncjac_square, guessiv, args=(option_price, S, K, r, t, option_type), method='L-BFGS-B', bounds=ivbounds, tol=1.0E-22, 
                                options={'ftol':1.0E-22, 'gtol':1.0E-22, 'iprint':-1}, jac=True)
    elif method==NEWTONCG:
        result = optimize.minimize(objfuncjac_square, guessiv, args=(option_price, S, K, r, t, option_type), method='Newton-CG', jac=True)
    else:
        raise('wrong method for optimization')
    # result = optimize.minimize(objfuncjac_square, guessiv, args=(option_price, S, K, r, t, option_type), method='Newton-CG', jac=True, options={'xtol':1.0E-12, 'disp':-1})
    # result = optimize.minimize(min_func, 1, args=(option_price, S, K, r, t, option_type), method='Nelder-Mead', bounds=[(0.01, 20)], tol=1.0E-22, jac=vegav, 
    #                            options={'ftol':1.0E-22, 'gtol':1.0E-22, 'iprint':7867634})
    
    # TODO Need a way to detect failure
    
    return result.x[0]

# Vectorize the baw_implied_volatility function
# v_implied_volatility = np.vectorize(implied_volatility)

def getallimpliedvolatilities(prices:np.ndarray, strikes:np.ndarray, maturities:np.ndarray, guessivs:np.ndarray, spot:float, riskfree:float, optiontype, method):
    ivs = [implied_volatility(prices[x], spot, strikes[x], riskfree, maturities[x], optiontype, guessiv=guessivs[x], method=method) for x in range(len(prices))] 
    return ivs


def find_vol(target_value, S, K, r, T, option_type):
    MAX_ITERATIONS = 4000
    PRECISION = 1.0e-7
    nevernegativesigma = .10
    for i in range(0, MAX_ITERATIONS):
        price = black_scholes(nevernegativesigma, S, K, r, T, option_type)
        vega = vegav(nevernegativesigma, S, K, r, T)
        objective = objfunc_square(nevernegativesigma, target_value, S, np.array([K]), r, T, option_type)
        jac = -2*(target_value-price)*vega
        if (abs(objective) < PRECISION):
            return nevernegativesigma
        nevernegativesigma = max(0.05, nevernegativesigma + objective/jac) # f(x) / f'(x)
    return nevernegativesigma # value wasn't found, return best guess so far


def grid_search(option_price, S, K, r, T, option_type, objective_function, vvec):
    
    objectives = objective_function(vvec**2, option_price, S, K, r, T, option_type)
    
    return objectives
    