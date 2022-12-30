# Disclaimer
This is an early stage of the code. It does what I need it to do for now. If ever someone is interested, I can clean up the code and provide better documentation and better functionnality. 

# robust_efficient_iv
This package will:
- compute the Black-Scholes Implied Volatility for calls and puts
- using method='L-BFGS-B'
- The objective function is mostly well behaved, but has flat sections at machine precision. A good starting value is necessary, or multiple starting values.

## Starting Values
For now, I need this for calculating my IV from bid and ask quotes. I can use the broker's calculated IV as a starting value. 
### Future updates on starting values
I intend to have a preliminary step with several starting values, and then an optimization from the best point, using the other points has bounds for the optimization.

## implied_volatility(...)
uses optimize.minimize to find the Implied Volatility.

## objfunc_square(...)
Simply use the square error on pricing as objective function

## black_scholesv(...)
Uses numba jit to compute option prices

## ndtr_numba(...)
Uses numba jit to quickly compute gaussian cumulative density 
(This is from: https://github.com/cuemacro/teaching/blob/master/pythoncourse/notebooks/numba_example.ipynb)

