import numpy as np

def ar1(x):
    """
    AR1 - Allen and Smith AR(1) model estimation.
    Syntax: g,a,mu2=ar1(x);

    Input:  x - time series, 1-D np array.

    Output: g - estimate of the lag-one autocorrelation.
         a - estimate of the noise variance.[unbiased_var(x)~=a^2/(1-g^2)]
         mu2 - estimated square on the mean.

    AR1 uses the algorithm described by Allen and Smith 1995, except
    that Matlab's 'fzero' is used rather than Newton-Raphson.

    Fzero in general can be rather picky - although
    I haven't had any problem with its implementation
    here, I recommend occasionally checking the output
    against the simple estimators in AR1NV.

    Alternative AR(1) estimatators: ar1cov, ar1nv, arburg, aryule

    Original version written by Eric Breitenberger.

    Updated,optimized&stabilized by Aslak Grinsted 2003-2007
    """
    N = len(x)
    m = x.mean()
    x_ctr = x - m

    # Lag zero and one covariance estimates
    c0 = x*x / N
    c1 = x[0:-2]*x[1:-1] / (N-1)

    A = c0 * N**2
    B = -c1*N-c0*N**2-2*c0+2*c1-c1*N**2+c0*N
    C = N*(c0+c1*N-c1)
    D = B**2 - 4*A*C

    if D > 0:
        g = (-B-np.sqrt(D))/(2*A)
    else:
        print('REDNOISE:unboundAr1','Can not place an upperbound on the' +
        ' unbiased AR1.\n\t\t -Series too short or too large trend.')
        g = float('NaN')

    mu2 = -1/N+(2/N**2)*((N-g^N)/(1-g)-g*(1-g**(N-1))/(1-g)) # allen&smith96(footnote4)
    c0t = c0/(1-mu2)
    a = np.sqrt((1-g**2)*c0t)
    return g,a,mu2
