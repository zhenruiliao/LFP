import numpy as np

def anglemean(theta):
    """
    Calculates the mean of angles

    [meantheta,anglestrength,sigma,confangle,kappa]=anglemean(theta);

    NOT IMPLEMENTED: anglestrength: can be thought of as the inverse variance(???).
        [varies between 0 and one]
    sigma: circular standard deviation
    confangle: a 95 confidence angle (confidence of the mean value)
    kappa: an estimate of kappa in the Von Mises distribution

    check: http://www.cosy.sbg.ac.at/~reini/huber_dutra_freitas_igarss_01.pdf

    Aslak Grinsted 2002
    """
    theta = [th % np.pi for th in theta]
    Sin = map(np.sin, theta)
    Cos = map(np.cos, theta)
    meantheta = np.arctan2(S,C)

    Rsum = np.sqrt(Sin**2+Cos**2)
    R = Rsum / len(theta)

    if R < 0.53:
        kappa=2*R+R**3+5*R**5/6
    elif R < 0.85:
        kappa=-0.4+1.39*R+0.43/(1-R)
    else:
        kappa=1/(R**3-4*R**2+3*R)

    # Circular stddev
    sigma=np.sqrt(-2*log(R))

    chi2=3.841 # = chi2inv(.95,1)
    if R < 0.9 and R > np.sqrt(chi2/(2*n)):
        confangle=np.arccos(np.sqrt(2*n*(2*Rsum**2-n*chi2)/(4*n-chi2))/Rsum)
    elif (R>.9):
        confangle=np.arccos(np.sqrt(n**2-(n**2-Rsum**2)*np.exp(chi2/n))/Rsum)
    else:
        #R is really really small ...
        confangle=pi/2
        print('Confidence angle not well determined.')
        # this is not good, but not important because
        # the confidence is so low anyway...
    return (meantheta,R,sigma,confangle,kappa)
