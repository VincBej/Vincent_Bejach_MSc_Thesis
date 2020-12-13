# Vincent Bejach. MSc student, TU Delft.
# 13-04-2019
# The functions in this file are used to define pulses shapes (used for the drive of qubits)

from math import cos, sin, pi, sqrt, exp
import numpy as np

__name__ = 'pulseshapes'

## Window functions
def rectangle_window( t, args ):
    """
    Computes the value of a rectangle modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)

    Returns
    -------
    res: the value of the modulation shape defined by args at time t.
    """
    t_end = args['t_end']
    if t <= t_end:
        res = 1
    else:
        res = 0
    return res


def step_window( t, args ):
    """
    Computes the value of a step modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']

    if t <= t_end/3.:
        res = 1
    elif t > t_end/3. and t <= 2*t_end/3.:
        res = 2
    elif t>2*t_end/3. and t <= t_end:
        res = 1
    else:
        res = 0

    return res


def triangular_window( t, args ):
    """
    Computes the value of a triangular (or Bartlet) modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res = 1 - abs(t)/(t_end/2)
    else:
        res = 0
    return res


def hann_window( t, args ):
    """
    Computes the value of a Hann modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res = 0.5 + 0.5*cos(pi*abs(t) / (t_end/2) )
    else:
        res = 0
    return res


def sine_window( t, args ):
    """
    Computes the value of a sine modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res = sin(pi*abs(t-t_end/2) / (t_end) )
    else:
        res = 0
    return res


def hamming_window( t, args ):
    """
    Computes the value of a Hamming modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res =  0.54 + 0.46*cos( pi*abs(t) / (t_end/2) )
    else:
        res = 0
    return res


def hamming_window_alt( t, args ):
    """
    Computes the value of a Hamming modulation shape at a given time. The difference with hamming_window is that at t=0, the amplitude value is maximum.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res =  0.54 + 0.46*cos( pi*abs(t-t_end/2) / (t_end/2) )
    else:
        res = 0
    return res


def blackman_window( t, args ):
    """
    Computes the value of a Blackman modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if t <= t_end:
        res =  0.42 + 0.5*cos( pi*abs(t) / (t_end/2) ) + 0.08*cos( 2*pi*abs(t) / (t_end/2) )
    else:
        res = 0
    return res


def optimized_blackman_window( t, args ):
    """
    Computes the value of a optimised Blackman modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'c'        : value of the shape's free parameter
    """
    t_end = args['t_end']
    c     = args['c']

    if t <= t_end:
        res = (0.5-2*c) + 0.5*cos(pi*abs(t) / (t_end/2) ) + 2*c*cos(2*pi*abs(t) / (t_end/2) )
    else:
        res = 0
    return res


def papoulis_window( t, args ):
    """
    Computes the value of a Papoulis modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    if (t <= t_end):
        res = 1/pi * abs(sin(pi*abs(t) / (t_end/2))) + (1-abs(t) / (t_end/2))*cos(pi*(t) / (t_end/2) )
    else:
        res = 0
    return res


from scipy.special import i0
def kaiser0_window( t, args ):
    """
    Computes the value of a 0th order Kaiser modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'alpha'    : value of the shape's free parameter
    """
    t_end = args['t_end']
    alpha = args['alpha']
    # alpha = 4
    z = ( 2*(t) / (t_end) ) - 1
    if t<=t_end:
        res = i0( pi * alpha * sqrt( 1 - z**2 ) )
        res = res / i0( pi * alpha )
    else:
        res = 0
    return res

from scipy.special import i1
def kaiser1_window( t, args ):
    """
    Computes the value of a 1st order Kaiser modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'gamma'    : value of the shape's free parameter
    """
    t_end=args['t_end']
    gamma = args['gamma']
    racine = sqrt( 1 - (t/t_end)**2 )
    if t<=t_end:
        res = i1( gamma * racine ) / ( i1(gamma) * racine )
    else:
        res = 0
    return res


def SFT3F_window( t, args ):
    """
    Computes the value of a SFT3F modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.26526
    c1 = -0.5
    c2 = 0.23474
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*(t_cor)) / (t_end) ) + c2 *cos( 2 * (2*pi*(t_cor)) / (t_end) )
    else:
        res = 0
    return res


def SFT4F_window( t, args ):
    """
    Computes the value of a SFT4F modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.21706
    c1 = -0.42103
    c2 = 0.28294
    c3 = -0.07897
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def SFT5F_window( t, args ):
    """
    Computes the value of a SFT5F modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.1881
    c1 = -0.36923
    c2 = 0.28702
    c3 = -0.13077
    c4 = 0.02488
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end ) + c4 *cos( 4 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def SFT3M_window( t, args ):
    """
    Computes the value of a SFT3M modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.28235
    c1 = -0.52105
    c2 = 0.19659
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*(t_cor)) / (t_end) ) + c2 *cos( 2 * (2*pi*(t_cor)) / (t_end) )
    else:
        res = 0
    return res


def SFT4M_window( t, args ):
    """
    Computes the value of a SFT4M modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.241906
    c1 = -0.460841
    c2 = 0.255381
    c3 = -0.041872
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def SFT5M_window( t, args ):
    """
    Computes the value of a SFT5M modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 0.209671
    c1 = -0.407331
    c2 = 0.281225
    c3 = -0.092669
    c4 = 0.0091036
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end ) + c4 *cos( 4 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def HFT90D_window( t, args ):
    """
    Computes the value of a HFT90D modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 1
    c1 = -1.942604
    c2 = 1.340318
    c3 = -0.440811
    c4 = 0.043097
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end ) + c4 *cos( 4 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def HFT116D_window( t, args ):
    """
    Computes the value of a HFT116D modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 1
    c1 = -1.9575375
    c2 = 1.4780705
    c3 = -0.6367431
    c4 = 0.1228389
    c5 = -0.0066288
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end ) + c4 *cos( 4 * (2*pi*t_cor) / t_end ) + c5 *cos( 5 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


def HFT169D_window( t, args ):
    """
    Computes the value of a HFT169D modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    c0 = 1
    c1 = -1.97441842
    c2 = 1.65409888
    c3 = -0.95788186
    c4 = 0.33673420
    c5 = -0.06364621
    c6 = 0.00521942
    c7 = -0.00010599
    if t<=t_end:
        t_cor = t-t_end/2   # Corrected to account for the time starting in the negatives
        res = c0 + c1 *cos( 1 * (2*pi*t_cor) / t_end ) + c2 *cos( 2 * (2*pi*t_cor) / t_end ) + c3 *cos( 3 * (2*pi*t_cor) / t_end ) + c4 *cos( 4 * (2*pi*t_cor) / t_end ) + c5 *cos( 5 * (2*pi*t_cor) / t_end ) + c6 *cos( 6 * (2*pi*t_cor) / t_end ) + c7 *cos( 7 * (2*pi*t_cor) / t_end )
    else:
        res = 0
    return res


from math import log
def gaussian_window( t, args ):
    """
    Computes the value of a Gaussian modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'a'        : value of the shape's free parameter
    """
    from math import sqrt, log

    truncation_threshold = 0.005

    t0    = args['t_end']/2.0   # Only used here to revocer value of sigma
    a     = args['a']
    sigma = sqrt( - a * t0**2 / log(0.005) )

    res = exp( -a * (t/sigma)**2 )

    if res >= truncation_threshold and abs(t) <= 2*t0:   # Covers the whole shape
        return res
    else:
        return 0


def half_gaussian_window( t, args ):
    """
    Computes the value of a half-Gaussian modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'a'        : value of the shape's free parameter
    """
    from math import sqrt, log

    truncation_threshold = 0.005

    t0    = args['t_end']#/2.0
    a     = args['a']
    sigma = sqrt( - a * t0**2 / log(0.005) )

    res = exp( -a * ((t)/sigma)**2 )

    if res >= truncation_threshold and t <= t0:
        return res
    else:
        return 0


def hermite_window( t, args ):
    """
    Computes the value of a Hermite modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'        : end time of the desired shape (in s)
        'rabi_freq'    : Rabi frequency of the pulse (in Hz.rad)
        'hermite_sigma': value of the shape's width parameter
        'hermite_a'    : value of the shape's first free parameter
        'hermite_b'    : value of the shape's second free parameter
    """
    sigma = args['hermite_sigma']

    a         = args['a']
    a_hermite = args['hermite_a']
    b_hermite = args['hermite_b']

    res = ( a_hermite - b_hermite * (t/sigma)**2 ) * exp(- a *(t/sigma)**2 )

    return res


def REBURP_shape( t, args ):
    """
    Computes the value of a REBURP modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']
    a = np.zeros(16)

# For 64 time slices
    a[0] = 0.48
    a[1] = -1.03
    a[2] = 1.09
    a[3] = -1.59
    a[4] = 0.86
    a[5] = -0.44
    a[6] = 0.27
    a[7] = -0.17
    a[8] = 0.10
    a[9] = -0.08
    a[10] = 0.04
    a[11] = -0.04
    a[12] = 0.01
    a[13] = -0.02
    a[14] = 0.0
    a[15] = -0.02
# For 256 time slices
    # a[0] = 0.49
    # a[1] = -1.02
    # a[2] = 1.11
    # a[3] = -1.57
    # a[4] = 0.83
    # a[5] = -0.42
    # a[6] = 0.26
    # a[7] = -0.16
    # a[8] = 0.1
    # a[9] = -0.07
    # a[10] = 0.04
    # a[11] = -0.03
    # a[12] = 0.01
    # a[13] = -0.02
    # a[14] = 0.0
    # a[15] = -0.01

    omega = (2*pi)/t_end
    # omega = 1/t_end

    res = a[0]
    for i in range(1,16):
        res = res + a[i] * cos(i*omega*t)

    # res = res * omega

    # if t<=t_end:
    #     return res
    # else:
    #     return 0
    return res


def UBURP_shape( t, args ):
    """
    Computes the value of a UBURP modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
    """
    t_end = args['t_end']

    a = np.zeros(21)

# For 64 time slices
    a[0] = 0.27
    a[1] = -1.42
    a[2] = -0.33
    a[3] = -1.72
    a[4] = 4.47
    a[5] = -1.33
    a[6] = -0.04
    a[7] = -0.34
    a[8] = 0.5
    a[9] = -0.33
    a[10] = 0.18
    a[11] = -0.21
    a[12] =  0.24
    a[13] = -0.14
    a[14] = 0.07
    a[15] = -0.06
    a[16] = 0.06
    a[17] = -0.04
    a[18] = 0.03
    a[19] = -0.03
    a[20] = 0.02
# For 256 time slices
    # a[0] = 0.27
    # a[1] = -1.42
    # a[2] = -0.37
    # a[3] = -1.84
    # a[4] = 4.4
    # a[5] = -1.19
    # a[6] = 0.0
    # a[7] = -0.37
    # a[8] = 0.5
    # a[9] = -0.31
    # a[10] = 0.18
    # a[11] = -0.21
    # a[12] = 0.23
    # a[13] = -0.12
    # a[14] = 0.07
    # a[15] = -0.06
    # a[16] = 0.06
    # a[17] = -0.04
    # a[18] = 0.03
    # a[19] = -0.02
    # a[20] = 0.02


    omega = (2*pi)/t_end
    # omega = 1/t_end

    res = a[0]
    for i in range(1,21):
        res = res + a[i] * cos(i*omega*t)

    res = res #* 2 #omega

    # if t<=t_end:
    #     return res
    # else:
    #     return 0
    return res


def slepian_window(t, args):
    """
    Computes the value of a Slepian modulation shape at a given time.

    Parameters
    ----------
    t: time point at which to compute the value of the modulation shape
    args: dictionnary containing the arguments used by the shape computations. args must contain the keys:
        't_end'    : end time of the desired shape (in s)
        'rabi_freq': Rabi frequency of the pulse (in Hz.rad)
        'alpha'    : value of the shape's free parameter
        'nb_qubits': number of qubits in the array
    """
    from scipy.signal.windows import dpss

    t_end     = args['t_end']
    alpha     = args['alpha']
    nb_qubits = args['nb_qubits']

    omega0_max = 2*np.pi * (10.0e9 + (nb_qubits-1)*0.1e9) # Valid only when the lowest frequency is 1.0 GHz
    #TODO: Add smarter way to recover the proper frequency to consider here
    nb_time_pts = np.ceil(t_end / ( 1./( omega0_max ) / 100 )).astype(int)

    res_full = dpss(nb_time_pts, alpha)
    res      = res_full[ (nb_time_pts * t/t_end).astype(int) ]

    return res


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ## Data structure
shapes = {
    'rectangle'    : rectangle_window,
    'hann'         : hann_window,
    'hamming'      : hamming_window,
    # # # 'hamming_alt'  : hamming_window_alt,
    'blackman'     : blackman_window,
    'opt_blackman' : optimized_blackman_window,
    'sine'         : sine_window,
    'triangular'   : triangular_window,
    'papoulis'     : papoulis_window,
    # 'kaiser0'      : kaiser0_window,
    # # # # # 'kaiser1'      : kaiser1_window,
    # # 'SFT3F'        : SFT3F_window,
    # # 'SFT4F'        : SFT4F_window,
    # # 'SFT5F'        : SFT5F_window,
    # # 'SFT3M'        : SFT3M_window,
    # # 'SFT4M'        : SFT4M_window,
    # # 'SFT5M'        : SFT5M_window,
    # # 'HFT90D'       : HFT90D_window,
    # # 'HFT116D'      : HFT116D_window,
    # # 'HFT169D'      : HFT169D_window,
    'gaussian'     : gaussian_window,
    'hermite'      : hermite_window,
    'half_gaussian': half_gaussian_window,
    'half_gaussian2': half_gaussian_window,     # Same shape as the half-gaussian 1, but the other way around (amplitude rises instead of decreasing) (thus no need to integrate it, the pulse duration is the same as for the "classical" half-gaussian
    # # # 'REBURP'       : REBURP_shape,
    # # # 'UBURP'        : UBURP_shape,
    # # # 'slepian'        : slepian_window,
}   # Dictionary of pulse shapes.


