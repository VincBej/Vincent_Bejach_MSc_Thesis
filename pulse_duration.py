import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from math import cos, sin, pi, sqrt, exp
import pickle


## Integral functions definitions

def integr_hamming(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 0.54 + 0.46 * cos( pi*(t)/ (x/2) ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_hamming_alt(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 0.54 + 0.46 * cos( pi*(t-x/2)/ (x/2) ), -x/2, x/2 )

    return rotation_angle/rabi_frequency - integral


def integr_triangle(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 1-abs(t)/(x/2), -x/2, x/2 )

    return rotation_angle/rabi_frequency - integral


def integr_hann(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 0.5 + 0.5*cos( pi*(t-x/2)/ (x/2) ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_blackman(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 0.42 + 0.5*cos( pi*abs(t) / (x/2) ) + 0.08*cos( 2*pi*abs(t) / (x/2) ), -x/2, x/2 )

    return rotation_angle/rabi_frequency - integral


def integr_opt_blackman(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : (0.5-2*c) + 0.5*cos(pi*abs(t-x/2) / (x/2) ) + 2*c*cos(2*pi*abs(t-x/2) / (x/2) ), -x/2, x/2)

    return rotation_angle/rabi_frequency - integral


def integr_papoulis(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 1/pi * abs(sin(pi*abs(t) / (x/2) )) + (1-abs(t)/(x/2))*cos(pi*(t) / (x/2)), -x/2, x/2 )

    return rotation_angle/rabi_frequency - integral


def integr_rectangle(x, *arg):
    """
    Computes the residual between the integral of the modulation shape and the target rotation angle.
    Function used in the computation of the pulse duration.

    Parameters
    ----------
    x  : integration variable (duration of the pulse)
    arg: tuple containing the parameters of the shapes. For more generality, values for the possible free parameters of the functions are also expected. In case the desired shape does not have free parameters, any dummy value can be provided.
     The tuple is organised as follows: c (optimsed blackman shape parameter), alpha (0th order Kaiser shape parameter), gamma (1st order Kaiser shape parameter), a (gaussian and gaussian-like shape parameter), rabi_frequency, rotation_angle.

     Returns
     -------
     residual: residual between rotation_angle/rabi_frequency and integral of the shape over the duration x
    """
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 1, 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_step(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : 1 if t<=x/3 else (2 if t>x/3 and t<=2*x/3 else (1 if t>2*x/3 and t<=x else 0) ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_gaussian(x, *arg):   # Computes half of the required pulse length
    from math import sqrt, log
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : (exp( -a * (t/x)**2 )) if exp(-a*(t/x)**2) >= 0.005 else (0) , 0, sqrt( - (x**2)/a * log(0.005) ) )

    return 2* ((rotation_angle)/(2*rabi_frequency) - integral)     # Factor 2 and rotation_angle/2 instead of rotation_angle because this is only one half of the shape


def integr_half_gaussian(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : (exp( -a * (t/x)**2 )) if (exp(-a*(t/x)**2) >= 0.005 and t>=0) else (0) , 0, 2*sqrt( - (x**2)/a * log(0.005) ) )

    return rotation_angle/rabi_frequency - integral


def integr_hermite(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg
    if abs(rotation_angle - np.pi) < abs(rotation_angle - np.pi/2):
        # The target rotation angle is closer to pi
        a_hermite = 1
        b_hermite = 0.956
    else:
        # The target rotation angle is closer to pi/2
        a_hermite = 1
        b_hermite = 0.667

    t0        = 50e-9

    integral, err = quad( lambda t : ( a_hermite - b_hermite * (t/x)**2 ) * exp(- a *(t/x)**2 ) , -t0, t0 )

    return (rotation_angle)/(rabi_frequency) - integral


def integr_sine(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : sin(pi*abs(t) / (x) ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_kaiser0(x, *arg):
    from scipy.special import i0

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : i0( pi * alpha * sqrt( 1 - ( ( 2*(t) / (x) ) - 1 )**2 ) ) / i0( pi * alpha ), 0, x )

    return rotation_angle/rabi_frequency - integral


from scipy.special import i1
def integr_kaiser1(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : i1( gamma * sqrt( 1-(t/x)**2 ) ) / ( i1(gamma) * sqrt( 1-(t/x)**2 ) ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT3F(x, *arg):
    c0 = 0.26526
    c1 = -0.5
    c2 = 0.23474

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT4F(x, *arg):
    c0 = 0.21706
    c1 = -0.42103
    c2 = 0.28294
    c3 = -0.07897

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 2 * (3*pi*t) / x ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT5F(x, *arg):
    c0 = 0.1881
    c1 = -0.36923
    c2 = 0.28702
    c3 = -0.13077
    c4 = 0.02488

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 2 * (3*pi*t) / x ) + c4 *cos( 2 * (4*pi*t) / x ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT3M(x, *arg):
    c0 = 0.28235
    c1 = -0.52105
    c2 = 0.19659

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT4M(x, *arg):
    c0 = 0.241906
    c1 = -0.460841
    c2 = 0.255381
    c3 = -0.041872

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 3 * (2*pi*t) / x ), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_SFT5M(x, *arg):
    c0 = 0.209671
    c1 = -0.407331
    c2 = 0.281225
    c3 = -0.092669
    c4 = 0.0091036

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 3 * (2*pi*t) / x ) + c4 *cos( 4 * (2*pi*t) / x ) , 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_HFT90D(x, *arg):
    c0 = 1.0
    c1 = -1.942604
    c2 = 1.340318
    c3 = -0.440811
    c4 = 0.043097

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 3 * (2*pi*t) / x ) + c4 *cos( 3 * (2*pi*t) / x ) , 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_HFT116D(x, *arg):
    c0 = 1.0
    c1 = -1.9575375
    c2 = 1.4780705
    c3 = -0.6367431
    c4 = 0.1228389
    c5 = -0.0066288

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 3 * (2*pi*t) / x ) + c4 *cos( 4 * (2*pi*t) / x ) + c5 *cos( 5 * (2*pi*t) / x ) , 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_HFT169D(x, *arg):
    c0 = 1.0
    c1 = -1.97441842
    c2 = 1.65409888
    c3 = -0.95788186
    c4 = 0.33673420
    c5 = -0.06364621
    c6 = 0.00521942
    c7 = -0.00010599

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    integral, err = quad( lambda t : c0 + c1 *cos( 1 * (2*pi*t) / x ) + c2 *cos( 2 * (2*pi*t) / x ) + c3 *cos( 3 * (2*pi*t) / x ) + c4 *cos( 4 * (2*pi*t) / x ) + c5 *cos( 5 * (2*pi*t) / x ) + c6 *cos( 6 * (2*pi*t) / x ) + c7 *cos( 7 * (2*pi*t) / x ) , 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_UBURP(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    a = np.zeros(21)
# For 256 time slices
    a[0] = 0.27
    a[1] = -1.42
    a[2] = -0.37
    a[3] = -1.84
    a[4] = 4.4
    a[5] = -1.19
    a[6] = 0.0
    a[7] = -0.37
    a[8] = 0.5
    a[9] = -0.31
    a[10] = 0.18
    a[11] = -0.21
    a[12] = 0.23
    a[13] = -0.12
    a[14] = 0.07
    a[15] = -0.06
    a[16] = 0.06
    a[17] = -0.04
    a[18] = 0.03
    a[19] = -0.02
    a[20] = 0.02

    omega = 2*pi/x

    integral, err = quad( lambda t : a[0] + a[1] * cos(1*omega*t) + a[2] * cos(2*omega*t) + a[3] * cos(3*omega*t) + a[4] * cos(4*omega*t) + a[5] * cos(5*omega*t) + a[6] * cos(6*omega*t) + a[7] * cos(7*omega*t) + a[8] * cos(8*omega*t) + a[9] * cos(9*omega*t) + a[10] * cos(10*omega*t) + a[11] * cos(11*omega*t) + a[12] * cos(12*omega*t) + a[13] * cos(13*omega*t) + a[14] * cos(14*omega*t) + a[15] * cos(15*omega*t) + a[16] * cos(16*omega*t) + a[17] * cos(17*omega*t) + a[18] * cos(18*omega*t) + a[19] * cos(19*omega*t) + a[20] * cos(20*omega*t), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_REBURP(x, *arg):
    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    a = np.zeros(16)
# For 256 time slices
    a[0] = 0.49
    a[1] = -1.02
    a[2] = 1.11
    a[3] = -1.57
    a[4] = 0.83
    a[5] = -0.42
    a[6] = 0.26
    a[7] = -0.16
    a[8] = 0.1
    a[9] = -0.07
    a[10] = 0.04
    a[11] = -0.03
    a[12] = 0.01
    a[13] = -0.02
    a[14] = 0.0
    a[15] = -0.01

    omega = 2*pi/x

    integral, err = quad( lambda t : a[0] + a[1] * cos(1*omega*t) + a[2] * cos(2*omega*t) + a[3] * cos(3*omega*t) + a[4] * cos(4*omega*t) + a[5] * cos(5*omega*t) + a[6] * cos(6*omega*t) + a[7] * cos(7*omega*t) + a[8] * cos(8*omega*t) + a[9] * cos(9*omega*t) + a[10] * cos(10*omega*t) + a[11] * cos(11*omega*t) + a[12] * cos(12*omega*t) + a[13] * cos(13*omega*t) + a[14] * cos(14*omega*t) + a[15] * cos(15*omega*t), 0, x )

    return rotation_angle/rabi_frequency - integral


def integr_slepian(x, *arg):
    from scipy.integrate import simps
    from scipy.signal.windows import dpss

    c, alpha, gamma, a, rabi_frequency, rotation_angle = arg

    #TODO: Add smarter way to recover the proper frequency to consider here
    omega0_max  = 2*np.pi*10.1e9     # 10.1GHz (valid for 2 qubits simulation)
    nb_time_pts = np.ceil(x / ( 1./( omega0_max ) / 100 )).astype(int)

    integral = simps(dpss(nb_time_pts, alpha), np.linspace(-x/2, x/2, nb_time_pts) )

    return rotation_angle/rabi_frequency - integral



integr = {'hamming'       : integr_hamming,
          'hamming_alt'   : integr_hamming_alt,
          'triangular'    : integr_triangle,
          'hann'          : integr_hann,
          'blackman'      : integr_blackman,
          'opt_blackman'  : integr_opt_blackman,
          'gaussian'      : integr_gaussian,
          'papoulis'      : integr_papoulis,
          'sine'          : integr_sine,
          'rectangle'     : integr_rectangle,
          'step'          : integr_step,
          'gaussian'      : integr_gaussian,
          'half_gaussian' : integr_half_gaussian,
          'half_gaussian2': integr_half_gaussian,
          'hermite'       : integr_hermite,
          'kaiser0'       : integr_kaiser0,
          'kaiser1'       : integr_kaiser1,
          'SFT3F'         : integr_SFT3F,
          'SFT4F'         : integr_SFT4F,
          'SFT5F'         : integr_SFT5F,
          'SFT3M'         : integr_SFT3M,
          'SFT4M'         : integr_SFT4M,
          'SFT5M'         : integr_SFT5M,
          'HFT90D'        : integr_HFT90D,
          'HFT116D'       : integr_HFT116D,
          'HFT169D'       : integr_HFT169D,
          'UBURP'         : integr_UBURP,
          'REBURP'        : integr_REBURP,
          'slepian'       : integr_slepian,
         }  # Dictionary of pulse shapes functions for integration purposes (ie without the case where it is equal to 0)


## Shape functions with finite rise time
from qubit_utilities import *
from pulseshapes import *
from scipy.interpolate import interp1d


def filtered_shape(end_time, shape, args):
    """
    Returns an array containing the value of the filtered shape function over its whole duration.

    Parameters
    ----------
    end_time: total duration of the pulse (in ns)
    shape   : string specifying the name of th shape
    args    : tuple containing the parameters of the shapes. For more generality, values for the possible free parameters of the functions are also expected. In case the desired shape does not have free parameters, any dummy value can be provided.
     The tuple is organised as follows: c (optimsed blackman shape parameter), alpha (0th order Kaiser shape parameter), gamma (1st order Kaiser shape parameter), a (gaussian and gaussian-like shape parameter), rabi_frequency, rotation_angle.

     Returns
     -------

    """
    from qubit_utilities import prepare_filter

    c, alpha, gamma, a, rabi_frequency, rotation_angle, omega, omega0_max, nb_time_pts = args

    # args = (c, alpha, gamma, rabi_frequency, rotation_angle)

    if shape == 'kaiser0':
        times = np.linspace(0, end_time, nb_time_pts)
    else:
        times = np.linspace(-end_time/2, end_time/2, nb_time_pts)

    # Define the arguments for the shape function
    args = {'c':c,
            'alpha':alpha,
            'gamma':gamma,
            'rabi_freq':rabi_frequency,
            't_end'    : end_time,
    }

    unfiltered_shape = np.zeros(nb_time_pts)
    if shape == 'slepian':
        from scipy.signal.windows import dpss
        unfiltered_shape = dpss( nb_time_pts, alpha )
    else:
        i = 0
        for t in times:
            unfiltered_shape[i] = shapes[shape](t, args)
            i+=1

    b, a = prepare_filter(omega0_max/(2*np.pi) - 400e6, omega0_max/(2*np.pi) + 400e6, nb_time_pts/end_time, order=3)

    filtered_shape = lfilter(b, a, unfiltered_shape)

    return filtered_shape


def integr_shape_rt(x, *arg):
    from scipy.integrate import simps

    c, alpha, gamma, a, rabi_frequency, rotation_angle, omega0, omega0_max, nb_time_pts, shape = arg

    arg = (c, alpha, gamma, a, rabi_frequency, rotation_angle, omega0, omega0_max, nb_time_pts)

    filtered_shape_values = filtered_shape(x, shape, arg)

    integral = simps( filtered_shape_values, np.linspace(-x/2, x/2, nb_time_pts) )
    # integral = simps( filtered_shape_values, np.linspace(0, x, nb_time_pts) )

    return rotation_angle/rabi_frequency - integral



## Pulse computation

def compute_pulse_duration_rt( rotation_angle, shape, rabi_frequency, nb_time_pts, omega0, omega0_max, arg ):
    """
    Compute the duration required to perform a desired rotation with a shaped pulse, including the modeling of rise time effect on the AWG.

    \param rotation_angle: desired rotation angle, in radians
    \param shape         : string specifying the name of the shape of the pulse
    \param rabi_frequency: amplitude of the pulse, in hertz
    \param arg           : tuple containing the values of the parameters c, alpha and gamma for optimised Blackman, Kaiser 0 and Kaiser 1 shapes respectively

    \return pulse_duration: Pulse duration required to perform the rotation with the given pulse, in seconds.
    """

    # Add rotation angle and Rabi frequency to arg
    c, alpha, gamma, a = arg
    arg_new = (c, alpha, gamma, a, rabi_frequency, rotation_angle, omega0, omega0_max, nb_time_pts, shape)

    # Solve for pulse duration
    vfunc = np.vectorize(integr_shape_rt)

    try:
        pulse_duration, = fsolve( vfunc, compute_pulse_duration(rotation_angle, shape, rabi_frequency, arg), xtol=1.0e-12, args=arg_new )
    except TypeError:
        pulse_duration, = fsolve( vfunc, 100e-9, xtol=1.0e-12, args=arg_new )

    return pulse_duration


def compute_pulse_duration( rotation_angle, shape, rabi_frequency, arg ):
    """
    Compute the duration required to perform a desired rotation with a shaped pulse.

    \param rotation_angle: desired rotation angle, in radians
    \param shape         : string specifying the name of the shape of the pulse
    \param rabi_frequency: amplitude of the pulse, in hertz
    \param arg           : tuple containing the values of the parameters c, alpha and gamma for optimised Blackman, Kaiser 0 and Kaiser 1 shapes respectively

    \return pulse_duration: Pulse duration required to perform the rotation with the given pulse, in seconds.
    """

    # Add rotation angle and Rabi frequency to arg
    c, alpha, gamma, a = arg
    arg = (c, alpha, gamma, a, rabi_frequency, rotation_angle)

    # Solve for pulse duration
    if shape == "half_gaussian2":
        # Avoids having to recompute the duration for the backward half-gaussian pulse (since it is the same as for the forward one)
        func = integr.get("half_gaussian")
    else:
        func = integr.get(shape)

    vfunc = np.vectorize(func)

    pulse_duration, = fsolve(vfunc, 20.0e-9, xtol=1.0e-12, args=arg)

    return pulse_duration


def load_sigma(rotation_angle, chosen_shape, rabi_frequency, arg, t0=1):

    if rotation_angle == np.pi/2:
        angle_str = "piOver2"
    elif rotation_angle == np.pi:
        angle_str = "pi"
    rabi_str = str( int( (rabi_frequency/(2*np.pi)) * 10**(-6) ) )

    c, alpha, gamma, a = arg

    filename = "/home/vincent/Documents/Master_Thesis/Simulations_Beginning/Utilities/pulse_duration_database/"+"sigma_"+angle_str+"_Rabi-"+rabi_str+"MHz_Shape-"+chosen_shape


    if 'gaussian' in chosen_shape:
        # Compute pulse_length
        filename = filename+"-a_"+str( '%.3f' % a )

    elif chosen_shape == 'hermite':
        t0 = t0 * 10**9
        filename = filename+"_t0-"+str( '%.2f' % t0 )

    filename = filename+".pckl"

    f = open(filename, 'rb')
    sigma = pickle.load(f)
    f.close()
    # print("Pulse duration loaded.")

    return sigma