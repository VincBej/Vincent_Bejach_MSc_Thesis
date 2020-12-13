## Imports
import qutip as qt

import numpy as np
from scipy.signal import iirfilter, lfilter

# User defined modules
from pulseshapes import *
from pulse_duration import *


## Bloch-Siegert shift
def compute_BS_shift( omega0, omega, rabi_freq, order=0 ):
    """
    Compute the shift in resonance frequency of the qubit due to Bloch-Siegert shift (formulae taken from Steck 2007 and Shirley 1965)

    Parameters
    ----------
    omega0   : original resonance frequency
    omega    : drive frequency
    rabi_freq: Rabi frequency
    order    : order of the correction to be computed (so far, only 0, 1, 4 and 6 are implemented)

    Returns
    -------
    delta_omega_bs : Correction to resonance frequency due to Bloch-Siegert shift
    """
    if order == 0:
        delta_omega_bs = rabi_freq**2 / (4*omega0)

    elif order == 1:
        delta_omega_bs = (rabi_freq/4)**2 / ( 2*(omega0 + omega) )

    elif order == 4:
        a = 0.25*rabi_freq
        delta_omega_bs = a**2/omega0 + a**4/(4*omega0**3)

    elif order == 6:
        a = 0.25*rabi_freq
        delta_omega_bs = a**2/omega0 + a**4/(4*omega0**3) - (35*a**6)/(32*omega0**5)

    else:
        raise NotImplementedError("Order "+str(order)+" has not been implemented.")

    return delta_omega_bs


## AC Stark shift
def compute_Stark_shift(omega0, omega, rabi_freq):
    """
    General expression for the AC Stark shift. Projects the value from the dressed rotation axis to the uncoupled one.

    Parameters
    ----------
    omega0   : original resonance frequency
    omega    : drive frequency
    rabi_freq: Rabi frequency

    Returns
    -------
    delta_omega_s : Correction to resonance frequency due to AC Stark shift
    """
    from math import cos, sin, atan, sqrt

    delta = omega - omega0

    if delta == 0:
        return 0

    rabi_eff  = sqrt( rabi_freq**2 + delta**2 )
    prefactor = abs(delta) / rabi_eff

    if delta > 0:
        res = prefactor * (-rabi_eff+delta)
    else:
        res = prefactor * (rabi_eff+delta)

    return res


## Qubit resonance frequency shifts
def objective_fn_freq_shift(omega, omega0_initial, rabi, correction_type):
    """
    Objective function used in the optimisation part of the frequency shift correction algorithm.
    """

    omega0_dressed = omega0_initial + iterate_freq_shifts( omega0_initial, omega, rabi, 6, len(omega0_initial), len(omega), iter_limit=1, test=True, correction_type=correction_type)

    residual = np.abs( omega0_dressed - omega )

    return residual


def compute_total_freq_shift(omega0, omega, rabi, nb_qubits, nb_drives, i_slice=1, nb_time_slices=2, order=6, correction_type='all'):
    """
    Compute the total frequency shift caused by nb_drives driving pulses on each of the nb_qubits qubits considered (returns an array of one shift in resonance frequency per qubit)
    """
    total_shift = np.zeros( nb_qubits )

    for i_qubit in range(nb_qubits):
        for i_drive in range(nb_drives):
            if i_slice >= nb_time_slices:
                s_shift_of_pulse  = 0
                bs_shift_of_pulse = 0
            else:

                if correction_type == 'stark':
                    s_shift_of_pulse  = compute_Stark_shift(omega0[i_qubit], omega[i_drive], rabi[i_drive])
                    bs_shift_of_pulse = 0

                elif correction_type == 'bs':
                    s_shift_of_pulse  = 0
                    bs_shift_of_pulse = compute_BS_shift(omega0[i_qubit], omega[i_drive], rabi[i_drive], order)

                elif correction_type == 'all':
                    s_shift_of_pulse  = compute_Stark_shift(omega0[i_qubit], omega[i_drive], rabi[i_drive])
                    bs_shift_of_pulse = compute_BS_shift(omega0[i_qubit], omega[i_drive], rabi[i_drive], order)

                elif correction_type == 'none':
                    s_shift_of_pulse  = 0
                    bs_shift_of_pulse = 0

                else:
                    raise NotImplementedError("The correction specified ("+correction_type+") is currently not implemented.")

            total_shift[i_qubit] = total_shift[i_qubit] + s_shift_of_pulse + bs_shift_of_pulse

    return total_shift


def iterate_freq_shifts(omega0, omega, rabi, order, nb_qubits, nb_drives, i_slice=1, nb_time_slices=2, iter_limit=1, test=False, correction_type='all'):
    """
    Computes several iterations of frequency shifts until convergence is reached.
    Not used anymore: the methods calling this function do it with a maximum iteration number of 1 (it then effectively only returns the original computation). The reason the iteration was discarted is that the change was too small to have significant impact over the 1 ns length of a time slice.

    Returns
    -------
    omega0 : Update of resonance frequencies of the qubits after the iterations have converged (or hit the iteration limit)
    """
    import matplotlib.pyplot as plt

    improvement_large = True
    counter           = 0
    omega0_initial    = np.copy(omega0)
    omega_initial     = np.copy(omega)
    omega_prev_iter   = np.copy(omega)

    while improvement_large and counter < iter_limit:

        omega0 = omega0_initial + compute_total_freq_shift( omega0_initial, omega_prev_iter, rabi, nb_qubits, nb_drives, i_slice, nb_time_slices, order, correction_type )
        omega  = omega_initial + compute_total_freq_shift( omega0_initial, omega_prev_iter, rabi, nb_qubits, nb_drives, i_slice, nb_time_slices, order, correction_type )

        if test == False:
            if max( np.abs( omega - omega_prev_iter ) ) < 2*np.pi * 100e3:
                improvement_large = False
                # print("End condition")

        counter+=1

        omega_prev_iter = np.copy( omega )

    return omega0 - omega0_initial


## Multiqubit hamiltonians (legacy)
def compute_multiqubit_hamil( template, nb_qubits, index_template, dim_I=2 ):
    """
    Compute a Hamiltonian of the form IxIx...xHxIx...xI, where there in total nb_qubits elements in the tensor product, and the index of the one that is not I is given by index_template.
    """
    if index_template == 0:
            H = template
    else:
        H = qt.qeye(dim_I)

    for i in range(1, nb_qubits):
        if i == index_template:
            H = qt.tensor( H, template )
        else:
            H = qt.tensor( H, qt.qeye(dim_I) )

    return np.array( list(H) )[:,0]


def compute_free_evolution_hamiltonian( nb_qubits, nb_drives, omega, omega0, hbar ):
    H_res = qt.qzero( [ [2,2] ] )

    for index_qubit in range(nb_qubits):
        H_qubit_tmp = qt.Qobj( np.zeros( (2,2) ) )

        for index_drive in range(nb_drives):
            delta = omega[index_drive] - omega0[index_qubit]
            H_qubit_tmp += -hbar * delta * ( qt.sigmam()*qt.sigmap() )

        H_res = H_res + compute_multiqbt_hamil( H_qubit_tmp, nb_qubits, index_qubit, 2 )

    return H_res


## Signal analysis functions
def compute_FT( drive_settings, duration=0 ):
    """
    Compute and returns the Fourier transform of the pulse defined by the parameters contained in the dictionary drive_settings
    Parameters
    ----------
    drive_settings       : dictionnary containing the parameters used to define the driving pulse

    Returns
    -------
    fft_signal: Real frequency components of the Fourier transform of the pulse
    freqs     : Frequencies corresponding to the FT elements
    """
    from pulse_duration import load_sigma, compute_pulse_duration

    ## Recover useful input
    try:
        rotation_array = drive_settings['rotation']
    except:
        rotation_array = drive_settings['rotation_angle']
    try:
        rabi_array     = drive_settings['rabi']
    except:
        rabi_array     = drive_settings['rabi_frequency']
    shape_array    = drive_settings['shape']
    omega_array    = drive_settings['omega']
    if 'arg' in drive_settings:
        arg = drive_settings['arg']
        a_gauss = arg[3]
    else:
        a_gauss = 1
        arg     = (0., 0., 0.1, a_gauss)


    ## Prepare driving pulse
    drive_freq = np.max(omega_array)
    # Temporary values
    rotation = rotation_array[0]
    rabi     = rabi_array[0]
    shape    = shape_array[0]
    omega    = omega_array[0]

    if duration == 0:
        # The duration is not specified, so it must be computed to fit the desired rotation
        if shape == 'gaussian':
            sigma = compute_pulse_duration(rotation, 'gaussian', rabi, arg)
            t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

            end_time = 2*t0
        elif shape == 'half_gaussian':
            sigma = compute_pulse_duration(rotation, 'half_gaussian', rabi, arg)
            t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

            end_time = t0
        elif shape == 'hermite':
            end_time = 100e-9
        else:
            end_time = compute_pulse_duration(rotation, shape, rabi, arg)

    else:
        # A duration has been specified
        end_time = duration

    nb_samples = int(drive_freq*8 * end_time)
    # Array to contain the full driving signal
    signal = np.zeros(nb_samples)

    for i in range(len(omega_array)):
        rotation = rotation_array[i]
        rabi     = rabi_array[i]
        shape    = shape_array[i]
        omega    = omega_array[i]


        drive_freq = np.max(omega_array)
        a_gauss = 1
        # arg        = (0., 0., 0.1, a_gauss)

        if duration == 0:
            # The duration is not specified, so it must be computed to fit the desired rotation
            if shape == 'gaussian':
                sigma = compute_pulse_duration(rotation, 'gaussian', rabi, arg)
                t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

                end_time = 2*t0
            elif shape == 'half_gaussian':
                sigma = compute_pulse_duration(rotation, 'half_gaussian', rabi, arg)
                t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

                end_time = t0
            elif shape == 'hermite':
                end_time = 100e-9
            else:
                end_time = compute_pulse_duration(rotation, shape, rabi, arg)

        else:
            # A duration has been specified
            end_time = duration

        sampling_rate = drive_freq*8   # In Hz.rad
        nb_samples    = int(sampling_rate * end_time)

        if shape == 'kaiser0':
            times_tmp = np.linspace( 0, end_time, nb_samples, endpoint=False )
        else:
            times_tmp = np.linspace( -end_time/2, end_time/2, nb_samples, endpoint=False )
        args = {'t_end':end_time,
                    'c'    :0.0,
                    'sigma':np.sqrt(2),
                    'alpha':0.0,
                    'gamma':0.1,
                    'a':a_gauss
            }
        a = rabi

        if shape == 'hermite':
            args['hermite_a']     = 1
            args['hermite_b']     = 0.956
            index_hermite         = 0
            args['hermite_sigma'] = load_sigma( rotation, 'hermite', rabi, arg, end_time/2 )
        elif shape == 'kaiser0':
            args['alpha'] = arg[1]
        elif shape == 'opt_blackman':
            args['c'] = arg[0]

        # Compute samples of the shaped signal
        signal_tmp = np.zeros(nb_samples)
        for j in range(0, nb_samples):
            signal_tmp[j] = shapes.get(shape)(times_tmp[j], args) * a * np.sin(omega * times_tmp[j])

        # Scaling for BURP shapes
        if shape == 'UBURP':
            scaling_UBURP = 1./(4*0.27)
            signal_tmp = signal_tmp * scaling_UBURP
        elif shape == 'REBURP':
            scaling_REBURP = 1./(2*0.48)
            signal_tmp = signal_tmp * scaling_REBURP

        signal = signal + signal_tmp


    ## Compute FFT
    sampling_interval = times_tmp[-1] - times_tmp[-2]
    fft_signal = np.fft.fft(signal, signal.size*45)

    freqs  = np.fft.fftfreq( len(signal)*45 ) * 1./sampling_interval    # Freqs is expressed in Hz

    bin_width = freqs[1] - freqs[0]
    bin_width2 = sampling_rate / len(fft_signal)
    print( "Debug: bin width1 = "+str(bin_width) )
    print( "Debug: bin width2 = "+str(bin_width2) )

    normalisation_factor = 1./fft_signal.size
    norm_fft_signal      = np.abs(fft_signal) * normalisation_factor    # FT is normalised in terms of energy

    return norm_fft_signal[:norm_fft_signal.size//2], freqs[:norm_fft_signal.size//2]  # Return the positive frequency components


def compute_energy_spectral_density( drive_settings, duration=0 ):
    """
    Compute and returns the energy spectral density of the pulse defined by the parameters contained in the dictionary drive_settings.
    Parameters
    ----------
    drive_settings       : dictionnary containing the parameters used to define the driving pulse

    Returns
    -------
    fft_signal: Real frequency components of the Fourier transform of the pulse
    freqs     : Frequencies corresponding to the FT elements
    """
    from pulse_duration import load_sigma, compute_pulse_duration

    ## Recover useful input
    try:
        rotation_array = drive_settings['rotation']
    except:
        rotation_array = drive_settings['rotation_angle']
    try:
        rabi_array     = drive_settings['rabi']
    except:
        rabi_array     = drive_settings['rabi_frequency']
    shape_array    = drive_settings['shape']
    omega_array    = drive_settings['omega']
    if 'arg' in drive_settings:
        arg = drive_settings['arg']
        a_gauss = arg[3]
    else:
        a_gauss = 1
        arg     = (0., 0., 0.1, a_gauss)


    ## Prepare driving pulse
    drive_freq = np.max(omega_array)
    # Temporary values
    rotation = rotation_array[0]
    rabi     = rabi_array[0]
    shape    = shape_array[0]
    omega    = omega_array[0]

    if duration == 0:
        # The duration is not specified, so it must be computed to fit the desired rotation
        if shape == 'gaussian':
            sigma = compute_pulse_duration(rotation, 'gaussian', rabi, arg)
            t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

            end_time = 2*t0
        elif shape == 'half_gaussian':
            sigma = compute_pulse_duration(rotation, 'half_gaussian', rabi, arg)
            t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

            end_time = t0
        elif shape == 'hermite':
            end_time = 100e-9
        else:
            end_time = compute_pulse_duration(rotation, shape, rabi, arg)

    else:
        # A duration has been specified
        end_time = duration

    nb_samples = int(drive_freq*8 * end_time)
    # Array to contain the full driving signal
    signal = np.zeros(nb_samples)

    for i in range(len(omega_array)):
        rotation = rotation_array[i]
        rabi     = rabi_array[i]
        shape    = shape_array[i]
        omega    = omega_array[i]


        drive_freq = np.max(omega_array)
        a_gauss = 1

        if duration == 0:
            # The duration is not specified, so it must be computed to fit the desired rotation
            if shape == 'gaussian':
                sigma = compute_pulse_duration(rotation, 'gaussian', rabi, arg)
                t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

                end_time = 2*t0
            elif shape == 'half_gaussian':
                sigma = compute_pulse_duration(rotation, 'half_gaussian', rabi, arg)
                t0 = sqrt( - (sigma**2)/a_gauss * log(0.005) )

                end_time = t0
            elif shape == 'hermite':
                end_time = 100e-9
            else:
                end_time = compute_pulse_duration(rotation, shape, rabi, arg)

        else:
            # A duration has been specified
            end_time = duration

        sampling_rate = drive_freq*8   # In Hz.rad
        nb_samples    = int(sampling_rate * end_time)

        if shape == 'kaiser0':
            times_tmp = np.linspace( 0, end_time, nb_samples, endpoint=False )
        else:
            times_tmp = np.linspace( -end_time/2, end_time/2, nb_samples, endpoint=False )
        args = {'t_end':end_time,
                    'c'    :0.0,
                    'sigma':np.sqrt(2),
                    'alpha':0.0,
                    'gamma':0.1,
                    'a':a_gauss
            }
        a = rabi

        if shape == 'hermite':
            args['hermite_a']     = 1
            args['hermite_b']     = 0.956
            index_hermite         = 0
            args['hermite_sigma'] = load_sigma( rotation, 'hermite', rabi, arg, end_time/2 )
        elif shape == 'kaiser0':
            args['alpha'] = arg[1]
        elif shape == 'opt_blackman':
            args['c'] = arg[0]

        # Compute samples of the shaped signal
        signal_tmp = np.zeros(nb_samples)
        for j in range(0, nb_samples):
            signal_tmp[j] = shapes.get(shape)(times_tmp[j], args) * a * np.sin(omega * times_tmp[j])

        # Scaling for BURP shapes
        if shape == 'UBURP':
            scaling_UBURP = 1./(4*0.27)
            signal_tmp = signal_tmp * scaling_UBURP
        elif shape == 'REBURP':
            scaling_REBURP = 1./(2*0.48)
            signal_tmp = signal_tmp * scaling_REBURP

        signal = signal + signal_tmp

    ## Compute power density
    sampling_interval = times_tmp[-1] - times_tmp[-2]
    fft_signal = np.fft.fft(signal, signal.size*45)

    freqs  = np.fft.fftfreq( len(signal)*45 ) * 1./sampling_interval    # Freqs is expressed in Hz

    s = fft_signal * np.conjugate(fft_signal)

    return s[:s.size//2], freqs[:s.size//2]  # Return the positive frequency components


## Average gate fidelity computation
def compute_single_qubit_target_unitary(angle, axis):
    """
    Computes the target unitary of the one-qubit rotation of angle angle around the axis defined by axis.

    Parameters
    ----------
    angle: value of the rotation angle (in rad)
    axis : tuple defining the rotation axis using the cartesian X, Y and Z axes. For instance, the X and Z axis are represented by (1,0,0) and (0,0,1) respectively

    Returns
    -------
    target: unitary matrix (numpy array) taht represented the ideal operation
    """

    X = np.array( [[0,1], [1,0]], dtype=np.complex )
    Y = np.array( [[0,-1j], [1j,0]], dtype=np.complex )
    Z = np.array( [[1,0], [0,-1]], dtype=np.complex )
    tmp = axis[0]*X + axis[1]*Y + axis[2]*Z

    target = cos(angle/2) * np.eye(2) - 1j*sin(angle/2) * tmp

    return target


def compute_multiqubit_target_unitary(angles, axes):
    """
    Computes the target unitary of the operation that rotates qubit i by the angle defined by the i-th element of angles, around the axis defined by the i-th element of axes.
    Parameters
    ----------
    angles: array containing the values of the rotation angle of each qubit
    axes  : array of tuples defining the rotation axes of the rotation of each qubit. Each rotation axis is defined using thecartesian X, Y and Z axes (e.g. the X and Z axis are represented by (1,0,0) and (0,0,1) respectively)

    Returns
    -------
    target: unitary matrix (numpy array) taht represented the ideal operation

    """
    target  = qt.qeye(2)
    i_qubit = 0

    for angle in angles:
        tmp = compute_single_qubit_target_unitary(angle, axes[i_qubit])
        # if target == qt.qeye(2):
        if i_qubit == 0:
            target = qt.Qobj(tmp)
        else:
            target = qt.tensor(target, qt.Qobj(tmp))

        i_qubit+=1

    return target


## Miscellaneous
# Computes the coefficient of a lowpass elliptic filter
def prepare_filter(lowcut, highcut, fs, order=3):
    """
    Compute the coefficient of a lowpass elliptic filter simulating the effect of rise time in the generation of pulse shapes.

    Parameters
    ----------
    lowcut: low cutoff frequency
    highcut: high cutoff frequency (used by lowpass)
    fs: sampling frequency
    order: order of the lowpass elliptic filter.

    Returns
    b, a: coefficients of the designed lowpass elliptic filter
    """
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = highcut / nyq
    # Settings of lowpass filter (to remove high frequency components responsible for sharp edges)
    rp    = 0.1
    rs    = 3.9
    b, a = iirfilter(order, [high], rp=rp, rs=rs, btype='low', ftype='ellip', analog=False)
    return b, a


def correct_frequencies( freqs, drive_settings ):
    import matplotlib.pyplot as plt
    """
    Compute the frequency shifts (AC Stark and Bloch-Siegert shifts) induced by the driving pulses defined by the parameters in drive_settings, and apply them to the frequencies in freqs.

    Parameters
    ----------
    freqs         : array containing the frequencies to be corrected (in Hz)
    drive_settings: dictionnary containing the parameters used to define the driving pulse

    Returns
    -------
    freqs_modified: array containing the corrected frequencies (f_corrected = f_init + frequency_shifts)
    """
    rabi     = drive_settings['rabi']
    omega0   = drive_settings['omega0']

    freqs_modified = np.zeros( len(freqs) )
    freqs = (freqs) * 2*np.pi

    correction_to_plot = list()

    for i_freq in range(len(freqs_modified)):
        if freqs[i_freq] >= 0:  # Compute correction only for the positive components of the FT (redundant now, since rfft is used instead of fft)
            delta = freqs[i_freq] - omega0

            correction = compute_BS_shift(omega0, freqs[i_freq], rabi, 6) + compute_Stark_shift(omega0, freqs[i_freq], rabi)
            freqs_modified[i_freq] = freqs[i_freq] + correction

            correction_to_plot.append(correction)

        else:
            freqs_modified[i_freq] = freqs[i_freq]
            correction_to_plot.append(0)

    return freqs_modified
