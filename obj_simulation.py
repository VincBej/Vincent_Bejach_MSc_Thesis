# Vincent Bejach. MSc student, TU Delft.
# 2019

## Imports
# Modules
import sys
sys.path.append( "/usr/local/lib/python2.7/dist-packages/c_solver-1.1-py2.7-linux-x86_64.egg/c_solver/" )   # DM_solver
import ME_solver as me            # DM_solver
# import c_solver.ME_solver as me   # DM_solver (server)
import qutip as qt

import pickle
import numpy as np

from scipy.signal.windows import dpss
import matplotlib.pyplot as plt

# User defined modules
from pulseshapes import *
from pulse_duration import *
from qubit_utilities import *


## Class definition
class arbitrary_pulse():
    """
    Class representing a shaped pulse of arbitrary duration and Rabi frequency.
    """
    def __init__(self, nb_qubits, nb_drives, rabi_frequency, length_pulse, chosen_shape, omega0, omega, phase, arg=(0.11, 5, 8, 1.0), rise_time=True ):
        """
        Initialises the arbitrary_pulse class. A rotation angle is not specified. In order to create a pulse that implements a specific rotation angle, use the shaped_pulse class.

        Parameters
        ----------
        nb_qubits      : Number of qubits considered in the simulation
        nb_drives      : Number of driving pulses. If nb_qubits > nb_drives, the drives will be applied to the n_drives first qubits
        rabi_frequency : Rabi frequencies of each driving pulse.
        length_pulse   : Pulse duration of all the driving pulses.
        chosen_shape   : Array containing the amplitude modulation shape of each of the driving pulses
        omega0         : Array containing the resonance frequencies of the qubits
        omega          : Array containing the frequencies of each of the driving pulses
        phase          : Array containing the initial phase of each of the driving pulses
        arg            : Tuple containing the shape function arguments: (c, alpha, gamma, a) for optimised blackman, kaiser 0, kaiser1 and gaussian modulation shapes
        rise_time      : Boolean determining if the effects of pulse rise time are to be applied directly by the constructor

        """
        from math import floor, ceil
        from scipy.integrate import simps

        ## Store input values
        self.nb_drives          = nb_drives
        # self.rotation_angle     = rotation_angle
        self.rabi_frequency = rabi_frequency
        print("Initial Rabi frequency")
        print(self.rabi_frequency/(2*np.pi*10**6))
        self.length_pulse   = length_pulse
        print("Pulse duration: "+str(self.length_pulse*10**9)+" ns")
        self.chosen_shape       = chosen_shape
        self.omega              = omega
        self.omega0             = omega0
        self.arg                = arg

        c, alpha, gamma, a = arg

        print("\n"+self.chosen_shape[0].capitalize() )


        ## Number of time points and time slices
        # Time pts
        self.nb_time_pts = np.ceil(self.length_pulse / ( 1./( max(self.omega) ) / 100 )).astype(int)

        # Time slices
        if 'UBURP' in list(chosen_shape) or 'REBURP' in list(chosen_shape):
            nb_time_slices = 63    # Manual modification specific to BURP pulses
            length_slice = int(self.length_pulse / nb_time_slices)
        else:
            length_slice   = 1.0e-9       # A slice is precisely 1.0 ns
            nb_time_slices = int(ceil(self.length_pulse / length_slice))

        self.length_slice      = length_slice
        self.nb_time_slices    = nb_time_slices
        self.nb_time_pts_slice = int(self.nb_time_pts / self.nb_time_slices)

        self.omega_array  = np.ones( (self.nb_drives, self.nb_time_slices) )
        self.omega0_array = np.ones( (nb_qubits, self.nb_time_slices) )

        for i_drive in range(self.nb_drives):
            self.omega_array[i_drive] = self.omega_array[i_drive] * omega[i_drive]
        for i_qubit in range(nb_qubits):
            self.omega0_array[i_qubit] = self.omega0_array[i_qubit] * omega0[i_qubit]


        ## Pulse data
        # Define args for the pulse shape
        c, alpha, gamma, a = arg
        args = { 't_end': self.length_pulse,
                'c':c,
                'alpha':alpha,
                'gamma':gamma,
                'a'    :a,
                'nb_qubits':nb_qubits,
                'shape_fn' :shapes.get(chosen_shape[0])
        }

        if 'hermite' in list(chosen_shape):
            if abs(self.rotation_angle[0] - np.pi) < abs(self.rotation_angle[0] - np.pi/2):
                # The rotation angle is closer to pi, so we use the polynomial coefficients defined for a pi-pulse
                args['hermite_a'] = 1
                args['hermite_b'] = 0.956

            else:
                # # The rotation angle is closer to pi/2, so we use the polynomial coefficients defined for a pi/2-pulse
                args['hermite_a'] = 1
                args['hermite_b'] = 0.667

            args['hermite_sigma'] = sigma
            times = np.linspace( -self.length_pulse/2, self.length_pulse/2, self.nb_time_pts )

        elif 'half_gaussian' in list(chosen_shape) or 'half_gaussian2' in list(chosen_shape) or 'UBURP' in list(chosen_shape) or 'REBURP' in list(chosen_shape) or 'kaiser0' in list(chosen_shape) or 'step' in list(chosen_shape):
            times_hg = np.linspace(0, self.length_pulse, self.nb_time_pts )

        else:
            times = np.linspace( -self.length_pulse/2, self.length_pulse/2, self.nb_time_pts )

        # Initialisation of data structures
        values = np.zeros( (self.nb_drives, self.nb_time_pts) )

        # Computation of pulse data - Library shape functions
        if 'slepian' in list(chosen_shape):
            index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('slepian') ]
            for i_shape in index_shape:
                values[i_shape] = dpss( self.nb_time_pts, alpha )


        # Computation of pulse data - Custom shape functions
        for index_time in range(self.nb_time_pts):
            # If a special case is present, recovers the indices of this shape in chosen_shape and for those indices computes the shape with the custom time scale
            if 'half_gaussian' in list(chosen_shape) or 'half_gaussian2' in list(chosen_shape):
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.startswith('half_gaussian') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if ( 'UBURP' in list(chosen_shape) ) or ( 'REBURP' in list(chosen_shape) ) :
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('BURP') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if ( 'kaiser0' in list(chosen_shape) ) or ( 'kaiser0' in list(chosen_shape) ) :
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('kaiser0') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if 'step' in list(chosen_shape):
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('step') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            # Compute values for all shapes with regular time scale (most common case)
            for i_shape in range(len(chosen_shape)):
                if chosen_shape[i_shape] != 'slepian' and  chosen_shape[i_shape] != 'kaiser0' and chosen_shape[i_shape] != 'REBURP' and chosen_shape[i_shape] != 'UBURP' and chosen_shape[i_shape] != 'half_gaussian' and chosen_shape[i_shape] != 'half_gaussian2':
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times[index_time], args)


        # Pulse storage
        unfiltered_shape_values = np.zeros( (self.nb_drives, self.nb_time_pts) )

        for i_shape in range(len(chosen_shape)):
            if chosen_shape[i_shape] == 'kaiser1':
            # Computing the full kaiser 1 shape
                shape_values_tmp = np.concatenate( (np.flip(values[i_shape][0:self.nb_time_pts:2]), values[i_shape][0:self.nb_time_pts:2]) )
                unfiltered_shape_values[i_shape] = np.copy(shape_values_tmp)
            elif chosen_shape[i_shape] == 'half_gaussian2':
            # Computing the backward half-gaussian shape (basically just sorting the forward one in reverse order)
                unfiltered_shape_values[i_shape] = np.flip( values[i_shape] )
            else:
            # All the other pulse shapes
                unfiltered_shape_values[i_shape][:] = np.copy( values[i_shape][:] )


            # Calibration of BURP pulses to the proper rotation angle
            if chosen_shape[i_shape] == 'UBURP':
                scaling_UBURP = 1./(4*0.27)
                unfiltered_shape_values[i_shape] = unfiltered_shape_values[i_shape] * scaling_UBURP
            elif chosen_shape[i_shape] == 'REBURP':
                scaling_REBURP = 1./(2*0.48)
                unfiltered_shape_values[i_shape] = unfiltered_shape_values[i_shape] * scaling_REBURP

        self.pulse_data = unfiltered_shape_values


        ## Phase
        self.phase = np.ones( (nb_drives, self.nb_time_slices) )
        for i_drive in range(self.nb_drives):
            self.phase[i_drive,:] = self.phase[i_drive,:] * phase[i_drive]



class shaped_pulse():
    """
    Class representing the driving pulses to be applied to the system. It contains the caracteristics of each of the driving pulses as well as a separate pulse data per qubit in the simulation.
    """
    def __init__(self, nb_qubits, nb_drives, rotation_angle, fixed_parameter, chosen_shape, omega0, omega, phase, arg=(0.11, 5, 8, 1.0), rise_time=True, fixed_pulse_dur=False ):
        """
        Initialises the shaped_pulse class.

        Parameters
        ----------
        nb_qubits      : Number of qubits considered in the simulation
        nb_drives      : Number of driving pulses. If nb_qubits > nb_drives, the drives will be applied to the n_drives first qubits
        rotation_angle : Array containing the target rotation angles of each of the driving pulses
        fixed_parameter: Parameter fixed for the generation of the pulse.
                         If the element is an array, it corresponds to the Rabi frequencies of each driving pulse.
                         If the element is a number, it corresponds to the pulse duration of all the driving pulses.
        chosen_shape   : Array containing the amplitude modulation shape of each of the driving pulses
        omega0         : Array containing the resonance frequencies of the qubits
        omega          : Array containing the frequencies of each of the driving pulses
        phase          : Array containing the initial phase of each of the driving pulses
        arg            : Tuple containing the shape function arguments: (c, alpha, gamma, a) for optimised blackman, kaiser 0, kaiser1 and gaussian modulation shapes
        rise_time      : Boolean determining if the effects of pulse rise time are to be applied directly by the constructor
        fixed_pulse_dur: Boolean determining if the pulse must be created using a pre-specified pulse duration (and hence adapt its Rabi frequency) or using a fixed Rabi frequency (and hence adapt its duration)

        """
        from math import floor, ceil
        from scipy.optimize import fsolve
        from scipy.integrate import simps

        ## Store input values
        self.nb_drives          = nb_drives
        self.rotation_angle     = rotation_angle
        if isinstance(fixed_parameter, np.ndarray):
            self.rabi_frequency = fixed_parameter
            print("Initial Rabi frequency")
            print(self.rabi_frequency/(2*np.pi*10**6))
        else:
            self.length_pulse   = fixed_parameter
            print("Pulse duration: "+str(self.length_pulse*10**9)+" ns")
        self.chosen_shape       = chosen_shape
        self.omega              = omega
        self.omega0             = omega0
        self.arg                = arg

        c, alpha, gamma, a = arg

        print("\n"+self.chosen_shape[0].capitalize() )


        ## Pulse length
        if not fixed_pulse_dur:
            # Only if the pulse is created in the fixed-Rabi, variable-pulse duration configuration

            for i_drive in range(len(omega)):
                length_pulse_array   = np.zeros(len(omega))

                if 'gaussian' not in chosen_shape[i_drive] and 'hermite' not in chosen_shape[i_drive] and 'UBURP' not in chosen_shape[i_drive] and 'REBURP' not in chosen_shape[i_drive]:
                    # Compute length of pulse i_drive
                    length_pulse = compute_pulse_duration(self.rotation_angle[i_drive], chosen_shape[i_drive], self.rabi_frequency[i_drive], arg)

                    #debug
                    if 'opt_blackman' in chosen_shape[i_drive]:
                        print("debug")
                        print(length_pulse)

                elif chosen_shape[i_drive] == 'gaussian':
                    # What we compute  in the case of the gaussian shape is the width of the pulse leading to the desired area. The proper length of the pulse is computed right after by truncating the shape at 0.5% of its peak value (to avoid infinite pulse)
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], 'gaussian', self.rabi_frequency[i_drive], arg)
                    t0 = sqrt( - (sigma**2)/a * log(0.005) )

                    length_pulse = 2*t0

                elif 'half_gaussian' in chosen_shape[i_drive] or 'half_gaussian2' in chosen_shape[i_drive]:
                    # Same comment
                    # sigma = load_sigma( rotation_angle[i_drive], 'half_gaussian', rabi_frequency[i_drive], arg )
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], chosen_shape[i_drive], self.rabi_frequency[i_drive], arg)
                    t0 = sqrt( - (sigma**2)/a * log(0.005) )
                    print(t0)

                    length_pulse = t0

                elif chosen_shape[i_drive] == 'hermite':
                    # Same comment
                    t0_hermite = 50e-9
                    length_pulse = 2*t0_hermite
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], 'hermite', self.rabi_frequency[i_drive], arg)
                    # sigma = load_sigma( rotation_angle[i_drive], 'hermite', rabi_frequency[i_drive], arg, length_pulse/2 )

                elif chosen_shape[i_drive] == 'UBURP' or chosen_shape[i_drive] == 'REBURP':
                    length_pulse = (2*np.pi) / self.rabi_frequency[i_drive]

                # If the current shape is not a special shape, simply store the loaded pulse length in the proper array
                length_pulse_array[i_drive] = length_pulse


                # Check the pulse durations to see if the driving speeds must be adapted to have all pulses have the same duration
            if len(set(length_pulse_array)) > 1: # If not all the pulse durations are the same
                # Select longer required pulse time and set it as the duration of all other pulses
                length_pulse = max(length_pulse_array)

            else:
                length_pulse = length_pulse_array[0]

            self.length_pulse = length_pulse

        else:
            # Only if the pulse is created in the variable-Rabi, fixed-pulse duration configuration

            for i_drive in range(len(omega)):
                if chosen_shape[i_drive] == 'gaussian':
                    # What we compute  in the case of the gaussian shape is the width of the pulse leading to the desired area. The proper length of the pulse is computed right after by truncating the shape at 0.5% of its peak value (to avoid infinite pulse)
                    t0 = self.length_pulse/2
                    sigma = sqrt( -(a * t0**2)/(log(0.005)) )

                elif 'half_gaussian' in chosen_shape[i_drive] or 'half_gaussian2' in chosen_shape[i_drive]:
                    # Same comment
                    t0 = self.length_pulse
                    sigma = sqrt( -(a * t0**2)/(log(0.005)) )

                elif chosen_shape[i_drive] == 'hermite':
                    # Same comment
                    if abs(self.rotation_angle[0] - np.pi) < abs(self.rotation_angle[0] - np.pi/2):
                        a_hermite = 1
                        b_hermite = 0.956
                    else:
                        a_hermite = 1
                        b_hermite = 0.667
                    vfunc = np.vectorize( lambda x: (a_hermite - b_hermite*(self.length_pulse/x)**2) * exp( -a_hermite * (self.length_pulse/x)**2 ) )

                    sigma, = fsolve(vfunc, 20e-9, xtol=1.0e-12)



        ## Number of time points and time slices
        # Time pts
        self.nb_time_pts = np.ceil( 1.25*(self.length_pulse * max(self.omega/(2*np.pi)) * 100 ) ).astype(int) # 100 steps per period (increased by 25% as a safety margin)

        # Time slices
        if 'UBURP' in list(chosen_shape) or 'REBURP' in list(chosen_shape):
            nb_time_slices = 63    # Manual modification specific to BURP pulses
            length_slice = int(self.length_pulse / nb_time_slices)
        else:
            length_slice   = 1.0e-9       # A slice is precisely 1.0 ns
            nb_time_slices = int(ceil(self.length_pulse / length_slice))

        self.length_slice      = length_slice
        self.nb_time_slices    = nb_time_slices
        self.nb_time_pts_slice = int(self.nb_time_pts / self.nb_time_slices)

        self.omega_array  = np.ones( (self.nb_drives, self.nb_time_slices) )
        self.omega0_array = np.ones( (nb_qubits, self.nb_time_slices) )

        for i_drive in range(self.nb_drives):
            self.omega_array[i_drive] = self.omega_array[i_drive] * omega[i_drive]
        for i_qubit in range(nb_qubits):
            self.omega0_array[i_qubit] = self.omega0_array[i_qubit] * omega0[i_qubit]


        ## Pulse data
        # Define args for the pulse shape
        c, alpha, gamma, a = arg
        args = { 't_end': self.length_pulse,
                'c':c,
                'alpha':alpha,
                'gamma':gamma,
                'a'    :a,
                'nb_qubits':nb_qubits,
                'shape_fn' :shapes.get(chosen_shape[0])
        }

        if 'hermite' in list(chosen_shape):
            if abs(self.rotation_angle[0] - np.pi) < abs(self.rotation_angle[0] - np.pi/2):
                # The rotation angle is closer to pi, so we use the polynomial coefficients defined for a pi-pulse
                args['hermite_a'] = 1
                args['hermite_b'] = 0.956

            else:
                # # The rotation angle is closer to pi/2, so we use the polynomial coefficients defined for a pi/2-pulse
                args['hermite_a'] = 1
                args['hermite_b'] = 0.667

            args['hermite_sigma'] = sigma
            times = np.linspace( -self.length_pulse/2, self.length_pulse/2, self.nb_time_pts )

        elif 'half_gaussian' in list(chosen_shape) or 'half_gaussian2' in list(chosen_shape) or 'UBURP' in list(chosen_shape) or 'REBURP' in list(chosen_shape) or 'kaiser0' in list(chosen_shape) or 'step' in list(chosen_shape):
            times_hg = np.linspace(0, self.length_pulse, self.nb_time_pts )

        else:
            times = np.linspace( -self.length_pulse/2, self.length_pulse/2, self.nb_time_pts )

        # Initialisation of data structures
        values = np.zeros( (self.nb_drives, self.nb_time_pts) )

        # Computation of pulse data - Library shape functions
        if 'slepian' in list(chosen_shape):
            index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('slepian') ]
            for i_shape in index_shape:
                values[i_shape] = dpss( self.nb_time_pts, alpha )


        # Computation of pulse data - Custom shape functions
        for index_time in range(self.nb_time_pts):
            # If a special case is present, recovers the indices of this shape in chosen_shape and for those indices computes the shape with the custom time scale
            if 'half_gaussian' in list(chosen_shape) or 'half_gaussian2' in list(chosen_shape):
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.startswith('half_gaussian') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if ( 'UBURP' in list(chosen_shape) ) or ( 'REBURP' in list(chosen_shape) ) :
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('BURP') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if ( 'kaiser0' in list(chosen_shape) ) or ( 'kaiser0' in list(chosen_shape) ) :
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('kaiser0') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            if 'step' in list(chosen_shape):
                index_shape = [ i for i, word in enumerate(chosen_shape) if word.endswith('step') ]
                for i_shape in index_shape:
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times_hg[index_time], args)

            # Compute values for all shapes with regular time scale (most common case)
            for i_shape in range(len(chosen_shape)):
                if chosen_shape[i_shape] != 'slepian' and  chosen_shape[i_shape] != 'kaiser0' and chosen_shape[i_shape] != 'REBURP' and chosen_shape[i_shape] != 'UBURP' and chosen_shape[i_shape] != 'half_gaussian' and chosen_shape[i_shape] != 'half_gaussian2':
                    values[i_shape][index_time] = shapes.get(self.chosen_shape[i_shape])(times[index_time], args)


        # Pulse storage
        unfiltered_shape_values = np.zeros( (self.nb_drives, self.nb_time_pts) )

        for i_shape in range(len(chosen_shape)):
            if chosen_shape[i_shape] == 'kaiser1':
            # Computing the full kaiser 1 shape
                shape_values_tmp = np.concatenate( (np.flip(values[i_shape][0:self.nb_time_pts:2]), values[i_shape][0:self.nb_time_pts:2]) )
                unfiltered_shape_values[i_shape] = np.copy(shape_values_tmp)
            elif chosen_shape[i_shape] == 'half_gaussian2':
            # Computing the backward half-gaussian shape (basically just sorting the forward one in reverse order)
                unfiltered_shape_values[i_shape] = np.flip( values[i_shape] )
            else:
            # All the other pulse shapes
                unfiltered_shape_values[i_shape][:] = np.copy( values[i_shape][:] )


            # Calibration of BURP pulses to the proper rotation angle
            if chosen_shape[i_shape] == 'UBURP':
                scaling_UBURP = 1./(4*0.27)
                unfiltered_shape_values[i_shape] = unfiltered_shape_values[i_shape] * scaling_UBURP
            elif chosen_shape[i_shape] == 'REBURP':
                scaling_REBURP = 1./(2*0.48)
                unfiltered_shape_values[i_shape] = unfiltered_shape_values[i_shape] * scaling_REBURP

        self.pulse_data = unfiltered_shape_values


        ## Phase
        self.phase = np.ones( (nb_drives, self.nb_time_slices) )
        for i_drive in range(self.nb_drives):
            self.phase[i_drive,:] = self.phase[i_drive,:] * phase[i_drive]


        ## Rise time
        if rise_time:
            self.add_pulse_rise_time(fixed_pulse_duration=fixed_pulse_dur)


        ## Rabi frequency
        if fixed_pulse_dur:
            # Initiate the temporary storage data structure
            rabi_freq = np.zeros( (self.nb_drives) )

            # Compute the update Rabi frequency to keep the pulse duration fixed
            for i_drive in range(self.nb_drives):
                integral_tmp = simps( self.pulse_data[i_drive], np.linspace(-self.length_pulse/2, self.length_pulse/2, self.nb_time_pts) )

                rabi_freq[i_drive] = self.rotation_angle[i_drive] / integral_tmp

            # Store the updated Rabi frequencies of each drive
            self.rabi_frequency = rabi_freq

            print("Rabi frequency")
            print(self.rabi_frequency/(2*np.pi*10**6))



    def add_pulse_rise_time( self, fixed_pulse_duration=False ):
        """
        Adds finite rise time to the amplitude modulated pulse, using third order lowpass filtering.
        It modifies the pulse amplitude data, as well as the pulse duration (and derived variables, e.g. number of time points, of time slices, etc) to conserve the target rotation angle.

        Parameter
        ---------
        fixed_pulse_duration: Boolean value that indicates whether the addition of rise time must be done with or without modifying the total pulse duration (and hence the number of time points and other relevant values).
        """
        from math import floor, ceil
        from scipy.signal import lfilter
        _, _, _, a = self.arg

        ## Prepare filter coefficients
        order = 3
        a_filter = np.zeros( (self.nb_drives, order+1) )
        b_filter = np.zeros( (self.nb_drives, order+1) )


        ## Apply filtering to pulse shape
        for index_qubit in range(self.nb_drives):
            if fixed_pulse_duration:
                length_pulse_unfiltered = self.length_pulse
            else:
                length_pulse_unfiltered = compute_pulse_duration( self.rotation_angle[index_qubit], self.chosen_shape[index_qubit], self.rabi_frequency[index_qubit], self.arg )    # The filtering is done on the unfiltered pulse, so it must use the unfiltered length to compute

            b_tmp, a_tmp            = prepare_filter(max(self.omega)/(2*np.pi) - 400e6, max(self.omega0)/(2*np.pi) + 400e6, 2*self.nb_time_pts/length_pulse_unfiltered, order)
            a_filter[index_qubit,:]        = a_tmp
            b_filter[index_qubit,:]        = b_tmp

        filtered_shape_values = np.zeros( (self.nb_drives, self.nb_time_pts) )
        for i_shape in range(self.nb_drives):
            filtered_shape_values[i_shape] = lfilter( b_filter[i_shape,:], a_filter[i_shape,:], self.pulse_data[i_shape] )

        self.pulse_data   = filtered_shape_values


        ## Update pulse length and associated number of time points
        if not fixed_pulse_duration:
            # Only if the pulse is created in the fixed-Rabi, variable-pulse duration configuration
            length_pulse_array   = np.zeros(self.nb_drives)

            if 'gaussian' not in self.chosen_shape and 'half_gaussian' not in self.chosen_shape and 'half_gaussian2' not in self.chosen_shape and 'hermite' not in self.chosen_shape and 'UBURP' not in self.chosen_shape and 'REBURP' not in self.chosen_shape:
                for i_drive in range(self.nb_drives):
                    # Compute length of pulse i_drive
                    length_pulse = compute_pulse_duration_rt( self.rotation_angle[i_drive], self.chosen_shape[i_drive], self.rabi_frequency[i_drive], self.nb_time_pts, self.omega0[i_drive], max(self.omega0), self.arg )


                    # If the current shape is not a special shape, simply store the loaded pulse length in the proper array
                    length_pulse_array[i_drive] = length_pulse

            elif 'gaussian' in self.chosen_shape:
                for i_drive in range(self.nb_drives):
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], 'gaussian', self.rabi_frequency[i_drive], self.arg)

                    t0 = sqrt( - (sigma**2)/a * log(0.005) )

                    length_pulse_array[i_drive] = 2*t0

            elif 'half_gaussian' in self.chosen_shape or 'half_gaussian2' in self.chosen_shape:
                for i_drive in range(self.nb_drives):
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], self.chosen_shape[i_drive], self.rabi_frequency[i_drive], self.arg)
                    t0 = sqrt( - (sigma**2)/a * log(0.005) )
                    print(t0)

                    length_pulse_array[i_drive] = t0

            elif 'hermite' in self.chosen_shape:
                for i_drive in range(self.nb_drives):
                    t0_hermite = 50e-9
                    length_pulse = 2*t0_hermite
                    sigma = compute_pulse_duration(self.rotation_angle[i_drive], 'hermite', self.rabi_frequency[i_drive], self.arg)

                    length_pulse_array[i_drive] = length_pulse

            else:
                nb_time_pts_tmp    = self.nb_time_pts
                nb_time_slices_old = self.nb_time_slices

            if len(set(length_pulse_array)) > 1: # If not all the pulse durations are the same
                # Select longer required pulse time and set it as the duration of all other pulses
                length_pulse = max(length_pulse_array)

            else:
                length_pulse = length_pulse_array[0]


            ## Store updated values
            # Pulse variables
            self.length_pulse = length_pulse
            nb_time_pts_tmp   = self.nb_time_pts
            # self.nb_time_pts  = np.ceil(self.length_pulse / ( 1./( max(self.omega) ) / 100 )).astype(int)
            self.nb_time_pts  = np.ceil( 1.25*(self.length_pulse * max(self.omega/(2*np.pi)) * 100 ) ).astype(int) # 100 steps per period (increased by 25% for more accuracy. Can be lowered if memory space becomes a concern.)
            nb_time_slices_old = self.nb_time_slices  # For later comparison
            # Slice variables
            if 'UBURP' in list(self.chosen_shape) or 'REBURP' in list(self.chosen_shape):
                self.nb_time_slices    = 63    # Manual modification specific to BURP pulses
                self.length_slice      = int(self.length_pulse / self.nb_time_slices)
                self.nb_time_pts_slice = int(self.nb_time_pts / self.nb_time_slices)
            else:
                self.length_slice      = 1.0e-9       # A slice is precisely 1.0 ns
                self.nb_time_slices    = int(ceil(self.length_pulse / self.length_slice))
                self.nb_time_pts_slice = int(self.nb_time_pts / self.nb_time_slices)


            ## Interpolate pulse data to yield the proper number of time points
            if nb_time_pts_tmp != self.nb_time_pts:
                pulse_data = np.zeros( (self.nb_drives, self.nb_time_pts) )

                times_tmp = np.linspace(-self.length_pulse/2, self.length_pulse/2, nb_time_pts_tmp)
                times     = np.linspace(-self.length_pulse/2, self.length_pulse/2, self.nb_time_pts)

                for i_drive in range(self.nb_drives):
                    pulse_fn = interp1d( times_tmp, self.pulse_data[i_drive][:], assume_sorted=True, kind='cubic' )

                    for i in range(self.nb_time_pts):
                        pulse_data[i_drive][i] = pulse_fn(times[i])

                self.pulse_data = pulse_data


            ## Update arrays storing evolution of omega, omega0 and phase to account for extended pulse duration
            if self.nb_time_slices > nb_time_slices_old:
                omega_array_new  = np.ones( (self.nb_drives, self.nb_time_slices) )
                omega0_array_new = np.ones( (len(self.omega0), self.nb_time_slices) )
                phase_new        = np.ones( (self.nb_drives, self.nb_time_slices) )

                for i_drive in range(self.nb_drives):
                    # Create new structures of appropriate size
                    omega_array_new[i_drive][:nb_time_slices_old]  = self.omega[i_drive]
                    phase_new[i_drive,:nb_time_slices_old]         = self.phase[i_drive][0]
                for i_qubit in range(len(self.omega0)):
                    omega0_array_new[i_qubit][:nb_time_slices_old] = self.omega0[i_qubit]
                    # Complete them to account for the extended pulse duration due to the rise time
                    for i in range(self.nb_time_slices - nb_time_slices_old):
                        omega_array_new[i_drive][nb_time_slices_old+i]  = self.omega[i_drive]
                        omega0_array_new[i_drive][nb_time_slices_old+i] = self.omega0[i_drive]
                        phase_new[i_drive, nb_time_slices_old+i]        = self.phase[i_drive][0]

                # Store updated arrays
                self.omega_array  = omega_array_new
                self.omega0_array = omega0_array_new
                self.phase        = phase_new


    def add_frequency_shifts(self, omega0, correction_type):
        """
        Adds the effect of coupling between the qubits and the driving pulses on the qubits' resonance frequencies (AC Stark shift, Bloch-Siegert shift), and stores the time evolution of the resonance frequencies.
        Implements a correction scheme to choose the drive frequency of each of the pulse such that the shifts are minimised, and tracks the remaining shift over time.

        Parameters
        ----------
        omega0         : Array containing the resonance frequencies of the qubits
        correction_type: String specifying which shift(s) must be accounted for in the correction scheme. Both AC Stark and Bloch-Siegert shifts are however applied to the qubit's resonance frequencies in each case.
            'all'  : all shifts (AC Stark and Bloch-Siegert)
            'stark': only AC Stark shift
            'bs'   : only Bloch-Siegert
            'none' : no shift (no correction is applied)

        """
        from scipy.optimize import leastsq
        from scipy.integrate import simps
        from math import floor, ceil

        omega0_initial       = np.copy(omega0)
        omega_initial        = np.copy(self.omega)
        omega_previous_slice = np.copy(omega_initial)
        nb_qubits            = len(omega0)
        times                = np.linspace(0, self.length_pulse, self.nb_time_pts)

        # Begin storage
        omega_tmp  = np.zeros( (self.nb_drives, self.nb_time_slices) )
        omega0_tmp = np.zeros( (nb_qubits, self.nb_time_slices) )
        phi        = np.zeros( (self.nb_drives, self.nb_time_slices) )
        # End storage

        for i_slice in range( self.nb_time_slices ):
            i_shape_value = int(i_slice * self.nb_time_pts_slice + int(floor(0.5*self.nb_time_pts_slice)))
            shaped_rabi = np.multiply( self.rabi_frequency, self.pulse_data[:,i_shape_value] )

            # Define arguments of objective function
            args_freq_shift = (omega0_initial, shaped_rabi, correction_type)

            # Optimise drive frequency so that shifted resonance frequencies are equal to the drive frequencies
            omega, _, infordict, mesg, ier = leastsq( objective_fn_freq_shift, x0=omega_previous_slice, args=args_freq_shift, full_output=True )

            # Check the optimisation has succeeded
            if ier > 4:
                raise RuntimeWarning("The optimisation of the drive frequency to compensate the frequency shift has not converged. Results may be inaccurate.")

            # Compute shifted resonance frequencies
            omega0 = omega0_initial + iterate_freq_shifts(omega0_initial, omega, shaped_rabi, 6, nb_qubits, self.nb_drives, i_slice, self.nb_time_slices, 1, test=True, correction_type='all')


            omega_previous_slice = np.copy(omega)

            # Begin storage
            for i in range(nb_qubits):
                omega0_tmp[i][i_slice] = omega0[i]
            for i in range(self.nb_drives):
                omega_tmp[i][i_slice]  = omega[i]
            # End storage

            # Compute phase offset to ensure the pulse is continuous (remove phase error)
            omega_integr = np.ones( (self.nb_drives,self.nb_time_pts) )
            for i in range(self.nb_time_slices):
                index_low = i * self.nb_time_pts_slice
                index_high = (i+1) * self.nb_time_pts_slice
                for i_drive in range(self.nb_drives):
                    omega_integr[i_drive][index_low:index_high] = omega_tmp[i_drive][i]

            if i_slice >= 1:
                for i in range(self.nb_drives):
                    # Phase correction is currently 0
                    phi[i][i_slice] = 0

        self.omega_array  = omega_tmp
        self.omega0_array = omega0_tmp
        self.phase        = self.phase + phi


    def plot_frequency_evolution(self, fig_nb1, fig_nb2):
        """
        Plots the evolution of both the resonance frequencies of the simulated qubits and the frequencies of each of the driving pulses., as well as the phase of each pulse.

        Parameters
        ----------
        fig_nb1: number of the matplotlib.figure object in which the evolution of the frequencies is to be plotted.
        fig_nb2: number of the matplotlib.figure object in which the evolution of the phases is to be plotted.
        """
        plt.figure(fig_nb1)

        for i_drive in range(self.nb_drives):
            ax = plt.subplot(self.nb_drives, 1, i_drive+1)
            ax.ticklabel_format(useOffset=False)
            plt.plot( self.omega_array[i_drive]/(2*np.pi*10**9), label='Drive freq' )
            plt.plot( self.omega0_array[i_drive]/(2*np.pi*10**9), label='Resonance freq' )
            plt.xlabel("Time slice [/]", fontsize=20)
            plt.ylabel("Frequency [GHz]", fontsize=20)
            plt.title("Qubit "+str(i_drive))

            plt.grid()

        plt.legend(fontsize=20, loc='best')

        plt.figure(fig_nb2)
        for i_drive in range(self.nb_drives):
            plt.subplot(self.nb_drives, 1, i_drive+1)
            plt.plot( self.phase[i_drive] )
            plt.xlabel("Time slice [/]", fontsize=20)
            plt.ylabel("Phase [/]", fontsize=20)
            plt.title("Qubit "+str(i_drive))

            plt.grid()


    def plot_pulse_data(self, fig_nb):
        """
        Plots the enveloppe of the driving pulse in the time domain.

        Parameters
        ----------
        fig_nb: number of the matplotlib.figure object in which the evolution is to be plotted.
        """
        plt.figure(fig_nb)
        for i in range(self.nb_drives):
            plt.subplot(self.nb_drives,1,i+1)
            plt.plot(self.pulse_data[i], label='p'+str(i))
            plt.grid()
            plt.legend()

        plt.show()



class n_qubits_simulation():
    """
    General class representing a time-evolution simulation of a linear array of qubits driven my multiple microwave pulses at the same time.
    """

    def __init__(self, nb_qubits, omega0, rotating_frame=True, rate_coupling=100e3):
        """
        Constructor of the n_qubit_simulation class. An exchange coupling of 100 kHz is simulated by default.

        Parameters
        ----------
        nb_qubits: Number of qubits considered in the simulation
        omega0   : Array containing the resonance frequencies of the qubits
        rotating_frame: Boolean indicating whether the simulation is to be done in the rotating frame of not
        rate_coupling: rate of the exchange coupling between adjacent qubits, expressed in Hz.

        """
        self.solver_obj  = me.VonNeumann(2**nb_qubits)
        self.nb_qubits   = nb_qubits
        self.omega0      = omega0
        self.t_start     = 0
        self.t_end       = 0
        self.total_nb_time_pts = 0

        # Spin Hamiltonian
        if not rotating_frame:
            for i_qubit in range(nb_qubits):
                H_0 = compute_multiqubit_hamil( qt.Qobj( [ [0,0],[0,1] ] ) * self.omega0[i_qubit]/(2*np.pi), nb_qubits, i_qubit )
                self.solver_obj.add_H0( H_0 )

        # Pulse Hamiltonian
        H_mw    = list()
        for i_qubit in range(nb_qubits):
            tmp = compute_multiqubit_hamil( qt.sigmap(), nb_qubits, i_qubit )
            H_mw.append(tmp)

        self.H_mw = H_mw

        # Exchange coupling Hamiltonian
        for i_coupl in range(nb_qubits-1):
            tmp_coupl_x = compute_multiqubit_hamil( qt.tensor(qt.sigmax(), qt.sigmax()), nb_qubits-1, i_coupl )
            tmp_coupl_y = compute_multiqubit_hamil( qt.tensor(qt.sigmay(), qt.sigmay()), nb_qubits-1, i_coupl )
            tmp_coupl_z = compute_multiqubit_hamil( qt.tensor(qt.sigmaz(), qt.sigmaz()), nb_qubits-1, i_coupl )

            tmp_coupl = (rate_coupling/4) * (tmp_coupl_x + tmp_coupl_y + tmp_coupl_z)

            self.solver_obj.add_H0( tmp_coupl )


    def add_mw_pulse(self, mw_pulse, rotating_frame=True, RWA=False):
        """
        Adds a microwave pulse to the simulation, while making the rotating-wave approximation (RWA) or not.

        Parameters
        ----------
        mw_pulse      : shaped_pulse class instance representing the drive applied to the linear array of qubits
        rotating_frame: Boolean indicating whether the pulses will be added to the solver in the rotating frame or in the laboratory frame
        RWA           : Boolean indicating whether to make the RWA or not (True = RWA applied, False = RWA not applied)

        """
        from math import floor, ceil

        # Print pulse details
        print( "Pulse duration: "+str( (mw_pulse.length_pulse*10**9) )+" ns" )
        print( "Number of points in pulse: "+str( mw_pulse.nb_time_pts ) )
        print( "Number of slices in pulse: "+str(mw_pulse.nb_time_slices) )
        print( "Number of points in each slice: "+str( mw_pulse.nb_time_pts_slice ) )

        for i_slice in range( mw_pulse.nb_time_slices ):
            i_shape_value = int(i_slice * mw_pulse.nb_time_pts_slice + int(floor(0.5*mw_pulse.nb_time_pts_slice)))
            rabi_f        = mw_pulse.rabi_frequency/(2*np.pi)
            shaped_rabi   = np.multiply( rabi_f, mw_pulse.pulse_data[:,i_shape_value] )

            if i_slice < mw_pulse.nb_time_slices:
                # Pulses are still on
                for i_pulse in range(mw_pulse.nb_drives):
                    # For each pulse

                    for i_qubit in range(self.nb_qubits) :
                        # Add a the effect of the pulse on each qubit

                        if rotating_frame:
                            pulse = me.microwave_RWA()

                            pulse.init( shaped_rabi[i_pulse]*np.pi, mw_pulse.phase[i_pulse][i_slice], (mw_pulse.omega_array[i_pulse][i_slice] - mw_pulse.omega0_array[i_qubit][i_slice])/(2*np.pi), i_slice*mw_pulse.length_slice, min( (i_slice+1)*mw_pulse.length_slice, mw_pulse.length_pulse ), self.H_mw[i_qubit] )

                            self.solver_obj.add_H1_MW_RF_obj_RWA( pulse )

                        else:
                            pulse = me.microwave_pulse()
                            pulse.init_normal( shaped_rabi[i_pulse]*np.pi, mw_pulse.phase[i_pulse][i_slice], mw_pulse.omega_array[i_pulse][i_slice]/(2*np.pi*10**9), i_slice*mw_pulse.length_slice, min( (i_slice+1)*mw_pulse.length_slice, mw_pulse.length_pulse ), self.H_mw[i_qubit] )

                            self.solver_obj.add_H1_MW_RF_obj( pulse )

                        if RWA == False:

                            if rotating_frame:
                                pulse_cr = me.microwave_RWA()

                                pulse_cr.init( shaped_rabi[i_pulse]*np.pi, -mw_pulse.phase[i_pulse][i_slice], (-mw_pulse.omega_array[i_pulse][i_slice] - mw_pulse.omega0_array[i_qubit][i_slice])/(2*np.pi), i_slice*mw_pulse.length_slice, min( (i_slice+1)*mw_pulse.length_slice, mw_pulse.length_pulse ), self.H_mw[i_qubit] )

                                self.solver_obj.add_H1_MW_RF_obj_RWA( pulse_cr )


        # Set final pulse length and number of time points
        self.t_end             = self.t_end + mw_pulse.length_pulse
        self.total_nb_time_pts = self.total_nb_time_pts + mw_pulse.nb_time_pts


    def calc_time_evolution(self, psi0, t_start, t_end):
        """
        Compute the time evolution of the system.

        Parameters
        ----------
        psi0   : Array containing the initial density matrix of the system
        t_start: Time at which the simulation starts
        t_end  : Time at which the simulation ends

        """
        self.solver_obj.calculate_evolution(psi0, t_start, t_end, self.total_nb_time_pts)


    def get_state_fidelity(self, target):
        """
        Returns the state fidelity of the operation compared to a target state.

        Parameters
        ----------
        target: target state to compare with. target is an instance of QuTip Quobj.

        Returns
        -------
        fid: fidelity value (ranging between 0 and 1)
        """
        # Recover all density matrices of the time evolution
        tmp_dm = self.solver_obj.get_all_density_matrices()

        # Select the last non-NaN of those
        isNan = True
        i = 2
        while isNan == True:
            end_state = tmp_dm[-i]
            if np.isnan(end_state).any():
                print("\n"+str(i)+" is NaN. Switching to previous one.\n")
                i+=1
            else:
                isNan= False

        # Compute fidelity
        fid = qt.fidelity( qt.Qobj( end_state, dims=[ [2 for i in range(self.nb_qubits)],[2 for i in range(self.nb_qubits)] ] ), qt.ket2dm(target) )

        return fid


    def get_unitary_fidelity(self, target_unitary):
        """
        Returns the average gate fidelity of the process (unitary fidelity). The formula used is taken from Pedersen et al. (2007) (https://www.sciencedirect.com/science/article/pii/S0375960107003271?via%3Dihub)

        Parameters
        ----------
        target_unitary: target unitary to compare with. target_unitary is an instance of QuTip Quobj.

        Returns
        -------
        fid: value of unitary fidelity (ranging between 0 and 1)
        """
        import scipy as sp

        unitary = self.solver_obj.get_unitary()

        # Dimension of the Hilbert spaceover which the unitary operates
        dim = len(target_unitary[0])
        temp_m = np.matmul( sp.conjugate( sp.transpose(target_unitary) ), unitary )
        fid = (sp.trace(np.matmul(temp_m,sp.conjugate(sp.transpose(temp_m))))+np.abs(sp.trace(temp_m))**2.)/(dim*(dim+1.))

        return np.real(fid)


    def plot_expectations(self, fig_nb):
        """
        Plots the X, Y and Z operator expectation values for each qubit in the simulation.
        Parameters
        ----------
        fig_nb: number of the matplotlib.figure object in which the evolution is to be plotted.
        """
        X = list()
        Y = list()
        Z = list()

        for i_qubit in range(self.nb_qubits):
            x_tmp = compute_multiqubit_hamil( qt.sigmax(), self.nb_qubits, i_qubit )
            y_tmp = compute_multiqubit_hamil( qt.sigmay(), self.nb_qubits, i_qubit )
            z_tmp = compute_multiqubit_hamil( qt.sigmaz(), self.nb_qubits, i_qubit )

            X.append(x_tmp)
            Y.append(y_tmp)
            Z.append(z_tmp)

        operators = np.array( X+Y+Z, dtype=np.complex )

        expect = self.solver_obj.return_expectation_values(operators)

        plt.figure(fig_nb)

        times = np.linspace( self.t_start, self.t_end, self.total_nb_time_pts ) * (10**9)

        ax_x = plt.subplot(3,1,1)
        ax_x.tick_params(labelsize=18)
        ax_x.set_ylabel("X [/]", fontsize=20)
        ax_x.set_ylim(-1.05, 1.05)
        ax_x.grid()

        ax_y = plt.subplot(3,1,2, sharex=ax_x)
        ax_y.tick_params(labelsize=18)
        ax_y.set_ylabel("Y [/]", fontsize=20)
        ax_y.set_ylim(-1.05, 1.05)
        ax_y.grid()

        ax_z = plt.subplot(3,1,3, sharex=ax_x)
        ax_z.tick_params(labelsize=18)
        ax_z.set_xlabel("Time [ns]", fontsize=20)
        ax_z.set_ylabel("Z [/]", fontsize=20)
        ax_z.set_ylim(-1.05, 1.05)
        ax_z.grid()

        for i_qubit in range(self.nb_qubits):
            ax_x.plot( times, expect[i_qubit][:-1]  )
            ax_y.plot( times, expect[self.nb_qubits + i_qubit][:-1] )
            ax_z.plot( times, expect[2*self.nb_qubits + i_qubit][:-1], label='Qubit '+str(i_qubit) )

        plt.legend(fontsize=20, loc='best')


    def plot_bloch(self, fig_nb):
        """
        Plots the evolution of the simulation on the Bloch sphere. The end state of each qubit is indicated by an arrow
        Parameters
        ----------
        fig_nb: number of the matplotlib.figure object in which the evolution is to be plotted.
        """
        fig = plt.figure(fig_nb)
        ax = plt.subplot(projection='3d')
        ax.axis('square')

        b = qt.Bloch(fig=fig, axes=ax)

        step = self.total_nb_time_pts / 100     # 100 points in the plot
        expect = self.return_expectations()

        for i_qubit in range(self.nb_qubits):
            b.add_points( [ expect[i_qubit][::step], expect[self.nb_qubits+i_qubit][::step], expect[2*self.nb_qubits+i_qubit][::step] ] , 'l')

            b.add_vectors([expect[i_qubit][-1], expect[self.nb_qubits+i_qubit][-1], expect[2*self.nb_qubits+i_qubit][-1]])

        b.render(fig=fig, axes=ax)

        plt.suptitle("Bloch sphere representation of the qubits trajectory", fontsize=25)



    def return_expectations(self):
        """
        Returns the X, Y and Z operator expectation values for each qubit in the simulation.
        The values are ordered in a single array in the following way:
        expect = ( X_qubit-0, X_qubit-1, ..., X_qubit-n, Y_qubit-0, Y_qubit-1, ..., Y_qubit-n, Z_qubit-0, Z_qubit-1, ..., Z_qubit-n )

        The expectations are computed using the dedicated method built-in DM_solver.
        """
        X = list()
        Y = list()
        Z = list()

        for i_qubit in range(self.nb_qubits):
            x_tmp = compute_multiqubit_hamil( qt.sigmax(), self.nb_qubits, i_qubit )
            y_tmp = compute_multiqubit_hamil( qt.sigmay(), self.nb_qubits, i_qubit )
            z_tmp = compute_multiqubit_hamil( qt.sigmaz(), self.nb_qubits, i_qubit )

            X.append(x_tmp)
            Y.append(y_tmp)
            Z.append(z_tmp)

        operators = np.array( X+Y+Z, dtype=np.complex )

        expect = self.solver_obj.return_expectation_values(operators)

        return expect


## Main
if __name__ == '__main__':
    rotation_angle = np.array( [np.pi/2, np.pi/2] )
    rabi_frequency = np.array( [15e6, 15e6] ) * 2*np.pi
    phase          = np.array( [0.0, 0.0] )
    arg            = (0.13, 7, 8, 1.0) # Free parameters of the optimised blackman, kaiser 0, kaiser 1 and gaussian shapes, respectively
    rotating_frame = True

    # Create simulation
    sim = n_qubits_simulation( 2, np.array( [10.0e9, 10.1e9] ) * 2*np.pi, rotating_frame=rotating_frame )

    # Initialise driving pulse (fixed Rabi frequency)
    pulse = shaped_pulse( sim.nb_qubits, 2, rotation_angle, rabi_frequency, np.array( ['rectangle', 'rectangle'] ), np.array( [10.0e9, 10.1e9] ) * 2*np.pi, np.array( [10.0e9, 10.1e9] ) * 2*np.pi, phase, arg, rise_time=True, fixed_pulse_dur=False )

    # Initialise driving pulse (fixed pulse duration)
    # pulse = shaped_pulse( sim.nb_qubits, 2, rotation_angle, 50e-09, np.array( ['hamming', 'hamming'] ), np.array( [10.0e9, 10.1e9] ) * 2*np.pi, np.array( [10.0e9, 10.1e9] ) * 2*np.pi, phase, arg, rise_time=True, fixed_pulse_dur=True )

    # Add modifications and add the pulse to the simulation
    pulse.add_frequency_shifts(sim.omega0, 'none')
    sim.add_mw_pulse(pulse, rotating_frame=rotating_frame, RWA=True)

    # Compute time evolution
    qubit = np.array( list( qt.ket2dm(qt.tensor( qt.basis(2,0), qt.basis(2,0) ) ) ) )[:,0]
    sim.calc_time_evolution( qubit, 0, sim.t_end )

    # Compute state fidelity (X rotation)
    if rotation_angle[0] == np.pi/2:
        tmp = (qt.basis(2,0) - 1j*qt.basis(2,1)).unit()    # X rotation - pi/2
    elif rotation_angle[0] == np.pi:
        tmp = qt.basis(2,1)                                # X rotation - pi
    elif rotation_angle[0] == 2*np.pi:
        tmp = qt.basis(2,0)                                # X rotation - 2pi
    else:
        raise NotImplementedError("Thisrotation angle does not have a tagert state built in. You need to mnually specify it.")

    target = qt.tensor( tmp, tmp )
    state_fid = sim.get_state_fidelity(target)

    # Compute unitary fidelity
    target_unitary = compute_multiqubit_target_unitary( rotation_angle, np.array( [ [1,0,0], [1,0,0] ] ) )  # X rotation
    avg_fid = sim.get_unitary_fidelity( np.array(list(target_unitary), dtype=np.complex)[:,0] )

    print("Average gate fidelity: "+str( (avg_fid*100) ))

    if rotation_angle[0] == np.pi/2:
        angle_str = 'piOver2'
    elif rotation_angle[0] == np.pi:
        angle_str = 'pi'
    elif rotation_angle[0] == 2*np.pi:
        angle_str = '2pi'

    sim.plot_expectations(4)
    plt.suptitle( "Fidelity: "+str( '%.5f' %(avg_fid*100) )+"%, Rotation: "+angle_str+", Rabi: "+str( '%.1f' %(rabi_frequency[0]/(2*np.pi*10**6)) )+" MHz, Shape: "+pulse.chosen_shape[0], fontsize=25 )

    # sim.plot_bloch(5)
    # pulse.plot_frequency_evolution(6,7)
    # pulse.plot_pulse_data(8)

    plt.show()

