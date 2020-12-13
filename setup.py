#!/usr/bin/env python

from distutils.core import setup

setup(name='multiQ',
	version='1.0',
	author='Vincent A Bejach',
	modules=[ 'obj_driving', 'pulseshapes', 'qubit_utilities', 'pulse_duration' ]
	)
