"""Functions that can be used for model fitting.

NOTES
-----
- FOOOF currently (only) uses the exponential and gaussian functions.
- Linear & Quadratic functions are from previous versions of FOOOF.
    - They are left available for easy swapping back in, if desired.
"""

import numpy as np

from fooof.core.errors import InconsistentDataError

###################################################################################################
###################################################################################################

def gaussian_function(xs, *params):
    """Gaussian fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.

    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """

    ys = np.zeros_like(xs)

    for ctr, hgt, wid in zip(*[iter(params)] * 3):

        ys = ys + hgt * np.exp(-(xs-ctr)**2 / (2*wid**2))

    return ys



def expo_nk_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component without a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, exp) that define Lorentzian function:
        y = 10^off * (1/(x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function, without a knee.
    """

    offset, exp = params
    ys = offset - np.log10(xs**exp)

    return ys

def expo_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component with a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = 10^offset * (1/(knee + x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """

    offset, knee, exp = params
    ys = offset - np.log10(1+10**(exp*(np.log10(xs)-np.log10(knee))))

    return ys


def two_exp(xs, *params):
    """2 exponents fitting function, for fitting aperiodic component with two exponents and 'knee'

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, 1st exponent, knee frequency, 2nd exponent, flattening frequency) that define the function:

        y = offset - log10((x/knee)^n1 + (x/knee)^n2) 

    Returns
    -------
    ys : 1d array
        Output values for 2 exponents function.
    """
    
    knee_off, knee, exp1, exp2 = params
    ys = knee_off - np.log10(10**(exp1*(np.log10(xs)-np.log10(knee))) + 10**(exp2*(np.log10(xs)-np.log10(knee)))) 
    return ys


def two_exp_flattening(xs, *params):
    """2 exponents + flattening fitting function, for fitting aperiodic component with two exponents, 'knee', and flattening part until 500 Hz.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, 1st exponent, knee frequency, 2nd exponent, flattening frequency) that define the function:

        y = offset - log10((x/knee)^n1 + (x/knee)^n2) + log10((x/knee)^n2 + flattening_freq)


    Returns
    -------
    ys : 1d array
        Output values for 2 exponents + flattening function.
    """

    
    offset, knee, flat_fr, exp1, exp2 = params    # o = power at z1;  z1 = knee_freq; z2 = flat_fr; d1 = exp1, d2 = exp2
    if knee > flat_fr:
        knee_copy = knee.copy()
        flat_fr_copy = flat_fr.copy()
        knee = flat_fr_copy
        flat_fr = knee_copy
    ys = offset - exp2*((np.log10(flat_fr)-np.log10(knee))/2) + np.log10(10**((exp2/(-2))*(np.log10(flat_fr)-np.log10(xs))) + 10**(((exp2*(np.log10(xs)-np.log10(flat_fr)))/(-2)))) - np.log10(10**(((exp2*(np.log10(knee)-np.log10(xs)))/(-2))) + 10**((exp1-(exp2/2))*(np.log10(xs)-np.log10(knee))))
    return ys    



def three_exponents(xs, *params):

    """ 3 exponents fitting function, for fitting aperiodic component with 3 exponents, 'knee', and flattening part until 500 Hz.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, 1st exponent, knee frequency, 2nd exponent, flattening frequency) that define the function:

        Y = b - log((xs/knee)^(-s2) + (xs/knee)^(s3)) + log((xs/flat_freq)^(-s1) + (xs/flat_freq)^s2)
        where:
        exp1 = s1 - s2
        exp2 = s1 + s3
        exp3 = s3 - s2 
    Returns
    -------
    ys : 1d array
        Output values for 3 exponents function.
    """

    offset, knee , flat_freq, exp1, exp2, exp3 = params    # offset = power at knee frequency  
    if knee > flat_freq:
        knee_copy = knee.copy()
        flat_freq_copy = flat_freq.copy()
        knee = flat_freq_copy
        flat_freq = knee_copy
    ys = offset - exp2*((np.log10(flat_freq)-np.log10(knee))/2) + np.log10(10**((exp3-(exp2/2))*(np.log10(flat_freq)-np.log10(xs))) + 10**(((exp2*(np.log10(xs)-np.log10(exp2)))/(-2)))) - np.log10(10**(((exp2*(np.log10(knee)-np.log10(xs)))/(-2))) + 10**((exp1-(exp2/2))*(np.log10(xs)-np.log10(knee))))
    return ys


def linear_function(xs, *params):
    """Linear fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define linear function.

    Returns
    -------
    ys : 1d array
        Output values for linear function.
    """

    offset, slope = params
    ys = offset + (xs*slope)

    return ys


def quadratic_function(xs, *params):
    """Quadratic fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define quadratic function.

    Returns
    -------
    ys : 1d array
        Output values for quadratic function.
    """

    offset, slope, curve = params
    ys = offset + (xs*slope) + ((xs**2)*curve)

    return ys


def get_pe_func(periodic_mode):
    """Select and return specified function for periodic component.

    Parameters
    ----------
    periodic_mode : {'gaussian'}
        Which periodic fitting function to return.

    Returns
    -------
    pe_func : function
        Function for the periodic component.

    Raises
    ------
    ValueError
        If the specified periodic mode label is not understood.

    """

    if periodic_mode == 'gaussian':
        pe_func = gaussian_function
    else:
        raise ValueError("Requested periodic mode not understood.")

    return pe_func


def get_ap_func(aperiodic_mode):
    """Select and return specified function for aperiodic component.

    Parameters
    ----------
    aperiodic_mode : {'fixed', 'knee'}
        Which aperiodic fitting function to return.

    Returns
    -------
    ap_func : function
        Function for the aperiodic component.

    Raises
    ------
    ValueError
        If the specified aperiodic mode label is not understood.
    """

    if aperiodic_mode == 'fixed':
        ap_func = expo_nk_function
    elif aperiodic_mode == '2exp':
        ap_func = two_exp
    elif aperiodic_mode == 'flat_1exp':
        ap_func = expo_function
    elif aperiodic_mode == '2exp_flat':
        ap_func = two_exp_flattening 
    elif aperiodic_mode == '3exp':
        ap_func = three_exponents
    
    else:
        raise ValueError("Requested aperiodic mode not understood.")

    return ap_func


def infer_ap_func(aperiodic_params):
    """Infers which aperiodic function was used, from parameters.

    Parameters
    ----------
    aperiodic_params : list of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee' m 'knee_1exp'}
        Which kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    InconsistentDataError
        If the given parameters are inconsistent with any available aperiodic function.
    """

    if len(aperiodic_params) == 2:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) ==4:
        aperiodic_mode = '2exp'
    elif len(aperiodic_params) == 3:
        aperiodic_mode = 'flat_1exp'
    elif len(aperiodic_params) == 5:
        aperiodic_mode = '2exp_flat'
    elif len(aperiodic_params) == 6:
        aperiodic_mode = '3exp'

    
    else:
        raise InconsistentDataError("The given aperiodic parameters are "
                                    "inconsistent with available options.")

    return aperiodic_mode
