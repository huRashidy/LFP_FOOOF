"""FOOOF Object - base object which defines the model.

Private Attributes
==================
Private attributes of the FOOOF object are documented here.

Data Attributes
---------------
_spectrum_flat : 1d array
    Flattened power spectrum, with the aperiodic component removed.
_spectrum_peak_rm : 1d array
    Power spectrum, with peaks removed.

Model Component Attributes
--------------------------
_ap_fit : 1d array
    Values of the isolated aperiodic fit.
_peak_fit : 1d array
    Values of the isolated peak fit.

Internal Settings Attributes
----------------------------
_ap_percentile_thresh : float
    Percentile threshold for finding peaks above the aperiodic component.
_ap_guess : list of [float, float, float]
    Guess parameters for fitting the aperiodic component.
_ap_bounds : tuple of tuple of float
    Upper and lower bounds on fitting aperiodic component.
_cf_bound : float
    Parameter bounds for center frequency when fitting gaussians.
_bw_std_edge : float
    Bandwidth threshold for edge rejection of peaks, in units of gaussian standard deviation.
_gauss_overlap_thresh : float
    Degree of overlap (in units of standard deviation) between gaussian guesses to drop one.
_gauss_std_limits : list of [float, float]
    Peak width limits, converted to use for gaussian standard deviation parameter.
    This attribute is computed based on `peak_width_limits` and should not be updated directly.
_maxfev : int
    The maximum number of calls to the curve fitting function.
_error_metric : str
    The error metric to use for post-hoc measures of model fit error.

Run Modes
---------
_debug : bool
    Whether the object is set in debug mode.
    This should be controlled by using the `set_debug_mode` method.
_check_data, _check_freqs : bool
    Whether to check added inputs for incorrect inputs, failing if present.
    Frequency data is checked for linear spacing.
    Power values are checked for data for NaN or Inf values.
    These modes default to True, and can be controlled with the `set_check_modes` method.

Code Notes
----------
Methods without defined docstrings import docs at runtime, from aliased external functions.
"""

import warnings
from copy import deepcopy
import scipy
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from fooof.core.utils import unlog
from fooof.core.items import OBJ_DESC
from fooof.core.info import get_indices
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm
from fooof.core.modutils import copy_doc_func_to_method
from fooof.core.utils import group_three, check_array_dim
from fooof.core.funcs import gaussian_function, get_ap_func, infer_ap_func
from fooof.core.jacobians import jacobian_gauss
from fooof.core.errors import (FitError, NoModelError, DataError,
                               NoDataError, InconsistentDataError)
from fooof.core.strings import (gen_settings_str, gen_results_fm_str,
                                gen_issue_str, gen_width_warning_str)
from fooof.utils import  interpolate_spectrum

from fooof.plts.fm import plot_fm
from fooof.utils.data import trim_spectrum
from fooof.utils.params import compute_gauss_std
from fooof.data import FOOOFSettings, FOOOFRunModes, FOOOFMetaData, FOOOFResults
from fooof.data.conversions import model_to_dataframe
from fooof.sim.gen import gen_freqs, gen_aperiodic, gen_periodic, gen_model
from statsmodels.tools.eval_measures import aic_sigma , bic_sigma , aicc_sigma , hqic_sigma
import matplotlib.pyplot as plt
###################################################################################################
###################################################################################################

class FOOOF():
    """Model a physiological power spectrum as a combination of aperiodic and periodic components.

    WARNING: FOOOF expects frequency and power values in linear space.

    Passing in logged frequencies and/or power spectra is not detected,
    and will silently produce incorrect results.

    Parameters
    ----------
    peak_width_limits : tuple of (float, float), optional, default: (0.5, 12.0)
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
    max_n_peaks : int, optional, default: inf
        Maximum number of peaks to fit.
    min_peak_height : float, optional, default: 0
        Absolute threshold for detecting peaks.
        This threshold is defined in absolute units of the power spectrum (log power).
    peak_threshold : float, optional, default: 2.0
        Relative threshold for detecting peaks.
        This threshold is defined in relative units of the power spectrum (standard deviation).
    aperiodic_mode : {'fixed', 'knee'}
        Which approach to take for fitting the aperiodic component.
    verbose : bool, optional, default: True
        Verbosity mode. If True, prints out warnings and general status updates.

    Attributes
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectrum : 1d array
        Power values, stored internally in log10 scale.
    freq_range : list of [float, float]
        Frequency range of the power spectrum, as [lowest_freq, highest_freq].
    freq_res : float
        Frequency resolution of the power spectrum.
    fooofed_spectrum_ : 1d array
        The full model fit of the power spectrum, in log10 scale.
    aperiodic_params_ : 1d array
        Parameters that define the aperiodic fit. As [Offset, (Knee), Exponent].
        The knee parameter is only included if aperiodic component is fit with a knee.
    peak_params_ : 2d array
        Fitted parameter values for the peaks. Each row is a peak, as [CF, PW, BW].
    gaussian_params_ : 2d array
        Parameters that define the gaussian fit(s).
        Each row is a gaussian, as [mean, height, standard deviation].
    r_squared_ : float
        R-squared of the fit between the input power spectrum and the full model fit.
    error_ : float
        Error of the full model fit.
    n_peaks_ : int
        The number of peaks fit in the model.
    has_data : bool
        Whether data is loaded to the object.
    has_model : bool
        Whether model results are available in the object.

    Notes
    -----
    - Commonly used abbreviations used in this module include:
      CF: center frequency, PW: power, BW: Bandwidth, AP: aperiodic
    - Input power spectra must be provided in linear scale.
      Internally they are stored in log10 scale, as this is what the model operates upon.
    - Input power spectra should be smooth, as overly noisy power spectra may lead to bad fits.
      For example, raw FFT inputs are not appropriate. Where possible and appropriate, use
      longer time segments for power spectrum calculation to get smoother power spectra,
      as this will give better model fits.
    - The gaussian params are those that define the gaussian of the fit, where as the peak
      params are a modified version, in which the CF of the peak is the mean of the gaussian,
      the PW of the peak is the height of the gaussian over and above the aperiodic component,
      and the BW of the peak, is 2*std of the gaussian (as 'two sided' bandwidth).
    """
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.01,
                 peak_threshold=2.0, aperiodic_mode='fixed', verbose=True):
        """Initialize object with desired settings."""

        # Set input settings
        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.peak_threshold = peak_threshold
        self.aperiodic_mode = aperiodic_mode
        self.verbose = verbose

        ## PRIVATE SETTINGS
        # Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        #   Points are selected at a low percentile value to restrict to non-peak points
        self._ap_percentile_thresh = 0.025
        # Guess parameters for aperiodic fitting, [offset, knee, exponent]
        #   If offset guess is None, the first value of the power spectrum is used as offset guess
        #   If exponent guess is None, the abs(log-log slope) of first & last points is used
        self._ap_guess = (None, 4, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here, if desired
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        #    offset,knee_param, exp1, exp2,f  = params
        self._ap_bounds = ((-np.inf,0,0, 0,0,0), (np.inf,np.inf,np.inf,np.inf,np.inf, np.inf))
        # Threshold for how far a peak has to be from edge to keep.
        #   This is defined in units of gaussian standard deviation
        self._bw_std_edge = 1.0
        # Degree of overlap between gaussians for one to be dropped
        #   This is defined in units of gaussian standard deviation
        self._gauss_overlap_thresh = 0.7
        # Parameter bounds for center frequency when fitting gaussians, in terms of +/- std dev
        self._cf_bound = 1.5
        # The error metric to calculate, post model fitting. See `_calc_error` for options
        #   Note: this is for checking error post fitting, not an objective function for fitting
        self._error_metric = 'AP'

        ## PRIVATE CURVE_FIT SETTINGS
        # The maximum number of calls to the curve fitting function
        self._maxfev = 5000
        # The tolerance setting for curve fitting (see scipy.curve_fit - ftol / xtol / gtol)
        #   Here reduce tolerance to speed fitting. Set value to 1e-8 to match curve_fit default
        self._tol = 0.00001

        ## RUN MODES
        # Set default debug mode - controls if an error is raised if model fitting is unsuccessful
        self._debug = False
        # Set default data checking modes - controls which checks get run on input data
        #   check_freqs: checks the frequency values, and raises an error for uneven spacing
        self._check_freqs = True
        #   check_data: checks the power values and raises an error for any NaN / Inf values
        self._check_data = True

        # Set internal settings, based on inputs, and initialize data & results attributes
        self._reset_internal_settings()
        self._reset_data_results(True, True, True)


    @property
    def has_data(self):
        """Indicator for if the object contains data."""

        return True if np.any(self.power_spectrum) else False


    @property
    def has_model(self):
        """Indicator for if the object contains a model fit.

        Notes
        -----
        This check uses the aperiodic params, which are:

        - nan if no model has been fit
        - necessarily defined, as floats, if model has been fit
        """

        return True if not np.all(np.isnan(self.aperiodic_params_)) else False


    @property
    def n_peaks_(self):
        """How many peaks were fit in the model."""

        return self.peak_params_.shape[0] if self.has_model else None


    def _reset_internal_settings(self):
        """Set, or reset, internal settings, based on what is provided in init.

        Notes
        -----
        These settings are for internal use, based on what is provided to, or set in `__init__`.
        They should not be altered by the user.
        """

        # Only update these settings if other relevant settings are available
        if self.peak_width_limits:

            # Bandwidth limits are given in 2-sided peak bandwidth
            #   Convert to gaussian std parameter limits
            self._gauss_std_limits = tuple(bwl / 2 for bwl in self.peak_width_limits)

        # Otherwise, assume settings are unknown (have been cleared) and set to None
        else:
            self._gauss_std_limits = None


    def _reset_data_results(self, clear_freqs=False, clear_spectrum=False, clear_results=False):
        """Set, or reset, data & results attributes to empty.

        Parameters
        ----------
        clear_freqs : bool, optional, default: False
            Whether to clear frequency attributes.
        clear_spectrum : bool, optional, default: False
            Whether to clear power spectrum attribute.
        clear_results : bool, optional, default: False
            Whether to clear model results attributes.
        """

        if clear_freqs:
            self.freqs = None
            self.freq_range = None
            self.freq_res = None

        if clear_spectrum:
            self.power_spectrum = None

        if clear_results:

            self.aperiodic_params_ = np.array([np.nan] * \
                (2 if self.aperiodic_mode == 'fixed' else 3))
            self.gaussian_params_ = np.empty([0, 3])
            self.peak_params_ = np.empty([0, 3])
            self.r_squared_ = np.nan
            self.error_ = np.nan

            self.fooofed_spectrum_ = None

            self._spectrum_flat = None
            self._spectrum_peak_rm = None
            self._ap_fit = None
            self._peak_fit = None


    def add_data(self, freqs, power_spectrum, freq_range=None, clear_results=True, knee_fix=None, exp1_fix=None, exp2_fix=None, off_fix=None, off_pr=0.0001, knee_pr=0.0001, exp1_pr=0.0001, exp2_pr=0.0001):
        """Add data (frequencies, and power spectrum values) to the current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array
            Power spectrum values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to.
            If not provided, keeps the entire range.
        clear_results : bool, optional, default: True
            Whether to clear prior results, if any are present in the object.
            This should only be set to False if data for the current results are being re-added.

        Notes
        -----
        If called on an object with existing data and/or results
        they will be cleared by this method call.
        """

        # If any data is already present, then clear previous data
        # Also clear results, if present, unless indicated not to
        #   This is to ensure object consistency of all data & results
        self._reset_data_results(clear_freqs=self.has_data,
                                 clear_spectrum=self.has_data,
                                 clear_results=self.has_model and clear_results)

        self.freqs, self.power_spectrum, self.freq_range, self.freq_res = \
            self._prepare_data(freqs, power_spectrum, freq_range, 1)

        ##added###############
        self.off_fix = off_fix
        self.knee_fix = knee_fix
        self.exp1_fix = exp1_fix
        self.exp2_fix = exp2_fix
        
        self.off_pr = off_pr
        self.knee_pr = knee_pr
        self.exp1_pr = exp1_pr
        self.exp2_pr = exp2_pr
        ##added###############
        


    def add_settings(self, fooof_settings):
        """Add settings into object from a FOOOFSettings object.

        Parameters
        ----------
        fooof_settings : FOOOFSettings
            A data object containing the settings for a FOOOF model.
        """

        for setting in OBJ_DESC['settings']:
            setattr(self, setting, getattr(fooof_settings, setting))

        self._check_loaded_settings(fooof_settings._asdict())


    def add_meta_data(self, fooof_meta_data):
        """Add data information into object from a FOOOFMetaData object.

        Parameters
        ----------
        fooof_meta_data : FOOOFMetaData
            A meta data object containing meta data information.
        """

        for meta_dat in OBJ_DESC['meta_data']:
            setattr(self, meta_dat, getattr(fooof_meta_data, meta_dat))

        self._regenerate_freqs()


    def add_results(self, fooof_result):
        """Add results data into object from a FOOOFResults object.

        Parameters
        ----------
        fooof_result : FOOOFResults
            A data object containing the results from fitting a FOOOF model.
        """

        self.aperiodic_params_ = fooof_result.aperiodic_params
        self.gaussian_params_ = fooof_result.gaussian_params
        self.peak_params_ = fooof_result.peak_params
        self.r_squared_ = fooof_result.r_squared
        self.error_ = fooof_result.error

        self._check_loaded_results(fooof_result._asdict())


    def report(self, freqs=None, power_spectrum=None, freq_range=None,
               plt_log=False, plot_full_range=False, **plot_kwargs):
        """Run model fit, and display a report, which includes a plot, and printed results.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Desired frequency range to fit the model to.
            If not provided, fits across the entire given range.
        plt_log : bool, optional, default: False
            Whether or not to plot the frequency axis in log space.
        plot_full_range : bool, default: False
            If True, plots the full range of the given power spectrum.
            Only relevant / effective if `freqs` and `power_spectrum` passed in in this call.
        **plot_kwargs
            Keyword arguments to pass into the plot method.
            Plot options with a name conflict be passed by pre-pending `plot_`.
            e.g. `freqs`, `power_spectrum` and `freq_range`.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        self.fit(freqs, power_spectrum, freq_range)
        self.plot(plt_log=plt_log,
                  freqs=freqs if plot_full_range else plot_kwargs.pop('plot_freqs', None),
                  power_spectrum=power_spectrum if \
                      plot_full_range else plot_kwargs.pop('plot_power_spectrum', None),
                  freq_range=plot_kwargs.pop('plot_freq_range', None),
                  **plot_kwargs)
        self.print_results(concise=False)



    def fit(self, freqs=None, power_spectrum=None, freq_range=None):
        """Fit the full power spectrum as a combination of periodic and aperiodic components.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to. If not provided, keeps the entire range.

        Raises
        ------
        NoDataError
            If no data is available to fit.
        FitError
            If model fitting fails to fit. Only raised in debug mode.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """
        # If freqs & power_spectrum provided together, add data to object.
        if freqs is not None and power_spectrum is not None:
            self.add_data(freqs, power_spectrum, freq_range)
        # If power spectrum provided alone, add to object, and use existing frequency data
        #   Note: be careful passing in power_spectrum data like this:
        #     It assumes the power_spectrum is already logged, with correct freq_range
        
        elif isinstance(power_spectrum, np.ndarray):
            self.power_spectrum = power_spectrum
        
        # Check that data is available
        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")

        # Check and warn about width limits (if in verbose mode)
        if self.verbose:
            self._check_width_limits()

        # In rare cases, the model fails to fit, and so uses try / except
        try:

            # If not set to fail on NaN or Inf data at add time, check data here
            #   This serves as a catch all for curve_fits which will fail given NaN or Inf
            #   Because FitError's are by default caught, this allows fitting to continue
            if not self._check_data:
                if np.any(np.isinf(self.power_spectrum)) or np.any(np.isnan(self.power_spectrum)):
                    raise FitError("Model fitting was skipped because there are NaN or Inf "
                                "values in the data, which preclude model fitting.")

            # Fit the aperiodic component
            self.current_ap_fit_params = None #added
            self.noise_peaks = None #added
            self.aperiodic_params_ = self._robust_ap_fit(self.freqs, self.power_spectrum)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Find peaks, and fit them with gaussians

            self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

            # Calculate the peak fit
            #   Note: if no peaks are found, this creates a flat (all zero) peak fit
            self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))

            # Create peak-removed (but not flattened) power spectrum
            self._spectrum_peak_rm = self.power_spectrum - self._peak_fit
            #self.auto_aperiodic(self.freqs, self._spectrum_peak_rm)
            # Run final aperiodic fit on peak-removed power spectrum
            #   This overwrites previous aperiodic fit, and recomputes the flattened spectrum
            self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._spectrum_peak_rm)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)
            self._spectrum_flat = self.power_spectrum - self._ap_fit
            add_iterations = 20
            if np.all(self.gaussian_params_ != [0, 0, 0]):
                for it in range(1, add_iterations+1):
    
                    self._spectrum_flat = self.power_spectrum - self._ap_fit
                    if it < add_iterations:
                        self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))

                    else:
                        self.gaussian_params_ = np.zeros_like(self.gaussian_params_)
                        self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))
                        
                    self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))
                    self._spectrum_peak_rm = self.power_spectrum - self._peak_fit

                    if it < add_iterations:
                        self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._spectrum_peak_rm)
                        self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_) # either remove this from the last iteration or, reiterate gaussians again


            # Create full power_spectrum model fit
            self.fooofed_spectrum_ = self._peak_fit + self._ap_fit
            # Convert gaussian definitions to peak parameters
            self.peak_params_ = self._create_peak_params(self.gaussian_params_)
            self.ap_fit = self._ap_fit
            # Calculate R^2 and error of the model fit
            self._calc_r_squared()
            self._calc_error()

        except FitError:

            # If in debug mode, re-raise the error
            if self._debug:
                raise

            # Clear any interim model results that may have run
            #   Partial model results shouldn't be interpreted in light of overall failure
            self._reset_data_results(clear_results=True)

            # Print out status
            if self.verbose:
                print("Model fitting was unsuccessful.")


    def print_settings(self, description=False, concise=False):
        """Print out the current settings.

        Parameters
        ----------
        description : bool, optional, default: False
            Whether to print out a description with current settings.
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_settings_str(self, description, concise))


    def print_results(self, concise=False):
        """Print out model fitting results.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_results_fm_str(self, concise))


    @staticmethod
    def print_report_issue(concise=False):
        """Prints instructions on how to report bugs and/or problematic fits.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_issue_str(concise))


    def get_settings(self):
        """Return user defined settings of the current object.

        Returns
        -------
        FOOOFSettings
            Object containing the settings from the current object.
        """

        return FOOOFSettings(**{key : getattr(self, key) \
                             for key in OBJ_DESC['settings']})


    def get_run_modes(self):
        """Return run modes of the current object.

        Returns
        -------
        FOOOFRunModes
            Object containing the run modes from the current object.
        """

        return FOOOFRunModes(**{key.strip('_') : getattr(self, key) \
                             for key in OBJ_DESC['run_modes']})


    def get_meta_data(self):
        """Return data information from the current object.

        Returns
        -------
        FOOOFMetaData
            Object containing meta data from the current object.
        """

        return FOOOFMetaData(**{key : getattr(self, key) \
                             for key in OBJ_DESC['meta_data']})


    def get_data(self, component='full', space='log'):
        """Get a data component.

        Parameters
        ----------
        component : {'full', 'aperiodic', 'peak'}
            Which data component to return.
                'full' - full power spectrum
                'aperiodic' - isolated aperiodic data component
                'peak' - isolated peak data component
        space : {'log', 'linear'}
            Which space to return the data component in.
                'log' - returns in log10 space.
                'linear' - returns in linear space.

        Returns
        -------
        output : 1d array
            Specified data component, in specified spacing.

        Notes
        -----
        The 'space' parameter doesn't just define the spacing of the data component
        values, but rather defines the space of the additive data definition such that
        `power_spectrum = aperiodic_component + peak_component`.
        With space set as 'log', this combination holds in log space.
        With space set as 'linear', this combination holds in linear space.
        """

        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")
        assert space in ['linear', 'log'], "Input for 'space' invalid."

        if component == 'full':
            output = self.power_spectrum if space == 'log' else unlog(self.power_spectrum)
        elif component == 'aperiodic':
            output = self._spectrum_peak_rm if space == 'log' else \
                unlog(self.power_spectrum) / unlog(self._peak_fit)
        elif component == 'peak':
            output = self._spectrum_flat if space == 'log' else \
                unlog(self.power_spectrum) - unlog(self._ap_fit)
        else:
            raise ValueError('Input for component invalid.')

        return output


    def get_model(self, component='full', space='log'):
        """Get a model component.

        Parameters
        ----------
        component : {'full', 'aperiodic', 'peak'}
            Which model component to return.
                'full' - full model
                'aperiodic' - isolated aperiodic model component
                'peak' - isolated peak model component
        space : {'log', 'linear'}
            Which space to return the model component in.
                'log' - returns in log10 space.
                'linear' - returns in linear space.

        Returns
        -------
        output : 1d array
            Specified model component, in specified spacing.

        Notes
        -----
        The 'space' parameter doesn't just define the spacing of the model component
        values, but rather defines the space of the additive model such that
        `model = aperiodic_component + peak_component`.
        With space set as 'log', this combination holds in log space.
        With space set as 'linear', this combination holds in linear space.
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available, can not proceed.")
        assert space in ['linear', 'log'], "Input for 'space' invalid."

        if component == 'full':
            output = self.fooofed_spectrum_ if space == 'log' else unlog(self.fooofed_spectrum_)
        elif component == 'aperiodic':
            output = self._ap_fit if space == 'log' else unlog(self._ap_fit)
        elif component == 'peak':
            output = self._peak_fit if space == 'log' else \
                unlog(self.fooofed_spectrum_) - unlog(self._ap_fit)
        else:
            raise ValueError('Input for component invalid.')

        return output


    def get_params(self, name, col=None):
        """Return model fit parameters for specified feature(s).

        Parameters
        ----------
        name : {'aperiodic_params', 'peak_params', 'gaussian_params', 'error', 'r_squared'}
            Name of the data field to extract.
        col : {'CF', 'PW', 'BW', 'offset', 'knee', 'exponent'} or int, optional
            Column name / index to extract from selected data, if requested.
            Only used for name of {'aperiodic_params', 'peak_params', 'gaussian_params'}.

        Returns
        -------
        out : float or 1d array
            Requested data.

        Raises
        ------
        NoModelError
            If there are no model fit parameters available to return.

        Notes
        -----
        If there are no fit peak (no peak parameters), this method will return NaN.
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available to extract, can not proceed.")

        # If col specified as string, get mapping back to integer
        if isinstance(col, str):
            col = get_indices(self.aperiodic_mode)[col]

        # Allow for shortcut alias, without adding `_params`
        if name in ['aperiodic', 'peak', 'gaussian']:
            name = name + '_params'

        # Extract the request data field from object
        out = getattr(self, name + '_')

        # Periodic values can be empty arrays and if so, replace with NaN array
        if isinstance(out, np.ndarray) and out.size == 0:
            out = np.array([np.nan, np.nan, np.nan])

        # Select out a specific column, if requested
        if col is not None:

            # Extract column, & if result is a single value in an array, unpack from array
            out = out[col] if out.ndim == 1 else out[:, col]
            out = out[0] if isinstance(out, np.ndarray) and out.size == 1 else out

        return out


    def get_results(self):
        """Return model fit parameters and goodness of fit metrics.

        Returns
        -------
        FOOOFResults
            Object containing the model fit results from the current object.
        """

        return FOOOFResults(**{key.strip('_') : getattr(self, key) \
            for key in OBJ_DESC['results']})


    @copy_doc_func_to_method(plot_fm)
    def plot(self, plot_peaks=None, plot_aperiodic=True, freqs=None, power_spectrum=None,
             freq_range=None, plt_log=False, add_legend=True, save_fig=False, file_name=None,
             file_path=None, ax=None, data_kwargs=None, model_kwargs=None,
             aperiodic_kwargs=None, peak_kwargs=None, **plot_kwargs):

        plot_fm(self, plot_peaks=plot_peaks, plot_aperiodic=plot_aperiodic, freqs=freqs,
                power_spectrum=power_spectrum, freq_range=freq_range, plt_log=plt_log,
                add_legend=add_legend, save_fig=save_fig, file_name=file_name,
                file_path=file_path, ax=ax, data_kwargs=data_kwargs, model_kwargs=model_kwargs,
                aperiodic_kwargs=aperiodic_kwargs, peak_kwargs=peak_kwargs, **plot_kwargs)


    @copy_doc_func_to_method(save_report_fm)
    def save_report(self, file_name, file_path=None, plt_log=False,
                    add_settings=True, **plot_kwargs):

        save_report_fm(self, file_name, file_path, plt_log, add_settings, **plot_kwargs)


    @copy_doc_func_to_method(save_fm)
    def save(self, file_name, file_path=None, append=False,
             save_results=False, save_settings=False, save_data=False):

        save_fm(self, file_name, file_path, append, save_results, save_settings, save_data)


    def load(self, file_name, file_path=None, regenerate=True):
        """Load in a FOOOF formatted JSON file to the current object.

        Parameters
        ----------
        file_name : str or FileObject
            File to load data from.
        file_path : Path or str, optional
            Path to directory to load from. If None, loads from current directory.
        regenerate : bool, optional, default: True
            Whether to regenerate the model fit from the loaded data, if data is available.
        """

        # Reset data in object, so old data can't interfere
        self._reset_data_results(True, True, True)

        # Load JSON file, add to self and check loaded data
        data = load_json(file_name, file_path)
        self._add_from_dict(data)
        self._check_loaded_settings(data)
        self._check_loaded_results(data)

        # Regenerate model components, based on what is available
        if regenerate:
            if self.freq_res:
                self._regenerate_freqs()
            if np.all(self.freqs) and np.all(self.aperiodic_params_):
                self._regenerate_model()


    def copy(self):
        """Return a copy of the current object."""

        return deepcopy(self)


    def set_debug_mode(self, debug):
        """Set debug mode, which controls if an error is raised if model fitting is unsuccessful.

        Parameters
        ----------
        debug : bool
            Whether to run in debug mode.
        """

        self._debug = debug


    def set_check_modes(self, check_freqs=None, check_data=None):
        """Set check modes, which controls if an error is raised based on check on the inputs.

        Parameters
        ----------
        check_freqs : bool, optional
            Whether to run in check freqs mode, which checks the frequency data.
        check_data : bool, optional
            Whether to run in check data mode, which checks the power spectrum values data.
        """

        if check_freqs is not None:
            self._check_freqs = check_freqs
        if check_data is not None:
            self._check_data = check_data


    # This kept for backwards compatibility, but to be removed in 2.0 in favor of `set_check_modes`
    def set_check_data_mode(self, check_data):
        """Set check data mode, which controls if an error is raised if NaN or Inf data are added.

        Parameters
        ----------
        check_data : bool
            Whether to run in check data mode.
        """

        self.set_check_modes(check_data=check_data)


    def set_run_modes(self, debug, check_freqs, check_data):
        """Simultaneously set all run modes.

        Parameters
        ----------
        debug : bool
            Whether to run in debug mode.
        check_freqs : bool
            Whether to run in check freqs mode.
        check_data : bool
            Whether to run in check data mode.
        """

        self.set_debug_mode(debug)
        self.set_check_modes(check_freqs, check_data)


    def to_df(self, peak_org):
        """Convert and extract the model results as a pandas object.

        Parameters
        ----------
        peak_org : int or Bands
            How to organize peaks.
            If int, extracts the first n peaks.
            If Bands, extracts peaks based on band definitions.

        Returns
        -------
        pd.Series
            Model results organized into a pandas object.
        """

        return model_to_dataframe(self.get_results(), peak_org)


    def _check_width_limits(self):
        """Check and warn about peak width limits / frequency resolution interaction."""

        # Check peak width limits against frequency resolution and warn if too close
        if 1.5 * self.freq_res >= self.peak_width_limits[0]:
            print(gen_width_warning_str(self.freq_res, self.peak_width_limits[0]))


    
    def _simple_ap_fit(self, freqs, power_spectrum):
        """
        Fit the aperiodic component of the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.
        """
        #guess parameters for 1exp and flat+1exp
        off_guess = [power_spectrum[0] if not self._ap_guess[0] else self._ap_guess[0]]
        kne_guess = [self._ap_guess[1]] if self.aperiodic_mode == 'knee_1exp' else []
        exp_guess = [np.abs((self.power_spectrum[-1] - self.power_spectrum[0]) /
                            (np.log10(self.freqs[-1]) - np.log10(self.freqs[0])))
                     if not self._ap_guess[2] else self._ap_guess[2]]
        
        if self.aperiodic_mode == 'fixed' or self.aperiodic_mode == 'flat_1exp':
            if self.aperiodic_mode == 'flat_1exp':
                ap_bounds = ((-np.inf, self.freq_range[0], 0), (np.inf, self.freq_range[1], np.inf)) 
            elif self.aperiodic_mode == 'fixed':
                ap_bounds = tuple(bound[0:2] for bound in self._ap_bounds)

            # Collect together guess parameters
            guess = np.array(off_guess + kne_guess + exp_guess)
            if self.current_ap_fit_params is not None: 
                guess = self.current_ap_fit_params 
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                        freqs, power_spectrum, p0=guess,
                                        maxfev=self._maxfev, bounds=ap_bounds,
                                        ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                        check_finite=False)
            except RuntimeError as excp:
                error_msg = "Model fitting failed due to not finding parameters in the simple aperiodic component fit."
                raise FitError(error_msg) from excp

        if self.aperiodic_mode == '2exp':
            # Select subregion of PSD
            f_mask = np.logical_and(freqs >= self.freq_range[0], freqs <= self.freq_range[1])
            freqs = freqs[f_mask]
            power_spectrum = power_spectrum[f_mask]

            # Calculate guess parameters for the 1st exponent
            freq_res = freqs[1] - freqs[0]

            start_index1 = round(13 / freq_res)
            end_index1 = round(45 / freq_res)
            f1 = freqs[start_index1:end_index1]
            u1 = np.log10(f1)
            p1 = power_spectrum[start_index1:end_index1]
            guess_exp1 = [(p1[-1] - p1[0]) / (u1[0] - u1[-1])]
            off1 = (guess_exp1[0] * u1[0]) + p1[0]

            # Calculate guess parameters for the 2nd exponent, knee, and power at knee
            start_index2 = round(75 / freq_res)
            end_index2 = round(120/ freq_res) 
            f2 = freqs[start_index2:end_index2]
            u2 = np.log10(f2)
            p2 = power_spectrum[start_index2:end_index2]
            guess_exp2 = [(p2[-1] - p2[0]) / (u2[0] - u2[-1])]
            off2 = (guess_exp2[0] * u2[0] + p2[0])
            knee_guess = [((off2 - off1) / (guess_exp2[0] - guess_exp1[0]))]
            knee_off_guess = [np.log10(2) + (off1 - (guess_exp1[0] * knee_guess[0]))]
            knee_guess = [10**(knee_guess[0])]
            guess = np.array(knee_off_guess + knee_guess + guess_exp1 + guess_exp2)
            ap_bounds = ((-np.inf, self.freq_range[0], 0, 0), (np.inf, self.freq_range[1], np.inf, np.inf))
            
            #avoid error, if guess is outside of ap_bounds #added
            for g in range(0, len(guess)):
                if guess[g] < ap_bounds[0][g]:
                    guess[g] = ap_bounds[0][g]
                elif guess[g] > ap_bounds[1][g]:
                    guess[g] = ap_bounds[1][g]

            if self.current_ap_fit_params is not None: #added
                guess = self.current_ap_fit_params #added

            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                guess[1] = self.knee_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                guess[2] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[2] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[2] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                guess[3] = self.exp2_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[3] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                guess[0] = self.off_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################


            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                    freqs, power_spectrum, p0=guess,
                                                    maxfev=self._maxfev, bounds=ap_bounds,
                                                    ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                                    check_finite=False)
            except RuntimeError as excp:
                error_msg = ("Model fitting failed due to not finding parameters in "
                            "the simple aperiodic component fit.")
                raise FitError(error_msg) from excp

            if aperiodic_params[2] > aperiodic_params[3]:
                exp1_copy = aperiodic_params[2].copy()
                exp2_copy = aperiodic_params[3].copy()
                aperiodic_params[2] = exp2_copy
                aperiodic_params[3] = exp1_copy

        if self.aperiodic_mode == "2exp_flat":
            f_mask = np.logical_and(freqs >= self.freq_range[0], freqs <= self.freq_range[1])
            freqs = freqs[f_mask]
            power_spectrum = power_spectrum[f_mask]

            # Calculate guess parameters for exp1
            freq_res = freqs[1] - freqs[0]
            start_index1 = round(13 / freq_res)
            end_index1 = round(45 / freq_res)
            f1 = freqs[start_index1:end_index1]
            u1 = np.log10(f1)
            p1 = power_spectrum[start_index1:end_index1]
            guess_exp1 = [(p1[-1] - p1[0]) / (u1[0] - u1[-1])]
            off1 = (guess_exp1[0] * u1[0]) + p1[0]

            # Calculate guess parameters for exp2 and knee
            start_index2 = round(75 / freq_res)
            end_index2 = round(180 / freq_res)
            f2 = freqs[start_index2:end_index2]
            u2 = np.log10(f2)
            p2 = power_spectrum[start_index2:end_index2]
            guess_exp2 = [(p2[-1] - p2[0]) / (u2[0] - u2[-1])]
            new_guess_exp2 = [np.abs(guess_exp2[0])]
            off2 = (guess_exp2[0] * u2[0] + p2[0])

            knee_guess = [(off2 - off1) / (guess_exp2[0] - guess_exp1[0])]
            
            
            # Calculate guess parameters for flat_freq, power at knee frequency
            start_index3 = round(300 / freq_res)
            end_index3 = round(495/ freq_res)
            f3 = freqs[start_index3:end_index3]
            u3 = np.log10(f3)
            p3 = power_spectrum[start_index3:end_index3]
            flat_off_guess = [np.mean(p3)]
            flat_freq_guess = [(flat_off_guess[0] - off2)/(-1*new_guess_exp2[0])]
            knee_off_guess = [np.log10(2) + (off1 - (guess_exp1[0] * knee_guess[0]))]
            
            guess = np.array(knee_off_guess + knee_guess + flat_freq_guess + guess_exp1 + new_guess_exp2) 
            ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0), (np.inf, self.freq_range[1], self.freq_range[1], np.inf , np.inf))
            if self.freq_range[1] < 400:
                flat_freq_guess = [300]
                guess = np.array(knee_off_guess + knee_guess + flat_freq_guess + guess_exp1 + new_guess_exp2) 
                ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0), (np.inf, 490, 490, np.inf , np.inf))


            #avoid error, if guess is outside of ap_bounds #added
            for g in range(0, len(guess)):
                if guess[g] < ap_bounds[0][g]:
                    guess[g] = ap_bounds[0][g]
                elif guess[g] > ap_bounds[1][g]:
                    guess[g] = ap_bounds[1][g]
            if self.current_ap_fit_params is not None: #added
                guess = self.current_ap_fit_params #added

            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                guess[1] = self.knee_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                guess[3] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[3] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                guess[4] = self.exp2_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[4] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[4] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                guess[0] = self.off_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################

            if guess[2] < guess[1]:
                guess[2] = guess[1] + 30
            if guess[2] > self.freq_range[1] and self.freq_range[1] > 400:
                guess[2] = 0.8*self.freq_range[1]

            aperiodic_params, _ = curve_fit(
                get_ap_func(self.aperiodic_mode),
                freqs, power_spectrum, p0=guess,
                maxfev=self._maxfev, bounds=ap_bounds,
                ftol=self._tol, xtol=self._tol, gtol=self._tol,
                check_finite=False
            )
                
            if aperiodic_params[1] > aperiodic_params[2]:
                knee_copy = aperiodic_params[1].copy()
                flat_freq_copy = aperiodic_params[2].copy()
                aperiodic_params[1] = flat_freq_copy
                aperiodic_params[2] = knee_copy

            
        if self.aperiodic_mode == "3exp":
            f_mask = np.logical_and(freqs >= self.freq_range[0], freqs <= self.freq_range[1])
            freqs = freqs[f_mask]
            power_spectrum = power_spectrum[f_mask]
            original_psd = power_spectrum.copy()  # vis

            # Calculate guess parameters for exp1
            freq_res = freqs[1] - freqs[0]
            start_index1 = round(13 / freq_res)
            end_index1 = round(45 / freq_res)
            f1 = freqs[start_index1:end_index1]
            u1 = np.log10(f1)
            p1 = power_spectrum[start_index1:end_index1]
            guess_exp1 = [(p1[-1] - p1[0]) / (u1[0] - u1[-1])]
            off1 = (guess_exp1[0] * u1[0]) + p1[0]

            # Calculate guess parameters for exp2, exp3, knee, flat_freq, power at knee frequency for fitting untill 500 Hz
            start_index2 = round(75 / freq_res)
            end_index2 = round(180 / freq_res)
            f2 = freqs[start_index2:end_index2]
            u2 = np.log10(f2)
            p2 = power_spectrum[start_index2:end_index2]
            guess_exp2 = [(p2[-1] - p2[0]) / (u2[0] - u2[-1])]
            new_guess_exp2 = [np.abs(guess_exp2[0])]
            off2 = (guess_exp2[0] * u2[0] + p2[0])

            knee_guess = [(off2 - off1) / (guess_exp2[0] - guess_exp1[0])]
            
            start_index3 = round(300 / freq_res)
            end_index3 = round(495/ freq_res)
            f3 = freqs[start_index3:end_index3]
            u3 = np.log10(f3)
            p3 = power_spectrum[start_index3:end_index3]
            flat_off_guess = [np.mean(p3)]
            flat_freq_guess = [(flat_off_guess[0] - off2)/(-1*new_guess_exp2[0])]
            
            exp3_guess = [0]
            knee_off_guess = [np.log10(2) + (off1 - (guess_exp1[0] * knee_guess[0]))]
            
            #flat_fr_logged_guess = [2.3]  # Adjust this initial guess if needed
            guess = np.array(knee_off_guess + knee_guess + flat_freq_guess + guess_exp1 + new_guess_exp2 + exp3_guess) 
            ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0, 0), (np.inf, self.freq_range[1], self.freq_range[1], np.inf , np.inf, np.inf))
            #guess flat_freq if fitting range is <400 Hz
            if self.freq_range[1] < 400:
                flat_freq_guess = [300]
                guess = np.array(knee_off_guess + knee_guess + flat_freq_guess + guess_exp1 + new_guess_exp2 + exp3_guess)  
                ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0, 0), (np.inf, 490, 490, np.inf , np.inf, np.inf))
            #avoid error, if guess is outside of ap_bounds #added
            for g in range(0, len(guess)):
                if guess[g] < ap_bounds[0][g]:
                    guess[g] = ap_bounds[0][g]
                elif guess[g] > ap_bounds[1][g]:
                    guess[g] = ap_bounds[1][g]
            
            if self.current_ap_fit_params is not None: #added
                guess = self.current_ap_fit_params #added

            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                guess[1] = self.knee_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                guess[3] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[3] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                guess[4] = self.exp2_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[4] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[4] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                guess[0] = self.off_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################
            if guess[2] < guess[1]:
                guess[2] = guess[1] + 30
            if guess[2] > self.freq_range[1] and self.freq_range[1] > 400:
                guess[2] = 0.8*self.freq_range[1]
            aperiodic_params, _ = curve_fit(
                get_ap_func(self.aperiodic_mode),
                freqs, power_spectrum, p0=guess,
                maxfev=self._maxfev, bounds=ap_bounds,
                ftol=self._tol, xtol=self._tol, gtol=self._tol,
                check_finite=False
            )
            if aperiodic_params[1] > aperiodic_params[2]:
                knee_copy = aperiodic_params[1].copy()
                flat_freq_copy = aperiodic_params[2].copy()
                aperiodic_params[1] = flat_freq_copy
                aperiodic_params[2] = knee_copy

        self.current_ap_fit_params = aperiodic_params #added
        return aperiodic_params

    def _robust_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum robustly, ignoring outliers.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.

        Raises
        ------
        FitError
            If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_ap_fit(freqs, power_spectrum)
        initial_fit = gen_aperiodic(freqs, popt)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = power_spectrum - initial_fit

        # Flatten outliers, defined as any points that drop below 0
        flatspec[flatspec < 0] = 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._ap_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = power_spectrum[perc_mask]

        # Get bounds for aperiodic fitting, dropping knee bound if not set to fit knee
        if self.aperiodic_mode == '2exp':
            ap_bounds = ((-np.inf, self.freq_range[0], 0, 0), (np.inf, self.freq_range[1], np.inf, np.inf))
            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                popt[1] = self.knee_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                popt[2] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[2] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[2] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                popt[3] = self.exp2_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[3] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                popt[0] = self.off_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################
        elif self.aperiodic_mode == '3exp':
            ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0, 0), (np.inf, self.freq_range[1], self.freq_range[1], np.inf , np.inf, np.inf))
            if self.freq_range[1] < 400:
                ap_bounds = ((-np.inf,self.freq_range[0], self.freq_range[0], 0 , 0, 0), (np.inf, 490, 490, np.inf , np.inf, np.inf))
            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                popt[1] = self.knee_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                popt[3] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[3] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                popt[4] = self.exp2_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[4] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[4] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                popt[0] = self.off_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################
        elif self.aperiodic_mode == '2exp_flat':
            ap_bounds = ((-np.inf, self.freq_range[0], self.freq_range[0], 0, 0), (np.inf, self.freq_range[1], self.freq_range[1], np.inf, np.inf))
            if self.freq_range[1] < 400:
                ap_bounds = ((-np.inf, self.freq_range[0], self.freq_range[0], 0, 0), (np.inf, 490, 490, np.inf, np.inf))
            ######added optional fixation of ap parameters ##############################
            if self.knee_fix is not None:
                popt[1] = self.knee_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[1] = self.knee_fix - self.knee_pr
                if lower_bounds_l[1] < self.freq_range[0]:
                    lower_bounds_l[1] = self.freq_range[0]
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                upper_bounds_l[1] = self.knee_fix + self.knee_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp1_fix is not None:
                popt[3] = self.exp1_fix
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[3] = self.exp1_fix - self.exp1_pr
                upper_bounds_l[3] = self.exp1_fix + self.exp1_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.exp2_fix is not None:
                popt[4] = self.exp2_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[4] = self.exp2_fix - self.exp2_pr
                upper_bounds_l[4] = self.exp2_fix + self.exp2_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            if self.off_fix is not None:
                popt[0] = self.off_fix 
                lower_bounds_l = list(ap_bounds[0])
                upper_bounds_l = list(ap_bounds[1])
                lower_bounds_l[0] = self.off_fix - self.off_pr
                upper_bounds_l[0] = self.off_fix + self.off_pr
                ap_bounds = (tuple(lower_bounds_l), tuple(upper_bounds_l))
            ######added optional fixation of ap parameters ##############################
        elif self.aperiodic_mode == 'flat_1exp':
            #ap_bounds = tuple(bound[0:3] for bound in self._ap_bounds)  # Assuming _ap_bounds already correctly structured for 'knee_1exp'
            ap_bounds = ((-np.inf, (self.freq_range[0]), 0), (np.inf, (self.freq_range[1]), np.inf)) 
        elif self.aperiodic_mode == 'fixed':
            ap_bounds = tuple(bound[0:2] for bound in self._ap_bounds)
        else:
            raise ValueError(f"Unknown aperiodic mode: {self.aperiodic_mode}")

        # Second aperiodic fit - using results of first fit as guess parameters
        #  See note in _simple_ap_fit about warnings
        for g in range(0, len(popt)):
                if popt[g] < ap_bounds[0][g]:
                    popt[g] = ap_bounds[0][g]
                elif popt[g] > ap_bounds[1][g]:
                    popt[g] = ap_bounds[1][g]
        aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                        freqs_ignore, spectrum_ignore, p0=popt,
                                        maxfev=self._maxfev, bounds=ap_bounds,
                                        ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                        check_finite=False)
        if self.aperiodic_mode == '2exp_flat' or self.aperiodic_mode == '3exp':
            if aperiodic_params[1] > aperiodic_params[2]:
                    knee_copy = aperiodic_params[1].copy()
                    flat_freq_copy = aperiodic_params[2].copy()
                    aperiodic_params[1] = flat_freq_copy
                    aperiodic_params[2] = knee_copy
        elif self.aperiodic_mode == '2exp':
            if aperiodic_params[2] > aperiodic_params[3]:
                exp1_copy = aperiodic_params[2].copy()
                exp2_copy = aperiodic_params[3].copy()
                aperiodic_params[2] = exp2_copy
                aperiodic_params[3] = exp1_copy
                                        
        self.current_ap_fit_params = aperiodic_params #added
        return aperiodic_params




##################################################inserted#########################################################################################



    def rem_electric_noise(self, power_spectrum, freqs):
        
        """
        Remove harmonics and electric noise from the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        
        power_spectrum : 1d array
            Power spectrum filtered from noise.
        """
        maxfevs = 5000 # The maximum number of calls to the curve fitting function
        _tol = 0.00001 #  Here reduce tolerance to speed fitting. Set value to 1e-8 to match curve_fit default


        power_spectrum = np.log10(power_spectrum)
        
        # first of all: remove electric noise and its harmonics
        noise_bands = [[43, 57], [130, 170], [12.5, 22]]
        correnction_bands = [[48.5, 51.5], [148, 152], [13.5, 21.5]]
        noise_std_limits = [0, 4]
        
        
        #second: fit gaussian peaks at bands having harmonics or electric noise
        ap_mode = 'fixed'
        ap_bounds_n = ((-np.inf, -np.inf), (np.inf, np.inf))

        
        for noise_band, correction_band in zip(noise_bands, correnction_bands):
            # calculate guess parameters
            freq_res = freqs[1] - freqs[0]
            start_index1 = round(noise_band[0]/freq_res)
            end_index1 = round(noise_band[1]/freq_res)
            f1 = freqs[start_index1:end_index1]
            u1 = np.log10(f1)
            p1 = power_spectrum[start_index1:end_index1]
            guess_exp1 = [(p1[-1] - p1[0])/(u1[0]-u1[-1])]
            off1 = (guess_exp1[0]*u1[0]) + p1[0]


            f_mask_n = np.logical_and(freqs >= noise_band[0], freqs <= noise_band[1])
            freqs_n = freqs[f_mask_n]
            f_mask_correction = np.logical_and(freqs >= correction_band[0], freqs <= correction_band[1])
            freqs_correction = freqs[f_mask_correction]
            psd_n = power_spectrum[f_mask_n]

            n_off_guess = [off1]
            n_exp_guess = guess_exp1
            noise_guess = np.array(n_off_guess + n_exp_guess) 
            noise_ap_params0, _ = curve_fit(get_ap_func(ap_mode),
                                                freqs_n, psd_n, p0=noise_guess,
                                                maxfev=maxfevs, bounds=ap_bounds_n,
                                                ftol=_tol, xtol=_tol, gtol=_tol,
                                                check_finite=False)
            
            initial_noise_fit = noise_ap_params0[0] - np.log10(freqs_n**noise_ap_params0[1])
            # flatten and remove outliers
            flatspec = psd_n - initial_noise_fit
            flatspec[flatspec < 0] = 0 # Flatten outliers, defined as any points that drop below 0
            perc_thresh = np.percentile(flatspec, 0.025) # -> remove top 2.5% of power values, don't be confused that it rounds to 0
            perc_mask = flatspec <= perc_thresh
            freqs_n_ignore = freqs_n[perc_mask]
            psd_n_ignore = psd_n[perc_mask]

            noise_ap_params1, _ = curve_fit(get_ap_func(ap_mode),
                                                freqs_n_ignore, psd_n_ignore, p0=noise_ap_params0,
                                                maxfev=maxfevs, bounds=ap_bounds_n,
                                                ftol=_tol, xtol=_tol, gtol=_tol,
                                                check_finite=False)
            out_rem_noise_fit = noise_ap_params1[0] - np.log10(freqs_n**noise_ap_params1[1])
            flat_noise_spec = psd_n - out_rem_noise_fit
            flat_noise_with_peak = psd_n - out_rem_noise_fit # copy for later

            
            # calculate noise gaussian peak
            guess_n_peak = np.empty([0, 3])
            max_ind = np.argmax(flat_noise_spec)
            max_height = flat_noise_spec[max_ind]
            guess_freq = freqs_n[max_ind]
            guess_height = max_height
            if guess_height > 0.01: # min_peak_height arbitrarily chosen to be 0.2
                if not max_height <= 0.1 * np.std(flat_noise_spec): # relitive threshold arbitrarily chosen to be 1
                    half_height = 0.5 * max_height
                    le_ind = next((val for val in range(max_ind - 1, 0, -1)
                                if flat_noise_spec[val] <= half_height), None)
                    ri_ind = next((val for val in range(max_ind + 1, len(flat_noise_spec), 1)
                                if flat_noise_spec[val] <= half_height), None)

                    try:
                        short_side = min([abs(ind - max_ind) \
                            for ind in [le_ind, ri_ind] if ind is not None])

                        fwhm = short_side * 2 * freq_res
                        guess_std = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
                    except ValueError:
                        guess_std = np.mean(noise_std_limits)*2

                    if guess_std < noise_std_limits[0]:
                        guess_std = noise_std_limits[0]
                    if guess_std > noise_std_limits[1]:
                        guess_std = noise_std_limits[1]
        
                    guess_n_peak = np.vstack((guess_n_peak, (guess_freq, guess_height, guess_std)))
                    peak_gauss = gaussian_function(freqs_n, guess_freq, guess_height, guess_std)
                    flat_noise_spec = flat_noise_spec - peak_gauss
        
        
                if len(guess_n_peak) > 0:
                    
                    lo_bound = [[peak[0] - 2 * 0.5 * peak[2], 0, noise_std_limits[0]] # 1.5 = self._cf_bound
                                for peak in guess_n_peak]
                    hi_bound = [[peak[0] + 2 * 0.5 * peak[2], np.inf, noise_std_limits[1]]
                                for peak in guess_n_peak]

                    lo_bound = [bound if bound[0] > noise_band[0] else \
                        [noise_band[0], *bound[1:]] for bound in lo_bound]
                    hi_bound = [bound if bound[0] < noise_band[1] else \
                        [noise_band[1], *bound[1:]] for bound in hi_bound]

                    gaus_param_bounds = (tuple(item for sublist in lo_bound for item in sublist),
                                        tuple(item for sublist in hi_bound for item in sublist))
            
                    # Flatten guess, for use with curve fit
                    guess_n_peak = np.ndarray.flatten(guess_n_peak)

                    # fit_noise_peak
                    noise_params, _ = curve_fit(gaussian_function, freqs_n, flat_noise_with_peak,
                                        p0=guess_n_peak, maxfev=maxfevs, bounds=gaus_param_bounds,
                                        ftol=_tol, xtol=_tol, gtol=_tol,
                                        check_finite=False, jac=jacobian_gauss)
                    
                    noise_peak  = gaussian_function(freqs_n, noise_params[0], noise_params[1], noise_params[2])
                    psd_n_peak_rem = psd_n - noise_peak


                    noise_ap_params2, _ = curve_fit(get_ap_func(ap_mode),
                                            freqs_n, psd_n_peak_rem, p0=noise_ap_params1,
                                            maxfev=maxfevs, bounds=ap_bounds_n,
                                            ftol=_tol, xtol=_tol, gtol=_tol,
                                            check_finite=False)
                    iters_noise_fit = noise_ap_params2[0] - np.log10(freqs_n**noise_ap_params2[1])

                    n_iterations = 12
                    for it in range(1, n_iterations +1):
                        if np.all(noise_params != [0, 0, 0]):
                            flat_noise_with_peak = psd_n - iters_noise_fit
                            noise_params, _ = curve_fit(gaussian_function, freqs_n, flat_noise_with_peak,
                                        p0=noise_params, maxfev=maxfevs, bounds=gaus_param_bounds,
                                        ftol=_tol, xtol=_tol, gtol=_tol,
                                        check_finite=False, jac=jacobian_gauss)                          
                            noise_peak  = gaussian_function(freqs_n, noise_params[0], noise_params[1], noise_params[2])

                            
                            psd_n_peak_rem = psd_n - noise_peak
                            noise_ap_params2, _ = curve_fit(get_ap_func(ap_mode),
                                            freqs_n, psd_n_peak_rem, p0=noise_ap_params2,
                                            maxfev=maxfevs, bounds=ap_bounds_n,
                                            ftol=_tol, xtol=_tol, gtol=_tol,
                                            check_finite=False)
                            iters_noise_fit = noise_ap_params2[0] - np.log10(freqs_n**noise_ap_params2[1])
            

                    power_spectrum[f_mask_n] = power_spectrum[f_mask_n] - noise_peak  #remove the noise peak from the original power spectrum

        power_spectrum = 10**power_spectrum
        return power_spectrum

    
    def _fit_peaks(self, flat_iter):
        """
        Iteratively fit peaks to flattened spectrum.

        Parameters
        ----------
        flat_iter : 1d array
        Flattened power spectrum values.

        Returns
        -------
        gaussian_params : 2d array
        Parameters that define the gaussian fit(s).
        Each row is a gaussian, as [mean, height, standard deviation].
        """
        # Define frequency bands ranges and corresponding parameters

        self.std_limits = np.empty([0 , 2])
        self.cf_bounds = np.empty([0 , 2])
        theta_range = [self.freq_range[0], 19]
        sgamma_range = [19, 48]
        fgamma_range = [51, 90]
        ripple_range = [100, 200]
        list_of_ranges = [ripple_range, fgamma_range, theta_range, sgamma_range]  
        list_of_thresholds = [1, 1, 1, 1] #equivalent to peak_threshold in the original FOOOF library
        abs_thresholds = [ 0.05, 0.08, 0.1, 0.05] #equivalent to min_peak_height in the original FOOOF library 
        list_of_peak_widths = [ [10, 30], [3, 35], [2, 5], [5, 20]] #equivalent to peak_width_limits in the original FOOOF library
        allowed_ranges = [[125, 160], [52, 80], [5, 9.5], [25, 40]] #fitting ranges for different frequency bands
        # Initialize matrix of guess parameters for gaussian fitting
        guess = np.empty([0, 3])


        #Parameters for frequency range higher than 400 Hz
        if self.freq_range[1] >= 400:
            additional_peak_range = [205, 495]
            list_of_ranges = [ripple_range, fgamma_range, theta_range, sgamma_range, additional_peak_range]
            list_of_thresholds = [1, 1, 1, 1, 1]
            abs_thresholds = [0.05, 0.08, 0.1, 0.05, 0.05]
            list_of_peak_widths = [ [10, 30], [3, 35], [2, 5], [5, 20], [50, 150]]
            allowed_ranges = [[125, 160], [52, 80], [5, 9.5], [25, 47], [220, 380]] 

        
        
        
        elif self.freq_range[1] <= 120:
            # for 4-100Hz:
            list_of_ranges = [fgamma_range, theta_range, sgamma_range]  
            list_of_thresholds = [1, 1, 1]
            abs_thresholds = [0.08, 0.1, 0.05] 
            list_of_peak_widths = [[3, 35], [2, 5], [5, 20]]
            allowed_ranges = [[52, 80], [5, 9.5], [25, 47]]
        
        # Find peak: Loop through, finding a candidate peak, and fitting with a guess gaussian
        # Stopping procedures: limit on # of peaks, or relative or absolute height thresholds
        for allowed_range, current_range, current_threshold, current_width, abs_threshold in zip(allowed_ranges, list_of_ranges, list_of_thresholds, list_of_peak_widths, abs_thresholds):
            current_range_indeces = (self.freqs >= current_range[0]) & (self.freqs <= current_range[1])
            current_band = flat_iter[current_range_indeces]
            current_freqs = self.freqs[current_range_indeces]

            #new change for better peak estimation
            allowed_range_indeces = (self.freqs >= allowed_range[0]) & (self.freqs <= allowed_range[1])
            allowed_band = flat_iter[allowed_range_indeces]
            allowed_freqs = self.freqs[allowed_range_indeces]
            max_ind = np.argmax(allowed_band)
            guess_freq = allowed_freqs[max_ind]
            max_ind = int(np.round((guess_freq - current_range[0])/self.freq_res))
            max_height = current_band[max_ind]
            guess_height = max_height
            guess_freq = current_freqs[max_ind]

            # Stop searching for peaks once height drops below height threshold
            if max_height <= current_threshold * np.std(current_band):
                continue
            if not guess_height > abs_threshold: # if 0 -> not absolute threshold
                continue 

            half_height = 0.5 * max_height
            le_ind = next((val for val in range(max_ind - 1, 0, -1) if current_band[val] <= half_height), None)
            ri_ind = next((val for val in range(max_ind + 1, len(current_band), 1) if current_band[val] <= half_height), None)

            # Guess bandwidth procedure: estimate the width of the peak
            try:
                # Get an estimated width from the shortest side of the peak
                #   We grab shortest to avoid estimating very large values from overlapping peaks
                # Grab the shortest side, ignoring a side if the half max was not found
                short_side = min([abs(ind - max_ind) \
                    for ind in [le_ind, ri_ind] if ind is not None])

                # Use the shortest side to estimate full-width, half max (converted to Hz)
                #   and use this to estimate that guess for gaussian standard deviation
                fwhm = short_side * 2 * self.freq_res
                guess_std = compute_gauss_std(fwhm)
            except ValueError:
                # This procedure can fail (very rarely), if both left & right inds end up as None
                #   In this case, default the guess to the average of the peak width limits
                guess_std = np.mean(current_width)  
            # Check that guess value isn't outside preset limits - restrict if so
            #   Note: without this, curve_fitting fails if given guess > or < bounds
            if guess_std < current_width[0]/2:
                guess_std = current_width[0]/2

            if guess_std > current_width[1]/2:
                guess_std = current_width[1]/2

            gauss_limit = [0,0]
            gauss_limit[0] = current_width[0]/2
            gauss_limit[1] = current_width[1]/2
            self.std_limits = np.vstack((self.std_limits, gauss_limit))
            self.cf_bounds = np.vstack((self.cf_bounds, allowed_range))
            # Collect guess parameters and subtract this guess gaussian from the data
            guess = np.vstack((guess, (guess_freq, guess_height, guess_std)))
            peak_gauss = gaussian_function(current_freqs, guess_freq, guess_height, guess_std)
            current_band = current_band - peak_gauss
            flat_iter[current_range_indeces] = current_band

        # Check peaks based on edges, and on overlap, dropping any that violate requirements
        # If there are peak guesses, fit the peaks, and sort results
        if len(guess) > 0:
            gaussian_params = self._fit_peak_guess(guess)
            gaussian_params = gaussian_params[gaussian_params[:, 0].argsort()]
        else:
            gaussian_params = np.empty([0, 3])

        return gaussian_params


   


    def _fit_peak_guess(self, guess):
        """
        Fits a group of peak guesses with a fit function.

        Parameters
        ----------
        guess : 2d array, shape=[n_peaks, 3]
            Guess parameters for gaussian fits to peaks, as gaussian parameters.

        Returns
        -------
        gaussian_params : 2d array, shape=[n_peaks, 3]
            Parameters for gaussian fits to peaks, as gaussian parameters.
        """

        # Set the bounds for CF, enforce positive height value, and set bandwidth limits
        #   Note that 'guess' is in terms of gaussian std, so +/- BW is 2 * the guess_gauss_std
        #   This set of list comprehensions is a way to end up with bounds in the form:
        #     ((cf_low_peak1, height_low_peak1, bw_low_peak1, *repeated for n_peaks*),
        #      (cf_high_peak1, height_high_peak1, bw_high_peak, *repeated for n_peaks*))
        #     ^where each value sets the bound on the specified parameter
        num_peaks = guess.shape[0]
    
        # Initialize self.std_limits if it is empty
                # Debugging output to check the input sizes
        if self.std_limits.shape[0] != num_peaks:
            raise ValueError("The number of peaks in 'guess' does not match the number of std limits provided.")
        # Set the bounds for CF, enforce positive height value, and set bandwidth limits
        lo_bound = [[cf[0], 0, std_lim[0]] ## Default was 2* self.cf_bound
                    for cf, peak, std_lim in zip(self.cf_bounds , guess, self.std_limits)]
        hi_bound = [[cf[1], np.inf, std_lim[1]]
                    for cf, peak, std_lim in zip(self.cf_bounds , guess, self.std_limits)]

        
        # Check that CF bounds are within frequency range
        lo_bound = [bound if bound[0] > self.freq_range[0] else \
            [self.freq_range[0], *bound[1:]] for bound in lo_bound]
        hi_bound = [bound if bound[0] < self.freq_range[1] else \
            [self.freq_range[1], *bound[1:]] for bound in hi_bound]

        # Unpacks the embedded lists into flat tuples
        gaus_param_bounds = (tuple(item for sublist in lo_bound for item in sublist),
                            tuple(item for sublist in hi_bound for item in sublist))

        # Flatten guess, for use with curve fit

        for a in range(0, len(guess)):
            peak_a = guess[a]
            hi_bound_a = hi_bound[a]
            lo_bound_a = lo_bound[a]
            for b in range(0, 3):
                p_b = peak_a[b]
                hi_b = hi_bound_a[b]
                lo_b = lo_bound_a[b]
                if p_b < lo_b:
                    guess[a][b] = lo_b
                if p_b > hi_b:
                    guess[a][b] = hi_b
        guess_flat = np.ndarray.flatten(guess)

        # Check if the initial guess is within the bounds
        for i, g in enumerate(guess_flat):
            if not (gaus_param_bounds[0][i] <= g <= gaus_param_bounds[1][i]):
                raise ValueError(f"Initial guess {g} at index {i} is out of bounds: "
                                f"{gaus_param_bounds[0][i]}, {gaus_param_bounds[1][i]}")

        # Fit the peaks
        try:
            gaussian_params, _ = curve_fit(gaussian_function, self.freqs, self._spectrum_flat,
                                        p0=guess_flat, maxfev=self._maxfev, bounds=gaus_param_bounds,
                                        ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                        check_finite=False, jac=jacobian_gauss)
        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding "
                        "parameters in the peak component fit.")
            raise FitError(error_msg) from excp
        except LinAlgError as excp:
            error_msg = ("Model fitting failed due to a LinAlgError during peak fitting. "
                        "This can happen with settings that are too liberal, leading, "
                        "to a large number of guess peaks that cannot be fit together.")
            raise FitError(error_msg) from excp

        # Re-organize params into 2d matrix
        gaussian_params = np.array(group_three(gaussian_params))
        
        return gaussian_params


    def _create_peak_params(self, gaus_params):
        """Copies over the gaussian params to peak outputs, updating as appropriate.

        Parameters
        ----------
        gaus_params : 2d array
            Parameters that define the gaussian fit(s), as gaussian parameters.

        Returns
        -------
        peak_params : 2d array
            Fitted parameter values for the peaks, with each row as [CF, PW, BW].

        Notes
        -----
        The gaussian center is unchanged as the peak center frequency.

        The gaussian height is updated to reflect the height of the peak above
        the aperiodic fit. This is returned instead of the gaussian height, as
        the gaussian height is harder to interpret, due to peak overlaps.

        The gaussian standard deviation is updated to be 'both-sided', to reflect the
        'bandwidth' of the peak, as opposed to the gaussian parameter, which is 1-sided.

        Performing this conversion requires that the model has been run,
        with `freqs`, `fooofed_spectrum_` and `_ap_fit` all required to be available.
        """

        peak_params = np.empty((len(gaus_params), 3))

        for ii, peak in enumerate(gaus_params):

            # Gets the index of the power_spectrum at the frequency closest to the CF of the peak
            ind = np.argmin(np.abs(self.freqs - peak[0]))

            # Collect peak parameter data
            peak_params[ii] = [peak[0], self.fooofed_spectrum_[ind] - self._ap_fit[ind],
                               peak[2] * 2]

        return peak_params


    def _drop_peak_cf(self, guess):
        """Check whether to drop peaks based on center's proximity to the edge of the spectrum.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].
        """

        cf_params = guess[:, 0]
        bw_params = guess[:, 2] * self._bw_std_edge

        # Check if peaks within drop threshold from the edge of the frequency range
        keep_peak = \
            (np.abs(np.subtract(cf_params, self.freq_range[0])) > bw_params) & \
            (np.abs(np.subtract(cf_params, self.freq_range[1])) > bw_params)

        # Ensure self.std_limits is a numpy array
        self.std_limits = np.array(self.std_limits)
        self.cf_bounds = np.array(self.cf_bounds)
        # Drop peaks that fail the center frequency edge criterion
        guess = guess[keep_peak]
        self.std_limits = self.std_limits[keep_peak]
        self.cf_bounds = self.cf_bounds[keep_peak]

        return guess, self.std_limits , self.cf_bounds


    def _drop_peak_overlap(self, guess):
        """Checks whether to drop gaussians based on amount of overlap.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].
        std_limits : list of lists
            Filtered standard deviation limits corresponding to the kept peaks.

        Notes
        -----
        For any gaussians with an overlap that crosses the threshold,
        the lowest height guess Gaussian is dropped.
        """

        # Ensure guess is a numpy array
        

        # Ensure guess is a numpy array
        guess = np.array(guess)

        # Ensure std_limits is a numpy array
        self.std_limits = np.array(self.std_limits)

        # Debugging output to check the structure of guess and std_limits
        # Check if the dimensions are correct
        if guess.ndim != 2 or guess.shape[1] != 3:
            raise ValueError("Expected 'guess' to be a 2D array with shape [n_peaks, 3].")

        if self.std_limits.ndim != 2 or self.std_limits.shape[1] != 2:
            raise ValueError("Expected 'std_limits' to be a 2D array with shape [n_peaks, 2].")

        if len(guess) != len(self.std_limits):
            raise ValueError("The number of peaks in 'guess' does not match the number of std limits provided.")

        # Sort the peak guesses by increasing frequency and sort std_limits in the same order
        sorted_indices = np.argsort(guess[:, 0])
        guess = guess[sorted_indices]
        self.std_limits = self.std_limits[sorted_indices]
        self.cf_bounds = self.cf_bounds[sorted_indices]

        # Debugging output to check the sorted guess and std_limits
        # Calculate standard deviation bounds for checking amount of overlap
        #   The bounds are the gaussian frequency +/- gaussian standard deviation
        bounds = [[peak[0] - peak[2] * self._gauss_overlap_thresh,
                    peak[0] + peak[2] * self._gauss_overlap_thresh] for peak in guess]

        # Debugging output to check the bounds
        # Loop through peak bounds, comparing current bound to that of next peak
        #   If the left peak's upper bound extends pass the right peaks lower bound,
        #   then drop the Gaussian with the lower height
        drop_inds = []
        for ind, b_0 in enumerate(bounds[:-1]):
            b_1 = bounds[ind + 1]

            # Check if bound of current peak extends into next peak
            if b_0[1] > b_1[0]:
                # If so, get the index of the gaussian with the lowest height (to drop)
                drop_inds.append([ind, ind + 1][np.argmin([guess[ind][1], guess[ind + 1][1]])])

        # Drop any peaks guesses that overlap too much, based on threshold
        keep_peak = np.array([ind not in drop_inds for ind in range(len(guess))], dtype=bool)
        guess = guess[keep_peak]
        self.std_limits = self.std_limits[keep_peak]
        self.cf_bounds = self.cf_bounds[keep_peak]

        # Check if the dimensions are correct
        if guess.ndim != 2 or guess.shape[1] != 3:
            raise ValueError("Expected 'guess' to be a 2D array with shape [n_peaks, 3].")

        if self.std_limits.ndim != 2 or self.std_limits.shape[1] != 2:
            raise ValueError("Expected 'std_limits' to be a 2D array with shape [n_peaks, 2].")

        if len(guess) != len(self.std_limits):
            raise ValueError("The number of peaks in 'guess' does not match the number of std limits provided.")

        ##############edit this return later you don't need to do this #################
        return guess, self.std_limits , self.cf_bounds



    def _calc_r_squared(self):
        """Calculate the r-squared goodness of fit of the model, compared to the original data."""

        r_val = np.corrcoef(self.power_spectrum, self.fooofed_spectrum_)
        self.r_squared_ = r_val[0][1] ** 2


    def _calc_error(self, metric=None):
        """Calculate the overall error of the model fit, compared to the original data.

        Parameters
        ----------
        metric : {'MAE', 'MSE', 'RMSE'}, optional
            Which error measure to calculate:
            * 'MAE' : mean absolute error
            * 'MSE' : mean squared error
            * 'RMSE' : root mean squared error

        Raises
        ------
        ValueError
            If the requested error metric is not understood.

        Notes
        -----
        Which measure is applied is by default controlled by the `_error_metric` attribute.
        """

        
        # If metric is not specified, use the default approach
        metric = self._error_metric if not metric else metric
        if metric == 'AP':
            self.error_ = np.abs(self.power_spectrum - self._ap_fit).mean()
        if metric == 'AP_squared':
            self.error_ = np.mean((self.power_spectrum - self._ap_fit)**2)
        if metric == 'MAE':
            self.error_ = np.abs(self.power_spectrum - self.fooofed_spectrum_).mean()

        elif metric == 'MSE':
            self.error_ = ((self.power_spectrum - self.fooofed_spectrum_) ** 2).mean()

        elif metric == 'RMSE':
            self.error_ = np.sqrt(((self.power_spectrum - self.fooofed_spectrum_) ** 2).mean())



    def _prepare_data(self, freqs, power_spectrum, freq_range, spectra_dim=1):
        """Prepare input data for adding to current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power values, which must be input in linear space.
            1d vector, or 2d as [n_power_spectra, n_freqs].
        freq_range : list of [float, float]
            Frequency range to restrict power spectrum to. If None, keeps the entire range.
        spectra_dim : int, optional, default: 1
            Dimensionality that the power spectra should have.

        Returns
        -------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power spectrum values, in log10 scale.
            1d vector, or 2d as [n_power_specta, n_freqs].
        freq_range : list of [float, float]
            Minimum and maximum values of the frequency vector.
        freq_res : float
            Frequency resolution of the power spectrum.

        Raises
        ------
        DataError
            If there is an issue with the data.
        InconsistentDataError
            If the input data are inconsistent size.
        """

        # Check that data are the right types
        if not isinstance(freqs, np.ndarray) or not isinstance(power_spectrum, np.ndarray):
            raise DataError("Input data must be numpy arrays.")

        # Check that data have the right dimensionality
        if freqs.ndim != 1 or (power_spectrum.ndim != spectra_dim):
            raise DataError("Inputs are not the right dimensions.")

        # Check that data sizes are compatible
        if freqs.shape[-1] != power_spectrum.shape[-1]:
            raise InconsistentDataError("The input frequencies and power spectra "
                                        "are not consistent size.")

        # Check if power values are complex
        if np.iscomplexobj(power_spectrum):
            raise DataError("Input power spectra are complex values. "
                            "FOOOF does not currently support complex inputs.")

        # Force data to be dtype of float64
        #   If they end up as float32, or less, scipy curve_fit fails (sometimes implicitly)
        if freqs.dtype != 'float64':
            freqs = freqs.astype('float64')
        if power_spectrum.dtype != 'float64':
            power_spectrum = power_spectrum.astype('float64')

        # Check frequency range, trim the power_spectrum range if requested
        power_spectrum = self.rem_electric_noise(power_spectrum , freqs)

        if freq_range:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, freq_range)

        # Check if freqs start at 0 and move up one value if so
        #   Aperiodic fit gets an inf if freq of 0 is included, which leads to an error
        if freqs[0] == 0.0:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, [freqs[1], freqs.max()])
            if self.verbose:
                print("\nFOOOF WARNING: Skipping frequency == 0, "
                      "as this causes a problem with fitting.")

        # Calculate frequency resolution, and actual frequency range of the data
        freq_range = [freqs.min(), freqs.max()]
        freq_res = freqs[1] - freqs[0]

        # Log power values
        power_spectrum = np.log10(power_spectrum)

        ## Data checks - run checks on inputs based on check modes

        if self._check_freqs:
            # Check if the frequency data is unevenly spaced, and raise an error if so
            freq_diffs = np.diff(freqs)
            if not np.all(np.isclose(freq_diffs, freq_res)):
                raise DataError("The input frequency values are not evenly spaced. "
                                "The model expects equidistant frequency values in linear space.")
        if self._check_data:
            # Check if there are any infs / nans, and raise an error if so
            if np.any(np.isinf(power_spectrum)) or np.any(np.isnan(power_spectrum)):
                error_msg = ("The input power spectra data, after logging, contains NaNs or Infs. "
                             "This will cause the fitting to fail. "
                             "One reason this can happen is if inputs are already logged. "
                             "Inputs data should be in linear spacing, not log.")
                raise DataError(error_msg)

        return freqs, power_spectrum, freq_range, freq_res


    def _add_from_dict(self, data):
        """Add data to object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of data to add to self.
        """

        # Reconstruct object from loaded data
        for key in data.keys():
            setattr(self, key, data[key])


    def _check_loaded_results(self, data):
        """Check if results have been added and check data.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If results loaded, check dimensions of peak parameters
        #   This fixes an issue where they end up the wrong shape if they are empty (no peaks)
        if set(OBJ_DESC['results']).issubset(set(data.keys())):
            self.peak_params_ = check_array_dim(self.peak_params_)
            self.gaussian_params_ = check_array_dim(self.gaussian_params_)


    def _check_loaded_settings(self, data):
        """Check if settings added, and update the object as needed.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If settings not loaded from file, clear from object, so that default
        # settings, which are potentially wrong for loaded data, aren't kept
        if not set(OBJ_DESC['settings']).issubset(set(data.keys())):

            # Reset all public settings to None
            for setting in OBJ_DESC['settings']:
                setattr(self, setting, None)

            # If aperiodic params available, infer whether knee fitting was used,
            if not np.all(np.isnan(self.aperiodic_params_)):
                self.aperiodic_mode = infer_ap_func(self.aperiodic_params_)

        # Reset internal settings so that they are consistent with what was loaded
        #   Note that this will set internal settings to None, if public settings unavailable
        self._reset_internal_settings()


    def _regenerate_freqs(self):
        """Regenerate the frequency vector, given the object metadata."""

        self.freqs = gen_freqs(self.freq_range, self.freq_res)


    def _regenerate_model(self):
        """Regenerate model fit from parameters."""

        self.fooofed_spectrum_, self._peak_fit, self._ap_fit = gen_model(
            self.freqs, self.aperiodic_params_, self.gaussian_params_, return_components=True)
