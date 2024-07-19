"""
Implements a modified version of the FOOOF (now called ParamSpectra) algorithm to fit the power spectrum of EEG data by Donoghue et al. 2020 https://www.nature.com/articles/s41593-020-00744-x
This changes allows the user to preidentify the band-ranges where peaks would be found, and eliminates the need to tune parameters such as peak width, peak prominence, peak threshold.

Key idea is to first fit an aperiodic component L(F) = b - log(k+F^X) where b=offset, k=knee, X=exponent, F=frequency
Then, subtract the aperiodic component from the power spectrum to get the peak spectrum, and fit the peaks with gaussians.
For this implementation, we fit a sum of Gaussians whose peaks are each constrained to the predefined band-ranges.
Details of the original algorithm are here: https://fooof-tools.github.io/fooof/auto_tutorials/plot_01-ModelDescription.html#sphx-glr-auto-tutorials-plot-01-modeldescription-py

Example Usage:

import numpy as np
import path.to.refactored_parameterized_spectra as bandparams
power_spectrum = np.load('example_spectrum.npy') # some power spectrum in linear space
freqs = np.load('example_freqs.npy')
param_spectra = bandparams.ParamSpectra(band_basis='standard',  max_n_peaks=6, log_freqs=True,  l_freq=0.3, h_freq=250, prominence=0.5, linenoise=60, aperiodic_mode='knee', verbose=1)
param_spectra.fit(freqs, power_spectrum)
print(param_spectra.get_params_out())
plt.figure()
plt.plot(freqs, np.log10(power_spectrum), label='original spectrum')
plt.plot(freqs, param_spectra.modeled_spectrum_, label='parameterized spectrum')
plt.plot(freqs, param_spectra._ap_fit, label='aperiodic component')
plt.plot(freqs, param_spectra._peak_fit+np.mean(param_spectra._ap_fit), label='periodic component')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Parameterized Spectrum using standard bands')
plt.show()

"""
import warnings
import os
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import numpy as np
import json
import pandas as pd
import time
import argparse
import fooof
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

import mtbi_detection.features.feature_utils as fu
import mtbi_detection.data.data_utils as du
import mtbi_detection.data.transform_data as td

LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'features')
TDPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'psd_transform')
CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

class SpecParamError(Exception):
    """Base class for custom errors."""

class FitError(SpecParamError):
    """Error for a failure to fit."""

class NoDataError(SpecParamError):
    """Error for if data is missing."""

class DataError(SpecParamError):
    """Error for if there is a problem with the data."""

class InconsistentDataError(SpecParamError):
    """Error for if the data is inconsistent."""

class IncompatibleSettingsError(SpecParamError):
    """Error for if settings are incompatible."""

class NoModelError(SpecParamError):
    """Error for if the model is not fit."""

class ParamSpectra():
    # implement FOOOF but allow for predefined bandwindows
    def __init__(self, band_basis='standard',  max_n_peaks=6, log_freqs=True,  l_freq=0.3, h_freq=250, prominence=0.5, linenoise=60, aperiodic_mode='knee', verbose=1):
        """Model a power spectrum as a combination of aperiodic and periodic components.

        WARNING: frequency and power values inputs must be in linear space.

        Passing in logged frequencies and/or power spectra is not detected,
        and will silently produce incorrect results.

        Parameters
        ----------
        band_basis : str or list of [float, float] pairs, optional, default: 'standard'.
            Possible options (see feature_utils.make_bands( for implementation details): 'custom' 'standard', 'log-standard', 'linear', 'log10', 'custom_low'
        max_n_peaks : int, optional, default: 6, only used if bands is a specified as "log" or "linear"
            Maximum number of peaks to fit.
        log_freqs : bool, optional, default: True
            Whether or not to convert the input frequencies into natural log space.
        aperiodic_mode : {'fixed', 'knee'}
            Which approach to take for fitting the aperiodic component.
       
        l_freq : float, optional, default: 0.3
            The lowest frequency to consider in the aperiodic fit.
        h_freq : float, optional, default: 250
            The highest frequency to consider in the aperiodic fit. NOTE: must be <= sampling frequency/2 (Nyquist frequency)
        
        prominence : float, optional, default: 0.5
            The prominence value to use for peak finding of the powerline harmonics
        linenoise : float, optional, default: 60
            The base frequency of the powerline noise.
        
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
        modeled_spectrum_ : 1d array
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

        band_basis = _convert_band_basis(band_basis, max_n_peaks)
        self.band_basis = band_basis
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.aperiodic_mode = aperiodic_mode
        self.verbose = verbose
        self.log_freqs = log_freqs
        self.linenoise = linenoise
        self.prominence = prominence
        self.max_n_peaks = max_n_peaks

        
        self.bands = fu.make_bands(band_basis, min_freq=l_freq, fs=h_freq*2)

        # drop bands that are above or below the frequency range
        self.bands = [band for band in self.bands if band[0] < h_freq and band[1] > l_freq]
        ## PRIVATE SETTINGS
        # Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        #   Points are selected at a low percentile value to restrict to non-peak points
        self._ap_percentile_thresh = 0.025
        # Guess parameters for aperiodic fitting, [offset, knee, exponent]
        #   If offset guess is None, the first value of the power spectrum is used as offset guess
        #   If exponent guess is None, the abs(log-log slope) of first & last points is used
        self._ap_guess = (None, self.h_freq-self.l_freq, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        ## I replace the lower bound of the knee with 0
        self._ap_bounds = ((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf))
        # Threshold for how far a peak has to be from edge to keep.
        #   This is defined in units of gaussian standard deviation
        self._bw_std_edge = 1.0
        # Degree of overlap between gaussians for one to be dropped
        #   This is defined in units of gaussian standard deviation
        self._gauss_overlap_thresh = 0.75
        # Parameter bounds for center frequency when fitting gaussians, in terms of +/- std dev
        self._cf_bound = 1.5
        # The error metric to calculate, post model fitting. See `_calc_error` for options
        #   Note: this is for checking error post fitting, not an objective function for fitting
        self._error_metric = 'MAPE'

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
        # self._reset_internal_settings()
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
            self.gaussian_params_ = np.empty([0, 3]) # amplitude, mean, std
            self.peak_params_ = np.empty([0, 3]) # CF, PW, BW
            self.r_squared_ = np.nan
            self.error_ = np.nan

            self.modeled_spectrum_ = None

            self._spectrum_flat = None
            self._spectrum_peak_rm = None
            self._ap_fit = None
            self._peak_fit = None
            self._noise_pks = None
            self._noise_ranges = None
        
    def add_data(self, freqs, power_spectrum, freq_range=None, clear_results=True):
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
        if self.verbose>2:
            print("Adding data to object")
        self.freqs, self.power_spectrum, self.freq_range, self.freq_res = \
            self._prepare_data(freqs, power_spectrum, freq_range, 1)

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
            Frequency range to restrict power spectrum to.
            If None, keeps the entire range.
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
                            "Model fitting does not currently support complex inputs.")

        # Force data to be dtype of float64
        #   If they end up as float32, or less, scipy curve_fit fails (sometimes implicitly)
        if freqs.dtype != 'float64':
            freqs = freqs.astype('float64')

        if power_spectrum.dtype != 'float64':
            power_spectrum = power_spectrum.astype('float64')

        # Check frequency range, trim the power_spectrum range if requested
        if freq_range:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, freq_range)

        # Check if freqs start at 0 and move up one value if so
        #   Aperiodic fit gets an inf if freq of 0 is included, which leads to an error
        if freqs[0] == 0.0:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, [freqs[1], freqs.max()])
            if self.verbose:
                print("\nFITTING WARNING: Skipping frequency == 0, "
                      "as this causes a problem with fitting.")

        # Calculate frequency resolution, and actual frequency range of the data
        freq_range = [freqs.min(), freqs.max()]
        freq_res = freqs[1] - freqs[0]


        if self.verbose>2:
            print("Removing powerline harmonics, max_harmonic: ", int(freqs.max()//self.linenoise))
        # remove powerline harmonics
        self._max_harmonic=int(freqs.max()//self.linenoise)
        if self._max_harmonic > 0:
            harmonics = [60*int(ii) for ii in range(1, self._max_harmonic+1)]
            noise_pks, noise_ranges = detect_powerline_harmonics_peak_widths(freqs, power_spectrum, harmonics=harmonics, prominence=self.prominence, verbose=self.verbose-1)
            if self.verbose>4:
                print(f"Nans before interpolated power_spectrum: {np.sum(np.isnan(power_spectrum))}, Infs before interpolated power_spectrum: {np.sum(np.isinf(power_spectrum))}")
            if len(noise_pks) > 0:
                for noise_range in noise_ranges:
                    if self.verbose>3:
                        print(f"Removing noise range: {noise_range}")
                    freqs, power_spectrum = fooof.utils.interpolate_spectrum(np.copy(freqs), np.copy(power_spectrum), interp_range=noise_range)
                    if self.verbose>4:
                        print(f"Nans in interpolated power_spectrum: {np.sum(np.isnan(power_spectrum))}, Infs in power_spectrum: {np.sum(np.isinf(power_spectrum))}")
            self._noise_pks = noise_pks
            self._noise_ranges = noise_ranges

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
                             "Input data should be in linear spacing, not log.")
                raise DataError(error_msg)

        return freqs, power_spectrum, freq_range, freq_res
   
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
        ap_bounds = self._ap_bounds if self.aperiodic_mode == 'knee' \
            else tuple(bound[0::2] for bound in self._ap_bounds)

        # Second aperiodic fit - using results of first fit as guess parameters
        #  See note in _simple_ap_fit about warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs_ignore, spectrum_ignore, p0=popt,
                                                maxfev=self._maxfev, bounds=ap_bounds,
                                                ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                                check_finite=False)
        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding "
                         "parameters in the robust aperiodic fit.")
            raise FitError(error_msg) from excp
        except TypeError as excp:
            error_msg = ("Model fitting failed due to sub-sampling "
                         "in the robust aperiodic fit.")
            raise FitError(error_msg) from excp

        if self.verbose>4:
            print(f"Fit aperiodic component with {self.aperiodic_mode} mode, params shape: {aperiodic_params.shape}")
        return aperiodic_params

    def _simple_ap_fit(self, freqs, power_spectrum):
        # Get the guess parameters and/or calculate from the data, as needed
        #   Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [power_spectrum[0] if not self._ap_guess[0] else self._ap_guess[0]]
        kne_guess = [self._ap_guess[1]] if self.aperiodic_mode == 'knee' else []
        exp_guess = [np.abs((self.power_spectrum[-1] - self.power_spectrum[0]) /
                            (np.log10(self.freqs[-1]) - np.log10(self.freqs[0])))
                     if not self._ap_guess[2] else self._ap_guess[2]]

        # Get bounds for aperiodic fitting, dropping knee bound if not set to fit knee
        ap_bounds = self._ap_bounds if self.aperiodic_mode == 'knee' \
            else tuple(bound[0::2] for bound in self._ap_bounds)

        if self.verbose>4:
            print(f"Is ap_guess none? {self._ap_guess[0] is None}, {self._ap_guess[1] is None}, {self._ap_guess[2] is None}")
            print(f"Simple fit aperiodic component with {self.aperiodic_mode} mode, bounds: {ap_bounds}")
            print(f"power_spectrum shape {power_spectrum.shape} , off guess: {off_guess}, kne guess: {kne_guess}, exp guess: {exp_guess}")
        # Collect together guess parameters
        guess = np.array(off_guess + kne_guess + exp_guess)
        if self.verbose>4:
            print(f"Guess: {guess}")

        # Ignore warnings that are raised in curve_fit
        #   A runtime warning can occur while exploring parameters in curve fitting
        #     This doesn't effect outcome - it won't settle on an answer that does this
        #   It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                freqs, power_spectrum, p0=guess,
                                                maxfev=self._maxfev, bounds=ap_bounds,
                                                ftol=self._tol, xtol=self._tol, gtol=self._tol,
                                                check_finite=False)
        except FitError as excp:
            error_msg = ("Model fitting failed due to not finding parameters in "
                         "the simple aperiodic component fit.")
            raise FitError(error_msg) from excp

        return aperiodic_params
    
    def fit(self, freqs=None, power_spectrum=None, freq_range=None, prominence=0.5):
        """Fit the full power spectrum as a combination of periodic and aperiodic components.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to.
            If not provided, keeps the entire range.
        prominence : scalar input for finding powerline peaks

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
        # Fit the aperiodic component
        # If freqs & power_spectrum provided together, add data to object.
        self.prominence = prominence
        if self.verbose > 1:
            print(f"Fitting model on freqs: {freqs.shape}, power_spectrum: {power_spectrum.shape}")
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
        # if self.verbose:
        #     self._check_width_limits()
        if self.verbose:
            print(f"Using bands: {self.bands} to fit")
        
        try:
            # If not set to fail on NaN or Inf data at add time, check data here
            #   This serves as a catch all for curve_fits which will fail given NaN or Inf
            #   Because FitError's are by default caught, this allows fitting to continue
            if not self._check_data:
                if np.any(np.isinf(self.power_spectrum)) or np.any(np.isnan(self.power_spectrum)):
                    raise FitError("Model fitting was skipped because there are NaN or Inf "
                                   "values in the data, which preclude model fitting.")
            if self.verbose:
                print(f"Fit aperiodic component with {self.aperiodic_mode} mode")
            self.aperiodic_params_ = self._robust_ap_fit(self.freqs, self.power_spectrum)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Find peaks, and fit them with gaussians
            # st = time.time()
            self.gaussian_params_ = constrained_gaussian_fit(self.freqs, self._spectrum_flat, self.bands, log_freqs=self.log_freqs)
            # print(f"Fit Gaussians in {time.time()-st} seconds")
            # Calculate the peak fit
            #   Note: if no peaks are found, this creates a flat (all zero) peak fit
            if self.log_freqs:
                self._peak_fit = constrained_sum_of_gaussians(np.log(self.freqs), self.bands, self.gaussian_params_)
            else:
                self._peak_fit = constrained_sum_of_gaussians(self.freqs, self.bands, self.gaussian_params_)
            
            self._spectrum_peak_rm = self.power_spectrum - self._peak_fit

            # Run final aperiodic fit on peak-removed power spectrum
            #   This overwrites previous aperiodic fit, and recomputes the flattened spectrum
            self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._spectrum_peak_rm)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_)
            self._spectrum_flat = self.power_spectrum - self._ap_fit

            # Create full power_spectrum model fit
            self.modeled_spectrum_ = self._peak_fit + self._ap_fit

            # Convert gaussian definitions to peak parameters
            self.peak_params_ = self._create_peak_params(self.gaussian_params_)
            if self.verbose:
                print(f"Peak parameters: {self.peak_params_}")
                for ii, (cf, pw, bw) in enumerate(self.peak_params_):
                    if self.log_freqs:
                        cf = np.exp(cf)
                        bw = np.exp(bw)
                    print(f"Peak {ii}: CF: {cf}, PW: {pw}, BW: {bw}")

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

    def _create_peak_params(self, gaus_params):
        """Copies over the gaussian params to peak outputs, updating as appropriate.

        Parameters
        ----------
        gaus_params : 2d array (n_peaks, 3) where the 3 are (amplitude, center, standard deviation)
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
        with `freqs`, `modeled_spectrum_` and `_ap_fit` all required to be available.
        """

        gaus_copy = np.copy(gaus_params).reshape(-1, 3)
        peak_params = np.empty(gaus_copy.shape)
        for ii, peak in enumerate(gaus_copy):

            # Gets the index of the power_spectrum at the frequency closest to the CF of the peak
            ind = np.argmin(np.abs(self.freqs - peak[1]))

            # Collect peak parameter data
            peak_params[ii] = [peak[1], self.modeled_spectrum_[ind] - self._ap_fit[ind],
                               peak[2] * 2]

        return peak_params
    
    def _calc_r_squared(self):
        """Calculate the r-squared goodness of fit of the model, compared to the original data."""

        r_val = np.corrcoef(self.power_spectrum, self.modeled_spectrum_)
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

        if metric == 'MAE':
            self.error_ = np.abs(self.power_spectrum - self.modeled_spectrum_).mean()

        elif metric == 'MSE':
            self.error_ = ((self.power_spectrum - self.modeled_spectrum_) ** 2).mean()

        elif metric == 'RMSE':
            self.error_ = np.sqrt(((self.power_spectrum - self.modeled_spectrum_) ** 2).mean())
        
        elif metric == 'MAPE':
            self.error_ = np.abs((self.power_spectrum - self.modeled_spectrum_)/self.power_spectrum).mean()
        else:
            error_msg = "Error metric '{}' not understood or not implemented.".format(metric)
            raise ValueError(error_msg)

    def get_params_out(self):
        # check if it's fitted
        if not self.has_model:
            raise NoModelError("No model available to get parameters from.")
        
        guess_noise_pks = [self.linenoise*int(ii) for ii in range(1, self._max_harmonic+1)]
        out_noise_pks = [None]* self._max_harmonic
        out_noise_wids = [None] * self._max_harmonic # half width of the noise peak

        if len(self._noise_pks) > 0:
            for ii, noise_pk in enumerate(self._noise_pks):
                # find the closest noise peak in the noise_pks
                ndx = np.argmin(np.array([np.abs(noise_pk - guess_noise_pk) for guess_noise_pk in guess_noise_pks]))
                out_noise_pks[ndx] = noise_pk
                out_noise_wids[ndx] = (self._noise_ranges[ii][1] - self._noise_ranges[ii][0])/2



        # get the parameters
        params_out = {'aperiodic_params': self.aperiodic_params_,
                        'gaussian_params': self.gaussian_params_,
                        'noise_pks': np.array(out_noise_pks),
                        'noise_wids': np.array(out_noise_wids),
                        'peak_params': self.peak_params_.flatten(),
                        'r_squared': self.r_squared_,
                        'error': self.error_}
        return params_out

def trim_spectrum(freqs, power_spectra, f_range):
    """Extract a frequency range from power spectra.

    Parameters
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectra : 1d or 2d array
        Power spectral density values.
    f_range: list of [float, float]
        Frequency range to restrict to, as [lowest_freq, highest_freq].

    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    power_spectra_ext : 1d or 2d array
        Extracted power spectral density values.

    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high.
    It does not round to below or above f_low and f_high, respectively.


    Examples
    --------
    Using a simulated spectrum, extract a frequency range:

    >>> from specparam.sim import sim_power_spectrum
    >>> freqs, powers = sim_power_spectrum([1, 50], [1, 1], [10, 0.5, 1.0])
    >>> freqs, powers = trim_spectrum(freqs, powers, [3, 30])
    """

    # Create mask to index only requested frequencies
    f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])

    # Restrict freqs & spectra to requested range
    #   The if/else is to cover both 1d or 2d arrays
    freqs_ext = freqs[f_mask]
    power_spectra_ext = power_spectra[f_mask] if power_spectra.ndim == 1 \
        else power_spectra[:, f_mask]

    return freqs_ext, power_spectra_ext

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
    elif aperiodic_mode == 'knee':
        ap_func = expo_function
    else:
        raise ValueError("Requested aperiodic mode not understood.")

    return ap_func

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
    # inner 
    inner_arg = knee + xs**exp
    # replace anything <=0 with nearest neighbor
    inner_arg[inner_arg <= 0] = np.min(inner_arg[inner_arg > 0])
    ys = offset - np.log10(inner_arg)

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

def gen_aperiodic(freqs, aperiodic_params, aperiodic_mode=None):
    """Generate aperiodic values.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector to create aperiodic component for.
    aperiodic_params : list of float
        Parameters that define the aperiodic component.
    aperiodic_mode : {'fixed', 'knee'}, optional
        Which kind of aperiodic component to generate.
        If not provided, is inferred from the parameters.

    Returns
    -------
    ap_vals : 1d array
        Aperiodic values, in log10 spacing.
    """

    if not aperiodic_mode:
        aperiodic_mode = infer_ap_func(aperiodic_params)

    ap_func = get_ap_func(aperiodic_mode)

    ap_vals = ap_func(freqs, *aperiodic_params)

    return ap_vals

def infer_ap_func(aperiodic_params):
    """Infers which aperiodic function was used, from parameters.

    Parameters
    ----------
    aperiodic_params : list of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee'}
        Which kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    InconsistentDataError
        If the given parameters are inconsistent with any available aperiodic function.
    """

    if len(aperiodic_params) == 2:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) == 3:
        aperiodic_mode = 'knee'
    else:
        raise InconsistentDataError("The given aperiodic parameters are "
                                    "inconsistent with available options.")

    return aperiodic_mode

def detect_powerline_harmonics_peak_widths(frequencies, psd, harmonics=[60, 120, 180, 240], threshold=None, prominence=0.5,verbose=0):
    """
    Detect peaks at specified powerline harmonics in a power spectral density (PSD) and return their widths.

    Parameters:
    - frequencies: 1D array, frequency values (linear scale)
    - psd: 1D array, corresponding power spectral density values (linear scale)
    - harmonics: List of integers, powerline harmonics to detect (e.g., [60, 120, 180]).
    - threshold: Threshold for peak detection.

    Returns:
    - peaks: List of detected peaks.
    - widths: List of widths corresponding to the detected peaks.
    """

    # Find peaks in the PSD
    peaks, _ = find_peaks(np.log10(psd), height=threshold, prominence=prominence)
    if verbose>0:
        print(f"Found {len(peaks)} peaks")
        if verbose>1 and len(peaks)>0:
            print(f"Peak frequencies: {frequencies[peaks]}")

    detected_peaks = []

    for peak in peaks:
        # Check if the peak is at a powerline harmonic
        freq = frequencies[peak]
        for harmonic in harmonics:
            if np.isclose(freq, harmonic, rtol=5e-2):
                detected_peaks.append(peak)
                break

    # Calculate widths for the detected peaks
    if len(detected_peaks) > 0:
        widths, _, _, _ = peak_widths(psd, detected_peaks, rel_height=0.5)
        len_freqs = len(frequencies)-1
        freq_ranges = [[frequencies[max(int(pk-wd), 0)], frequencies[min(int(pk+wd), len_freqs)]]for pk, wd in zip(detected_peaks, widths)]
    else:
        freq_ranges = []
    freq_peaks = [frequencies[pk] for pk in detected_peaks]
    return freq_peaks, freq_ranges


def compute_gauss_std(fwhm):
    """Compute the gaussian standard deviation, given the full-width half-max.

    Parameters
    ----------
    fwhm : float
        Full-width half-max.

    Returns
    -------
    float
        Calculated standard deviation of a gaussian.
    """

    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def gen_gaussian(x, amplitude, mean, std_dev):
    return amplitude * stats.norm.pdf(x, loc=mean, scale=std_dev)

## https://stackoverflow.com/questions/16082171/curve-fitting-by-a-sum-of-gaussian-with-scipy
def constrained_sum_of_gaussians(x, bands=None, *params):
    result = np.zeros_like(x)
    params = np.reshape(params, (len(bands), 3))  # Reshape the flat parameter array
    for idx in range(len(bands)):
        # mask = np.logical_and(x > l_bound, x <= h_bound)
        result += gen_gaussian(x, *params[idx])
    return result
    
def constrained_gaussian_fit(freqs, power_spectrum, bands, log_freqs=False):
    """
    Fits len(bands) number of gaussians to the power spectrum, with the constraint that the gaussians sum to the power spectrum.
    and that the means of the gaussians are within the bands and that the std_devs are the width of the bands.
    Args:
        freqs (np.array): The frequencies of the power spectrum
        power_spectrum (np.array): The power spectrum in log10 space
        bands (list of tuples): The bands to fit the gaussians to, e.g. [(0, 4), (5, 10)]
        log_freqs (bool): Whether to log (base e) the frequencies
    Returns:
        popt (np.array): The parameters of the gaussians of shape (3*len(bands),) where the parameters are (amplitude, mean, std_dev) for each gaussian
    """
    num_gaussians = len(bands)
    if log_freqs:
        freqs = np.log(freqs)
        bands = [(np.log(l), np.log(h)) for l, h in bands]

    my_func = lambda x, *params: constrained_sum_of_gaussians(x, bands, *params)
    initial_guess = []
    bounds = []
    for l_bound, h_bound in bands:
        band_vals = power_spectrum[np.logical_and(freqs > l_bound, freqs <= h_bound)]
        freqs_in_band = freqs[np.logical_and(freqs > l_bound, freqs <= h_bound)]
        initial_guess.extend([np.max(band_vals), np.mean(freqs_in_band), compute_gauss_std((h_bound-l_bound)/2)])
        bounds.extend([(-np.inf, np.inf), (l_bound, h_bound), (0, compute_gauss_std((h_bound-l_bound)))])
        # # Initialize parameters with random values (you can customize this)
        # initial_guess.extend([np.max(power_spectrum), np.mean(freqs), 1.0])
    
    bounds = ([b[0] for b in bounds], [b[1] for b in bounds])
    # param_names = ['amplitude', 'mean', 'std_dev']
    # print(f"Bounds: {[f'{n}_{i//3}:({l},{h})' for i, (n, l, h) in enumerate(zip(param_names*num_gaussians, bounds[0], bounds[1]))]}")
    # Perform curve fittin
    popt, _ = curve_fit(my_func, freqs, power_spectrum, p0=initial_guess, maxfev=10000, bounds = bounds) # params get reshaped into len(bands),3

    return popt

def return_modeled_spectrum(freqs, ap_params, gaus_params, bands):
    ap_vals = gen_aperiodic(freqs, ap_params)
    peak_vals = constrained_sum_of_gaussians(freqs, [(0, 4), (5, 10)], *gaus_params)
    return ap_vals + peak_vals

def select_datapathdict(datapathdict, num_load_subjs=None, random_load=True, which_psd_segment='avg'):
    """
    Given a dictionary of paths to npz files structured as {subj:{'avg': path}} (output of transform_data.main()), select a subset of the subjects to load
    Args:
        datapathdict (dict): Dictionary of paths to npz files
        num_load_subjs (int): The number of subjects to load
        random_load (bool): Whether to randomly select the subjects
        which_psd_segment (str): Which segment of the PSD to load ['avg', 'first', 'second']
    Returns:
        new_datapathdict (dict): The new datapathdict with only the selected subjects
    """
    subjs = [key for key in datapathdict.keys() if key.isnumeric()]
    non_subjs = [key for key in datapathdict.keys() if not key.isnumeric()]
    if len(subjs) == 0:
        raise ValueError(f"No subjects found in {datapathdict}")

    if num_load_subjs is not None:
        if random_load:
            subjs = list(np.random.choice(subjs, num_load_subjs, replace=False))
        else:
            subjs = subjs[:num_load_subjs]

    assert len(subjs) > 0, f"Number of subjects {len(subjs)} is 0"
    new_datapathdict = {subj: datapathdict[subj][which_psd_segment] for subj in subjs}
    for key in non_subjs:
        new_datapathdict[key] = datapathdict[key][which_psd_segment]
    return new_datapathdict

def get_nan_params(band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', l_freq=0.3, h_freq=250, verbose=0, **kwargs):
    """
    Given the parameters used to fit the power spectrum, return a dictionary of nan parameters 
    (useful for when the fitting fails)
    Args:
        band_basis (str): The basis for the bands
        max_n_peaks (int): The maximum number of peaks to fit
        aperiodic_mode (str): The mode of the aperiodic component
        l_freq (float): The lowest frequency to fit
        verbose (int): The verbosity level
    Returns:
        params_out (dict): The dictionary of nan parameters {'aperiodic_params': np.array, 'gaussian_params': np.array, 'noise_pks': np.array, 'noise_wids': np.array, 'peak_params': np.array, 'r_squared': float, 'error': float}
    """
    if verbose > 0:
        print(f"Creating nan params")
    band_basis = _convert_band_basis(band_basis, max_n_peaks)
    band_ranges = fu.make_bands(band_basis, min_freq=l_freq, fs=h_freq*2)
    n_bands = len(band_ranges)


    if aperiodic_mode == 'knee':
        aperiodic_mode = np.empty(3)
        aperiodic_mode[:] = np.nan
    elif aperiodic_mode == 'fixed':
        aperiodic_mode = np.empty(2)
    else:
        raise ValueError(f"Aperiodic mode {aperiodic_mode} not understood")
    gaussian_params = np.empty((n_bands*3))
    peak_params = np.empty((n_bands, 3))
    max_harmonic = h_freq//60
    noise_pks = np.empty((max_harmonic))
    noise_wids = np.empty((max_harmonic))
    r_squared = np.nan
    error = np.nan

    params_out = {'aperiodic_params': aperiodic_mode,
                    'gaussian_params': gaussian_params.flatten(),
                    'noise_pks': noise_pks,
                    'noise_wids': noise_wids,
                    'peak_params': peak_params.flatten(),
                    'r_squared': r_squared,
                    'error': error}
    return params_out

def _isolate_fit_joint_parallel(filename, freqs, cdx, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, l_freq=0.3, h_freq=250, log_freqs=False, verbose=0):
    """
    Function that runs sequentially to fit the open and closed data for a single channel of a single file
    """
    loaded_psd_data = td.load_single_subject_transform_data(filename, inner_keys=['open_power', 'closed_power']) # {'open_power': np.array, 'closed_power': np.array}
    open_power = loaded_psd_data['open_power']
    closed_power = loaded_psd_data['closed_power']
    ps_closed = ParamSpectra(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, prominence=prominence, log_freqs=log_freqs, verbose=verbose)
    try:
        ps_closed.fit(freqs, closed_power[cdx])
        closed_params = ps_closed.get_params_out()
    except:
        print(f"Failed to fit closed data for channel {cdx}, file {filename}")
        closed_params = get_nan_params(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    ps_open = ParamSpectra(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, prominence=prominence, log_freqs=log_freqs, verbose=verbose)
    try:
        ps_open.fit(freqs, open_power[cdx])
        open_params = ps_open.get_params_out()
    except:
        print(f"Failed to fit open data for channel {cdx}, file {filename}")
        open_params = get_nan_params(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    return open_params, closed_params

def _parallel_fit_psds(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, l_freq=0.3, h_freq=250,  log_freqs=False, n_channels=1, parallel=False, n_jobs=1, fdx=None, n_files=None, inner_jobs=1, verbose=0):
    """ 
    Function that runs in parallel to fit the open and closed data for a single file
    """
    if fdx is not None:
        if n_files is not None:
            print(f"Loading subj {fdx+1}/({n_files})")
            ftt = time.time()
    loaded_psd_data = td.load_single_subject_transform_data(filename, inner_keys=['open_power', 'closed_power']) # {'open_power': np.array, 'closed_power': np.array}
    open_power = loaded_psd_data['open_power']
    closed_power = loaded_psd_data['closed_power']
    assert open_power.shape[0] == closed_power.shape[0], f"Number of channels {n_channels} does not match number of power spectra {open_power.shape[0]} and {closed_power.shape[0]}"
    assert n_channels <= open_power.shape[0], f"Number of channels {n_channels} is greater than number of power spectra {open_power.shape[0]} and {closed_power.shape[0]}"

    def _fit_power(power):
        ps_closed = ParamSpectra(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, prominence=prominence, log_freqs=log_freqs, verbose=verbose)
        try:
            ps_closed.fit(freqs, power)
            params_out = ps_closed.get_params_out()
        except:
            params_out = get_nan_params(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        time.sleep(0)
        return params_out
        
    if parallel:
        
        if verbose > 0:
            print(f"Fitting {n_channels} channels in parallel")
            ptime = time.time()

        n_inner_jobs = min(min(max((os.cpu_count()//2)//n_jobs,1), n_channels), inner_jobs)
        open_fit_closed_fit = Parallel(n_jobs=n_inner_jobs, verbose=5)(delayed(_isolate_fit_joint_parallel)(filename, freqs, cdx, band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, prominence=prominence, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs, verbose=verbose) for cdx in range(n_channels))
        open_fit = [ofcf[0] for ofcf in open_fit_closed_fit]
        closed_fit = [ofcf[1] for ofcf in open_fit_closed_fit]

        if verbose > 0:
            print(f"Finished fitting in {time.time() - ptime} seconds")
    else:
        if verbose > 0:
            print(f"Fitting {n_channels} channels in serial")
            ptime = time.time()
            
        open_fit = [_fit_power(open_power[cdx]) for cdx in range(n_channels)]
        closed_fit = [_fit_power(closed_power[cdx]) for cdx in range(n_channels)]
        if verbose > 0:
            print(f"Finished fitting in {time.time() - ptime} seconds")
    # will return n_channels len list of dictionaries
    if fdx is not None:
        if n_files is not None:
            print(f"Finished loading file {fdx+1}/({n_files}) in {time.time() - ftt} seconds")
    time.sleep(0)
    return open_fit, closed_fit

def _extract_parameterization_params(**kwargs):
    out_dict = {
        'band_basis': kwargs['band_basis'],
        'max_n_peaks': kwargs['max_n_peaks'],
        'aperiodic_mode': kwargs['aperiodic_mode'],
        'prominence': kwargs['prominence'],
        'fs_baseline': kwargs['fs_baseline'],
        'l_freq': kwargs['l_freq'],
        'h_freq': kwargs['h_freq'],
        'log_freqs': kwargs['log_freqs'],
        'num_load_subjs': kwargs['num_load_subjs'],
        'random_load': kwargs['random_load'],
        'choose_subjs': kwargs['choose_subjs'],
    }
    return out_dict



def _extract_open_closed_params(**kwargs):
    open_closed_params = {}
    locd_keys = ['data_l_freq', 'data_h_freq', 'fs_baseline', 'order', 'notches', 'notch_width', 'num_subjs', 'verbose', 'reference_method', 'reference_channels', 'keep_refs', 'bad_channels', 'filter_ecg', 'late_filter_ecg', 'ecg_l_freq', 'ecg_h_freq', 'ecg_thresh', 'ecg_method', 'include_ecg']
    for key in locd_keys:
        if 'data_' in key:
            open_closed_params[key.replace('data_', '')] = kwargs[key]
        else:
            open_closed_params[key] = kwargs[key]
    return open_closed_params

def _extract_td_params(**kwargs):
    td_params = {}
    td_keys = ['interpolate_spectrum', 'freq_interp_method', 'which_segment', 'bandwidth']
    for key in td_keys:
        td_params[key] = kwargs[key]
    return td_params


def convert_open_closed_fits_to_df(open_closed_fits: ArrayLike, subjs: ArrayLike, channels: ArrayLike, band_basis='custom', max_n_peaks=5, l_freq=0.3, h_freq=250,  aperiodic_mode='knee', verbose=0) -> pd.DataFrame:
    """
    Given a list of fitted parameters, the subjs corresponding to the list and the ordered channels, returns a dataframe (n_subjs, n_features)
    Inputs:
        - open_closed_fits: list of tuples of open and closed fitted parameters
        - subjs: list of subjects 
        - channels: list of channels
        - band_basis: The basis for the bands ['custom', 'standard', 'log-standard', 'custom_low']
        - max_n_peaks: The maximum number of peaks to fit
        - l_freq: The lowest frequency to fit
        - h_freq: The highest frequency to fit
    Returns:
        - spectral_features: dataframe of spectral features (n_subjs, n_features)
    """
    assert len(open_closed_fits) == len(subjs), f"Number of open closed fits {len(open_closed_fits)} does not match number of subjects {len(subjs)}"
    assert len(channels) == len(open_closed_fits[0][0]), f"Number of channels {len(channels)} does not match number of channels in open closed fits {len(open_closed_fits[0][0])}"
    
    band_basis = _convert_band_basis(band_basis, max_n_peaks)
    bands_ranges = fu.make_bands(band_basis, min_freq=l_freq, fs=h_freq*2)
    n_channels = len(open_closed_fits[0][0])
    n_aps = 3 if aperiodic_mode == 'knee' else 2
    n_gauss = 3*len(bands_ranges)
    n_peaks = 3*len(bands_ranges)
    n_rsq = 1
    n_err = 1
    n_noise_wids = 4
    n_noise_pks = 4
    n_params = n_aps + n_gauss + n_peaks + n_rsq + n_err + n_noise_wids + n_noise_pks
    n_features=  n_channels*n_params # 2 is for open and closed
    dummy_num =  -1000000.512
    open_array = np.ones((len(subjs), n_features)) * dummy_num
    closed_array = np.ones((len(subjs), n_features)) * dummy_num
    column_names = []
    parameter_groups = ['aperiodic_params', 'gaussian_params', 'peak_params', 'r_squared', 'error', 'noise_wids', 'noise_pks']
    group_sizes = {'aperiodic_params': n_aps, 'gaussian_params': n_gauss, 'peak_params': n_peaks, 'r_squared': n_rsq, 'error': n_err, 'noise_wids': n_noise_wids, 'noise_pks': n_noise_pks}
    for sdx, (open_fit, closed_fit) in enumerate(open_closed_fits):
        filled_to = 0
        for cdx in range(n_channels):
            for pdx, param_group in enumerate(parameter_groups):
                if group_sizes[param_group] > 1:
                    assert len(list(open_fit[cdx][param_group])) == len(list(closed_fit[cdx][param_group])) == group_sizes[param_group], f"Number of parameters in {param_group} for channel {cdx} does not match"
                col_idx_start = filled_to
                col_idx_end = filled_to + group_sizes[param_group]
                open_array[sdx, col_idx_start:col_idx_end] = open_fit[cdx][param_group]
                closed_array[sdx, col_idx_start:col_idx_end] = closed_fit[cdx][param_group]
                filled_to = col_idx_end
                if sdx == 0:
                    if param_group == 'aperiodic_params':
                        group_features = ['ap_offset', 'ap_knee', 'ap_exp'] if n_aps == 3 else ['ap_offset', 'ap_exp']
                    elif param_group == 'gaussian_params':
                        group_features = []
                        for bdx in range(len(bands_ranges)):
                            group_features.extend([f'gauss_amp_{bands_ranges[bdx]}', f'gauss_mean_{bands_ranges[bdx]}', f'gauss_std_{bands_ranges[bdx]}'])
                    elif param_group == 'peak_params':
                        group_features = []
                        for bdx in range(len(bands_ranges)):
                            group_features.extend([f'peak_cf_{bands_ranges[bdx]}', f'peak_pw_{bands_ranges[bdx]}', f'peak_bw_{bands_ranges[bdx]}'])
                    elif param_group == 'r_squared':
                        group_features = ['rsq']
                    elif param_group == 'error':
                        group_features = ['mape']
                    elif param_group == 'noise_wids':
                        group_features = []
                        for pdx in range(n_noise_wids):
                            group_features.extend([f'noise_range_{pdx}_{pdx%2}'])
                    elif param_group == 'noise_pks':
                        group_features = []
                        for pdx in range(n_noise_pks):
                            group_features.extend([f'noise_pk_{pdx}'])
                    else:
                        raise ValueError(f"Parameter group {param_group} not understood")
                    column_names.extend([f'{feature}_{channels[cdx]}' for feature in group_features])
    assert len(column_names) == n_features, f"Number of columns {len(column_names)} does not match number of features {n_features}"
    open_closed_features =  np.concatenate([open_array, closed_array], axis=1)
    # assert that none of features are dummy_num
    assert not np.any(open_closed_features == dummy_num), f"Found dummy number {dummy_num} in open closed features"
    open_closed_feature_names = [f'Parameterized open_{feature}' for feature in column_names] + [f'Parameterized closed_{feature}' for feature in column_names]
    assert open_closed_features.shape == (len(subjs), len(open_closed_feature_names)), f"Shape of open closed features {open_closed_features.shape} does not match shape of open closed feature names {len(open_closed_feature_names)}"
    spectral_features = pd.DataFrame(open_closed_features, columns=open_closed_feature_names, index=subjs)
    return spectral_features
    

def _convert_band_basis(band_basis, max_n_peaks=None):
    """
    Convert the band basis to the correct format for feature_utils.make_bands
    """
    if max_n_peaks is not None and band_basis in ['standard', 'log-standard']:
        band_basis = f'{band_basis}{int(max_n_peaks)}'
    else:
        band_basis = band_basis
    return band_basis

#### Tests
def _test_periodic_fit():
    """
    Tests the fitting of the periodic component
    """
    freqs = np.linspace(0, 10, 100)
    power_spectrum = 2 * np.exp(-(freqs - 2)**2 / 1) + 1.5 * np.exp(-(freqs - 6)**2 / 0.5)
    bands = [(0, 4), (5, 10)]

    # # Specify the number of Gaussians
    num_gaussians = len(bands)

    # # Fit the data
    fit_params = constrained_gaussian_fit(freqs, power_spectrum, bands)

    # Extract the parameters for each Gaussian
    fit_params = np.reshape(fit_params, (num_gaussians, 3))

    fitted_spectrum = constrained_sum_of_gaussians(freqs, bands, *fit_params.flatten())
    assert np.allclose(fitted_spectrum, power_spectrum, atol=0.1), "Fitted spectrum does not match original spectrum"

def _test_convert_open_closed_fits_to_df():
    """
    Tests the conversion of open closed fits to a dataframe
    """
    # randomly load two subjects
    print("Testing convert open closed fits to df")
    loadpath = os.path.join(TDPATH, 'params/params0/')
    subjs = [key for key in os.listdir(loadpath) if key.isnumeric()]
    subjs = np.random.choice(subjs, 2, replace=False)
    filenames = [os.path.join(loadpath, subj, 'avg', f'open_closed_multitaper_psds_{subj}.npz') for subj in subjs]
    channelpath = os.path.join(loadpath, 'channels.npy')
    freqs = np.load(os.path.join(loadpath, 'common_freqs.npy'))
    channels = np.load(channelpath)
    open_closed_fits = [_parallel_fit_psds(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0) for filename in filenames]
    channels = channels[:2]
    out_df = convert_open_closed_fits_to_df(open_closed_fits, subjs, channels, band_basis='custom', max_n_peaks=5, l_freq=0.3, h_freq=250, aperiodic_mode='knee')
    print(f"Finished testing open closed fits to df, shape {out_df.shape}")

def _test_parallel_vs_non_parallel_fit_psds():
    """
    Tests the parallel vs non parallel fitting of the psds
    """
    # randomly load two subjects
    print("Testing convert open closed fits to df")
    loadpath ='/shared/roy/mTBI/data_transforms/loaded_transform_data/params/params5/'
    loadpath = os.path.join(TDPATH, 'params/params0/')
    subjs = [key for key in os.listdir(loadpath) if key.isnumeric()]
    subjs = np.random.choice(subjs, 2, replace=False)
    freqs = np.load(os.path.join(loadpath, 'common_freqs.npy'))
    filenames = [os.path.join(loadpath, subj, 'avg', f'open_closed_multitaper_psds_{subj}.npz') for subj in subjs]
    ptime = time.time()
    open_closed_fits = Parallel(n_jobs=len(subjs), verbose=5)(delayed(_parallel_fit_psds)(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0, parallel=True, inner_jobs=2) for filename in filenames)
    print(f"Parallel outer, Inner parallel time: {time.time() - ptime}")
    pnptime = time.time()
    open_closed_fits = Parallel(n_jobs=len(subjs), verbose=5)(delayed(_parallel_fit_psds)(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0, parallel=False) for filename in filenames)
    print(f"Parallel outer, linear inner time: {time.time() - pnptime}")

    nptime = time.time()
    open_closed_fits = Parallel(n_jobs=1, verbose=5)(delayed(_parallel_fit_psds)(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0, parallel=True, inner_jobs=len(subjs)*2) for filename in filenames)
    print(f"Linear outer, parallel inner time: {time.time()-nptime}")
    ntime = time.time()
    open_closed_fits = Parallel(n_jobs=1, verbose=5)(delayed(_parallel_fit_psds)(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0, parallel=False) for filename in filenames)
    print(f"Linear outer, linear inner time: {time.time()-ntime}")

    finaltime = time.time()

    print(f"Overall:\n\tParallel outer, parallel inner: {pnptime-ptime}\t Parallel outer, linear inner: {nptime-pnptime}") # parallel outer, linear inner fastest at this scale
    print(f"\tLinear outer, parallel inner: {ntime-nptime}\t Linear outer, linear inner: {finaltime-ntime}")

    print(f"Finished testing parallel vs non parallel fit psds")


def _test_fit():
    # randomly load one subject
    print("Testing fit")
    loadpath = os.path.join(TDPATH, 'params/params0/')
    subjs = [key for key in os.listdir(loadpath) if key.isnumeric()]
    subjs = np.random.choice(subjs, 2, replace=False)
    filenames = [os.path.join(loadpath, subj, 'avg', f'open_closed_multitaper_psds_{subj}.npz') for subj in subjs]
    channelpath = os.path.join(loadpath, 'channels.npy')
    channels = np.load(channelpath)
    freqs = np.load(os.path.join(loadpath, 'common_freqs.npy'))
    open_closed_fits = [_parallel_fit_psds(filename, freqs, band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, n_channels=2, verbose=0, log_freqs=True, parallel=False) for filename in filenames]
    channels = channels[:2]
    out_df = convert_open_closed_fits_to_df(open_closed_fits, subjs, channels, band_basis='custom', max_n_peaks=5, l_freq=0.3, h_freq=250,  aperiodic_mode='knee')
    print(f"Finished testing fit, shape {out_df.shape}")

def _test_get_nan_params():
    nan_params = get_nan_params(band_basis='custom', max_n_peaks=5, min_peak_height=0.0, aperiodic_mode='knee', l_freq=0.3, h_freq=250, verbose=0)
    n_bands = len(fu.make_bands('custom', min_freq=0.3, fs=500))
    assert len(nan_params['aperiodic_params']) == 3, f"Number of aperiodic params {len(nan_params['aperiodic_params'])} does not match 3"
    assert len(nan_params['gaussian_params']) == 3*n_bands, f"Number of gaussian params {len(nan_params['gaussian_params'])} does not match 3*5"
    assert len(nan_params['peak_params']) == 3*n_bands, f"Number of peak params {len(nan_params['peak_params'])} does not match 3*5"
    assert len(nan_params['noise_pks']) == 4, f"Number of noise pks {len(nan_params['noise_pks'])} does not match 4"
    assert len(nan_params['noise_wids']) == 4, f"Number of noise wids {len(nan_params['noise_wids'])} does not match 4"
    assert np.isnan(nan_params['r_squared']), f"R squared {nan_params['r_squared']} is not nan"
    assert np.isnan(nan_params['error']), f"Error {nan_params['error']} is not nan"
    print("Passed test get nan params")

def _run_tests():
    _test_periodic_fit()
    _test_fit()
    _test_get_nan_params()
    _test_convert_open_closed_fits_to_df()
    _test_parallel_vs_non_parallel_fit_psds()
    print("All tests passed")


def main(open_closed_params, transform_params, open_closed_path=LOCD_DATAPATH, featurepath=FEATUREPATH, num_load_subjs=151, n_jobs=1, random_load=False, internal_folder='data/internal/', which_psd_segment='avg', \
         band_basis='custom', max_n_peaks=5, aperiodic_mode='knee', prominence=0.5, fs_baseline=500, l_freq=0.3, h_freq=250,  choose_subjs='train', remove_noise=True, log_freqs=False, verbose=0):
    """
    Extracts the parameterized spectra as a dataframe from PSD data
    Inputs:
        - transform_params (dict): Parameters for the PSD transform
        - transform_datapath (str): Path to the transform data
        - featurepath (str): Path to save the feature data
        - num_load_subjs (int): Number of subjects to load
        - n_jobs (int): Number of jobs to run in parallel
        - random_load (bool): Whether to randomly load subjects
        - bands (str): Basis for the bands
        - max_n_peaks (int): Maximum number of peaks to fit
        - aperiodic_mode (str): Mode for the aperiodic component
        - prominence (float): Prominence for peak detection
        - fs (int): Sampling frequency
        - l_freq (float): Lower frequency bound
        - h_freq (float): Upper frequency bound
        - log_freqs (bool): Whether to log the frequencies
        - verbose (int): Verbosity level
    Outputs:
        - spectral_parameters (pd.DataFrame): Dataframe of the spectral parameters with subjects as the index
    
    """
    model_params = _extract_parameterization_params(band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, prominence=prominence, \
                                            fs_baseline=fs_baseline, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs,
                                            num_load_subjs=num_load_subjs, random_load=random_load, choose_subjs=choose_subjs)
    parameterized_savepath = os.path.join(featurepath, 'parameterized_psds')
    all_params = du.make_dict_saveable({**transform_params, **open_closed_params, **model_params})
    du.clean_params_path(parameterized_savepath)
    savepath, found_match = du.check_and_make_params_folder(parameterized_savepath, all_params)
    savefilename = os.path.join(savepath, 'spectral_parameters.csv')
    if found_match:
        if verbose > 0:
            print(f"Found match for {all_params} at {savepath}")
        spectral_parameters = pd.read_csv(savefilename, index_col=0)
        assert set(fu.select_subjects_from_dataframe(spectral_parameters, choose_subjs,internal_folder).index) == set(spectral_parameters.index), f"Subjects in dataframe {set(spectral_parameters.index)} do not match subjects in choose_subjs {set(fu.select_subjects_from_dataframe(spectral_parameters, choose_subjs,internal_folder).index)}"
    else:
        if verbose > 0:
            print(f"No match found for {all_params} at {savepath}")
            print(f"Running parameterized psd with model params {model_params}")
        num_load_subjs = model_params['num_load_subjs']
        random_load = model_params['random_load']

        transform_data_pathdict = td.main(locd_params=open_closed_params, locd_savepath=open_closed_path, n_jobs=1, as_paths=True, **transform_params)
        select_pathdict = select_datapathdict(fu.select_subjects_from_dict(transform_data_pathdict['subj_data'], choose_subjs, internal_folder=internal_folder), num_load_subjs=num_load_subjs, random_load=random_load, which_psd_segment=which_psd_segment)
        channels = transform_data_pathdict['channels']
        freqs = transform_data_pathdict['common_freqs']
        if h_freq is None:
            h_freq = fs_baseline//2
        subjs = [key for key in select_pathdict if key.isnumeric()]
        n_subjs = len(subjs)
        
        # parallelize the open and closed fitting for each subject
        open_closed_fits = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_parallel_fit_psds)(select_pathdict[subj], freqs, band_basis=band_basis, max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode, prominence=prominence, l_freq=l_freq, h_freq=h_freq, log_freqs=log_freqs,n_jobs=n_jobs, fdx=fdx, n_files=n_subjs, n_channels=len(channels), verbose=verbose) for fdx, subj in enumerate(subjs))
        # save the results: ap_params n_chan x ((offset, knee, exp), gaussian_params (n_bands, amp, mean, std), peak_params (n_bands, cf, pw, bw), r_squared (1), mape error (1), noise_ranges (4), noise_pks (4))
        
        spectral_parameters = convert_open_closed_fits_to_df(open_closed_fits, subjs, channels, band_basis=band_basis, max_n_peaks=max_n_peaks, l_freq=l_freq, h_freq=h_freq, aperiodic_mode=aperiodic_mode, verbose=verbose)
        if verbose>0:
            print(f"Saving parameterized spectral features to {savefilename}")
        spectral_parameters.to_csv(savefilename)
        with open(os.path.join(savepath, 'params.json'), 'w') as f:
            json.dump(all_params, f)

    if remove_noise:
        nonnoise_cols = [col for col in spectral_parameters.columns if 'noise' not in col]
        spectral_parameters = spectral_parameters[nonnoise_cols]

    return spectral_parameters

if __name__ == '__main__':
    # _run_tests()

    # _test_convert_open_closed_fits_to_df()
    parser = argparse.ArgumentParser(description='Fit power spectra to parameters')

    ## LOCD params
    parser.add_argument('--data_l_freq', type=float, default=0.3)
    parser.add_argument('--data_h_freq', type=float, default=None)
    parser.add_argument('--fs_baseline', type=float, default=500)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--notches', type=int, nargs='+', default=[60, 120, 180, 240])
    parser.add_argument('--notch_width', type=float, nargs='+', default=[2, 1, 0.5, 0.25])
    parser.add_argument('--num_subjs', type=int, default=151)
    parser.add_argument('--reference_method', type=str, default='CSD')
    parser.add_argument('--reference_channels', type=str, nargs='+', default=['A1', 'A2'])
    parser.add_argument('--keep_refs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bad_channels', type=str, nargs='+', default=['T1', 'T2'])
    parser.add_argument('--filter_ecg', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--late_filter_ecg', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ecg_l_freq', type=float, default=8)
    parser.add_argument('--ecg_h_freq', type=float, default=16)
    parser.add_argument('--ecg_thresh', type=str, default='auto')
    parser.add_argument('--ecg_method', type=str, default='correlation')
    parser.add_argument('--include_ecg', action=argparse.BooleanOptionalAction, default=True)

    ## Transform params
    parser.add_argument('--interpolate_spectrum', type=int, default=1000)
    parser.add_argument('--freq_interp_method', type=str, default='linear', choices=['linear', 'log', 'log10'])
    parser.add_argument('--which_segment', type=str, default='avg', choices=['first', 'second', 'avg'], help='Which segment to use for the multitaper')
    parser.add_argument('--bandwidth', type=float, default=1)


    ## Parameterization params

    parser.add_argument('--num_load_subjs', type=int, default=151, help='Number of subjects to load')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    parser.add_argument('--random_load', type=bool, default=False, help='Whether to randomly load subjects')
    parser.add_argument('--choose_subjs', type=str, default='train', help='Which subjects to choose')
    parser.add_argument('--band_basis', type=str, default='custom', help='Bands to fit the gaussians to, e.g. [(0, 4), (5, 10)]')
    parser.add_argument('--max_n_peaks', type=int, default=5, help='Maximum number of peaks to fit')
    parser.add_argument('--aperiodic_mode', type=str, default='knee', help='Aperiodic mode to fit')
    parser.add_argument('--l_freq', type=float, default=0.3, help='Low frequency to fit')
    parser.add_argument('--h_freq', type=float, default=250, help='High frequency to fit')
    parser.add_argument('--log_freqs', action=argparse.BooleanOptionalAction, default=True, help='Whether to log (base e) the frequencies')
    parser.add_argument('--prominence', type=float, default=0.5, help='Prominence to fit')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')

    args = parser.parse_args()
    uin = input(f"Running with args: {args}, continue? (y/n)")
    if uin == 'y':
        st = time.time()
        open_closed_params = _extract_open_closed_params(**vars(args))
        transform_params = _extract_td_params(**vars(args))
        parameterization_params = _extract_parameterization_params(**vars(args))
        param_df = main(open_closed_params, transform_params, **parameterization_params)
        print(f"Finished in {time.time()-st} seconds")

    else:
        print("Exiting")
