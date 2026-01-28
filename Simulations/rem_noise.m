function [psd] = rem_noise(freqs, psd)
    expected_notch_and_noise = [46, 55];
    
    settings_notch_and_noise = struct();
    settings_notch_and_noise.peak_width_limits = [0.5, 2];
    settings_notch_and_noise.max_n_peaks = 1;
    settings_notch_and_noise.min_peak_height = 0;
    settings_notch_and_noise.peak_threshold = 0;
    settings_notch_and_noise.aperiodic_mode = 'fixed';
    settings_notch_and_noise.verbose = true;
    
    f_range_Notch_or_Noise = [expected_notch_and_noise(1), expected_notch_and_noise(2)];
    fooof_results_N = fooof(freqs, psd, f_range_Notch_or_Noise, settings_notch_and_noise, true);

    if ~isempty(fooof_results_N.peak_params)
        Notch_width = [48, 52];
        freq_indices_of_interest = freqs >= Notch_width(1) & freqs <= Notch_width(2);
        fff = freqs(freq_indices_of_interest);
        offset = fooof_results_N.aperiodic_params(1);
        alpha = fooof_results_N.aperiodic_params(2);
        psd(freq_indices_of_interest) = 1./ (fff.^alpha)*10^offset;
    end 
end

