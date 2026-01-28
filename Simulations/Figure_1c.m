clear all;
clc;
addpath('C:\Users\guku064b\matlab\fooof_mat-main\fooof_mat');

%fooof_settings
settings_theta = struct();
settings_theta.peak_width_limits = [0.5, 8];
settings_theta.max_n_peaks = 10;
settings_theta.min_peak_height = 0.04;
settings_theta.peak_threshold = 0.3;
settings_theta.aperiodic_mode = 'fixed';
settings_theta.verbose = true; 

%pwelch_settings
pwelch_window = 3000;
overlap_percentage = 0.5;  % 50% overlap
num_overlap = round(overlap_percentage * pwelch_window);
nfft = 4000;

%plot_settings
cf_start = 6;
cf_end = 9;
bw_start = 2;
bw_end = 5;
lengh_in_pixels = 50; % change resolution here
cf_step = (cf_end - cf_start)/(lengh_in_pixels-1);
bw_step = (bw_end - bw_start)/(lengh_in_pixels-1);
cfs = cf_start:cf_step:cf_end;
bws = bw_start:bw_step:bw_end;

exponent = 1.2;
peak_power = 0.6;

Fs = 1000;         % sampling frequency, Hz
T = 30*60;             % signal duration, s
time   = 1/Fs:1/Fs:T;
N = round(Fs*T);    % number of samples

varrying_theta_results = struct();
varrying_theta_results.exponent = exponent;
varrying_theta_results.peak_power = peak_power;

n_loops = 20;
for l = 1:n_loops
    disp(['current_loop: ', num2str(l), ' out of ', num2str(n_loops)]);
    % generate random time signal
    x = randn(1, N);
    x = x(:);
    
    NumUniquePts = ceil((size(x, 1)+1)/2); % calculate the number of unique fft points
    k = 1:NumUniquePts; k = k(:); % vector with frequency indexes for 1/f, to avoid inf first value
    
    % prepare a vector with frequency indexes 
    freq_res = (Fs/2)/(NumUniquePts-1);
    freqs = 0:freq_res: Fs/2;
    freqs = freqs'; % make vertical, so that it matches shape of phase
    
    ground_truth_cfs = zeros(length(cfs), length(bws));
    ground_truth_bws = zeros(length(cfs), length(bws));
    iters_fooof_cfs = zeros(length(cfs), length(bws));
    iters_fooof_exp = zeros(length(cfs), length(bws));
    iters_fooof_error = zeros(length(cfs), length(bws));
    for i = 1:length(cfs)
        for j = 1:length(bws)
            % time signal generation
            X = fft(x);
            X = X(1:NumUniquePts, :);
    
            %%multiply with custom shaped function
            expn = exponent/2;
            offset = 14; %consistently observed in our tetrode datasets
            offset = offset/2;
            offset = offset - (log10(mean(abs(X))));
            
            cf = cfs(i);
            bw = bws(j);
            sd = bw/2;
            p = peak_power; 
            
            X = X.*(1 ./ (freqs.^expn) * 10^offset);
            X = X.* sqrt((10.^(p * exp(-(freqs - cf).^2 / (2 * sd^2))))); %functions have to be muliplied! 10^a + 10^b = 10^(a+b)
            
            %add 50hz noise
            cf = 50; %set this
            bw = 0.5; %set this
            sd = bw/2;
            p = 0.25; %set this
            X = X.* sqrt((10.^(p * exp(-(freqs - cf).^2 / (2 * sd^2)))));
            
            %add 150hz harmonic
            cf = 150; %set this
            bw = 0.5; %set this
            sd = bw/2;
            p = 0.25; %set this
            X = X.* sqrt((10.^(p * exp(-(freqs - cf).^2 / (2 * sd^2)))));
            
            noise_amp = randn(length(X), 1); %scaling determines noise of spectrum
            X = X.*noise_amp;
            X(1) = 0; % cant use ifft with inf value 
            
            % perform ifft
            if rem(size(x, 1), 2)	% odd length excludes Nyquist point 
                % reconstruct the whole spectrum
                X = [X; conj(X(end:-1:2, :))];
                
                % take ifft of X
                time_signal = real(ifft(X));   
            else                    % even length includes Nyquist point  
                % reconstruct the whole spectrum
                X = [X; conj(X(end-1:-1:2, :))];
                
                % take ifft of X
                time_signal = real(ifft(X));  
            end



            [psd_welch, freqs_welch] = pwelch(time_signal,hann(pwelch_window), num_overlap, nfft, Fs);
            psd_welch = rem_noise(freqs_welch, psd_welch);
        
            fooof_results = fooof(freqs_welch, psd_welch,  [1, 100], settings_theta, true);
    
            %disp(['cf: ', num2str(cfs(i)), '; bw: ', num2str(bws(j))]);
            %disp(fooof_results.peak_params);
            %disp('');
            %fooof_plot(fooof_results);
            ground_truth_cfs(i, j) = cfs(i);
            ground_truth_bws(i, j) = bws(j);
            iters_fooof_cfs(i, j) = fooof_results.peak_params(1);
            iters_fooof_exp(i, j) = fooof_results.aperiodic_params(2);
            iters_fooof_error(i, j) = fooof_results.error;
        end
    end
    if l == 1
        %varrying_theta_results.time_signals = time_signals;
        varrying_theta_results.groundt_cfs = ground_truth_cfs;
        varrying_theta_results.groundt_bws = ground_truth_bws;
        
        varrying_theta_results.iters_fooof_cfs = iters_fooof_cfs;
        iters_fooof_diff_cf = iters_fooof_cfs - ground_truth_cfs;
        varrying_theta_results.iters_cf_diff = iters_fooof_diff_cf;
        varrying_theta_results.iters_fooof_error = iters_fooof_error;
        iters_fooof_exp_diff = iters_fooof_exp - exponent;
        varrying_theta_results.iters_exp_diff = iters_fooof_exp_diff;
    else
        iters_fooof_diff_cf = iters_fooof_cfs - ground_truth_cfs;
        varrying_theta_results.iters_fooof_cfs = varrying_theta_results.iters_fooof_cfs + iters_fooof_cfs;
        varrying_theta_results.iters_cf_diff = varrying_theta_results.iters_cf_diff + iters_fooof_diff_cf;
        varrying_theta_results.iters_fooof_error = varrying_theta_results.iters_fooof_error + iters_fooof_error;
        iters_fooof_exp_diff = iters_fooof_exp -exponent;
        varrying_theta_results.iters_exp_diff = varrying_theta_results.iters_exp_diff + iters_fooof_exp_diff;
    end
end
varrying_theta_results.iters_fooof_cfs = varrying_theta_results.iters_fooof_cfs ./ n_loops;
varrying_theta_results.iters_cf_diff = varrying_theta_results.iters_cf_diff ./ n_loops;
varrying_theta_results.iters_fooof_error = varrying_theta_results.iters_fooof_error ./ n_loops;
varrying_theta_results.iters_exp_diff = varrying_theta_results.iters_exp_diff ./ n_loops;


%iters_fooof
error_matrix = varrying_theta_results.iters_cf_diff;
%error_matrix = varrying_theta_results.iters_exp_diff;
%error_matrix = abs(error_matrix);
% Plot using imagesc
figure;
imagesc(bws, cfs, error_matrix);
colorbar;
ylabel('Center Frequency');
xlabel('Bandwidth');
title('Errors in exp estimate, using normal FOOOF');
set(gca, 'YDir', 'normal');
axis tight;

save_path = 'C:\Users\guku064b\matlab\new plots\cf_vs_bw_theta_exp1.2_window3000_nfft4000_NORMAL_fooof\theta_cf_vs_bw\new_high_res_save_folder\theta_cf_vs_bw_timesignal_and_results_exp1.2_window3000_nfft4000_1-100Hz-50Hzsubstracted_NORMAL_fooof_average_of_20_50x50_HIGHER_NOISE.mat';
save(save_path, 'varrying_theta_results');
                                             
