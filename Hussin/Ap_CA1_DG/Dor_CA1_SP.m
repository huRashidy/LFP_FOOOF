CA1_paths = {'F:\MSc content\work\Dor\TT1\TT1_Cont_04-02_downsampled.mat';
            'F:\MSc content\work\Dor\TT11\TT11_Cont_04-02_downsampled.mat'; 
            'F:\MSc content\work\Dor\TT12\TT12_Cont_03-29_downsampled.mat';
            'F:\MSc content\work\Dor\TT16\TT16_Cont_03-25_downsampled.mat';
            };

DG_paths = {'F:\MSc content\work\Dor\TT1\TT1_Cont_04-10_downsampled.mat';
            'F:\MSc content\work\Dor\TT3\TT3_Cont_04-09_downsampled.mat'; 
            'F:\MSc content\work\Dor\TT5\TT5_Cont_03-27_downsampled.mat';
            'F:\MSc content\work\Dor\TT9\TT9_Cont_04-09_downsampled.mat';
            'F:\MSc content\work\Dor\TT11\TT11_Cont_04-12_downsampled.mat';
            'F:\MSc content\work\Dor\TT12\TT12_Cont_04-10_downsampled.mat';
            'F:\MSc content\work\Dor\TT16\TT16_Cont_04-05_downsampled.mat';
    };

notch_freq = 50;
lowpass_freq = 300;
window_size = 1000;
notch_range= [48,53]; 
harmonics = {[12,19] ,[48,53] , [148,151]};
f1=1; 
f2=200;
% Run FOOOF 
%for i = 1:3
ranges = [4,250];
settings1.peak_width_limits = [2,100];
settings1.max_n_peaks = 1;
settings1.peak_threshold = 0.2;
settings.peak_width_limits = [0.5,100];
settings.aperiodic_mode = 'knee'; 

CA1_ap = cell(size(CA1_paths,1),4);

for i = 1:size(CA1_paths,1)

    load(CA1_paths{i,:})
    electrode_idx= 0;
    dt = mean(diff(timestamps_downsample));
    fs = 1/dt;

    electrode_idx = maxpower(tetrode_data_downsample, window_size , fs, f1, f2);
    %disp('best electrodes are being determined...')
    filtered_electode_tt= preProcess(tetrode_data_downsample(:,electrode_idx),fs,lowpass_freq);
    [freq, new_psd]= detect_notches_and_harmonics(filtered_electode_tt, window_size , harmonics, fs);
    fooof_results = doAnalysis(freq, new_psd , ranges , settings);
    fooof_plot(fooof_results)
    title(['File: ', CA1_paths{i,:}]);

    % Convert aperiodic_params to a string
    aperiodic_params_str = sprintf('%.2f ', fooof_results.aperiodic_params);
    
    % Add a text box with the aperiodic parameters
    annotation('textbox', [0.15, 0.8, 0.1, 0.1], 'String', ['Aperiodic Params: ', aperiodic_params_str], 'FitBoxToText', 'on');

    CA1_ap{i,1} = fooof_results.aperiodic_params(1);
    CA1_ap{i,2} = fooof_results.aperiodic_params(2);
    CA1_ap{i,3} = fooof_results.aperiodic_params(3);
    CA1_ap{i,4} = fooof_results.aperiodic_params(4);

end

filename = 'Dor_CA1_ap.mat';
save(filename, 'CA1_ap')


DG_ap = cell(size(DG_paths,1),4);

for i = 1:size(DG_paths,1)

    load(DG_paths{i,:})
    electrode_idx= 0;
    dt = mean(diff(timestamps_downsample));
    fs = 1/dt;
    electrode_idx = maxpower(tetrode_data_downsample, window_size , fs, f1, f2);
    %disp('best electrodes are being determined...')
    filtered_electode_tt= preProcess(tetrode_data_downsample(:,electrode_idx),fs,lowpass_freq);
    [freq, new_psd]= detect_notches_and_harmonics(filtered_electode_tt, window_size , harmonics, fs);
    fooof_results = doAnalysis(freq, new_psd , ranges , settings);
    fooof_plot(fooof_results)
    title(['File: ', DG_paths{i,:}]);
    % Cnvert aperiodic_params to a string
    aperiodic_params_str = sprintf('%.2f ', fooof_results.aperiodic_params);
    
    % Add a text box with the aperiodic parameters
    annotation('textbox', [0.15, 0.8, 0.1, 0.1], 'String', ['Aperiodic Params: ', aperiodic_params_str], 'FitBoxToText', 'on');

    DG_ap{i,1} = fooof_results.aperiodic_params(1);
    DG_ap{i,2} = fooof_results.aperiodic_params(2);
    DG_ap{i,3} = fooof_results.aperiodic_params(3);
    DG_ap{i,4} = fooof_results.aperiodic_params(4);

end

filename = 'Dor_DG_ap.mat'
save(filename , 'DG_ap')

data1 = load('Dor_CA1_ap.mat'); % Replace 'file1.mat' with the actual file name
data2 = load('Dor_DG_ap.mat'); % Replace 'file2.mat' with the actual file name

% Assuming the data is stored in variables named 'mat1' and 'DG'
CA1 = data1.CA1_ap; % Change 'mat1' to the actual variable name in the file
DG = data2.DG_ap; % Change 'DG' to the actual variable name in the file

CA1_numeric = cell2mat(CA1);
DG_numeric = cell2mat(DG);

% Number of columns (parameters)
num_columns = size(CA1, 2);

% Perform t-tests and store p-values
p_values = zeros(1, num_columns);
for col = 1:num_columns
    [~, p_values(col)] = ttest2(CA1_numeric(:, col), DG_numeric(:, col));
end

% Plotting
figure;

params = {'knee offset' , 'knee log frequency', 'exponent 1', 'exponent 2'};
% Scatter plots
for col = 1:num_columns
    subplot(2, num_columns, col);
    scatter(ones(size(CA1_numeric, 1), 1), CA1_numeric(:, col), 'r', 'filled'); hold on;
    scatter(2 * ones(size(DG_numeric, 1), 1), DG_numeric(:, col), 'b', 'filled'); hold off;
    title(['Parameter ', params{col}]);
    xlabel('Group');
    ylabel(['Parameter ', params{col}]);
    xticks([1 2]);
    xticklabels({'CA1', 'SP'});
    legend(' CA1', 'SP');
end

% Box plots
for col = 1:num_columns
    subplot(2, num_columns, num_columns + col);
    boxplot([CA1_numeric(:, col); DG_numeric(:, col)], [ones(size(CA1_numeric, 1), 1); 2 * ones(size(DG_numeric, 1), 1)]);
    title(['Parameter ', params{col}, ' (p = ', num2str(p_values(col)), ')']);
    xlabel('Group');
    ylabel(['Parameter ', params{col}]);
    xticks([1 2]);
    xticklabels({'CA1', 'SP'});
end

% Display p-values
disp('P-values for each parameter:');
disp(p_values);


%}










