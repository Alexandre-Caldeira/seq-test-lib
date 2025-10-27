% function SinalEEG_Analysis() 
% SinalEEG_Analysis 
% % This unified MATLAB application: 
% • Loads or simulates EEG data 
% • Preprocesses the signal (DC removal, bandpass filtering) 
% • Computes FFT and magnitude-squared coherence (MSC) 
% • Computes beta-distribution based sequential test thresholds 
% • Runs a multi-stage sequential test with early stopping 
% • Evaluates performance metrics and visualizes the results 
% % Developed based on modern software engineering practices. 
% % Author: [Alexandre Gomes Caldeira] 
% % Date: 
% - v0.0: [14 02 2025 10:40] 
%   Project created, exposing incompatibilities...
%
% - v0.1: [07 03 2025 10:45] 
%   DataLoader with FFT functional for exp and sim! :)
%
% - v0.2: [07 03 2025 16:42] 
%   Preprocessing functional for exp and sim, signal recompute added
%
% - v0.3: [11 03 2025 21:55] 
%   MSC calculator now accepts variable parameters
%
% - v0.4: [] 
%   ORD calculator functional for exp and sim
%
% - v0.5: [] 
%   SHT (single hyp. test [Decisions, Time]) functional for exp and sim 
%
% - v0.6: [] 
%   add Metrics: [Pareto, ConfusionMatrix, Specificity, Repeatability]
%
% - v0.7: [] 
%   GST (group sequential test) functional for exp and sim
%
% - v0.8: [] 
%   GST metrics verified 
%
% - v0.9: []
%
% - v1: [] 
%

%% Clear workspace and initialize parameters
clearvars; 
% close all; 
clc;

%% Main Pipeline
% dtl = DataLoader('exp');
dtl = DataLoader('sim');

dtl.noiseMean = 15;
% dtl.channels = 3;
% dtl.nChannels = 3;
dtl = dtl.genSimulatedSignals();
dtl.inPath = 'C:\PPGEE\SBEB_CBA_24\CGST_figuras\Sinais_EEG\';
dtl.zanoteliStimulusIndex = 3;

%% Test Pipeline
dtl = DataLoader('exp');
ppc = PreProcessor();

% -1 excludes ESP stimulus (less than 20 secs => not enough for filter)
% Exam is ESP stimulus on subject Er,
%  measuring on FC for 120 seconds. => cant filter if lower than 48 windows
% random_electrode = 5;
% random_epoch = 11;
% random_stimulus = 6;
% random_subject = 4;

random_stimulus_fun = @() randi([1, numel(dtl.zanoteliStimulusNames)-1],1,1);
random_electrode_fun = @() randi([1, numel(dtl.zanoteliLeads)],1,1);
random_epoch_fun = @() randi([1, dtl.zanoteliSuggestedMMax(dtl.zanoteliStimulusIndex)],1,1);
random_subject_fun = @() randi([1, 11],1,1);

dtl.inspectExam()
dtl = dtl.resetExam(random_subject_fun(), random_stimulus_fun());
dtl.inspectExam()

random_stimulus = random_stimulus_fun();
random_electrode = random_stimulus_fun();
random_epoch = random_stimulus_fun();
random_subject = random_stimulus_fun();

new_random_leads = dtl.channels(randi([1, 16],1,5));
dtl.zanoteliLeads(new_random_leads)
dtl = dtl.resetChannels(new_random_leads);

original_epoch = dtl.signals(:,random_epoch,random_electrode);
original_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

ppc = ppc.zanoteliPreProcessing(dtl);
dtl.signals = ppc.processedSignals;
dtl = dtl.computeFFT();

processed_epoch = dtl.signals(:,random_epoch,1);
processed_freq_sample = dtl.SIGNALS(:,random_epoch,1); 

ppc = ppc.antunesFiltering(dtl);
dtl.signals = ppc.filteredSignals;
dtl = dtl.computeFFT();

filtered_epoch = dtl.signals(:,random_epoch,1);
filtered_freq_sample = dtl.SIGNALS(:,random_epoch,1); 

dtl.inspectExam()

disp(dtl)

%% Show Results
exam_time = (random_epoch*dtl.fs:random_epoch*dtl.fs+numel(original_epoch)-1)';
lead_name = dtl.zanoteliLeads(random_electrode);

figure(1)
subplot(231)
plot(exam_time,original_epoch)
grid on 
xlabel('Time [s]')
ylabel('Voltage [V]')
title(['Measures Aquired on' lead_name])

subplot(232)
plot(exam_time,processed_epoch)
grid on 
xlabel('Time [s]')
ylabel('Voltage [V]')
title(['Processed signal from' lead_name])

subplot(233)
plot(exam_time,filtered_epoch)
grid on 
xlabel('Time [s]')
ylabel('Voltage [V]')
title(['Filtered signal from' lead_name])

subplot(234)
stem(abs(original_freq_sample))
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['PSD aquired on' lead_name])

subplot(235)
stem(abs(processed_freq_sample))
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['PSD processed from' lead_name])

subplot(236)
stem(abs(filtered_freq_sample))
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['PSD filtered from' lead_name])

%% old stuff to be reworked:

% Sequential test settings
params.alpha = 0.05;           % Overall false-positive rate
% params.Mmin = []; params.Mmax = [];
params.K     = 7;              % Number of sequential stages
params.M     = dtl.duration / params.K;  % Window length (seconds) per stage

path_to_params = dtl.inPath;
addpath(path_to_params)

addpath('C:\PPGEE\Assessing CGST on ASSR\clean_code\assr-ord\garbage_in\seq_test\Numero_Deteccoes_consecutiva_H_recebidodePatricia14022025\Numero_Deteccoes_consecutiva_H')

Mmax= dtl.zanoteliSuggestedMMax(dtl.zanoteliStimulusIndex);
Mmax = 240;
load(['NDC_AlfaCorrigido_Mmax' num2str(Mmax) '_alfa_'  num2str(params.alpha) '_FPdesejado' num2str(params.alpha) '.mat'], ...
    'P')

params.P     = P; % parametros = [Min Mstep Mmax alfa_corrigido]
params.duration = dtl.duration;
params.nChannels = dtl.nChannels;
params.nBins = dtl.nBins;
params.testFrequencies = [dtl.signalFrequencies, dtl.noiseFrequencies];
params.flagNoise = numel(dtl.signalFrequencies);

fftSignals = dtl.SIGNALS;

% check if filter is working
figure
stem(abs(fftSignals(:,1,1)))

% 3. Compute MSC (here a simplified average per stage)
MSCvalues = computeMSC(fftSignals, params);

% 4. Compute sequential test thresholds

% findIndex is not found in the current folder or on the MATLAB path, but exists in:
    % C:\PPGEE\Assessing CGST on ASSR\new_code
addpath('C:\PPGEE\Assessing CGST on ASSR\new_code')
[aThresholds, gThresholds] = computeBetaThresholds(params.K, params.M, params);
params.aThresholds = aThresholds;
params.gThresholds = gThresholds;

% 5. Run the sequential test (decision process)
[decisions, stageMetrics] = sequentialTest(MSCvalues, params);

% 6. Evaluate performance metrics (e.g., FPR and TPR)
% Comment: 
% gets Attempt to grow array along ambiguous dimension.
% when numel(noiseFrequencies) > numel(signalFrequencies)
% but not otherwise
% why?
% perf = performanceMetrics(decisions, params);
% 
% % 7. Plot the results
% fignum=2;
% plotResults(aThresholds, gThresholds, stageMetrics, perf, params,fignum);
% 
% % 8. Print summary metrics
% fprintf('(FPR,pct.): %.4f\n', perf.FPR);
% fprintf('(TPR,pct.): %.4f\n', perf.TPR);
% fprintf('(TNR,pct.): %.4f\n', perf.TNR);
% fprintf('(FNR,pct.): %.4f\n', perf.FNR);
% 
% fprintf('(No decision...,pct.): %.4f\n', sum(decisions==0,'all')/numel(decisions));
% fprintf('(Stop: detected.,pct.): %.4f\n', sum(decisions==1,'all')/numel(decisions));
% fprintf('(Stop: futile.,pct.): %.4f\n', sum(decisions==-1,'all')/numel(decisions));
