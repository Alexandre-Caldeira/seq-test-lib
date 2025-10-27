%% Clear workspace and initialize parameters
clearvars; 
close all; 
clc;
% Load object with default exam
dtl = DataLoader('exp');
dtl = dtl.computeFFT();

%% Prep Pipeline

% Get random exam 
[~,random_subject,random_electrode,~] = random_exam_args(dtl);

% Force 70dB for testsing with 50 seconds
random_stimulus = 1;
random_epoch = 35;

% Reset data and declare exam on terminal
dtl = dtl.resetExam(random_subject, random_stimulus);
dtl.inspectExam()

% Save unfiltered data
original_epoch = dtl.signals(:,random_epoch,random_electrode);
original_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

% Filter data and recompute signals
ppc = PreProcessor().zanoteliPreProcessing(dtl).antunesFiltering(dtl);
dtl.signals = ppc.filteredSignals;
dtl = dtl.computeFFT();

% Save filtered data
filtered_epoch = dtl.signals(:,random_epoch,random_electrode);
filtered_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

%% Test Pipeline
M = 10;
ordc = ORDCalculator(dtl);
ordc = ordc.compute_msc_on_all_channels(M);

filtered_freq_ord = ordc.latestMSC;

% Show parameters
ordc.age()

%% Show Results
lead_name = dtl.zanoteliLeads(random_electrode);
random_epoch = random_epoch+2; % add 2 seconds removed during preprocessing
exam_time = (random_epoch*dtl.fs:random_epoch*dtl.fs+numel(original_epoch)-1)';

test_result = figure(2);
subplot(131)
stem(abs(original_freq_sample),'filled', 'MarkerSize',3,'LineWidth',0.1)
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
xlim([0,dtl.nBins])
title(['PSD aquired on' lead_name])

subplot(132)
stem(abs(filtered_freq_sample),'filled', 'MarkerSize',3,'LineWidth',0.1)
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
xlim([0,dtl.nBins])
title(['PSD processed from' lead_name])

subplot(133)
stem(filtered_freq_ord,'filled', 'MarkerSize',3,'LineWidth',0.1)
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['Last MSC filtered from' lead_name])
xlim([0,dtl.nBins])
set(test_result , 'WindowState', 'maximized');

exam = figure(3);
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
xlim([0,dtl.nBins])
ylim([0,min(1.10*max(ordc.MSC(:,:,random_electrode),[],'all'),1.09)])
grid on
hold on
set(exam, 'WindowState', 'maximized');
nWindows = size(ordc.MSC,2);
for window_index = 1:nWindows
    title_str =['Window #', num2str(window_index),' of ',...
        num2str(nWindows),': MSC filtered from ' cell2mat(lead_name)];
    title(title_str)

    s = stem(squeeze(ordc.MSC(:,window_index,random_electrode)), ...
        'filled', 'MarkerSize',3,'LineWidth',0.1, 'Color','#0072BD');

    t = stem(dtl.signalFrequencies, ...
        squeeze(ordc.MSC(dtl.signalFrequencies,window_index,random_electrode)), ...
        'filled', 'MarkerSize',3,'LineWidth',0.1, 'Color','red');

    drawnow
    if window_index<nWindows
        pause(1.3)
        delete(s)
        delete(t)
    end
end


