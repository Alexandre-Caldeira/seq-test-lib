%% Clear workspace and initialize parameters
clearvars; 
% close all; 
clc;

%% Test pipeline
% Load object with default exam and reset data
dtl = DataLoader('exp');

% Define size of vector with randomly selected exams
% and randomly select subjects and stimuli without replacement
% nSubj = 5;
% nStim = 3;
% dtl.selectedZanoteliSubjects = randperm(numel(dtl.zanoteliSubjects),nSubj);
% dtl.selectedZanoteliStimuli = randperm(numel(dtl.zanoteliStimulusNames),nStim);

% Or choose all
nSubj = 11;
nStim = 5;
dtl.selectedZanoteliSubjects = 1:numel(dtl.zanoteliSubjects);
dtl.selectedZanoteliStimuli = 1:numel(dtl.zanoteliStimulusNames);

% Load data and compute FFT
dtl = dtl.loadBulkEEGData(); %.computeBulkFFTs();

% Preprocess and filter all data
ppc = PreProcessor().bulkZanoteliPreprocess(dtl).bulkAntunesFilter(dtl);

% Reset SIGNALS to filtered for display
% dtl.groupSignals = ppc.groupProcessedSignals;
dtl.groupSignals = ppc.groupFilteredSignals;
dtl = dtl.computeBulkFFTs();

%% Show results
% Show object
dtl.age()
disp(dtl)

% Show 3 exam timeseries and PSD (from random random stimulus-subject-electrode)
randomly_shown_subj = dtl.selectedZanoteliSubjects(randperm(nSubj,3));
randomly_shown_stim = dtl.selectedZanoteliStimuli(randperm(nStim,3));
randomly_shown_electrode = randperm(dtl.nChannels,3);

figure(1)
for i = 1:3

    current_signal = cell2mat( ...
        dtl.groupSignals(randomly_shown_stim(i),randomly_shown_subj(i)));

    current_SIGNAL =  cell2mat( ...
        dtl.groupSIGNALS(randomly_shown_stim(i),randomly_shown_subj(i)));
    
    random_epoch = randi([1,size(current_SIGNAL,2)],1);

    current_signal = current_signal(:,random_epoch,randomly_shown_electrode(i));    
    lead_name = dtl.zanoteliLeads(randomly_shown_electrode(i));
    random_epoch = random_epoch+2; % add 2 seconds removed during preprocessing
    exam_time = (random_epoch*dtl.fs:random_epoch*dtl.fs+numel(current_signal)-1)';

    subplot(2,3,i)
    plot(exam_time,current_signal)
    grid on 
    xlabel('Time [ms]')
    ylabel('Voltage [V]')
    title(['Measures Aquired on' lead_name])

    subplot(2,3,i+3)
    stem(abs(current_SIGNAL(:,random_epoch,randomly_shown_electrode(i))), ...
        'filled', 'MarkerSize',3,'LineWidth',0.1)
    hold on
    stem(dtl.signalFrequencies, ...
        abs(current_SIGNAL(dtl.signalFrequencies,random_epoch,randomly_shown_electrode(i))), ...
        'filled', 'MarkerSize',3,'LineWidth',0.1, 'Color','red');
    grid on 
    xlabel('Frequency [Hz]')
    ylabel('Voltage [V]')
    title(['PSD processed from' lead_name])
    hold off

end

    


