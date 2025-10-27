%% Clear workspace and initialize parameters
clearvars; 
close all; 
clc;

%% Test Pipeline
dtl = DataLoader('exp','C:\Users\alexa\Desktop\Sinais_EEG\');
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

new_random_leads = dtl.channels(randi([1, 16],1,5));
measures_from = dtl.zanoteliLeads(new_random_leads)
dtl = dtl.resetChannels(new_random_leads);
random_electrode_fun = @() randi([1 dtl.nChannels],1,1);

random_stimulus = random_stimulus_fun();
random_electrode = dtl.channels(random_electrode_fun());
random_epoch = random_epoch_fun();
random_subject = random_subject_fun();

original_epoch = dtl.signals(:,random_epoch,random_electrode);
original_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

ppc = ppc.zanoteliPreProcessing(dtl);
dtl.signals = ppc.processedSignals;
dtl = dtl.computeFFT();

processed_epoch = dtl.signals(:,random_epoch,random_electrode);
processed_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

ppc = ppc.antunesFiltering(dtl);
dtl.signals = ppc.filteredSignals;
dtl = dtl.computeFFT();

filtered_epoch = dtl.signals(:,random_epoch,random_electrode);
filtered_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

dtl.inspectExam()

disp(dtl)

%% Show Results
lead_name = cell2mat(dtl.zanoteliLeads(random_electrode));
random_epoch = random_epoch+2; % add 2 seconds removed from preprocessed
exam_time = (10^-3)*(random_epoch*dtl.fs:random_epoch*dtl.fs+numel(original_epoch)-1)';


figure(1)   
% subplot(231)
% subplot(221)
% plot(exam_time,original_epoch, 'color', 0.15*[1 1 1])
% hold on 
% plot(exam_time,original_epoch,'.', 'color', 0.15*[1 1 1])
% grid on 
% xlabel('Time [s]')
% ylabel('Voltage [V]')
% title(['Measures Aquired on' lead_name])

% subplot(232)
% plot(exam_time,processed_epoch, 'color', 0.15*[1 1 1])
% hold on 
% plot(exam_time,processed_epoch,'.', 'color', 0.15*[1 1 1])
% grid on 
% xlabel('Time [s]')
% ylabel('Voltage [V]')
% title(['Processed signal from' lead_name])

% subplot(233)
figure(1)
plot(exam_time,filtered_epoch, 'color',0.65*[1 1 1])
hold on 
p = plot(exam_time,filtered_epoch,'.', 'color', 0.15*[1 1 1], ...
    'MarkerSize',10);
grid on 
xlabel('Time [s]')
ylabel('Voltage [V]')
title(['Filtered signal from ',lead_name])

s = sprintf('Measures (sampled at F_s = %d Hz)', dtl.fs);
legend(p,s)

% subplot(234)
% subplot(223)
% stem(abs(original_freq_sample),'filled', 'color', 0.15*[1 1 1], ...
%                                  'MarkerSize',2,'LineWidth',0.1)
% grid on 
% xlabel('Frequency [Hz]')
% ylabel('Measured Power')
% title(['PSD aquired on' lead_name])
% xlim([-25,1.1*size(original_freq_sample,1)])

% subplot(235)
% stem(abs(processed_freq_sample),'filled', 'color', 0.15*[1 1 1], ...
%                                  'MarkerSize',2,'LineWidth',0.1)
% grid on 
% xlabel('Frequency [Hz]')
% ylabel('Measured Power')
% title(['PSD processed from' lead_name])
% xlim([-25,1.1*size(processed_freq_sample,1)])

% subplot(236)
figure(2)
stem(abs(filtered_freq_sample),'filled', 'color', 0.65*[1 1 1], ...
                                 'MarkerSize',2,'LineWidth',0.1, ...
                                 'MarkerEdgeColor',0.15*[1 1 1])

hold on
b = stem(dtl.signalFrequencies, abs(filtered_freq_sample(dtl.signalFrequencies)), ...
    'filled', 'color',  0.65*[1 1 1],...
                                 'MarkerSize',3,'LineWidth',0.1, ...
                                 'MarkerEdgeColor','b');

r = stem(dtl.noiseFrequencies, abs(filtered_freq_sample(dtl.noiseFrequencies)), ...
    'filled', 'color',  0.65*[1 1 1],...
                                 'MarkerSize',3,'LineWidth',0.1, ...
                                 'MarkerEdgeColor','r');

legend([b,r],{'Stimulus','Noise'})
grid on 
xlabel('Frequency [Hz]')
ylabel('Power Estimate')
title(['PSD filtered from ',lead_name])
xlim([-25,1.1*size(filtered_freq_sample,1)])