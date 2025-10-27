%% Clear workspace and initialize parameters
clearvars; 
close all; 
clc;

%% Test Pipeline
dtl = DataLoader('sim');
dtl = dtl.resetDuration(55);
dtl = dtl.resetSNRfun(0, 0);
dtl = dtl.genSimulatedSignals();
ppc = PreProcessor();

random_stimulus_fun = @() randi([1, numel(dtl.zanoteliStimulusNames)-1],1,1);
random_epoch_fun = @() randi([1, dtl.zanoteliSuggestedMMax(dtl.zanoteliStimulusIndex)],1,1);

new_random_leads = dtl.channels(randi([1, 5],1,5));
measures_from = dtl.zanoteliLeads(new_random_leads)
dtl = dtl.resetChannels(new_random_leads);
random_electrode_fun = @() randi([1 dtl.nChannels],1,1);

random_electrode = dtl.channels(random_electrode_fun());
random_epoch = random_epoch_fun();

original_epoch = dtl.signals(:,random_epoch,random_electrode);
original_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

dtl.zanoteliStimulusIndex = 1;
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

disp(dtl)

%% Show Results
lead_name = dtl.zanoteliLeads(dtl.channels(random_electrode));
random_epoch = random_epoch+2; % add 2 seconds removed from preprocessed
exam_time = (random_epoch*dtl.fs:random_epoch*dtl.fs+numel(original_epoch)-1)';

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