%% Clear workspace and initialize parameters
clearvars; 
close all; 
clc;

%% Test Pipeline
dtl = DataLoader('exp');
dtl = dtl.computeFFT();
ppc = PreProcessor();
ppc = ppc.zanoteliPreProcessing(dtl);

random_electrode = randi([1, max(numel(dtl.zanoteliLeads))],1,1);
random_epoch = randi([1, dtl.zanoteliSuggestedMMax(dtl.zanoteliStimulusIndex)],1,1);

original_epoch = dtl.signals(:,random_epoch,random_electrode);
original_freq_sample = dtl.SIGNALS(:,random_epoch,random_electrode); 

dtl.signals = ppc.processedSignals;
dtl = dtl.computeFFT();

processed_epoch = dtl.signals(:,random_epoch,1);
processed_freq_sample = dtl.SIGNALS(:,random_epoch,1); 

%% Show Results
exam_time = (random_epoch:random_epoch+numel(original_epoch)-1)';

figure(1)
subplot(121)
plot(exam_time,original_epoch)
grid on 
xlabel('Time [ms]')
ylabel('Voltage [V]')
title(['Measures Aquired on' dtl.zanoteliLeads(1)])

subplot(122)
plot(exam_time,processed_epoch)
grid on 
xlabel('Time [ms]')
ylabel('Voltage [V]')
title(['Processed signal from' dtl.zanoteliLeads(1)])

figure(2)
subplot(121)
stem(abs(original_freq_sample))
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['PSD aquired on' dtl.zanoteliLeads(1)])

subplot(122)
stem(abs(processed_freq_sample))
grid on 
xlabel('Frequency [Hz]')
ylabel('Voltage [?]')
title(['PSD processed from' dtl.zanoteliLeads(1)])