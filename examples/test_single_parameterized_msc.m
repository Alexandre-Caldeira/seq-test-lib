%% Clear workspace and initialize parameters
clearvars; 
close all; 
clc;

% Get random exam 
[random_stimulus,random_subject,...
    random_electrode,random_epoch] = random_exam_args(DataLoader('exp'));

% Force 70dB for testsing with 50 seconds
% random_stimulus = 1;
% random_epoch = 35;

% Load object with default exam and reset data
dtl = DataLoader('exp').resetExam(random_subject, random_stimulus);

% Show random exam details on terminal
dtl.inspectExam()

% Preprocess and filter all data
ppc = PreProcessor().zanoteliPreProcessing(dtl).antunesFiltering(dtl);

% Reset SIGNALS to filteredSIGNALS
dtl.signals = ppc.filteredSignals;
dtl = dtl.computeFFT();

%% Test pipeline

% Compute objective response detector (MSC)
ordc = ORDCalculator(dtl);
ordc = ordc.compute_msc( ...
                startWindow=25, ...
                windowStepSize=32 ...
                );

%% Show results
% Show object
dtl.age()
disp(ordc)

lead_name = dtl.zanoteliLeads(random_electrode);

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

    


