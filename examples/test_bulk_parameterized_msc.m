%% Clear workspace and initialize parameters
clearvars; close all; clc;

% Load object with default exam and reset data
dtl = DataLoader('exp');

% Define size of vector with randomly selected exams
% and randomly select subjects and stimuli without replacement
% nSubj = 5;
% nStim = 3;
% dtl.selectedZanoteliSubjects = randperm(numel(dtl.zanoteliSubjects),nSubj);
% dtl.selectedZanoteliStimuli = randperm(numel(dtl.zanoteliStimulusNames)-1,nStim); % -1 to remove 'ESP'

% Or choose all
nSubj = 11;
nStim = 5;
dtl.selectedZanoteliSubjects = 1:numel(dtl.zanoteliSubjects);
dtl.selectedZanoteliStimuli = [3 5]; % 1:numel(dtl.zanoteliStimulusNames)-1; % -1 to remove 'ESP'

% Load data, preprocess and filter
dtl = dtl.loadBulkEEGData();
% ppc = PreProcessor().bulkZanoteliPreprocess(dtl);
ppc = PreProcessor().bulkZanoteliPreprocess(dtl).bulkAntunesFilter(dtl);

% Reset SIGNALS to filtered for display
% dtl.groupSignals = ppc.groupProcessedSignals;
dtl.groupSignals = ppc.groupFilteredSignals;
dtl = dtl.computeBulkFFTs();

%% Test pipeline
% Compute objective response detector (MSC)
% Comportamento:
% O que precisamos definir? Janelas! 
% (Onde cada começa, quantas amostras têm.)
%
% Calculamos a posição de inicio e fim delas com base em 4 parametros:
% 1. inicio: Instante de inicio do exame
% 2. tamanho: Numero de testes OU Numero de amostras por teste OU Intervalo entre testes;
% 3. paradaT (limita tempo): duração máxima do exame OU numero deamostras/janelas;
% 4. paradaI (limita dados): NDC, futilidade/detecção-CGST, variância das amostras (SNR). 
%

% Exemplo: 
% compute_bulk_msc( ...
% startWindows = 1:5:21, ...
% windowStepSizes = [5 18 24 32], ...
% channels = [1 8 16], ...
% lastWindowCalcMethod = 'flexible', ... % maxFromStart, maxFromLast, exactK, flexible
% stepType = 'fixedSize'... % minToK, minToMax, minToFix, withResampling, default = fixedSize
%     ...
%     );

% Specify epoch parameters and compute MSCs accordingly
% M= variable
% ordc = ORDCalculator(dtl).fit_epochs( ...
%     stimulusIndices = dtl.selectedZanoteliStimuli,...
%     subjectIndices = dtl.selectedZanoteliSubjects,...
%     startWindows = [1 5 15], ...
%     windowStepSizes = [5 18 24 32], ...
%     lastWindowCalcMethod = 'maxFromStart', ... % maxFromStart, maxFromLast, exactK, fromSizeType
%     sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
%     ... % then, compute on selected channels:
%     );
%
% ordc = ordc.bulk_compute_msc(channels = [1 8 16]);

% M = 40 
ordc = ORDCalculator(dtl).fit_epochs( ...
    stimulusIndices = dtl.selectedZanoteliStimuli,...
    subjectIndices = dtl.selectedZanoteliSubjects,...
    startWindows = [1], ...
    windowStepSizes = 32, ...
    single_or_bulk = 'bulk',...
    lastWindowCalcMethod = 'maxFromStart', ... % maxFromStart, maxFromLast, exactK, fromSizeType
    sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
    ... % then, compute on selected channels:
    );

ordc = ordc.bulk_compute_msc(channels = 1);

% K_stages
% ordc = ORDCalculator(dtl).fit_epochs( ...
%     stimulusIndices = dtl.selectedZanoteliStimuli,...
%     subjectIndices = dtl.selectedZanoteliSubjects,...
%     startWindows = [1 2], ...
%     windowStepSizes = 40, ...
%     K_stages = [8 2 5],...
%     lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
%     sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
%     ... % then, compute on selected channels:
%     );
% 
% ordc = ordc.bulk_compute_msc(channels = 1);


%% Validating results
a = ordc.groupMSC(ordc.stimulusIndices(1),ordc.subjectIndices(1),:);
for k = 1:numel(a)
if ~isempty(a{k})
break
end
end
b = cell2mat(ordc.groupMSC(ordc.stimulusIndices(1),ordc.subjectIndices(1),k));
% stem(abs(b(:,1)))
stem(abs(b(:,1,1)))
whos b
% stem(abs(b(:,25,1)))
% stem(abs(b(:,25,3)))
% stem(abs(b(:,25,4)))
% stem(abs(b(:,25,8)))
% stem(abs(b(:,25,16)))


%% Show results
% Show object
dtl.age()
disp(ordc)

filtered_freq_ord = ordc.latestMSC;

lead_name = dtl.zanoteliLeads(random_electrode);
random_epoch = random_epoch+2; % add 2 seconds removed during preprocessing

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


%% UTILS
% Count # of actual ords to compute
ne = 0;
for idx = 1:numel(ordc.epochs)
    if ~isempty(cell2mat(ordc.epochs(idx)))
        ne = ne+1;
    end
end
    

