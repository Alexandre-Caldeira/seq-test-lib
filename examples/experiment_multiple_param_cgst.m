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
nStim = 5; % 3 5
dtl.selectedZanoteliSubjects = 1:numel(dtl.zanoteliSubjects);
dtl.selectedZanoteliStimuli = 1:5; % 1:numel(dtl.zanoteliStimulusNames)-1; % -1 to remove 'ESP'

% Load data, preprocess and filter
dtl = dtl.loadBulkEEGData();
% ppc = PreProcessor().bulkZanoteliPreprocess(dtl);
% ppc = PreProcessor();
% ppc.groupProcessedSignals = dtl.groupSignals;
% ppc = ppc.bulkAntunesFilter(dtl);
% ppc = PreProcessor().bulkZanoteliPreprocess(dtl).bulkZanoteliPreprocess(dtl);

% ppc = PreProcessor().bulkZanoteliPreprocess(dtl).bulkAntunesFilter(dtl);

% .bulkZanoteliPreprocess(dtl).bulkAntunesFilter(dtl);

% Reset SIGNALS to filtered for display
% dtl.groupSignals = ppc.groupProcessedSignals;
% dtl.groupSignals = ppc.groupFilteredSignals;
dtl = dtl.computeBulkFFTs();

%%
% M = 40 
K_stages = [2 3 5 8 10];
ordc = ORDCalculator(dtl).fit_epochs( ...
    stimulusIndices = dtl.selectedZanoteliStimuli,...
    subjectIndices = dtl.selectedZanoteliSubjects,...
    startWindows = [1 6 10 25], ... % windowStepSizes = 52, ...
    K_stages = K_stages,...
    single_or_bulk = 'bulk',...
    lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
    sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
    ... % then, compute on selected channels:
    );

% 23 secs
% K_stages = 5;
% ordc = ORDCalculator(dtl).fit_epochs( ...
%     stimulusIndices = dtl.selectedZanoteliStimuli,...
%     subjectIndices = dtl.selectedZanoteliSubjects,...
%     startWindows = 10, ... % windowStepSizes = 52, ...
%     K_stages = K_stages,...
%     single_or_bulk = 'bulk',...
%     lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
%     sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
%     ... % then, compute on selected channels:
%     );

single_channel = 1:16;
ordc = ordc.bulk_compute_msc(channels = single_channel);
dtl.age()

%% Test pipeline
% Apply TEST(s) to objective response detector (MSC)
tester = ORDTester(ordc);
tester.desired_alpha = 0.05;

tester = tester.compute_bulk_beta_cgst_decisions();
tester.age()
% disp(tester)

% Comportamento (a implementar)
% O que precisamos testar? Janelas! 
% (qual limite para deteccao, quando parar, como sinalizar)
%
% Calculamos a posição de inicio e fim das janelas/testes com base em 4 parametros:
% 1. inicio: Instante de inicio do exame
% 2. tamanho: Numero de testes OU Numero de amostras por teste OU Intervalo entre testes;
% 3. paradaT (limita tempo): duração máxima do exame OU numero deamostras/janelas;
% 4. paradaI (limita dados): NDC, futilidade/detecção-CGST, variância das amostras (SNR). 

%% Show results

nParams = numel(tester.epochs(1,1,:));
nChannels = numel(tester.ord_calculator.channels);
tester.FP = zeros(nParams,nChannels);
tester.FN = zeros(nParams,nChannels);
tester.TP = zeros(nParams,nChannels);
tester.TN = zeros(nParams,nChannels);

for stimulusIndex = tester.stimulusIndices
    for subjectIndex = tester.subjectIndices
        for channel_idx = 1:numel(tester.ord_calculator.channels)
            for epoch_param_idx = 1:nParams 
                exam_fp = cell2mat(tester.groupFP(stimulusIndex, subjectIndex, epoch_param_idx));

                if ~isempty(exam_fp)
                    % If there was a detection on any stage, 
                    % assign 1 to that exam, and repeat for all
                    % frequencies. Then, sum all 1s.
                    % + Add previous results from other subjs
                    % To compute rate (pct), divide by #freqs and #subj

                    exam_fn = cell2mat(tester.groupFN(stimulusIndex, subjectIndex, epoch_param_idx));
                    exam_tp = cell2mat(tester.groupTP(stimulusIndex, subjectIndex, epoch_param_idx));
                    exam_tn = cell2mat(tester.groupTN(stimulusIndex, subjectIndex, epoch_param_idx));

                    

                    tester.FN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fn(tester.signalFrequencies,:,channel_idx)>0,2))...
                            + tester.FN(epoch_param_idx, channel_idx);

                    tester.TP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tp(tester.signalFrequencies,:,channel_idx)>0,2))...
                            + tester.TP(epoch_param_idx, channel_idx);

                    tester.FP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fp(tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + tester.FP(epoch_param_idx, channel_idx);

                    tester.TN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tn(tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + tester.TN(epoch_param_idx, channel_idx);
    
                end
            end
        end
    end
end


denom = numel(tester.noiseFrequencies)*numel(tester.subjectIndices)*numel(tester.stimulusIndices);
fp_rate = tester.FP/(denom);

tp_rate = tester.TP/(denom);

fn_rate = tester.FN/(denom);

tn_rate = tester.TN/(denom);

confmat = table( ...
    100*mean(fn_rate(end,:)), ...
    100*mean(fp_rate(end,:)), ...
    100*mean(tp_rate(end,:)), ...
    100*mean(tn_rate(end,:)), ...
    'VariableNames',{'fn','fp','tp','tn'})
