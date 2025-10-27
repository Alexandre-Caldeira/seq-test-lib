%% Clear workspace and initialize parameters
clearvars; close all; clc;

% Load object with default exam and reset data
dtl = DataLoader('exp', 'C:\Users\alexa\Desktop\Sinais_EEG\');

dtl.selectedZanoteliSubjects = 1:11; %:numel(dtl.zanoteliSubjects);
dtl.selectedZanoteliStimuli = 3; % 1:numel(dtl.zanoteliStimulusNames)-1; % -1 to remove 'ESP'

% Load data, preprocess and filter
dtl = dtl.loadBulkEEGData();
ppc = PreProcessor().bulkZanoteliPreprocess(dtl);%.bulkAntunesFilter(dtl);
dtl.groupSignals = ppc.groupProcessedSignals;
% dtl.groupSignals = ppc.groupFilteredSignals;

% Reset SIGNALS to filtered for display
dtl = dtl.computeBulkFFTs(); 

dtl.age();

%% import parameters from Vaz 2024SS
% caminho = 'C:\Users\alexa\Desktop\Numero_Deteccoes_consecutiva_H\';
% vaz_data = load([caminho,'NDC_AlfaCorrigido_Mmax240_alfa_0.05_FPdesejado0.05.mat'], ...
%     'alfa_corrigido', 'NDC_minimo','P', 'nRuns');

% vaz_startWindows = vaz_data.P(:,1)';
% vaz_windowSizes = vaz_data.P(:,2)';

% % filter number of stages
% vaz_windowSizes = 8:238;
% K_stages = zeros(size(vaz_windowSizes));
% for idx = 1:numel(K_stages)
%     K_stages(idx) = numel(1:vaz_windowSizes(idx):240);
% end

% no_go = [find(K_stages >5), find(K_stages<=1)];
% vaz_startWindows(find(K_stages >5)) = [];
% vaz_windowSizes(find(K_stages >5)) = [];
% 
% vaz_windowSizesS

% vaz_translated_Kstages = flip(unique(K_stages));
vaz_startWindows = 1:200;
% vaz_startWindows = 1:400;
% vaz_startWindows = 121:240;

%%
ordc = ORDCalculator(dtl).fit_epochs( ...
    stimulusIndices = dtl.selectedZanoteliStimuli,...
    subjectIndices = dtl.selectedZanoteliSubjects,...
    startWindows = vaz_startWindows, ... % windowStepSizes = 52, ...
    K_stages = 2:30, ...
    single_or_bulk = 'bulk',...
    lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
    sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
    ... % then, compute on selected channels:
    );

fprintf('\tEpochs are computed.\n')
ordc.age()

single_channel = 1:16;
ordc = ordc.bulk_compute_msc(channels = single_channel);
ordc.age()

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
nChannels = 16; %numel(tester.ord_calculator.channels);
tester.FP = zeros(nParams,nChannels);
tester.FN = zeros(nParams,nChannels);
tester.TP = zeros(nParams,nChannels);
tester.TN = zeros(nParams,nChannels);

for stimulusIndex = tester.stimulusIndices
    for subjectIndex = tester.subjectIndices
        for channel_idx = 1:numel(tester.ord_calculator.channels)
            nonemptyparams_idxs = [];
            for epoch_param_idx = 1:nParams 
                
                if size(tester.groupFP,1)==1
                    exam_fp = cell2mat(tester.groupFP(1, subjectIndex, epoch_param_idx));
                elseif size(tester.groupFP,2)==1
                    exam_fp = cell2mat(tester.groupFP(stimulusIndex, 1, epoch_param_idx));
                else
                    exam_fp = cell2mat(tester.groupFP(stimulusIndex, subjectIndex, epoch_param_idx));
                end

                if ~isempty(exam_fp)
                    % If there was a detection on any stage, 
                    % assign 1 to that exam, and repeat for all
                    % frequencies. Then, sum all 1s.
                    % + Add previous results from other subjs
                    % To compute rate (pct), divide by #freqs and #subj

                    if size(tester.groupFP,1)==1
                        exam_fn = cell2mat(tester.groupFN(1, subjectIndex, epoch_param_idx));
                        exam_tp = cell2mat(tester.groupTP(1, subjectIndex, epoch_param_idx));
                        exam_tn = cell2mat(tester.groupTN(1, subjectIndex, epoch_param_idx));

                    elseif size(tester.groupFP,2)==1
                        exam_fn = cell2mat(tester.groupFN(stimulusIndex, 1, epoch_param_idx));
                        exam_tp = cell2mat(tester.groupTP(stimulusIndex, 1, epoch_param_idx));
                        exam_tn = cell2mat(tester.groupTN(stimulusIndex, 1, epoch_param_idx));

                    else
                        exam_fn = cell2mat(tester.groupFN(stimulusIndex, subjectIndex, epoch_param_idx));
                        exam_tp = cell2mat(tester.groupTP(stimulusIndex, subjectIndex, epoch_param_idx));
                        exam_tn = cell2mat(tester.groupTN(stimulusIndex, subjectIndex, epoch_param_idx));

                    end
                    
                    test_channel_idx =tester.ord_calculator.channels(channel_idx);
                    
                    tester.FN(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_fn(tester.signalFrequencies,:,test_channel_idx)>0,2))...
                            + tester.FN(epoch_param_idx, test_channel_idx);

                    tester.TP(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_tp(tester.signalFrequencies,:,test_channel_idx)>0,2))...
                            + tester.TP(epoch_param_idx, test_channel_idx);

                    tester.FP(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_fp(tester.noiseFrequencies,:,test_channel_idx)>0,2))...
                            + tester.FP(epoch_param_idx, test_channel_idx);

                    tester.TN(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_tn(tester.noiseFrequencies,:,test_channel_idx)>0,2))...
                            + tester.TN(epoch_param_idx, test_channel_idx);

                    nonemptyparams_idxs = [nonemptyparams_idxs,...
                                            epoch_param_idx];
    
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
    100*mean(fn_rate(nonemptyparams_idxs,single_channel),'all'), ...
    100*mean(fp_rate(nonemptyparams_idxs,single_channel),'all'), ...
    100*mean(tp_rate(nonemptyparams_idxs,single_channel),'all'), ...
    100*mean(tn_rate(nonemptyparams_idxs,single_channel),'all'), ...
    'VariableNames',{'fn','fp','tp','tn'})
% 
% tester_checkpoint = struct();
% tester_checkpoint.epochs = tester.epochs ;
% tester_checkpoint.groupTP = tester.groupTP;
% tester_checkpoint.groupTN = tester.groupTN;
% tester_checkpoint.groupFP = tester.groupFP;
% tester_checkpoint.groupFN = tester.groupFN;
% tester_checkpoint.previous_cgst_thresholds = tester.previous_cgst_thresholds;
% tester_checkpoint.allTestFrequencies = tester.allTestFrequencies;
% 
% tester_checkpoint.selectedZanoteliSubjects = tester.dataloader.selectedZanoteliSubjects;
% tester_checkpoint.selectedZanoteliStimuli = tester.dataloader.selectedZanoteliStimuli;
% tester_checkpoint.epochs_index_metadata = tester.ord_calculator.epochs_index_metadata;
% 
% tester_checkpoint.vaz_translated_Kstages = vaz_translated_Kstages;
% tester_checkpoint.vaz_startWindows = vaz_startWindows;
% 

[Y,MO,D,H,MI,S] = datevec(datetime);
s = ['tester_checkpoint_1_T',...
    num2str(H),'_',num2str(MI),'_',num2str(fix(S)),...
    num2str(D),'_',num2str(MO),'_',num2str(Y),'.mat']
% save(s,'tester_checkpoint','-v7.3')


%%
figure(1)
subplot(211)
plot(100*fp_rate(nonemptyparams_idxs,single_channel),'.')
subplot(212)
plot(100*tp_rate(nonemptyparams_idxs,single_channel),'.')

figure(2)
subplot(211)
boxchart(100*fp_rate(nonemptyparams_idxs,single_channel))
subplot(212)
boxchart(100*tp_rate(nonemptyparams_idxs,single_channel))

% save(s,'-v7.3')
