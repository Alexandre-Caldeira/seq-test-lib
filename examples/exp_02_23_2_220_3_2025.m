%% Clear workspace and initialize parameters
clearvars; close all; clc;

% Load object with default exam and reset data
dtl = DataLoader('exp');%, 'C:\Users\alexa\Desktop\Sinais_EEG\');

dtl.selectedZanoteliSubjects = 1:11; %:numel(dtl.zanoteliSubjects);
dtl.selectedZanoteliStimuli = 5; % 1:numel(dtl.zanoteliStimulusNames)-1; % -1 to remove 'ESP'
dtl.zanoteliStimulusNames(dtl.selectedZanoteliStimuli)

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
% vaz_startWindows = 1:200;
% vaz_startWindows = 1:400;
% vaz_startWindows = 121:240;

vaz_startWindows_per_stimuli = cell(size(dtl.zanoteliSuggestedMMax));
for i = 1:numel(vaz_startWindows_per_stimuli)
    vaz_startWindows_per_stimuli{i} = 1:(dtl.zanoteliSuggestedMMax(i)-25);
end

vaz_startWindows = cell2mat(vaz_startWindows_per_stimuli(dtl.selectedZanoteliStimuli))


%%
ordc = ORDCalculator(dtl).fit_epochs( ...
    stimulusIndices = dtl.selectedZanoteliStimuli,...
    subjectIndices = dtl.selectedZanoteliSubjects,...
    startWindows = vaz_startWindows, ... % windowStepSizes = 52, ...
    K_stages = 2:5, ...
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

%%
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
                    
                    test_channel_idx = tester.ord_calculator.channels(channel_idx);
                    
                    tester.FN(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_fn(tester.signalFrequencies,:,test_channel_idx)>0 ...
                            * ~(exam_tp(tester.signalFrequencies,:,test_channel_idx)>0),2))...
                            + tester.FN(epoch_param_idx, test_channel_idx);

                    tester.TP(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_tp(tester.signalFrequencies,:,test_channel_idx)>0 ...
                            * ~(exam_fn(tester.signalFrequencies,:,test_channel_idx)>0),2))...
                            + tester.TP(epoch_param_idx, test_channel_idx);

                    tester.FP(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_fp(tester.noiseFrequencies,:,test_channel_idx)>0 ...
                            * ~(exam_tn(tester.noiseFrequencies,:,test_channel_idx)>0),2))...
                            + tester.FP(epoch_param_idx, test_channel_idx);

                    tester.TN(epoch_param_idx, test_channel_idx) = ...
                            sum(any(exam_tn(tester.noiseFrequencies,:,test_channel_idx)>0 ...
                            * ~(exam_fp(tester.noiseFrequencies,:,test_channel_idx)>0),2))...
                            + tester.TN(epoch_param_idx, test_channel_idx);

                    nonemptyparams_idxs = [nonemptyparams_idxs,...
                                            epoch_param_idx];

                    % Find the smallest detection time for all frequencies
                    % on this epoch, for these parameters, subj and stim:
                    %   First, retrieve epoch 
                    this_epoch = cell2mat(tester.epochs(stimulusIndex,...
                                                subjectIndex, epoch_param_idx));
                    %   Repeate for each test frequency
                    t = repmat(this_epoch(2:end),numel(tester.signalFrequencies),1);

                    %   Multiply the epoch time by binary detection mask 
                    %   (this leaves either 0 or the detection time)
                    t_detect = t.*(exam_tp(tester.signalFrequencies,2:end,test_channel_idx)>0);
                
                    %   Ignore non-detections, and find the minimal time
                    t_detect(t_detect==0) = NaN;
                    minimal_detection_time(epoch_param_idx, test_channel_idx) = min(t_detect,[],'all');
                    % average_detection_time(epoch_param_idx, test_channel_idx) = mean(t_detect,'all','omitnan');
    
                end
            end
        end
    end
end


denom = numel(tester.noiseFrequencies)*numel(tester.subjectIndices);
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

%% Setup export csv for statistical analysis
% desired table
%        mmin mmax windowsize ntests eegchannel stimlvl method tp fp tn fn
% param1
% param2
% param3
% ...

% table_size = table_rows*(param_idx-1)+channel_idx;

nChannels = numel(tester.ord_calculator.channels);
mmin = zeros(numel(nonemptyparams_idxs)*nChannels,1);
mstep = zeros(numel(nonemptyparams_idxs)*nChannels,1);
minwindowsize = zeros(numel(nonemptyparams_idxs)*nChannels,1);
mmax = zeros(numel(nonemptyparams_idxs)*nChannels,1);
ntests = zeros(numel(nonemptyparams_idxs)*nChannels,1);
eeg_channel = cell(numel(nonemptyparams_idxs)*nChannels,1);
stim_lvl = cell(numel(nonemptyparams_idxs)*nChannels,1);
method = cell(numel(nonemptyparams_idxs)*nChannels,1);
table_fp = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_tp = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_fn = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_tn = zeros(numel(nonemptyparams_idxs)*nChannels,1);

thisMethod ='cgst';

thisStimlvl = cell2mat(dtl.zanoteliStimulusNames(stimulusIndex));
% thisStimlvl = str2double(thisStimlvl(1:2));

table_rows = numel(epoch_param_idx);

for channel_idx = 1:numel(tester.ord_calculator.channels)
    thisEegChannel = cell2mat(dtl.zanoteliLeads(channel_idx));
    % thisEegChannel =channel_idx;

    for param_idx = 1:numel(nonemptyparams_idxs)

        epoch_idx =nonemptyparams_idxs(param_idx);
        test_channel_idx = tester.ord_calculator.channels(channel_idx);

        this_epoch = cell2mat(tester.epochs(stimulusIndex,subjectIndex, ...
                                                    epoch_idx));

        table_idx = table_rows*(param_idx-1)+channel_idx;

        mmin(table_idx) = min(this_epoch,[],'all');
        mstep(table_idx) = this_epoch(end)-this_epoch(end-1)+1;
        minwindowsize(table_idx) = min(diff(this_epoch));
        mmax(table_idx) = max(this_epoch,[],'all');

        ntests(table_idx) = numel(this_epoch);

        eeg_channel{table_idx} = thisEegChannel;
        % eeg_channel(table_idx) = thisEegChannel;

        stim_lvl{table_idx} = thisStimlvl;
        method{table_idx} = thisMethod;

        table_fp(table_idx) = fp_rate(epoch_idx, test_channel_idx);
        table_tp(table_idx) = tp_rate(epoch_idx, test_channel_idx);
        table_fn(table_idx) = fn_rate(epoch_idx, test_channel_idx);
        table_tn(table_idx) = tn_rate(epoch_idx, test_channel_idx);

    end
end

varnames = {'mmin','mmax','mstep','minwindowsize','ntests', ...
                'eegchannel','stimlvl','method','tp','fp','tn','fn'};

results_table = table(mmin,mmax,mstep,minwindowsize,ntests,eeg_channel, ...
                stim_lvl,method,table_fp,table_tp,table_fn,table_tn, ...
                'VariableNames',varnames);

head(results_table)
writetable(results_table,'30db_cgst.csv')
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
%%

figure1 = figure(3); axes1 = axes('Parent',figure1); hold(axes1,'on');
Intensidade = {'30dB'};
 % Intensidade = {'50dB'};
cor = ['m','r','g','y'];
% cor = ['m','r'];

addpath('C:\Users\alexa\Desktop\Numero_Deteccoes_consecutiva_H')

for metodo = 1:4
    % caminho1 = ['timeM_50db_M',num2str(metodo),'.mat'];
    % caminho2 = ['pareto_50db_M',num2str(metodo), '.mat'];
    caminho1 = ['timeM_30db_M',num2str(metodo),'.mat'];
    caminho2 = ['pareto_30db_M',num2str(metodo), '.mat'];
%     
    load(caminho1,'timeM')
    load(caminho2,'TXD','Mmax','parametros');
    
    % pontos e linhas pretos: SINGLE SHOT 50dB
    plot([0 1]*100, [Mmax Mmax],'-.b','linewidth',1)
    plot([TXD(1) TXD(1)], [min(timeM) max(timeM)],'-.b','linewidth',1)
    % SINGLE SHOT 30dB
%     plot([TXD(end) TXD(end)], [min(timeM) max(timeM)],'-.b','linewidth',1)
   


    % for ii = 1:size(parametros,1)
    %     plot(TXD(ii),timeM(ii),'.k','Markersize',6,'DisplayName',[num2str(parametros(ii,1)) '-' num2str(parametros(ii,2))])
    % end

    [p, idxs] = paretoFront([TXD,(-timeM)] ); 
    auxL = p(:,1)<0.5; 
    p(auxL,:) = [];
    idxs(auxL,:) = [];

    [~,ind] = sort(p(:,1));
    p = p(ind,:);
    idxs = idxs(ind,:);
    
    % Linha colorida:
    tipo = ['-o' cor(metodo)];
    linha(metodo) = plot(TXD(idxs),timeM(idxs),tipo,'Markersize',8,'linewidth',1.8) ;
    
    for ii = 1:size(idxs,1)

        [I] = find((TXD == TXD(idxs(ii))) & (timeM==timeM(idxs(ii))));
        I = I(1); 

        for jj = I  
            text(TXD(jj),timeM(idxs(ii))*.975,['\{' num2str(parametros(jj,1)) ',' num2str(parametros(jj,2)) '\}' ]);
        end    
    end
end

%%
set(axes1,'XMinorTick','on');
set(axes1,'YMinorTick','on');
box(axes1,'off');


xlim([min(TXD(idxs))*.95,max(TXD(idxs))*1.05])
ylim([min(timeM(idxs))*.95 Mmax*1.05])

grid on
hold on
title(['Curva de Pareto Intensidade ', (Intensidade{1}), ' SPL: Métodos NDC e NNDC'])
% title('Curva de Pareto - 30 dB')

legend([linha(1),linha(2),linha(3),linha(4)],'Método 1','Método 2','Método 3','Método 4')
xlabel('Taxa de deteção [%]')
ylabel('Tempo de exame [s]')

%%
tp_rate2 = tp_rate(tp_rate>0);
minimal_detection_time2 = minimal_detection_time(tp_rate>0);
% minimal_detection_time2 = minimal_detection_time(tp_rate>0);
tp_rate2 =tp_rate2(~isnan(minimal_detection_time2));
minimal_detection_time2 = minimal_detection_time2(~isnan(minimal_detection_time2));
% tp_rate2(isnan(minimal_detection_time2)) = [];
% tp_rate2(isnan(tp_rate2)) = [];
% figure(3)

[p, idxs] = paretoFront([100*tp_rate2,-minimal_detection_time2] );
% auxL = p(:,1)<0.5; 
% p(auxL,:) = [];
% idxs(auxL,:) = [];

[~,ind] = sort(p(:,1));
p = p(ind,:);
idxs = idxs(ind,:);

best_tp_time = minimal_detection_time2(idxs);
best_tp = 100*tp_rate2(idxs);

% new_best_tp_time = best_tp_time(find(best_tp == unique(best_tp)));
% [best_tp2,r] = unique(sort(best_tp));
% best_tp_time2 = best_tp_time(sort(r));

% best_tp_time(isnan(best_tp)) = [];
% best_tp(isnan(best_tp)) = [];




% Linha colorida:
% tipo = ['-o' cor(metodo)];
plot(100*tp_rate2,minimal_detection_time2,'k.')
hold on
% plot(best_tp_time,best_tp,'-o blue','Markersize',8,'linewidth',1.8) ;
plot(best_tp,best_tp_time,'-o blue','Markersize',8,'linewidth',1.8) ;
% plot(minimal_detection_time2(idxs),100*tp_rate2(idxs),'blue','Markersize',8,'linewidth',1.8) ;
xlabel('Detection Rate [%]')
ylabel('Exam Time [s]')
% save(s,'-v7.3')