%% Clear workspace and initialize parameters
clearvars; close all; clc;

%% Test pipeline
% Apply TEST(s) to objective response detector (MSC)
% tester = ORDTester(ordc);
% tester = tester.compute_beta_cgst_thresholds();

tester = ORDTester(ORDCalculator(DataLoader('sim')));

tester.dataloader = tester.dataloader.genBulkSim( ...
    groupNoiseMean=[-35 -35], groupNoiseStd=1.5*randi(5,1,11)).computeBulkFFTs(mode='sim');
tester.age()

% Pretty much instantly runs
% K_stages = 5;
% tester.ord_calculator = ORDCalculator(tester.dataloader).fit_epochs( ...
%     stimulusIndices = tester.dataloader.groupNoiseMean,...
%     subjectIndices = tester.dataloader.groupNoiseStd,...
%     startWindows = 1, ...
%     K_stages = K_stages, ...
%     single_or_bulk = 'bulk',...
%     lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
%     sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
%     ... % then, compute on selected channels:
%     );

% Takes at least about 860.52 seconds = 15 mins to run 
K_stages = [2 3 5 8 10];
tester.ord_calculator = ORDCalculator(tester.dataloader).fit_epochs( ...
    stimulusIndices = tester.dataloader.groupNoiseMean,...
    subjectIndices = tester.dataloader.groupNoiseStd,...
    startWindows = [1 6 10 25], ... % windowStepSizes = 52, ...
    K_stages = K_stages,...
    single_or_bulk = 'bulk',...
    lastWindowCalcMethod = 'exactK', ... % maxFromStart, maxFromLast, exactK, fromSizeType
    sizeType = 'fixedSize'... % minToMax, minToFix, withResampling, default = fixedSize
    ... % then, compute on selected channels:
    );

tester.ord_calculator = tester.ord_calculator.bulk_compute_msc();
tester.age()

tester.desired_alpha = 0.05;
tester = tester.compute_bulk_sim_beta_cgst_decisions();
tester.age()


%% Show results

nParams = numel(tester.epochs(1,1,:));
nChannels = numel(tester.ord_calculator.channels);
tester.FP = zeros(nParams,nChannels);
tester.FN = zeros(nParams,nChannels);
tester.TP = zeros(nParams,nChannels);
tester.TN = zeros(nParams,nChannels);

for noiseMean_idx = 1:numel(tester.dataloader.groupNoiseMean)
    for noiseVar_idx = 1:numel(tester.dataloader.groupNoiseStd)
        for channel_idx = 1:numel(tester.ord_calculator.channels)
            for epoch_param_idx = 1:nParams 
                exam_fp = cell2mat(tester.groupFP(noiseMean_idx, noiseVar_idx, epoch_param_idx));
                if ~isempty(exam_fp)
                    % If there was a detection on any stage, 
                    % assign 1 to that exam, and repeat for all
                    % frequencies. Then, sum all 1s.
                    % + Add previous results from other subjs
                    % To compute rate (pct), divide by #freqs and #subj

                    exam_fn = cell2mat(tester.groupFN(noiseMean_idx, noiseVar_idx, epoch_param_idx));
                    exam_tp = cell2mat(tester.groupTP(noiseMean_idx, noiseVar_idx, epoch_param_idx));
                    exam_tn = cell2mat(tester.groupTN(noiseMean_idx, noiseVar_idx, epoch_param_idx));

                    tester.FP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fp(tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + tester.FP(epoch_param_idx, channel_idx);

                    tester.FN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fn(tester.signalFrequencies,:,channel_idx)>0,2))...
                            + tester.FN(epoch_param_idx, channel_idx);

                    tester.TP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tp(tester.signalFrequencies,:,channel_idx)>0,2))...
                            + tester.TP(epoch_param_idx, channel_idx);

                    tester.TN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tn(tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + tester.TN(epoch_param_idx, channel_idx);
    
                end
            end
        end
    end
end

denom = numel(tester.noiseFrequencies)*numel(tester.dataloader.groupNoiseMean)...
                                      *numel(tester.dataloader.groupNoiseStd);
fn_rate = 100*tester.FN/(denom)

tn_rate = 100*tester.TN/(denom)

tp_rate = 100*tester.TP/(denom)

fp_rate = 100*tester.FP/(denom)

confmat = table(mean(fp_rate,'all'), ...
    mean(tp_rate(K_stages,:),'all'), ...
    mean(fn_rate(K_stages,:),'all'), ...
    mean(tn_rate(K_stages,:),'all'), 'VariableNames',{'fp','tp','fn','tn'})
