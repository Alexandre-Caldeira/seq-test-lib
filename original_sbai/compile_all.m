%%
clearvars;close all; clc;

data_strs = {'tester_checkpoint_1_T10_16_320_3_2025.mat',...
'tester_checkpoint_2_T10_47_320_3_2025.mat',...
'tester_checkpoint_3_T11_25_3520_3_2025.mat',...
'tester_checkpoint_4_T12_10_5620_3_2025.mat',...
'tester_checkpoint_5_T12_27_4020_3_2025.mat',...
'tester_checkpoint_6_T12_52_1920_3_2025.mat',...
'tester_checkpoint_7_T13_22_1620_3_2025.mat',...
'tester_checkpoint_8_T13_51_1920_3_2025.mat'};

disp(1)
disp(datetime())
res = load(cell2mat(data_strs(1)));
complete_groupFP = res.tester_checkpoint.groupFP;

%%

all_vaz_startWindows = {[1:30 240],...
    [31:60 240],...
    [61:90 240],...
    [91:120 240],...
    [121:150 240],...
    [151:180 240],...
    [181:210 240],...
    [211:239 240],...
    };

for idx = 2:numel(data_strs)
    disp(idx)
    disp(datetime())
    res = load(cell2mat(data_strs(idx)));
    % complete_groupFP(:,:,all_vaz_startWindows(idx),:) = ...
    %     res.tester_checkpoint.groupFP(:,:,cell2mat(all_vaz_startWindows(idx)),:);
    % 
    % clearvars -except complete_groupFP all_vaz_startWindows data_strs
    complete_groupFP = res.tester_checkpoint.groupFP;
% end

%%

nParams = numel(complete_groupFP(1,1,:));
nChannels = numel(1);
tester.FP = zeros(nParams,nChannels);
tester.FN = zeros(nParams,nChannels);
tester.TP = zeros(nParams,nChannels);
tester.TN = zeros(nParams,nChannels);

for stimulusIndex = res.tester_checkpoint.selectedZanoteliStimuli
    for subjectIndex = res.tester_checkpoint.selectedZanoteliSubjects
        for channel_idx = 1:numel(1)
            nonemptyparams_idxs = [];
            for epoch_param_idx = 1:nParams 
                
                if size(tester.groupFP,1)==1
                    exam_fp = cell2mat(complete_groupFP(1, subjectIndex, epoch_param_idx));
                elseif size(tester.groupFP,2)==1
                    exam_fp = cell2mat(complete_groupFP(stimulusIndex, 1, epoch_param_idx));
                else
                    exam_fp = cell2mat(complete_groupFP(stimulusIndex, subjectIndex, epoch_param_idx));
                end

                if ~isempty(exam_fp)
                    % If there was a detection on any stage, 
                    % assign 1 to that exam, and repeat for all
                    % frequencies. Then, sum all 1s.
                    % + Add previous results from other subjs
                    % To compute rate (pct), divide by #freqs and #subj
                    % 
                    % if size(tester.groupFP,1)==1
                    %     exam_fn = cell2mat(tester.groupFN(1, subjectIndex, epoch_param_idx));
                    %     exam_tp = cell2mat(tester.groupTP(1, subjectIndex, epoch_param_idx));
                    %     exam_tn = cell2mat(tester.groupTN(1, subjectIndex, epoch_param_idx));
                    % 
                    % elseif size(tester.groupFP,2)==1
                    %     exam_fn = cell2mat(tester.groupFN(stimulusIndex, 1, epoch_param_idx));
                    %     exam_tp = cell2mat(tester.groupTP(stimulusIndex, 1, epoch_param_idx));
                    %     exam_tn = cell2mat(tester.groupTN(stimulusIndex, 1, epoch_param_idx));
                    % 
                    % else
                    %     exam_fn = cell2mat(tester.groupFN(stimulusIndex, subjectIndex, epoch_param_idx));
                    %     exam_tp = cell2mat(tester.groupTP(stimulusIndex, subjectIndex, epoch_param_idx));
                    %     exam_tn = cell2mat(tester.groupTN(stimulusIndex, subjectIndex, epoch_param_idx));
                    % 
                    % end
                    % 
                    

                    % tester.FN(epoch_param_idx, channel_idx) = ...
                    %         sum(any(exam_fn(tester.signalFrequencies,:,channel_idx)>0,2))...
                    %         + tester.FN(epoch_param_idx, channel_idx);
                    % 
                    % tester.TP(epoch_param_idx, channel_idx) = ...
                    %         sum(any(exam_tp(tester.signalFrequencies,:,channel_idx)>0,2))...
                    %         + tester.TP(epoch_param_idx, channel_idx);

                    tester.FP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fp(tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + tester.FP(epoch_param_idx, channel_idx);

                    % tester.TN(epoch_param_idx, channel_idx) = ...
                    %         sum(any(exam_tn(tester.noiseFrequencies,:,channel_idx)>0,2))...
                    %         + tester.TN(epoch_param_idx, channel_idx);

                    nonemptyparams_idxs = [nonemptyparams_idxs,...
                                            epoch_param_idx];
                        %sub2ind(size(tester.FP), epoch_param_idx, channel_idx)];
    
                end
            end
        end
    end
end


denom = numel(res.tester_checkpoint.allTestFrequencies(9:end))...
    *numel(tester.selectedZanoteliSubjects)...
    *numel(res.tester_checkpoint.selectedZanoteliStimuli);
fp_rate = tester.FP/(denom)

% clearvars -except all_vaz_startWindows data_strs fp_rate


% tp_rate = tester.TP/(denom);

% fn_rate = tester.FN/(denom);

% tn_rate = tester.TN/(denom);

% confmat = table( ...
%     100*mean(fn_rate(nonemptyparams_idxs,single_channel)), ...
%     100*mean(fp_rate(nonemptyparams_idxs,single_channel)), ...
%     100*mean(tp_rate(nonemptyparams_idxs,single_channel)), ...
%     100*mean(tn_rate(nonemptyparams_idxs,single_channel)), ...
%     'VariableNames',{'fn','fp','tp','tn'})

%%
figure(1);
plot(100*fp_rate(nonemptyparams_idxs,1),'.')

drawnow
end
%%

% a = 3
% 
% b = 1:10
% n = rand(3);
% g = randi([110],1,20);
% c =[];
% for i = b
% 
%     c(i) = 2*a*i;   
%     clearvars -except a b c
%     a = c(1)-1;
% end