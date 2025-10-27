
%%
temp_tester = obj;
nParams = numel(temp_tester.epochs(1,1,:));
nChannels = numel(temp_tester.ord_calculator.channels);
temp_tester.FP = zeros(nParams,nChannels);
temp_tester.FN = zeros(nParams,nChannels);
temp_tester.TP = zeros(nParams,nChannels);
temp_tester.TN = zeros(nParams,nChannels);

for stimulusIndex = temp_tester.stimulusIndices
    for subjectIndex = temp_tester.subjectIndices
        for channel_idx = 1:numel(temp_tester.ord_calculator.channels)
            for epoch_param_idx = 1:nParams 
                
                if size(temp_tester.groupFP,1)==1
                    exam_fp = cell2mat(temp_tester.groupFP(1, subjectIndex, epoch_param_idx));
                elseif size(temp_tester.groupFP,2)==1
                    exam_fp = cell2mat(temp_tester.groupFP(stimulusIndex, 1, epoch_param_idx));
                else
                    exam_fp = cell2mat(temp_tester.groupFP(stimulusIndex, subjectIndex, epoch_param_idx));
                end

                if ~isempty(exam_fp)
                    % If there was a detection on any stage, 
                    % assign 1 to that exam, and repeat for all
                    % frequencies. Then, sum all 1s.
                    % + Add previous results from other subjs
                    % To compute rate (pct), divide by #freqs and #subj

                    if size(temp_tester.groupFP,1)==1
                        exam_fn = cell2mat(temp_tester.groupFN(1, subjectIndex, epoch_param_idx));
                        exam_tp = cell2mat(temp_tester.groupTP(1, subjectIndex, epoch_param_idx));
                        exam_tn = cell2mat(temp_tester.groupTN(1, subjectIndex, epoch_param_idx));

                    elseif size(temp_tester.groupFP,2)==1
                        exam_fn = cell2mat(temp_tester.groupFN(stimulusIndex, 1, epoch_param_idx));
                        exam_tp = cell2mat(temp_tester.groupTP(stimulusIndex, 1, epoch_param_idx));
                        exam_tn = cell2mat(temp_tester.groupTN(stimulusIndex, 1, epoch_param_idx));

                    else
                        exam_fn = cell2mat(temp_tester.groupFN(stimulusIndex, subjectIndex, epoch_param_idx));
                        exam_tp = cell2mat(temp_tester.groupTP(stimulusIndex, subjectIndex, epoch_param_idx));
                        exam_tn = cell2mat(temp_tester.groupTN(stimulusIndex, subjectIndex, epoch_param_idx));

                    end
                    
                    

                    temp_tester.FN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fn(temp_tester.signalFrequencies,:,channel_idx)>0,2))...
                            + temp_tester.FN(epoch_param_idx, channel_idx);

                    temp_tester.TP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tp(temp_tester.signalFrequencies,:,channel_idx)>0,2))...
                            + temp_tester.TP(epoch_param_idx, channel_idx);

                    temp_tester.FP(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_fp(temp_tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + temp_tester.FP(epoch_param_idx, channel_idx);

                    temp_tester.TN(epoch_param_idx, channel_idx) = ...
                            sum(any(exam_tn(temp_tester.noiseFrequencies,:,channel_idx)>0,2))...
                            + temp_tester.TN(epoch_param_idx, channel_idx);
    
                end
            end
        end
    end
end


denom = numel(temp_tester.noiseFrequencies)*numel(temp_tester.subjectIndices)*numel(temp_tester.stimulusIndices);
fp_rate = temp_tester.FP/(denom);

tp_rate = temp_tester.TP/(denom);

fn_rate = temp_tester.FN/(denom);

tn_rate = temp_tester.TN/(denom);

confmat = table( ...
    100*mean(fn_rate(end,:)), ...
    100*mean(fp_rate(end,:)), ...
    100*mean(tp_rate(end,:)), ...
    100*mean(tn_rate(end,:)), ...
    'VariableNames',{'fn','fp','tp','tn'})