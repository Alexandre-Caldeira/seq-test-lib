function perf = performanceMetrics(decisions, params) 
% performanceMetrics computes performance metrics for the sequential test. 
    
    nTestFreqs = numel(params.testFrequencies);
    flagNoise = params.flagNoise;
    NumTests = params.nChannels * nTestFreqs;

    is_noise = 1:nTestFreqs > flagNoise;
    mask = zeros(params.nChannels,params.K,nTestFreqs, 'logical');
    for i = 1:params.nChannels
        for j = 1:params.K
            mask(i,j,:) = is_noise;
        end
    end
    is_noise = mask;

    is_detection = decisions(params.testFrequencies,:,:) == 1;
    is_futility = decisions(params.testFrequencies,:,:) == -1;

    % Vectorized operations for each condition
    mask_TP = ~is_noise & is_detection;
    mask_FN = ~is_noise & is_futility;
    mask_FP = is_noise & is_detection;
    mask_TN = is_noise & is_futility;

    perf.TP = zeros(params.nChannels,params.K, nTestFreqs);
    perf.FN = zeros(params.nChannels,params.K, nTestFreqs);
    perf.FP = zeros(params.nChannels,params.K, nTestFreqs);
    perf.TN = zeros(params.nChannels,params.K, nTestFreqs);

    perf.TP(mask_TP) = 1;
    perf.FN(mask_FN) = 1;
    perf.FP(mask_FP) = 1;
    perf.TN(mask_TN) = 1;

    perf.TPRr = 100*sum_reduce(sum_reduce(perf.TP,3),1)/NumTests;
    perf.FNRr = 100*sum_reduce(sum_reduce(perf.FN,3),1)/NumTests;
    perf.FPRr = 100*sum_reduce(sum_reduce(perf.FP,3),1)/NumTests;
    perf.TNRr = 100*sum_reduce(sum_reduce(perf.TN,3),1)/NumTests;

    perf.TPR = 100*sum_reduce(sum_reduce(sum_reduce(perf.TP,3),1),2)/(NumTests*params.K);
    perf.FNR = 100*sum_reduce(sum_reduce(sum_reduce(perf.FN,3),1),2)/(NumTests*params.K);
    perf.FPR = 100*sum_reduce(sum_reduce(sum_reduce(perf.FP,3),1),2)/(NumTests*params.K);
    perf.TNR = 100*sum_reduce(sum_reduce(sum_reduce(perf.TN,3),1),2)/(NumTests*params.K);



    % 
    % for idx_freq = 1:nTestFreqs
    % 
    %         % SIGNAL
    %         if idx_freq < flagNoise && ...                         % not noise
    %             ( decisions(channel_n,k, idx_freq)==1 )                    % detected
    % 
    %             TP(channel_n,k,idx_freq) = TP(k,idx_freq)+1;
    %             t_decisao(vk,idx_freq) = ~sum(t_decisao(:,idx_freq),'all');
    % 
    %         elseif idx_freq < flagNoise && ...                     % not noise
    %                 ( decisions(channel_n,k, idx_freq)==-1 )               % gave up 
    % 
    %             FN(channel_n,k,idx_freq) = FN(k,idx_freq)+1;
    %             t_decisao(channel_n,k,idx_freq) = -1*(~sum(t_decisao(:,idx_freq),'all'));
    % 
    %         % NOISE
    %         elseif idx_freq >= flagNoise && ...                    % is noise
    %                 ( decisions(channel_n,k, idx_freq)==1 )               % detected
    % 
    %             FP(channel_n,k,idx_freq) = FP(k,idx_freq)+1;
    %             t_decisao(channel_n,k,idx_freq) = ~sum(t_decisao(:,idx_freq),'all');
    % 
    %         elseif idx_freq >= flagNoise && ...                    % is noise
    %                 ( decisions(channel_n,k, idx_freq)==-1 )               % gave up
    % 
    %             TN(channel_n,k,idx_freq) = TN(k,idx_freq)+1;
    %             t_decisao(channel_n,k,idx_freq) = -1*(~sum(t_decisao(:,idx_freq),'all'));
    % 
    %         end
    % 
    % end
    % 
    % nDetect = sum(decisions == 1, 'omitnan'); 
    % % Under a noise-only simulation, detections are false positives. 
    % perf.FPR = nDetect / NumTests; 
    % % For signal simulations, TPR would be similarly computed. 
    % % Here we provide a dummy value. 
    % perf.TPR = (nDetect) / NumTests; 
end