function plotResults(aThresholds, gThresholds, stageMetrics, perf, params,fignum) 
% plotResults displays the sequential test results and performance metrics. 
    figure(fignum); 
    subplot(2,1,1); 
    hold on; 
    for channel = 1:params.nChannels
        for freq = 1:numel(params.testFrequencies)
            % add if freq is noise or not
            if freq<=params.flagNoise
                plot(1:params.K, squeeze(stageMetrics(channel,:,freq)), ...
                    'b-o', 'LineWidth', 1.2);
            else
                % plot(1:params.K, squeeze(stageMetrics(channel,:,freq)), ...
                %     '-.', 'Color', 0.8*[1 1 1],'LineWidth', 0.8);
            end
           
        end
    end
    plot(1:params.K, aThresholds, 'g--', 'LineWidth', 2); 
    plot(1:params.K, gThresholds, 'r--', 'LineWidth', 2); 
    xlabel('Stage'); ylabel('Accumulated MSC'); 
    title('Sequential Test Accumulation'); 
    legend('Accumulated MSC', 'Detection Threshold', 'Futility Threshold'); 
    grid on;
    subplot(2,1,2);
    bar([perf.FPR, perf.TPR, perf.TNR, perf.FNR]);
    set(gca, 'XTickLabel', {'FPR','TPR','FNR','TNR'});
    ylabel('Rate');
    title('Performance Metrics');
    grid on;
end