function [decisions, stageMetrics] = sequentialTest(MSCvalues, params) 
% sequentialTest applies the decision process using the accumulated MSC values.
    % decisions = zeros(1, K); 
    stageMetrics = cumsum(MSCvalues,2); 
    decisions = zeros(size(stageMetrics));

    % % Compute stopping decisions
    % for k = 1:params.K
    %     decisions(stageMetrics(:,k,:) >= params.aThresholds(k)) = 1;
    %     decisions(stageMetrics(:,k,:) < params.gThresholds(k)) = -1;
    % end 

    for channel=1:params.nChannels
        for freq = params.testFrequencies
            for k = 1:params.K

                % Useful for debugging
                % if k==params.K && ( (decisions(channel, k-1, freq))==0 )
                    % disp('last chance!')
                    % k
                    % decs = decisions(channel, :, freq)
                    % this_met = stageMetrics(channel, k, freq)
                    % is_signal = params.aThresholds(k)
                    % is_noise = params.gThresholds(k)
                % end
            
                % if k>1 && ( (decisions(channel, k-1, freq))~=0 )%~isnan(decisions(channel, k, freq))%
                %     % Not first test and already decided, keep decision
                %     decisions(channel, k, freq) = decisions(channel, k-1, freq);
                %     % disp('...')
                %     % disp([channel, k, freq])
                %     % disp(decisions(channel, k, freq))
                % 
                % else
                    if stageMetrics(freq,k, channel) >= params.aThresholds(k)
                        % Enough evidence to detect signal
                        decisions(freq, k:params.K, channel) = 1;
                        decisions(freq, 1:k-1, channel) = NaN;
                        continue

                    elseif stageMetrics(freq, k, channel) < params.gThresholds(k)
                        % Enough evidence to stop trial (futile)
                        decisions(freq, k:params.K, channel) = -1;
                        decisions(freq, 1:k-1, channel) = NaN;
                        continue

                    elseif k==params.K 
                        % Last trial, and not enough evidence found for
                        % either detection criteria (ERROR!)
                        % decisions(channel, :, freq) = NaN;
                        error('Oops')
                        
                    end
                
                % end

            end
        end
    end


end

%
% onde stageMetrics(:,k,:) >= aThresholds(k), stageMetrics(:,k,:) = 1
%
% onde decisions(:,k+1,:) = decisions(:,1:k,:) if not zero