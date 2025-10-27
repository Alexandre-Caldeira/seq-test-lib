classdef PreProcessor
    % TODO: add all filters from (Antunes et al., 2016?) for further use 
    properties
        zanoteliGain  = 200;
        zanoteliCut

        filterFcLower;  % Lower cutoff frequency (Hz)
        filterFcUpper;  % Upper cutoff frequency (Hz)
        filterOrder;    % Butterworth filter order
        fs;      
        
        processedSignals
        filteredSignals

        % Stores timeseries from all selected subjects and stimuli
        groupProcessedSignals % stimulus-subject-indexed cell:
        groupFilteredSignals  % groupSignals{stimulus,subject} -> y(sample,window,channel)
    end

    methods
        function obj = PreProcessor()
            obj.zanoteliCut = 0.1/obj.zanoteliGain; % for artifact removal
        end

        function obj = zanoteliPreProcessing(obj, dataloader)
            % preprocessSignal removes the DC offset and applies a Butterworth bandpass filter.
            % expected datastructure is signal = zeros(fs, epochs, nchannels)
            Mmax = dataloader.zanoteliSuggestedMMax(dataloader.zanoteliStimulusIndex);
            obj.processedSignals = zeros( ...
                                    size(dataloader.signals,1), ...
                                    Mmax, ...
                                    size(dataloader.signals,3));

            % TODO: remover ALTISSIMO acoplamento aqui! deveria ser
            % interface na entrada e nao consumir o obj inteiro ou sla
            for channel = dataloader.channels
                % If mock needed: dataloader.signal = nan(dataloader.fs, dataloader.duration, dataloader.nChannel)
                x = squeeze(dataloader.signals(:,:,channel));
    
                %1segundo de sinal 
                nfft = dataloader.nfft;
                 
                %retirar componente DC por janela 
                % (fiz isso pq no processamento em tempo real é por janela)
                % tirar a média de cada de cada trecho - devido a remoção
                x = x - repmat(mean(x),nfft,1); 
                    
                % Excluir os dois primeiros segundos do inicio da coleta 
                x(:,1:2,:) = []; 
                    
                % Encontrar o valor máximo por canal 
                Vmax = max(abs(x),[],1);

                % Remover o ruído amplitude 
                ind = Vmax > obj.zanoteliCut;
                x = x(:,~ind);  

                % Limitar o tamanho para o valor máximo. 
                obj.processedSignals(:,:,channel) = x(:,1:Mmax);

            end

            % Remove DC offset
            % obj.processedSignals = signal - mean(signal, 2);
            % 
            % [b, a] = butter(obj.filterOrder, ... 
            %     [obj.filterFcLower, obj.filterFcUpper] / (obj.fs/2)); 
            % obj.processedSignals = filtfilt(b, a, obj.processedSignals); 

        end

        function obj = antunesFiltering(obj, dataloader)
            % antunesFiltering applies a Butterworth bandpass filter.
            obj.filteredSignals = zeros(size(obj.processedSignals));
            
            if size(obj.processedSignals,2)<48
                warning('Skipping filtering this exam, it has less than 48 windows!')
                obj.filteredSignals = obj.processedSignals;
                return
            end
                
            obj.fs = size(obj.processedSignals,1);
            obj.filterFcLower = 70;  
            obj.filterFcUpper = obj.fs/2 - 1;       
            obj.filterOrder   = 8;                  

            for channel = dataloader.channels
                
                x = squeeze(obj.processedSignals(:,:,channel));

                [b, a] = butter(obj.filterOrder, ...
                    [obj.filterFcLower, obj.filterFcUpper] / (obj.fs/2)); 
                
                obj.filteredSignals(:,:,channel) = filtfilt(b, a, x')'; 

            end

        end

        function obj = bulkZanoteliPreprocess(obj, dataloader)

            if matches(dataloader.mode(1:3),'exp', IgnoreCase=true)
            
                obj.groupProcessedSignals = cell(numel(dataloader.selectedZanoteliStimuli), ...
                                    numel(dataloader.selectedZanoteliSubjects));
    
                for stimulusIndex = dataloader.selectedZanoteliStimuli
                    for subjectIndex = dataloader.selectedZanoteliSubjects
    
                        % Reset signal data and parameters
                        dataloader.signals = cell2mat(dataloader.groupSignals(stimulusIndex,subjectIndex));
                        dataloader.fs = size(dataloader.signals, 1);
                        dataloader.nfft = dataloader.fs;
                        dataloader.nBins = floor(dataloader.fs/2)+1;
                        dataloader.zanoteliStimulusIndex = stimulusIndex;
                        dataloader.zanoteliSubjectIndex = subjectIndex;
    
                        obj = obj.zanoteliPreProcessing(dataloader);
    
                        obj.groupProcessedSignals{stimulusIndex,subjectIndex} = obj.processedSignals;
    
                    end
                end

            elseif matches(dataloader.mode(1:3),'sim', IgnoreCase=true) 
                
                 obj.groupProcessedSignals = cell(numel(dataloader.selectedZanoteliStimuli), ...
                                    numel(dataloader.selectedZanoteliSubjects));
    
                for stimulusIndex = dataloader.selectedZanoteliStimuli
                    for subjectIndex = dataloader.selectedZanoteliSubjects
    
                        % Reset signal data and parameters
                        dataloader.signals = cell2mat(dataloader.groupSignals(stimulusIndex,subjectIndex));
                        dataloader.fs = size(dataloader.signals, 1);
                        dataloader.nfft = dataloader.fs;
                        dataloader.nBins = floor(dataloader.fs/2)+1;
                        dataloader.zanoteliStimulusIndex = stimulusIndex;
                        dataloader.zanoteliSubjectIndex = subjectIndex;
    
                        obj = obj.zanoteliPreProcessing(dataloader);
    
                        obj.groupProcessedSignals{stimulusIndex,subjectIndex} = obj.processedSignals;
    
                    end
                end
                

            else
                error('Invalid dataloader mode!')

            end

        end

        function obj = bulkAntunesFilter(obj, dataloader)
             obj.groupFilteredSignals= cell(numel(dataloader.selectedZanoteliStimuli), ...
                                    numel(dataloader.selectedZanoteliSubjects));

            for stimulusIndex = dataloader.selectedZanoteliStimuli
                for subjectIndex = dataloader.selectedZanoteliSubjects


                    obj.processedSignals = cell2mat(obj.groupProcessedSignals(stimulusIndex,subjectIndex));
                    obj = obj.antunesFiltering(dataloader);

                    obj.groupFilteredSignals{stimulusIndex,subjectIndex} = obj.filteredSignals;

                end
            end

        end



    end
end