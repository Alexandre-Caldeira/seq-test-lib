classdef ORDCalculator
    properties
        dataloader

        MSC
        groupMSC
        latestMSC
        % multiParamMSC

        % Calculator Parameters
        startWindow
        windowStepSize
        lastWindow
        
        epochs = []
        epochs_index_metadata
        epochs_method
        K_stages    % total number of tests to be applied on the ORD
        nWindows    % number of Windows (may match K, or not)

        subjectIndices = [1:11];
        stimulusIndices = [1:5];
        channels

        startWindows = [1];
        windowStepSizes = [24 32];
        lastWindows = [50];

        % Utils
        timer   % obj timer
        id      % obj id
    end

    methods
        function obj = ORDCalculator(dataloader)
            obj.timer = tic;
            obj.id = [num2str(keyHash(obj.timer))];
            obj.dataloader = dataloader;
            
            obj.startWindow = 1;
            obj.windowStepSize = 24;
            obj.lastWindow = dataloader.duration;
            % obj.epochs = obj.startWindow:obj.windowStepSize:obj.lastWindow;

        end

        function obj = fit_epochs(obj, p)
             arguments
                obj % The ORDCalculator class

                % p: additional parameters, passed as Name-Value arguments
                %    declared below, including their default values.
                p.dataloader = obj.dataloader;
                p.channels = obj.dataloader.channels;
                p.nChannels = numel(obj.dataloader.channels);

                p.subjectIndices = obj.subjectIndices;
                p.stimulusIndices = obj.stimulusIndices;

                p.startWindows = obj.startWindows;
                p.windowStepSizes = obj.windowStepSizes;
                p.lastWindows = obj.lastWindows;

                % How will the index of the last window be computed?
                p.lastWindowCalcMethod {mustBeMember(p.lastWindowCalcMethod,...
                                    {'maxFromStart', 'maxFromLast', 'exactK', 'fromSizeType'})} = 'maxFromStart'; 
                
                % How will the number of samples in each window be computed?
                p.sizeType {mustBeMember(p.sizeType,...
                                    {'fixedSize', 'minToMax', 'minToFix', 'withResampling'})} = 'fixedSize';
                
                % Are epochs for a single ORD or bulk ORDs?
                p.single_or_bulk {mustBeMember(p.single_or_bulk,{'single', 'bulk'})} = 'single'
                p.K_stages = obj.K_stages;
                p.nWindows = obj.nWindows;
             end

             % Make single to bulk switch automatic (may reconsider later,
             % bad practice...!) TODO
             if numel(p.startWindows) > 1
                 p.single_or_bulk = 'bulk';
             end

             obj.subjectIndices = p.subjectIndices;
             obj.stimulusIndices = p.stimulusIndices;

             method = [p.single_or_bulk,'_',p.sizeType,'_',p.lastWindowCalcMethod];

             switch method
                 case 'bulk_fixedSize_maxFromStart' % zanoteli bulk
                     % User expects all windows to be equally sized. 
                     % User input is subjects, stimuli, stepSizes and startWindows
                     % other params will be ignored

                     % This is not quite right cell indexing allocation but
                     % works on MATLAB... fix later! TODO
                     obj.epochs = cell(numel(p.stimulusIndices), ...
                                        numel(p.subjectIndices), ...
                                        numel(p.startWindows), ...
                                        numel(p.windowStepSizes) ...
                                        );
                     obj.epochs_index_metadata = {'stimulusIndices',...
                        'subjectIndices', 'startWindows', 'windowStepSizes'};
                     obj.epochs_method = method;
                     

                     obj.nWindows = cell(size(obj.epochs));
                     obj.K_stages = cell(size(obj.epochs));

                     for stimulus_index = p.stimulusIndices
                         for subject_index = p.subjectIndices
                             this_exam = cell2mat(p.dataloader.groupSignals(stimulus_index,subject_index));
                             this_lastWindow = size(this_exam, 2);

                             for this_startWindow = p.startWindows
                                 for this_windowStepSize = p.windowStepSizes

                                     this_epoch = this_startWindow:this_windowStepSize:this_lastWindow;
                                     if numel(this_epoch) > 30 || numel(this_epoch) ==0
                                         if numel(this_epoch)> 30
                                            warning('skipping, large n of convs')
                                         end
                                         continue
                                    
                                     end
                                     
                                     obj.epochs{stimulus_index, ...
                                         subject_index, ...
                                         this_startWindow, ...
                                         this_windowStepSize} = this_epoch;
                                     
                                     obj.nWindows{stimulus_index, ...
                                         subject_index, ...
                                         this_startWindow, ...
                                         this_windowStepSize} = numel(this_epoch);

                                     obj.K_stages{stimulus_index, ...
                                         subject_index, ...
                                         this_startWindow, ...
                                         this_windowStepSize} = cell2mat(obj.nWindows(stimulus_index, ...
                                                                             subject_index, ...
                                                                             this_startWindow, ...
                                                                             this_windowStepSize));
                                end
                             end

                         end
                     end

                 case 'bulk_fixedSize_maxFromLast' % inverted zanoteli
                     % User expects all windows to be equally sized. 
                     % User input is subjects, stimuli, stepSizes and lastWindows
                     % User may pass startWindows
                     % other params will be ignored

                     obj.epochs = cell(numel(p.stimulusIndices), ...
                                        numel(p.subjectIndices), ...
                                        numel(p.startWindows), ...
                                        numel(p.windowStepSizes), ...
                                        numel(p.lastWindows)...
                                        );
                     obj.epochs_index_metadata = {'subjectIndices',...
                        'stimulusIndices', 'startWindows', ...
                        'windowStepSizes', 'lastWindows'};
                     obj.epochs_method = method;

                     obj.nWindows = cell(size(obj.epochs));
                     obj.K_stages = cell(size(obj.epochs));

                     for stimulus_index = p.stimulusIndices
                         for subject_index = p.subjectIndices
                             for this_firstWindow = p.startWindows
                                 for this_windowStepSize = p.windowStepSizes
                                    for this_lastWindow = p.lastWindows
                                        this_epoch = flip(p.this_lastWindow:-p.windowStepSize:this_firstWindow);
                                 
                                         obj.epochs{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_windowStepSize, ...
                                             this_lastWindow} = this_epoch;
                                         
                                         obj.nWindows{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_windowStepSize, ...
                                             this_lastWindow} = numel(this_epoch);

                                         obj.K_stages{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_windowStepSize, ...
                                             this_lastWindow} = cell2mat(obj.nWindows(stimulus_index, ...
                                                                         subject_index, ...
                                                                         this_firstWindow, ...
                                                                         this_windowStepSize, ...
                                                                         this_lastWindow));
                                    end
                                 end
                             end
                         end
                     end

                 case 'bulk_fixedSize_exactK' % chesnaye bulk published
                     % Requires 1 or more first windows and Ks
                     % User expects exactly K windows, starting at defined point 
                     % User input is subjects, stimuli and K
                     % User may pass startWindows
                     % other params will be ignored

                     obj.epochs = cell(numel(p.stimulusIndices), ...
                                         numel(p.subjectIndices), ...
                                         numel(p.startWindows), ...
                                         numel(p.K_stages));
                     obj.epochs_index_metadata = {'subjectIndices',...
                        'stimulusIndices', 'startWindows','K_stages'};
                     obj.epochs_method = method;

                     obj.nWindows = obj.epochs;
                     
                     mode = obj.dataloader.mode;
                    if matches(mode(1:3),"sim", IgnoreCase=true)
                        
                        for noiseMean_idx = 1:numel(obj.dataloader.groupNoiseMean)
                            for noiseVar_idx = 1:numel(obj.dataloader.groupNoiseStd)
            
                                this_exam = cell2mat(p.dataloader.groupSignals(noiseMean_idx,noiseVar_idx));
                                this_lastWindow = size(this_exam, 2);
                                 for this_firstWindow = p.startWindows %!
                                     for this_K = p.K_stages
                                         
                                         obj.epochs{noiseMean_idx, ...
                                             noiseVar_idx, ...
                                             this_firstWindow, ...
                                             this_K} = ceil(linspace(this_firstWindow,this_lastWindow,this_K));
                                         
                                         obj.nWindows{noiseMean_idx, ...
                                             noiseVar_idx, ...
                                             this_firstWindow, ...
                                             this_K} = this_K;
    
                                         obj.K_stages{noiseMean_idx, ...
                                             noiseVar_idx, ...
                                             this_firstWindow, ...
                                             this_K} = this_K;
                                     end
                                 end
                            end
                        end
        
                    elseif matches(mode(1:3),"exp", IgnoreCase=true)
                        for stimulus_index = p.stimulusIndices
                             for subject_index = p.subjectIndices
                                 this_exam = cell2mat(p.dataloader.groupSignals(stimulus_index,subject_index));
                                 this_lastWindow = size(this_exam, 2);
                                 
                                 for this_firstWindow = p.startWindows %!
                                     for this_K = p.K_stages
                                         
                                         obj.epochs{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_K} = ceil(linspace(this_firstWindow,this_lastWindow,this_K));
                                         
                                         obj.nWindows{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_K} = this_K;
    
                                         obj.K_stages{stimulus_index, ...
                                             subject_index, ...
                                             this_firstWindow, ...
                                             this_K} = this_K;
                                     end
                                 end
                             end
                         end
        
                    else
                        % Throw error
                        error('DataLoader mode input is invalid.'); 
                    end


                 case 'bulk_fizedSize_fromSizeType' % chesnaye bulk max tests
                     % User expects all windows to be equally sized. 
                     % User input is subjects, stimuli and stepSizes
                     % User may pass startWindows
                     % other params will be ignored
                     obj.epochs = cell(numel(p.stimulusIndices), ...
                                         numel(p.subjectIndices), ...
                                         numel(p.startWindows), ...
                                         numel(p.windowStepSizes));

                     obj.epochs_index_metadata = {'subjectIndices',...
                        'stimulusIndices', 'startWindows',...
                        'windowStepSizes'};
                     obj.epochs_method = method;

                     obj.nWindows = obj.epochs;
                     obj.K_stages = obj.epochs;

                     for stimulus_index = p.stimulusIndices
                         for subject_index = p.subjectIndices
                             this_exam = cell2mat(p.dataloader.groupSignals(stimulus_index,subject_index));    
                             this_lastWindow = size(this_exam, 2);
    
                             for this_firstWindow = p.startWindows %!    
                                 for this_windowStepSize = p.windowStepSizes

                                     this_epoch = this_firstWindow:this_windowStepSize:this_lastWindow;

                                     obj.epochs{stimulus_index, ...
                                         subject_index, ...
                                         this_firstWindow, ...
                                         this_windowStepSize} = this_epoch;
                                    
                                     obj.nWindows{stimulus_index, ...
                                         subject_index, ...
                                         this_firstWindow, ...
                                         this_windowStepSize} = numel(this_epoch);

                                     obj.K_stages{stimulus_index, ...
                                         subject_index, ...
                                         this_firstWindow, ...
                                         this_windowStepSize} = cell2mat(obj.nWindows(stimulus_index, ...
                                                                         subject_index, ...
                                                                         this_firstWindow, ...
                                                                         this_windowStepSize));
                                 end
                             end
                         end
                     end


                 case 'single_fixedSize_maxFromStart' % zanoteli single
                    obj.epochs = p.startWindows:p.windowStepSizes:p.lastWindows; 
                    obj.epochs_index_metadata = 'single-exam';
                    obj.epochs_method = method;
                    obj.nWindows = numel(obj.epochs)-1;
                    obj.K_stages = obj.nWindows;

                 case 'single_fixedSize_fromSizeType' % chesnaye single
                    obj.epochs = ceil(linspace(p.startWindows,p.lastWindows,p.K_stages));
                    obj.epochs_index_metadata = 'single-exam';
                    obj.epochs_method = method;
                    obj.nWindows = p.K_stages;
                    obj.K_stages = p.K_stages;

                 otherwise
                     error('This sizeType-lastWindowCalcMethod method pair was not implemented yet.')
             end
             
        end

        function obj = compute_msc(obj, p)
            arguments
                obj % The ORDCalculator class

                % p: additional parameters, passed as Name-Value arguments
                %    declared below, including their default values.
                p.dataloader = obj.dataloader;
                p.channels = obj.dataloader.channels;
                p.nChannels = numel(obj.dataloader.channels);

                p.startWindow = obj.startWindow;
                p.windowStepSize = obj.windowStepSize;
                p.lastWindow = obj.lastWindow;
                
                p.epochs = obj.epochs;

                % epochCalcMethod define how
                p.epochCalcMethod  {mustBeMember(p.epochCalcMethod,...
                                    {'zanoteli','chesnaye'})}  = 'zanoteli'
                p.nWindows = obj.nWindows;
                p.K_stages = obj.K_stages;
                                
            end

            if isempty(p.epochs)
                switch p.epochCalcMethod
                    case 'zanoteli'
                        % obj.epochs = p.startWindow:p.windowStepSize:p.lastWindow;
                        % p.nWindows = numel(obj.epochs)-1;
                        % p.K_stages = p.nWindows;
                        p.lastWindowCalcMethod = 'maxFromStart';
                        p.sizeType = 'fixedSize';
    
                    case 'chesnaye'    
                        % obj.epochs = ceil(linspace(p.startWindow,p.lastWindow,p.K_stages));
                        % p.nWindows = p.K_stages;
                        p.lastWindowCalcMethod = 'fromSizeType';
                        p.sizeType = 'fixedSize';
                end

                obj = obj.fit_epochs( ...
                    lastWindowCalcMethod=p.lastWindowCalcMethod, ...
                    sizeType = p.sizeType, ...
                    startWindows = p.startWindow, ...
                    windowStepSizes = p.windowStepSize,...
                    lastWindows = p.lastWindow,...
                    K_stages = p.K_stages...
                    );
                
                p.nWindows = obj.nWindows;
                p.K_stages = obj.K_stages;
                p.epochs = obj.epochs;

            end
            
            % IS THIS THE CORRECT/BEST PRACTICE ACCESS??
            Y  = p.dataloader.SIGNALS;           
            obj.MSC = zeros([p.dataloader.nBins, p.nWindows, p.nChannels]);

            for channel = p.channels
                for epoch_index = 1:p.nWindows-1
                    epochStart = p.epochs(epoch_index);

                    epochEnd = p.epochs(epoch_index+1)-1;
                    this_windowStepSize = epochEnd - epochStart+1;

                    current_epoch = squeeze(Y(:,epochStart:epochEnd,channel));
                    
                    obj = obj.zanotelli_msc_fft(current_epoch, this_windowStepSize);
                    obj.MSC(:, epoch_index, channel) = obj.latestMSC;
                end
            end

        end


        % STILL UNDER DEVELOPMENT
        function obj = bulk_compute_msc(obj,p)
            arguments
                obj % The ORDCalculator class

                % p: additional parameters, passed as Name-Value arguments
                %    declared below, including their default values.
                p.dataloader = obj.dataloader;
                p.channels = obj.dataloader.channels;
                p.nChannels = numel(obj.dataloader.channels);

                % parameters for exam reload (should this be here?)
                p.subjectIndices = obj.subjectIndices;
                p.stimulusIndices = obj.stimulusIndices; 
                p.epochs = obj.epochs;
                                
            end
            
            mode = obj.dataloader.mode;
            if matches(mode(1:3),"sim", IgnoreCase=true)
                % obj.parameterizedMSC
                obj.groupMSC = cell(size(p.epochs));
                obj.channels = p.channels;
                for noiseMean_idx = 1:numel(obj.dataloader.groupNoiseMean)
                    for noiseVar_idx = 1:numel(obj.dataloader.groupNoiseStd)
                        selected_epochs = p.epochs(noiseMean_idx, noiseVar_idx,:);
                        
                        for params_idx = 1:numel(selected_epochs)
                            if ~isempty(cell2mat(selected_epochs(params_idx)))
                
                            current_epoch = cell2mat(selected_epochs(params_idx));

                            epoch_idx = sub2ind(size(obj.epochs), ...
                                noiseMean_idx, noiseVar_idx, params_idx );

                            p.nWindows = cell2mat(obj.nWindows(epoch_idx));

                            p.dataloader.SIGNALS = cell2mat(p.dataloader.groupSIGNALS(noiseMean_idx,noiseVar_idx));
                            p.dataloader.nBins = size(p.dataloader.SIGNALS, 1);

                            obj = obj.compute_msc( ...
                                    dataloader = p.dataloader,...    
                                    channels = p.channels,...
                                    nChannels = p.nChannels, ...
                                    nWindows = p.nWindows, ...
                                    epochs = current_epoch ...
                                    );

                            obj.groupMSC{epoch_idx} = obj.MSC;


                            end 
                        end
                    end
                end

            elseif matches(mode(1:3),"exp", IgnoreCase=true)
                for stimulus_index = p.stimulusIndices
    
                    % obj.parameterizedMSC
                    obj.groupMSC = cell(size(p.epochs));
                    obj.channels = p.channels;
                   for stimulusIndex = p.stimulusIndices
                        for subjectIndex = p.subjectIndices
                            selected_epochs = p.epochs(stimulusIndex, subjectIndex,:);
        
                            for params_idx = 1:numel(selected_epochs)
                                if ~isempty(cell2mat(selected_epochs(params_idx)))
        
                                    current_epoch = cell2mat(selected_epochs(params_idx));
        
                                    epoch_idx = sub2ind(size(obj.epochs), ...
                                        stimulusIndex, subjectIndex, params_idx );
        
                                    p.nWindows = cell2mat(obj.nWindows(epoch_idx));
        
                                    p.dataloader.SIGNALS = cell2mat(p.dataloader.groupSIGNALS(stimulusIndex,subjectIndex));
                                    p.dataloader.nBins = size(p.dataloader.SIGNALS, 1);
        
                                    obj = obj.compute_msc( ...
                                            dataloader = p.dataloader,...    
                                            channels = p.channels,...
                                            nChannels = p.nChannels, ...
                                            nWindows = p.nWindows, ...
                                            epochs = current_epoch ...
                                            );
        
                                    obj.groupMSC{epoch_idx} = obj.MSC;
        
        
                                end 
                            end
                        end
                   end
                end
            end

        end

        function obj = compute_msc_on_all_channels(obj,varargin)
            if nargin>1
                obj.windowStepSize = varargin{1};
            end

            Y  = obj.dataloader.SIGNALS;            

            % Full MSC matrix is nBins x M x nChannels
            obj.nWindows = floor((size(Y,2)-1)/obj.windowStepSize);
            obj.epochs = obj.startWindow:obj.windowStepSize:obj.lastWindow;
            obj.MSC = zeros([obj.dataloader.nBins, obj.nWindows, ...
                                    obj.dataloader.nChannels]);
            
            for channel = obj.dataloader.channels
                for window_index = 1:numel(obj.epochs)-1
                    epochStart = obj.epochs(window_index);
                    epochEnd = obj.epochs(window_index+1)-1;                     
                
                    current_epoch = squeeze(Y(:,epochStart:epochEnd,channel));
    
                    obj = obj.zanotelli_msc_fft(current_epoch, obj.windowStepSize);
                    obj.MSC(:, window_index, channel) = obj.latestMSC;
                end
            end
        end
                

        function obj = zanotelli_msc_fft(obj,Y,M)
            if (size(Y,2) ~= M)
                error('Tamanho da janela diferente'); 
            end

            % y = y(1:tamanho_janela*M,1); 
            % %dividir em janela; 
            % y =reshape(y,tamanho_janela,M); 
            % 
            % %aplicar a fft; 
            % Y =fft(reshape(y,tamanho_janela,M)); %
            
            %MSC
            obj.latestMSC =  abs(sum(Y,2)).^2./(M*sum(abs(Y).^2,2));
        end


        function age(obj)
            fprintf( ...
                '\n\t [%s] This ORDCalculator was built %0.2f seconds ago.\n\n', ...
                datetime, round(toc(obj.timer),2))
        end

    end
    
end