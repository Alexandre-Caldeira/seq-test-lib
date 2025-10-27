classdef DataLoader
% IN: experimental .mat data path or MC sim parameters
% out: Y (freq. data for testing)

    properties
       
        % y (timeseries data, MC sim or exp aquisition)
        signals % Stores timeseries of the latest loaded exam
        SIGNALS % Stores frequency spectrum of the latest loaded exam
        
        % Stores timeseries from all selected subjects and stimuli
        groupSignals % stimulus-subject-indexed cell:
                     % groupSignals{stimulus,subject} -> y(sample,window,channel)

        % Stores frequency spectrum  from all selected subjects and stimuli
        groupSIGNALS % stimulus-subject-indexed cell: 
                     % groupSIGNALS{stimulus,subject} -> Y(sample,window,channel)

        signalFrequencies = [82    84    86    88    90    92    94    96];
        noiseFrequencies = [];
        
        % Processing Parameters
        channels = 1:16;  % Index for EEG leads (channels) in data
        fs       = 1000;  % Sampling frequency (Hz)
        nfft              % Number of FFT points per epoch (1 sec window in NIASv1)
        nBins             % Number of 'positive frequency' values on the spectrum
        nChannels         % length of channels vector
        duration = [60];  % exam duration [secs], per zanoteliStimulusIndex intensity
        totalSamples

        % For MC Simulation        
        noiseMean = -10
        noiseStd = 1
        groupNoiseMean = [0 -15 -30]
        groupNoiseStd  = [1   1   1]
        SNRfun

        % For ASSR EEG Data
        zanoteliSubjectIndex = 1;
        zanoteliStimulusIndex = 1;

        selectedZanoteliSubjects
        selectedZanoteliStimuli

        zanoteliSuggestedMMax = [50; 50; 240; 440; 440; 20];
        zanoteliStimulusNames = {'70dB'; '60dB'; '50dB';
                                 '40dB'; '30dB';'ESP'};
        zanoteliSubjects  = {'Ab'; 'An'; 'Bb'; 'Er'; 'Lu';...
                             'So'; 'Qu'; 'Vi'; 'Sa'; 'Ti';'Wr'};
        zanoteliLeads = {'FC'; 'F4'; 'T6'; 'P4'; 'T4'; 'Oz'; 'C4'; 'T5';...
                         'P3'; 'F7'; 'F3'; 'T3'; 'C3'; 'Fz'; 'Pz'; 'Cz'}

        % Utils
        timer   % obj timer
        id      % obj id
        
        inPath  = 'C:\PPGEE\SBEB_CBA_24\CGST_figuras\Sinais_EEG\' % where y comes from
        outPath % where Y is saved
        mode    % simulation mode 'sim' or 'exp'
        

    end
    
    methods
        function obj = DataLoader(mode,varargin)
            obj.timer = tic;
            obj.id = [num2str(keyHash(keyHash(mode)+obj.timer))];
            obj.outPath = [pwd,'\'];
            
            if nargin == 2
                obj.inPath = varargin{1};
            elseif nargin == 3
                obj.inPath = varargin{1};
                obj.outPath = varargin{2};
            end
    
            obj.nfft = obj.fs; 
            obj.nBins = floor(obj.fs/2)+1; % ...

            % Randomly selects noise frequencies (above 300 Hz) 
            % Wihouth replacement
            obj.noiseFrequencies = 300+randperm(200,numel(obj.signalFrequencies));
            
            % With replacement
            % obj.noiseFrequencies = randi([300 500],1, ...
            %                         numel(obj.signalFrequencies));

            % By default, SNR is:
            % Adiciona ruido gaussiano branco com SNR aleatoria 
            obj.SNRfun = @() obj.noiseMean+obj.noiseStd()*randn(1);

            obj.mode = mode;
            if matches(obj.mode(1:3),"sim", IgnoreCase=true)
                
                % Run MC Sim
                obj = obj.genSimulatedSignals();
                obj = obj.computeFFT();

            elseif matches(obj.mode(1:3),"exp", IgnoreCase=true)
                % Try to load exp data for validation

                %carregar o voluntário
                zanoteliSubjectIndex = obj.zanoteliSubjectIndex; 
                
                %intensidadde 
                zanoteliStimulusNames = cell2mat(obj.zanoteliStimulusNames(obj.zanoteliStimulusIndex));
                
                obj = obj.loadEEGData(zanoteliSubjectIndex, zanoteliStimulusNames);

            else
                % Throw error
                error('DataLoader mode input is invalid.'); 
            end

        end

        function obj = loadEEGData(obj, zanoteliSubjectIndex, zanoteliStimulusNames)
            % Check if obj is properly constructed
            if isempty(obj.inPath) 
                error('EEG data file path not specified.'); 
            end 
            
            subject_id = cell2mat(obj.zanoteliSubjects(zanoteliSubjectIndex));

            % Build filepath and load
            % data = load(filepath); 
            data = load([obj.inPath subject_id zanoteliStimulusNames], ...
                'x','Fs','binsM','freqEstim');
            
            if isfield(data, 'x') 
                obj.signals = data.x; 
                obj.totalSamples = size(obj.signals,2)*size(obj.signals,1);
            else 
                error('The EEG data file must contain variable "x".'); 
            end 

            if isfield(data, 'Fs') 
                obj.fs = data.Fs;
                obj.nfft = obj.fs;
                obj.nBins = floor(obj.fs/2)+1;
            else 
                error('The EEG data file must contain variable "x".'); 
            end

        end

        function obj = loadBulkEEGData(obj)
            obj.groupSignals = cell(numel(obj.selectedZanoteliStimuli), ...
                                    numel(obj.selectedZanoteliSubjects));
            
            for stimulusIndex = obj.selectedZanoteliStimuli
                current_stim = cell2mat(obj.zanoteliStimulusNames(stimulusIndex));

                for subjectIndex = obj.selectedZanoteliSubjects
                    obj = obj.loadEEGData(subjectIndex, current_stim);
                    obj.groupSignals{stimulusIndex,subjectIndex} = obj.signals;
                end
            end
        end        

        function obj = genSimulatedSignals(obj,p)
            arguments
                obj
                p.duration = obj.duration
            end
            % genSimulatedSignals generates simulated EEG signals (noise + sinusoidal signal). 
            % Example test usage:
            % dtl = DataLoader('sim');
            % stem(abs(dtl.SIGNALS(:,1,1)))
            % hold on
            % stem(dtl.signalFrequencies,abs(dtl.SIGNALS(dtl.signalFrequencies,1,1)),...
            % 'MarkerFaceColor','red',...
            % 'MarkerEdgeColor','red')

            obj.nChannels = numel(obj.channels);
            obj.totalSamples = obj.fs * p.duration * obj.nChannels; 
            t = (0:obj.totalSamples-1) / obj.fs;     
        
            obj.signals = 0;
            for i = 1:numel(obj.signalFrequencies)
                fo = obj.signalFrequencies(i)-1; 
                % fo subtracted by one, such that FFT bin for frequency 
                % will match the frequency because MATLAB index starts at 1

                obj.signals = obj.signals+sin(2*pi*fo*t);
            end
           
            obj.signals = reshape(obj.signals, [obj.fs, p.duration, obj.nChannels]);  
            for channel = 1:obj.nChannels
                for epoch = 1:p.duration

                    x = awgn( ... % add white gaussian noise
                            obj.signals(:,epoch,channel), ... % to this section of signal
                            obj.SNRfun()-rand()*sqrt(i),'measured','db')'; % with this SNR in measured dB

                    % Scale to uV
                    obj.signals(:,epoch,channel)  = (10^-6) * x;  
                end
            end

        end

        function obj = genBulkSim(obj,p)
            arguments
                obj

                p.groupNoiseMean = obj.groupNoiseMean
                p.groupNoiseStd  = obj.groupNoiseStd

            end

            obj.groupSignals = cell(numel(p.groupNoiseMean), numel(p.groupNoiseStd));
            
            for noiseMean_idx = 1:numel(p.groupNoiseMean)
                for noiseVar_idx = 1:numel(p.groupNoiseStd)
                    current_noiseMean = p.groupNoiseMean(noiseMean_idx);
                    current_noiseVar = p.groupNoiseStd(noiseVar_idx);

                    obj = obj.resetSNRfun(current_noiseMean, current_noiseVar);
                    obj = obj.genSimulatedSignals();

                    obj.groupSignals{noiseMean_idx,noiseVar_idx} = obj.signals;

                end
            end
            obj.groupNoiseMean = p.groupNoiseMean;
            obj.groupNoiseStd = p.groupNoiseStd;
        end

        function obj = computeFFT(obj) 
            % computeFFT computes the FFT along each window (row-wise) and 
            % returns only the positive frequencies. 
            
            obj.duration = size(obj.signals,2);
            obj.nChannels= numel(obj.channels);
            obj.SIGNALS = zeros(obj.nBins, obj.duration, obj.nChannels);

            for channel = obj.channels
                for epoch = 1:obj.duration
                    temp = fft(squeeze(obj.signals(:,epoch,channel)),obj.nfft, 1); 
                    obj.SIGNALS(:,epoch,channel) = temp(1:obj.nBins); 
                end
            end

        end

        function obj = computeBulkFFTs(obj,p)
            arguments
                obj
                
                p.mode = 'exp';
            end

            
            if matches(p.mode(1:3),"sim", IgnoreCase=true)
                for noiseMean_idx = 1:numel(obj.groupNoiseMean)
                    for noiseVar_idx = 1:numel(obj.groupNoiseStd)

                        obj.signals = cell2mat(obj.groupSignals(noiseMean_idx,noiseVar_idx));
    
                        obj = obj.computeFFT();
                        obj.groupSIGNALS{noiseMean_idx,noiseVar_idx} = obj.SIGNALS;
    
                    end
                end

            elseif matches(obj.mode(1:3),"exp", IgnoreCase=true)
                obj.groupSIGNALS = cell(numel(obj.selectedZanoteliStimuli), ...
                                    numel(obj.selectedZanoteliSubjects));

                for stimulusIndex = obj.selectedZanoteliStimuli
                    for subjectIndex = obj.selectedZanoteliSubjects
    
                        % Reset signal and parameters
                        obj.signals = cell2mat(obj.groupSignals(stimulusIndex,subjectIndex));
                        obj.fs = size(obj.signals, 1);
                        obj.nfft = obj.fs;
                        obj.nBins = floor(obj.fs/2)+1;
                        obj.zanoteliStimulusIndex = stimulusIndex;
                        obj.zanoteliSubjectIndex = subjectIndex;
    
                        obj = obj.computeFFT();
                        obj.groupSIGNALS{stimulusIndex,subjectIndex} = obj.SIGNALS;
                    end
                end

            else
                % Throw error
                error('DataLoader mode input is invalid.'); 
            end
            
        end

        function obj =resetSNRfun(obj,noiseMean,noiseStd)
            % Alguns exemplos:
            % noise_var_mean = 2^2;
            % noise_var = @() (noise_var_mean+ sqrt(noise_var_var)*randn(1));
            % noise_sd = @() randi([1,45],1)/10;

            % Test case:
            % dtl = DataLoader('sim');
            % vec = zeros(1,1e5);
            % for i = 1:numel(vec)
            % vec(i) = dtl.SNRfun();
            % end
            % histogram(vec)

            obj.noiseMean = noiseMean;
            obj.noiseStd = noiseStd;
            obj.SNRfun = @() obj.noiseMean+obj.noiseStd()*randn(1);

        end

        function obj = resetExam(obj, subjectIndex,stimulusIndex)
            obj.zanoteliSubjectIndex = subjectIndex;
            obj.zanoteliStimulusIndex = stimulusIndex;
            obj = obj.loadEEGData(obj.zanoteliSubjectIndex, ...
                         cell2mat(obj.zanoteliStimulusNames(obj.zanoteliStimulusIndex)));
            obj= obj.computeFFT();

        end

        function obj = resetSubject(obj, subjectIndex)
            % TODO: add inspect exam and recompute as optional
            obj.zanoteliSubjectIndex = subjectIndex;
            obj = obj.loadEEGData(obj.zanoteliSubjectIndex, ...
                         cell2mat(obj.zanoteliStimulusNames(obj.zanoteliStimulusIndex)));
            obj= obj.computeFFT();

        end

        function obj = resetStimulus(obj,stimulusIndex)
            obj.zanoteliStimulusIndex = stimulusIndex;
            obj = obj.loadEEGData(obj.zanoteliSubjectIndex, ...
                         cell2mat(obj.zanoteliStimulusNames(obj.zanoteliStimulusIndex)));
            obj = obj.computeFFT();

        end

        function obj = resetChannels(obj,channelsIndices)
            obj.channels = channelsIndices;
            obj.nChannels = numel(channelsIndices);
            obj = obj.computeFFT();

        end

        function obj = resetDuration(obj,newDuration)

            if matches(obj.mode(1:3),"exp", IgnoreCase=true) ...
                    && obj.zanoteliSuggestedMMax(obj.zanoteliStimulusIndex) < newDuration 
                warning(['Desired duration is larger than Zanoteli suggestion.',... 
                    'This might impact on some subjects lacking data on later epochs.', ...
                    'Consider lowering exam duration.'])
            end

            obj.duration = newDuration;

            if newDuration > size(obj.signals,2) 
                if matches(obj.mode(1:3),"exp", IgnoreCase=true)
                    error('Duration is largest than signals'' size (%s).',size(obj.signals))
                else
                    fprintf('\t [%s] Sim duration reset to %d.\n\n',datetime(),newDuration)
                    obj = obj.genSimulatedSignals();
                end

            else
                obj.signals = obj.signals(:,1:obj.duration,:);
                
            end            
            
            obj = obj.computeFFT();
            
        end

        % Utils:
        function age(obj)
            fprintf('\n\t [%s] This DataLoader was built %0.2f seconds ago.\n\n', ...
                datetime, round(toc(obj.timer),2))
        end

        function inspectExam(obj)
            fprintf(...
            '\n\tExam is %s stimulus on subject %s,\n\t measuring on %s for %s seconds.\n\n', ...
                cell2mat(obj.zanoteliStimulusNames(obj.zanoteliStimulusIndex)), ...
                cell2mat(obj.zanoteliSubjects(obj.zanoteliSubjectIndex,:)),...
                cell2mat(obj.zanoteliLeads(1)),...
                num2str(size(obj.signals, 2))...
                )
        end

    end
        
end
