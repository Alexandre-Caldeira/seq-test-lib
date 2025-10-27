classdef ORDTester 
    properties
        % Test Parameters     
        desired_alpha
        corrigir_alpha
        stageAlphas 
        stageGammas 
        previous_cgst_thresholds
        TP
        TN
        FP
        FN

        groupTP
        groupTN
        groupFP
        groupFN

        lastExam

        
        groupStageAlphas % stageAlphas{stimulus,subject,epoch} -> stageAlphas(K)
        groupStageGammas % groupStageGammas{stimulus,subject,epoch} -> stageGammas(K)

        signalFrequencies = [82    84    86    88    90    92    94    96];
        noiseFrequencies = [];
        allTestFrequencies
        nTestFrequencies
        noiseMask
        noiseFlag

        % Outputs
        groupDecisions
        decisions
        time

        % Exam/ORD data
        epochs = []
        epochs_index_metadata
        epochs_method
        K_stages    % total number of tests to be applied on the ORD
        nWindows    % number of Windows (may match K, or not)

        % Esses já existem no ORDCalculator: 
        % (deveria estar aqui?)
        subjectIndices = [1:11];
        stimulusIndices = [1:5];
        startWindows = [1];
        windowStepSizes = [24 32];
        lastWindows = [50];
        

        % (bad practice, fix later)
        dataloader
        ord_calculator
        
        %debugging features
        latest_stim
        latest_subj
        latest_epoch_idx
        latest_windowSize
        latest_K

        % Utils
        timer   % obj timer
        id      % obj id
    end


    methods
        function obj = ORDTester(ord_calculator,p)
            arguments
                ord_calculator

                % Optional inputs
                p.dataloader = [];
                p.K_stages = [];
                p.desired_alpha = 0.05;              

            end

            obj.timer = tic;
            obj.id = [num2str(keyHash(obj.timer))];
            obj.ord_calculator = ord_calculator;
            obj.desired_alpha = p.desired_alpha;

            if isempty(p.K_stages)
                p.K_stages = ord_calculator.K_stages;
            end

            if isa(p.dataloader, 'double')
                obj.dataloader = obj.ord_calculator.dataloader;
            else
                obj.dataloader = p.dataloader;
            end

            % obj.groupMSC = obj.ord_calculator.groupMSC;

            obj.signalFrequencies = obj.dataloader.signalFrequencies;
            obj.noiseFrequencies = obj.dataloader.noiseFrequencies;
            
            obj.allTestFrequencies = [obj.signalFrequencies, sort(obj.noiseFrequencies)];
            obj.nTestFrequencies = numel(obj.allTestFrequencies);
            obj.noiseFlag = numel(obj.signalFrequencies)+1; 
            obj.noiseMask = 1:obj.nTestFrequencies > obj.noiseFlag;

            obj.epochs = obj.ord_calculator.epochs;
            obj.epochs_index_metadata = obj.ord_calculator.epochs_index_metadata;
            obj.epochs_method = obj.ord_calculator.epochs_method;
            obj.K_stages    = obj.ord_calculator.K_stages;
            obj.nWindows    = obj.ord_calculator.nWindows; 

            obj.stimulusIndices = obj.dataloader.selectedZanoteliStimuli;
            obj.subjectIndices = obj.dataloader.selectedZanoteliSubjects;
            
            obj.previous_cgst_thresholds = cell(obj.dataloader.duration+1, ...
                max(cell2mat(obj.ord_calculator.K_stages(:)))+1);
        end

        function obj = compute_beta_cgst_thresholds(obj)
            warning('Test pending')

            obj.groupStageAlphas = cell(size(obj.epochs));
            obj.groupStageGammas = cell(size(obj.epochs));

            for stimulusIndex = obj.stimulusIndices
                for subjectIndex = obj.subjectIndices
                    selected_epochs = obj.epochs(stimulusIndex, subjectIndex,:);

                    for params_idx = 1:numel(selected_epochs)
                        if ~isempty(cell2mat(selected_epochs(params_idx)))

                            epoch_idx = sub2ind(size(obj.epochs), ...
                                stimulusIndex, subjectIndex, params_idx );

                            p.nWindows = cell2mat(obj.nWindows(epoch_idx));
                            p.K_stages = cell2mat(obj.K_stages(epoch_idx));
                            
                            M = p.nWindows;
                            K = p.K_stages;
                            
                            Mmax = K*M;
                            Mstep = p.nWindows;
                            Mmin = min(current_epoch,[],'all');
                            
                            % [alfa_corrigido,~,~,~] =funcao_NDC_alfaCorrigido_Mmax(1e4,Mmax,obj.desired_alpha,obj.desired_alpha)
                            % obj = obj.zanotelli_adjust_FP( ...
                            %     Mmax = Mmax,...
                            %     Mstep = Mstep, ...
                            %     Mmin = Mmin);

                            obj = obj.single_exam_beta_cgst_threshold(M,K, ...
                                corrigir_alpha = obj.corrigir_alpha ,...
                                Mmax = Mmax,...
                                Mstep = Mstep, ...
                                Mmin = Mmin);
                            % obj = obj.patient_beta_cgst_threshold(M,K);
                            
                            obj.groupStageAlphas{epoch_idx} = obj.stageAlphas;
                            obj.groupStageGammas{epoch_idx} = obj.stageGammas;

                        end
                    end
                end
            end
        end

        
        
        function obj = single_exam_beta_cgst_threshold(obj,M,K,p)
            arguments
                obj
                M
                K

                p.corrigir_alpha = obj.corrigir_alpha
                p.Mmin
                p.Mstep
                p.Mmax
                
            end
            
    
            if M<=size(obj.previous_cgst_thresholds,1) ...
                    && K<=size(obj.previous_cgst_thresholds,2)

                check_previous = cell2mat(obj.previous_cgst_thresholds(M,K));
                if ~isempty(check_previous)
                    obj.stageAlphas = check_previous(1,:);
                    obj.stageGammas = check_previous(2,:);
                    return
                end
            end

            if p.corrigir_alpha ~= 0
                obj = obj.zanotelli_adjust_FP( Mmax = p.Mmax,...
                                Mstep = p.Mstep, Mmin = p.Mmin);
            end
            
            alpha           = obj.desired_alpha;                        

            Alpha_k         = ones(1,K)*(alpha/K);    
            Gamma_k         = ((1-alpha)/K).*ones(1,K);
            Resolution      = (1/0.00001); %(1/0.0001);                 
            Xvalues         = 0:1/Resolution:1;            
            Null         	= betapdf(Xvalues, 1, M-1);
            Null            = Null/sum(Null);            
            Chi2_Norm       = Null/sum(Null);             
         
            k               = 1;                           
            aThresholds(k)	= 1 - Alpha_k(k).^(1./(M-1));  
            gThresholds(k)	= 1-(1- Gamma_k(k)).^(1./(M-1));
            TruncInd_Ra      = round(aThresholds(k)*Resolution);
            TruncInd_Rg      = round(gThresholds(k)*Resolution);           
            
            for k = 2:K
                NullTrunc                   = Null;                                                
                NullTrunc(TruncInd_Ra:end)  = zeros(1, length(NullTrunc(TruncInd_Ra:end)));    
                NullTrunc(1:TruncInd_Rg)    = zeros(1, length(NullTrunc(1:TruncInd_Rg)));
                
                Null2                       = conv(Chi2_Norm, NullTrunc);   
                Null2                       = Null2 / (sum(Null2) / (1 - sum(Gamma_k(1:(k-1))) - sum(Alpha_k(1:(k-1)))));
        
                TruncInd_Ra                 = ORDTester.findIndex(Null2, sum(Null2) - Alpha_k(k)); 
                aThresholds(k)              = TruncInd_Ra/Resolution;  
                TruncInd_Rg                 = ORDTester.findIndex(Null2, Gamma_k(k), 1);
                gThresholds(k)              = TruncInd_Rg/Resolution;
                Null                        = Null2; 
            end   


            obj.stageAlphas = aThresholds;
            obj.stageGammas = gThresholds;
            obj.previous_cgst_thresholds{M,K} = [aThresholds;gThresholds];
        end

        function obj = patient_beta_cgst_threshold(obj,M,K)
            
            alpha           = obj.desired_alpha;                        

            Alpha_k         = ones(1,K)*(alpha/K);    
            % Gamma_k         = ((1-alpha)/K).*ones(1,K);
            % x = 2/(K*(K+1));
            % sequence =-alpha + x * (1:K);
            % Gamma_k = flip(sequence);
            % Gamma_k = [(1-alpha)*0.80/(K-3)*ones(1,K-3) ((1-alpha)*0.20/3)*ones(1,3)];
            % Gamma_k = [(1-alpha)*0.99/(K-3)*ones(1,K-3) ((1-alpha)*0.01/3)*ones(1,3)];

            n=3;
            pct = 0.70;
            Gamma_k = [(1-alpha)*pct/(K-n)*ones(1,K-n) ((1-alpha)*(1-pct)/3)*ones(1,n)];            

            Resolution      = (1/0.00001); %(1/0.0001);                 
            Xvalues         = 0:1/Resolution:1;            
            Null         	= betapdf(Xvalues, 1, M-1);
            Null            = Null/sum(Null);            
            Chi2_Norm       = Null/sum(Null);             
         
            k               = 1;                           
            aThresholds(k)	= 1 - Alpha_k(k).^(1./(M-1));  
            gThresholds(k)	= 1-(1- Gamma_k(k)).^(1./(M-1));
            TruncInd_Ra      = round(aThresholds(k)*Resolution);
            TruncInd_Rg      = round(gThresholds(k)*Resolution);           
            
            for k = 2:K
                NullTrunc                   = Null;                                                
                NullTrunc(TruncInd_Ra:end)  = zeros(1, length(NullTrunc(TruncInd_Ra:end)));    
                NullTrunc(1:TruncInd_Rg)    = zeros(1, length(NullTrunc(1:TruncInd_Rg)));
                
                Null2                       = conv(Chi2_Norm, NullTrunc);   
                Null2                       = Null2 / (sum(Null2) / (1 - sum(Gamma_k(1:(k-1))) - sum(Alpha_k(1:(k-1)))));
        
                TruncInd_Ra                 = ORDTester.findIndex(Null2, sum(Null2) - Alpha_k(k)); 
                aThresholds(k)              = TruncInd_Ra/Resolution;  
                TruncInd_Rg                 = ORDTester.findIndex(Null2, Gamma_k(k), 1);
                gThresholds(k)              = TruncInd_Rg/Resolution;
                Null                        = Null2; 
            end   


            obj.stageAlphas = aThresholds;
            obj.stageGammas = gThresholds;
        end


        function [p,obj]  = validateDetectionThresholds(obj, p)
            arguments
                obj

                p.dataloader
                p.K_stages = 5
                p.SNRmean = 5
                p.SNRvar = 0.01
                p.duration = 50
            end


            p.dataloader = DataLoader('sim').resetSNRfun(p.SNRmean, p.SNRvar);
            p.dataloader = p.dataloader.genSimulatedSignals(duration = p.duration);

            p.ord_calculator = ORDCalculator(p.dataloader.computeFFT());
            p.ord_calculator = p.ord_calculator.compute_msc( ...
                K_stages= p.K_stages, ...
                epochCalcMethod='chesnaye', ...
                startWindow=1);

            p.epochs = p.ord_calculator.epochs;
            obj.decisions = zeros(size(p.ord_calculator.MSC));
            p.allTestFrequencies = [p.dataloader.signalFrequencies, ...
                sort(p.dataloader.noiseFrequencies)];

            M = p.epochs(end)-p.epochs(end-1)+1;
            obj = obj.single_exam_beta_cgst_threshold(M, p.K_stages);
            obj = obj.compute_beta_cgst_decisions( ...
                allTestFrequencies=p.allTestFrequencies, ...
                ord_calculator=p.ord_calculator);
        end
    
        function obj = compute_bulk_beta_cgst_decisions(obj,p)
            arguments
                obj % The ORDCalculator class

                % p: additional parameters, passed as Name-Value arguments
                %    declared below, including their default values.
                p.dataloader = obj.dataloader;
                p.channels = obj.dataloader.channels;
                p.nChannels = numel(obj.dataloader.channels);

                % parameters for exam reload (should this be here?)
                p.subjectIndices = obj.ord_calculator.subjectIndices;
                p.stimulusIndices = obj.ord_calculator.stimulusIndices; 
                p.epochs = obj.ord_calculator.epochs;
                p.corrigir_alpha = 0;
                                
            end
            % warning('Test pending')
            
           obj.corrigir_alpha = p.corrigir_alpha;
            % obj.parameterizedMSC
           obj.groupDecisions = cell(size(p.epochs));
           obj.groupTP = cell(size(p.epochs));
           obj.groupTN = cell(size(p.epochs));
           obj.groupFP = cell(size(p.epochs));
           obj.groupFN = cell(size(p.epochs));
          
           for stimulusIndex = p.stimulusIndices
               obj.latest_stim = stimulusIndex;
                            
                for subjectIndex = p.subjectIndices

                    obj.latest_subj = subjectIndex;

                    selected_epochs = p.epochs(stimulusIndex, subjectIndex,:);

                    for params_idx = 1:numel(selected_epochs)
                        if ~isempty(cell2mat(selected_epochs(params_idx)))

                            current_epoch = cell2mat(selected_epochs(params_idx));

                            epoch_idx = sub2ind(size(obj.epochs), ...
                                stimulusIndex, subjectIndex, params_idx );

                            
                            obj.latest_epoch_idx =  epoch_idx;      

                            p.nWindows = cell2mat(obj.nWindows(epoch_idx));
                            
                            p.K_stages = cell2mat(obj.K_stages(epoch_idx));

                            % if p.K_stages > 5
                            %     warning('large n of convs')
                            % end
                            pct = fix(100*params_idx/numel(selected_epochs));
                            if rem(pct,20)==0
                                fprintf('\n \t [%s] %d%% of current stim/subj (%d / %d).\n', ...
                                    datetime, pct, obj.latest_stim, obj.latest_subj)
                                                              
                            end

                            obj.ord_calculator.MSC = cell2mat( ...
                                obj.ord_calculator.groupMSC(stimulusIndex,subjectIndex,params_idx));
      
                            M = current_epoch(end) - current_epoch(end-1)+1;
                            K = p.K_stages;

                            % Mmax = size(obj.ord_calculator.MSC,2);
                            % Mstep = p.nWindows;
                            % Mmin = min(current_epoch,[],'all');
                            % 
                            % % [alfa_corrigido,~,~,~] =funcao_NDC_alfaCorrigido_Mmax(1e4,Mmax,obj.desired_alpha,obj.desired_alpha)
                            % obj = obj.zanotelli_adjust_FP( ...
                            %     Mmax = Mmax,...
                            %     Mstep = Mstep, ...
                            %     Mmin = Mmin);

                            obj.latest_windowSize = M;
                            obj.latest_K = K;

                            % obj = obj.single_exam_beta_cgst_threshold(M,K);
                            % obj = obj.patient_beta_cgst_threshold(M,K);
                            % t = tic();
                            Mmax = current_epoch(end);
                            Mstep = p.nWindows;
                            Mmin = min(current_epoch,[],'all');
                            

                            obj = obj.single_exam_beta_cgst_threshold(M,K, ...
                                corrigir_alpha = obj.corrigir_alpha ,...
                                Mmax = Mmax,...
                                Mstep = Mstep, ...
                                Mmin = Mmin);
                            % if K>10
                            %     disp(toc(t));
                            % end


                            obj = obj.compute_beta_cgst_decisions();

                            obj.groupDecisions{epoch_idx} = obj.decisions;
                            obj.groupTP{epoch_idx} = obj.TP;
                            obj.groupTN{epoch_idx} = obj.TN;
                            obj.groupFP{epoch_idx} = obj.FP;
                            obj.groupFN{epoch_idx} = obj.FN;

                        end 
                    end
                end
           end
        end

        function obj = zanotelli_adjust_FP(obj,p)
            arguments
                obj

                p.Mmin =1;
                p.Mstep = 2;
                p.nRuns  = 1000;
                p.Mmax = 20; %número máximo de janela
                p.alfa_teste = 0.05;
                p.FP_desejado =0.05;
            end
            
            %parâmetros defaul
            % fs = 64;
            tj = 32; %cada janela um segundo
            bin = 8;
            
            
            Ntotal = p.Mmax*tj; %número de pontos totais
            
            %Na simulação iremos estimar a aplicação do detector a cada janela
            ord = zeros(p.nRuns,p.Mmax); %armazena os valores dos detectores a cada experimento.
            
            for ii = 1: p.nRuns
                x = randn(Ntotal,1);
                x = reshape(x,tj,p.Mmax); %dividir em janelas
                %aplicar o detector a cada janela ------------------
                xfft = fft(x); %aplico uma ´única vez a FFT.
                for M = 2:p.Mmax %fazer para cada acrescimo de uma janela
                    ord(ii,M) = msc_fft(xfft(bin,1:M),M);
                end
            end

            Ninicial=1; 
            [NDC,~]  = estimarNDC(Ninicial,p.alfa_teste,p.FP_desejado, ord, p.Mmin,p.Mstep, p.Mmax);
            % NDC_minimo = NDC;
            % NDC_minimo = 1;
            
            %ajustar os valores crítico
            MM = p.Mmin:p.Mstep:p.Mmax;
            options = optimset('MaxIter', 50);
            cc = @(alfa) funcao_custo_v2(alfa, NDC, MM, ord, p.FP_desejado);
            [alfa, ~] = fmincg(cc,p.alfa_teste, options);
            obj.desired_alpha = alfa;

        end
        %     epoch = linspace()
        %     Mmax = obj.
        %     P = parametros_protocolo(Mmax);
        %     alfa_corrigido = nan*ones(size(P,1),1);
        %     cost_alfa = nan*ones(size(P,1),1);
        % 
        %     for ii = 1:size(P,1)
        %         Mmin = P(ii,1);
        %         Mstep = P(ii,2); 
        %         Mmax = P(ii,3);
        %         MM = Mmin:Mstep:Mmax;
        %         disp([num2str(ii*100/size(P,1)),'%'])
        % 
        %         det = ord(:,MM);
        %         alfa = 0.05;  %TAXA DE FALSO POSITIVO DE CADA TESTES
        %         options = optimset('MaxIter', 50);
        %         cc = @(alfa) funcao_custo(alfa ,MM, det, FP_desejado);                               
        %         [alfa, cost] = fmincg(cc,alfa, options);
        %         alfa_corrigido(ii) = alfa; 
        % 
        %         if ~isempty(cost)
        %             cost_alfa(ii) = cost(end);
        %         end
        %     end
        % end


        function obj = compute_bulk_sim_beta_cgst_decisions(obj,p)
           arguments
               obj % The ORDCalculator class

               % p: additional parameters, passed as Name-Value arguments
               %    declared below, including their default values.
               p.dataloader = obj.dataloader;
               p.channels = obj.dataloader.channels;
               p.nChannels = numel(obj.dataloader.channels);

               % parameters for exam reload (should this be here?)
               p.subjectIndices = obj.ord_calculator.subjectIndices;
               p.stimulusIndices = obj.ord_calculator.stimulusIndices; 
               p.epochs = obj.ord_calculator.epochs;

               p.nWindows = obj.ord_calculator.nWindows;
               p.K_stages = obj.ord_calculator.K_stages;
                                
           end
           % warning('Test pending')
            
           obj.epochs = p.epochs;
           obj.groupDecisions = cell(size(p.epochs));
           obj.groupTP = cell(size(obj.groupDecisions));
           obj.groupTN = cell(size(obj.groupDecisions));
           obj.groupFP = cell(size(obj.groupDecisions));
           obj.groupFN = cell(size(obj.groupDecisions));

           for noiseMean_idx = 1:numel(obj.dataloader.groupNoiseMean)
                for noiseVar_idx = 1:numel(obj.dataloader.groupNoiseStd)
                    selected_epochs = p.epochs(noiseMean_idx, noiseVar_idx,:);
                    
                    for params_idx = 1:numel(selected_epochs)
                        if ~isempty(cell2mat(selected_epochs(params_idx)))
            
                            current_epoch = cell2mat(selected_epochs(params_idx));
        
                            epoch_idx = sub2ind(size(p.epochs), ...
                                noiseMean_idx, noiseVar_idx, params_idx );
    
                            obj.ord_calculator.MSC = cell2mat( ...
                                obj.ord_calculator.groupMSC(noiseMean_idx,noiseVar_idx,params_idx));
      
                            M = current_epoch(end) - current_epoch(end-1)+1;
                            K = cell2mat(p.K_stages(epoch_idx));
        
                            % t = tic()
                            obj = obj.single_exam_beta_cgst_threshold(M,K);
                            % if K>10
                            %     fprintf('K = %d, took:\n',K)
                            %     disp(toc(t));
                            % end
                            obj = obj.compute_beta_cgst_decisions();

                            obj.groupTP{epoch_idx} = obj.TP;
                            obj.groupTN{epoch_idx} = obj.TN;
                            obj.groupFP{epoch_idx} = obj.FP;
                            obj.groupFN{epoch_idx} = obj.FN;

                        end 
                    end
                end
           end
        end

        function obj = compute_zanoteli_ndc_decisions()
        end

        function obj = compute_bazoni_ndc_decisions()
        end

        function obj = compute_antunes_ndc_decisions()
        end
        
        function obj = compute_beta_cgst_decisions(obj, p)
            arguments
                obj

                p.channels = obj.dataloader.channels
                p.allTestFrequencies = obj.allTestFrequencies
                p.ord_calculator = obj.ord_calculator
                p.noiseFlag = obj.noiseFlag
            end

            current_tests = p.ord_calculator.MSC;
            % current_decisions = zeros(size(current_tests));
            obj.lastExam = current_tests;%zeros(size(current_tests));

            % TP = zeros(Kmax,numel(freq_bins));

            obj.TP = zeros(size(obj.lastExam));
            obj.FP = zeros(size(obj.lastExam));
            obj.TN = zeros(size(obj.lastExam));
            obj.FN = zeros(size(obj.lastExam));
            
            % TN = zeros(Kmax,numel(freq_bins));
            % FN = zeros(Kmax,numel(freq_bins));

            for channel=p.channels
                for freq = p.allTestFrequencies
                    % obj.lastExam(freq,:,channel) = current_tests(freq,:,channel);
                    for k = 1:size(current_tests,2)
                        
        
                        % SIGNAL
                        if freq < p.allTestFrequencies(p.noiseFlag) && ...              % not noise
                            sum(obj.lastExam(freq,1:k,channel)) > obj.stageAlphas(k)    % detected
                
                            % obj.decisions(freq,k,channel) = obj.decisions(freq,k,channel) +1;
                            obj.TP(freq,k,channel) = obj.TP(freq,k,channel)+1;
                            % t_decisao(k,freq) = ~sum(t_decisao(:,freq),'all');
                    
                        elseif freq < p.allTestFrequencies(p.noiseFlag) && ...           % not noise
                                sum(obj.lastExam(freq,1:k,channel)) <= obj.stageGammas(k)% gave up 
                
                            % obj.decisions(freq,k,channel)  = obj.decisions(freq,k,channel) -1;
                            obj.FN(freq,k,channel) = obj.FN(freq,k,channel)+1;
                            % t_decisao(k,freq) = -1*(~sum(t_decisao(:,freq),'all'));
                
                        % NOISE
                        elseif freq >= p.allTestFrequencies(p.noiseFlag) && ...          % is noise
                                sum(obj.lastExam(freq,1:k,channel)) > obj.stageAlphas(k) % detected
                
                            obj.FP(freq,k,channel) = obj.FP(freq,k,channel)+1;
                            % t_decisao(k,freq) = ~sum(t_decisao(:,freq),'all');
                            
                        elseif freq >= p.allTestFrequencies(p.noiseFlag) && ...           % is noise
                                sum(obj.lastExam(freq,1:k,channel)) <= obj.stageGammas(k) % gave up
                
                            % obj.decisions(freq,k,channel)  = obj.decisions(freq,k,channel) +1;
                            obj.TN(freq,k,channel) = obj.TN(freq,k,channel)+1;
                            % t_decisao(k,freq) = -1*(~sum(t_decisao(:,freq),'all'));

                        else
                            if k==size(current_tests,2)
                                warning('neither detection nor stop')
                            end
                            
                        end
                
                    end
                end
            end

        end

        % function obj = compute_chesnaye_thresholds(obj)
        %     error('Implementation pending')
        % end
        % 
        % function obj = compute_zanoteli_thresholds(obj)
        %     error('Implementation pending')
        % end

        % UTILS
        function age(obj)
            fprintf( ...
                '\t [%s] This ORDTester was built %0.2f seconds ago.\n\n', ...
                datetime, round(toc(obj.timer),2))
        end


    end

    methods(Static)
        function Ind = findIndex(PDF1, Goal, GetFutil)         % GetFutil = 1 if futility
            L               = length(PDF1);
            increment       = round(0.05*L);
            Pos             = round(L/2);
            S               = sum(PDF1(1:Pos));
            increasing      = true;
            if S>Goal
                increasing  = false;
            end
            if abs(sum(PDF1(1:1))) >= Goal
                increment   = 0;
                Pos         = 0;
            end
            if sum(PDF1) <= Goal
                increment   = 0;
                if GetFutil
                    Pos	= 0;        % futility threshold
                else
                    Pos = L;        % efficacy threshold
                end
            end
            while increment>1
                if increasing
                    Pos = Pos+increment;
                else
                    Pos = Pos-increment;
                end
                S = sum(PDF1(1:Pos));
                if increasing
                    if S > Goal
                        increment = floor(increment/2);
                        increasing = false;
                    end
                else
                    if S < Goal
                        increment = floor(increment/2);
                        increasing = true;
                    end
                end
            end
            Ind = Pos;
        end
    end
end