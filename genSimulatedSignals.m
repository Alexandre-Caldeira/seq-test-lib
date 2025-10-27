function [signals, params] = genSimulatedSignals(params) 
% genSimulatedSignals generates simulated EEG signals (noise + sinusoidal signal). 
    duration = params.duration; % seconds 
    nChannels = params.nChannels;
    totalSamples = params.Fs * duration * nChannels; 
    
    t = (0:totalSamples-1) / params.Fs;
    
    % % White noise generation
    % noise = randn(1, totalSamples);
    % 
    % % Sinusoidal component (e.g., 80 Hz stimulation)
    % freqStim = 80;
    % signal = sin(2*pi*freqStim*t);
    % 
    % % Adjust amplitude based on SNR (convert dB to linear scale)
    % snr_linear = 10^(params.sim.signalSNR/20);
    % signal = signal * snr_linear;
    % 
    % % Combine the noise and signal
    % signals = noise + signal;
    % 
    % % For multi-channel simulation, replicate the signal (e.g., 2 channels)
    % % signals = repmat(signals, nChannels, 1);
    % signals = reshape(signals, [nChannels, params.Fs, duration]); 


    signals = 0;
    for i = 1:numel(params.signalFrequencies)
        fo = params.signalFrequencies(i); 
        signals = signals+sin(2*pi*fo*t);
    end
   
    signals = reshape(signals, [nChannels, params.Fs, duration]);    
    for janela = 1:duration
        for channel = 1:nChannels
            % Adiciona ruido gaussiano branco com SNR aleatoria 
            noise_mean = params.sim.signalSNR; % very high = -25

            % Alguns exemplos:
            % noise_var_mean = 2^2;
            % noise_var = @() (noise_var_mean+ sqrt(noise_var_var)*randn(1));
            % noise_sd = @() randi([1,45],1)/10;

            noise_sd = @() 1;
            snr = @() noise_mean+noise_sd()*randn(1);
            signals(channel,:,janela) = awgn(signals(channel,:,janela),snr(),'measured','db')';
        end
    end
end