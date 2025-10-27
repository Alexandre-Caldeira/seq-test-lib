function fftSignals = computeFFT(signals, params) 
% computeFFT computes the FFT along each window (row-wise) and 
% returns only the positive frequencies. 
    fftSignals = zeros(params.nChannels, params.nBins, params.duration );
    for channel = 1:params.nChannels
        temp = fft(squeeze(signals(channel,:,:)),params.nfft, 1); 
        fftSignals(channel,:,:) = temp(1:params.nBins,:); 
    end
end
