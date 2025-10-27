function procSignals = preprocessSignal(signals, params) 
% preprocessSignal removes the DC offset and applies a Butterworth bandpass filter. 
    procSignals = signals - mean(signals, 2); 
    [b, a] = butter(params.filter.order, ... 
        [params.filter.fcLower, params.filter.fcUpper] / (params.Fs/2)); 
    procSignals = filtfilt(b, a, procSignals); 
end
