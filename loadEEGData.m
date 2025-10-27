function signals = loadEEGData(filepath) 
% loadEEGData loads EEG data from the specified file. 
% The file should contain a variable named 'x' representing the EEG data. 

    if isempty(filepath) 
        error('EEG data file path not specified.'); 
    end 
    data = load(filepath); 
    
    if isfield(data, 'x') 
        signals = data.x; 
    else 
        error('The EEG data file must contain variable "x".'); 
    end 
end
