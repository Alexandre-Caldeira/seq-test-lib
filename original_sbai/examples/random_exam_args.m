function [random_stimulus,random_subject,random_electrode,random_epoch] = random_exam_args(dataloader)
%random_exam_args automate random args generation
%   Helper function to reduce boilerplate code

    random_stimulus_fun = @() randi([1, ...
        numel(dataloader.zanoteliStimulusNames)-1],1,1);

    random_stimulus = random_stimulus_fun();

    random_epoch_fun = @() randi([1, ...
        dataloader.zanoteliSuggestedMMax(random_stimulus)], ...
        1,1);

    random_epoch = random_epoch_fun();

    random_subject_fun = @() randi([1, 11],1,1);

    random_electrode_fun = @() randi([1, ...
        numel(dataloader.channels)],1,1);
    
    random_subject = random_subject_fun();
    random_electrode = random_electrode_fun();
    

end