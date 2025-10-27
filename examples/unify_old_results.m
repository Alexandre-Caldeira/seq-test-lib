%% Clean workspace and setup path
clearvars;closeall;clc;

addpath('C:\Users\alexa\Desktop\Numero_Deteccoes_consecutiva_H')

%% Load each method and save new table

nChannels = 1;
thisEegChannel = 'FC'


mmin = zeros(numel(nonemptyparams_idxs)*nChannels,1);
mstep = zeros(numel(nonemptyparams_idxs)*nChannels,1);
minwindowsize = zeros(numel(nonemptyparams_idxs)*nChannels,1);
mmax = zeros(numel(nonemptyparams_idxs)*nChannels,1);
ntests = zeros(numel(nonemptyparams_idxs)*nChannels,1);
eeg_channel = cell(numel(nonemptyparams_idxs)*nChannels,1);
stim_lvl = cell(numel(nonemptyparams_idxs)*nChannels,1);
method = cell(numel(nonemptyparams_idxs)*nChannels,1);
table_fp = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_tp = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_fn = zeros(numel(nonemptyparams_idxs)*nChannels,1);
table_tn = zeros(numel(nonemptyparams_idxs)*nChannels,1);


nome_metodo = {'zanoteli2020','bazoni2021','antunes2019','vaz2023'};

for metodo = 1:4
    % caminho1 = ['timeM_50db_M',num2str(metodo),'.mat'];
    % caminho2 = ['pareto_50db_M',num2str(metodo), '.mat'];
    caminho1 = ['timeM_30db_M',num2str(metodo),'.mat'];
    caminho2 = ['pareto_30db_M',num2str(metodo), '.mat'];
%     
    load(caminho1,'timeM')
    load(caminho2,'TXD','Mmax','parametros');

    K = numel(this_epoch);

end
