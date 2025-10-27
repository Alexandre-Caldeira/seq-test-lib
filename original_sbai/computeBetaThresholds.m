function [aThresholds, gThresholds] = computeBetaThresholds(K, M, params) 
% computeBetaThresholds calculates sequential detection and futility thresholds. 
    % aThresholds = zeros(1, K); 
    % gThresholds = zeros(1, K);
    % % Evenly distribute alpha and gamma for each stage
    % Alpha_k = ones(1, K) * (alpha / K);
    % Gamma_k = ((1 - alpha) / K) * ones(1, K);
    % 
    % resolution = 1e5;
    % Xvalues = linspace(0, 1, resolution+1);
    % Null = betapdf(Xvalues, 1, M-1);
    % Null = Null / sum(Null);
    % Chi2_Norm = Null;
    % 
    % % Stage 1 thresholds (closed-form)
    % aThresholds(1) = 1 - Alpha_k(1)^(1/(M-1));
    % gThresholds(1) = 1 - (1 - Gamma_k(1))^(1/(M-1));
    % 
    % TruncInd_Ra = round(aThresholds(1) * resolution);
    % TruncInd_Rg = round(gThresholds(1) * resolution);
    % 
    % for k = 2:K
    %     NullTrunc = Null;
    %     NullTrunc(TruncInd_Ra:end) = 0;
    %     NullTrunc(1:TruncInd_Rg) = 0;
    %     Null2 = conv(Chi2_Norm, NullTrunc);
    %     Null2 = Null2 / sum(Null2);
    %     cumsumNull2 = cumsum(Null2);
    %     idx_a = find(cumsumNull2 >= (1 - Alpha_k(k)), 1, 'first');
    %     idx_g = find(cumsumNull2 >= Gamma_k(k), 1, 'first');
    %     aThresholds(k) = idx_a / resolution;
    %     gThresholds(k) = idx_g / resolution;
    %     TruncInd_Ra = idx_a;
    %     TruncInd_Rg = idx_g;
    %     Null = Null2;



TotalAlpha      = params.alpha;                        
Alpha_k         = ones(1,K)*(TotalAlpha/K);     
Gamma_k         = ((1-TotalAlpha)/K).*ones(1,K);


aThresholds  = zeros(1,K);
gThresholds  = zeros(1,K);

Resolution      = (1/0.0001);                  
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
    TruncInd_Ra                 = findIndex(Null2, sum(Null2) - Alpha_k(k));            
    aThresholds(k)              = TruncInd_Ra/Resolution;                                    
    TruncInd_Rg                 = findIndex(Null2, Gamma_k(k), 1);
    gThresholds(k)              = TruncInd_Rg/Resolution;
    Null                        = Null2;
end

end
