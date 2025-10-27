clearvars;close all;clc;

K = 5;
alpha           = 0.05;
Alpha_k         = ones(1,K)*(alpha/K);
Gamma_k         = ((1-alpha)/K).*ones(1,K);


% Example usage:
K = 5;
patterns = {'linear', 'exponential', 'logarithmic', 'polynomial'};
figure(1);
for i = 1:length(patterns)
    seq = generate_sequence(K, patterns{i},alpha);
    subplot(2,2,i);
    bar(seq);
    title(sprintf('%s Growth', patterns{i}));
    xlabel('Position');
    ylabel('Value');
end

figure(2)
subplot(211)
stem(Alpha_k)
title('Alpha_k')
grid on

subplot(212)
stem(Gamma_k)
title('Gamma_k')
grid on