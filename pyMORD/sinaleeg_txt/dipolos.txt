function y = dipolos(x)

%função que gera os dipolos;
N = size(x,2); 
CC = [[[1:N]', zeros(N,1)];combnk(1:N,2)]; %todos os dipolos 
% indDipolo = [1:size(CC,1)]'; 
Ndipolo = size(CC,1); 

y = zeros(size(x,1),Ndipolo);
for nd = 1: Ndipolo    %para cada dipolo          
             
    if CC(nd,2) == 0 
        y(:,nd) = x(:,CC(nd,1)); 
    else
         y(:,nd)= x(:,CC(nd,1))-x(:,CC(nd,2)); 
    end
end



