%Generic function for multivariate detectors
 %input 
 %s:  (string) multiariavel detector
 %y: signal 
 %parameters
 %  parameters.fs -> sampling frequency
 %  parameters.tj -> stretch size (number of window points)
 %x: sinal noise (para o teste global)
 
 %Detector = {'MSC', 'CSM', 'LFT', 'MMSC', 'aMSC', 'pMSC', 'MCSM', 'aCSM', 'pCSM', 'MLFT', 'aLFT',...
%Exemplo 
% parameters.M=50;parameters.tj=32;
% x = randn(parameters.M*parameters.tj,1); 
% value = mord('aMSC',x,parameters) 
 
 function [value, F] = mord(s,y,parameters,x) 


  % Detectores univariados ---------------------------------------
if strcmp(s,'MSC') s ='aMSC';  parameters.N=1; end
if strcmp(s,'CSM') s ='aCSM';  parameters.N=1; end
if strcmp(s,'LFT') s ='aLFT';  parameters.N=1; end
 

%% MORD 
if strcmp(s,'aMSC') == 1 
   
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = aMSC(y,tj,fs,[]);
   else 
     [value] = aMSC(y,tj);  
   end
end


if strcmp(s,'pMSC') == 1    
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = pMSC(y,tj,fs,[]);
   else 
     [value] = pMSC(y,tj);  
   end
end

if strcmp(s,'MMSC') == 1    
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = MMSC(y,tj,fs,[]);
   else 
     [value] = MMSC(y,tj);  
   end
end


if strcmp(s,'aCSM') == 1 
   
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = aCSM(y,tj,fs,[]);
   else 
     [value] = aCSM(y,tj);  
   end
end

if strcmp(s,'pCSM') == 1 
   
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = pCSM(y,tj,fs,[]);
   else 
     [value] = pCSM(y,tj);  
   end
end


if strcmp(s,'MCSM') == 1 
   
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = MCSM(y,tj,fs,[]);
   else 
     [value] = MCSM(y,tj);  
   end
end

if strcmp(s,'aLFT') == 1 
   
    L = parameters.L;
    fs = parameters.fs;  
    fo = parameters.fo;   
    value = aLFT(y,L,fs,[],fo);
    % for i=1:length(fo)
    %     [value(i)] = aLFT(y,L,fs,[],fo(i));
    % end
    F = fo; 
end

if strcmp(s,'pLFT') == 1 
   
    L = parameters.L;
    fs = parameters.fs;  
    fo = parameters.fo;         
    [value] = pLFT(y,L,fs,[],fo);
    F = fo; 
end


if strcmp(s,'MLFT') == 1 
   
    L = parameters.L;
    fs = parameters.fs;  
    fo = parameters.fo;         
    [value] = MLFT(y,L,fs,[],fo);
    F = fo; 
end

if strcmp(s,'aGBT') == 1    
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = aGBT(y,x,tj,fs,[]);
   else 
     [value] = aGBT(y,x,tj);  
   end
end


if strcmp(s,'pGBT') == 1    
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = pGBT(y,x,tj,fs,[]);
   else 
     [value] = pGBT(y,x,tj);  
   end
end

if strcmp(s,'MGBT') == 1    
    tj = parameters.tj; 
   if nargout>1
    fs = parameters.fs;    
    [value,F] = MGBT(y,x,tj,fs,[]);
   else 
     [value] = MGBT(y,x,tj);  
   end
end


%% função dipolos 
if strcmp(s(1),'d') == 1   
   
    %gerar o dipolo
    xd = dipolos(y);
    if nargout>1
        [value, F] = mord(s(2:end),xd,parameters);
    else 
       [value] = mord(s(2:end),xd,parameters); 
    end
end


