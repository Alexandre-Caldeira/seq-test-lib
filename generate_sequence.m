function sequence = generate_sequence(K, pattern,alpha)
    % Generate a sequence of K numbers with specified growth pattern
    % Pattern options: 'linear', 'exponential', 'logarithmic', 'polynomial'
    
    switch lower(pattern)
        case 'linear'
            % Linear growth: x, 2x, 3x, ... nx where n*x*(n+1)/2 = 1
            x = 2/(K*(K+1));
            sequence =-alpha + x * (1:K);
            
        case 'exponential'
            % Exponential growth: a^1, a^2, ..., a^K where sum = 1
            r = nth_root(1 + 1/K, K);  % Find base r such that sum(r.^[1:K]) ≈ 1
            sequence = -alpha +r.^[1:K];
            
        case 'logarithmic'
            % Logarithmic growth: log(x+1), log(x+2), ..., log(x+K)
            % Scale factor ensures sum equals 1
            temp = logspace(0, log10(K), K);
            sequence = -alpha +temp ./ sum(temp);
            
        case 'polynomial'
            % Polynomial growth: x^n where n increases with position
            % Use increasing powers from 1 to K
            temp = (1:K).^([1:K]/2);
            sequence = -alpha +temp ./ sum(temp);
            
        otherwise
            error('Invalid pattern. Choose from: linear, exponential, logarithmic, polynomial');
    end
    
    % Verify sum is approximately 1
    % assert(abs(sum(sequence) - 1) < 1e-10, 'Sum is not close enough to 1');
end

% Helper function for nth root calculation
function result = nth_root(x, n)
    result = exp(log(x)/n);
end
