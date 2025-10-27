function z= my_polinomial_fun(x,optional)
% my_polinomial_fun computes a*x^b = x^2 by default
% Examples: 
%   my_polinomial_fun(2)
%   my_polinomial_fun(2, b=3)
%   my_polinomial_fun(2, a=5)
%   my_polinomial_fun(2, a=2, b=3)
%   my_polinomial_fun(2, a=15, b=1)
    arguments 
        x 
        optional.b = 2
        optional.a = 1
    end

    z = optional.a*x^optional.b;
end