function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  
  n=size(X,1);
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%

for i = 1:m
    hyp = hypothesis(theta, X(:,i));
    f = f - y(i)*log(hyp) - (1-y(i))*log(1-hyp);
    g = g + X(:,i).*(ones(n,1)*(hyp-y(i)));
end

end

function h = hypothesis(theta, x)
    h = 1./(1 + exp(-theta'*x));
end