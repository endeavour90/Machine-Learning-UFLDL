function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
    incomplete_cond_prob = bsxfun(@rdivide, exp(theta'*X), 1+sum(exp(theta'*X)));
    cond_prob = [incomplete_cond_prob; 1 - sum(incomplete_cond_prob)];
    log_cond_prob = log(cond_prob);
    index = sub2ind(size(log_cond_prob), y, 1:size(log_cond_prob,2));
    f = -sum(log_cond_prob(index));
    
    for k = 1 : size(theta,2)
        g(:,k) = -sum(bsxfun(@times, X, (y==k) - cond_prob(k,:)),2);
    end
    
    g=g(:); % make gradient a vector for minFunc
  
end


