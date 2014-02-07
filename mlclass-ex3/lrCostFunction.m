function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% to ease the computation, generate partial of result
hypotheses = sigmoid(X*theta);

% to ease the computation, generate partial of result
result_in_sum = - y.*log(hypotheses) - (1- y).*log(1 - hypotheses);

% compute J
J = sum(result_in_sum) / m;

% to ease the computation, generate partial of result
part1 = hypotheses - y;
part2 = X' * part1;

% set grad
grad = part2/m;

% regularize J
temp = theta(2:end,:); % theta_0 should not be penalized
penal_to_j = lambda/(2*m)*(temp'*temp);
J = J + penal_to_j;

% regularize grad
penal_to_grad = lambda/m*theta;

% theta_0 should not be penalized
grad(2:end,:) = grad(2:end,:) .+ penal_to_grad(2:end,:);

% below is alternative
% penal_to_grad(1) = 0;
% grad = grad .+ penal_to_grad;


% =============================================================

grad = grad(:);

end
