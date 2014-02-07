function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculate what to be summed first
error = X * theta - y;

% calculate first half of cost
cost = 1 / (2 * m) * sum(error.*error);

% calculate penalty for cost. theta starts from 1 instead of 0
theta_penalty = theta(2:end,:);
penalty = lambda / (2 * m) * sum(theta_penalty .* theta_penalty);

% calculate cost with upper result
J = cost + penalty;

% calculate what to be summed with upper error result
deriv = 1 / m * (X' * error);

% calculate penalty for deriv. theta starts from 1 instead of 0
deriv_penalty = lambda / m * theta;
deriv_penalty(1) = 0;

% calculate grad with upper result
grad = deriv + deriv_penalty;

% =========================================================================

grad = grad(:);

end
