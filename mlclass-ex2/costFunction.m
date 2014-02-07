function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of theta

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% calculate hypothesis matrix per X and theta
hypothesis = sigmoid(X*theta);

% calculate cost of all samples
for i=1:m
	J += -y(i,1)*log(hypothesis(i,1))-(1-y(i,1))*log((1-hypothesis(i,1)));
end;
J = J / m;

% calculate gradient
for j=1:n
	sum = 0;
	for i=1:m
		sum += (hypothesis(i,1)-y(i,1))*X(i,j);
	end;
	grad(j,1) = 1/m*sum;
end;

% =============================================================

end
