function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculate hypothesis matrix per X and theta
hypothesis = sigmoid(X*theta);

% calculate cost(before adding penalty) of all samples
sum = 0;
for i=1:m
	sum += -y(i,1)*log(hypothesis(i,1))-(1-y(i,1))*log((1-hypothesis(i,1)));
end
before_penal = sum / m;

% calculate penalty factor
sum = 0;
for i=2:n
	sum += theta(i)^2;
end
penal_factor = lambda/(2*m)*sum;

% calculate cost
J = before_penal + penal_factor;

% calculate gradient
for j=1:n
	sum = 0;
	for i=1:m
		sum += (hypothesis(i,1)-y(i,1))*X(i,j);
	end
	grad(j,1) = 1/m*sum;
	if j != 1 
		grad(j,1) += lambda/m*theta(j);
	endif
end




% =============================================================

end
