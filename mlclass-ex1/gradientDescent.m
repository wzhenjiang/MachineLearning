function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters	
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


	sum_num1 = 0;
	sum_num2 = 0;
	for x_row=1:m
		sum_num1 = sum_num1 + (X(x_row,:)*theta(:,1)-y(x_row))*X(x_row,1);
		sum_num2 = sum_num2 + (X(x_row,:)*theta(:,1)-y(x_row))*X(x_row,2);
	end;
	theta(1) = theta(1) - alpha/m*sum_num1;
	theta(2) = theta(2) - alpha/m*sum_num2;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end;

end
