function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% a1 = X
a1 = X;
a1_withzero = [ones(m)(:,1), X];	% Add x0

% calculate a2
z2 = a1_withzero * Theta1';
a2 = sigmoid(z2);
a2_withzero = [ones(m)(:,1), a2];

% calculate a3
z3 = a2_withzero * Theta2';
a3 = sigmoid(z3);

% convert y from vector into matrix
y_matrix = zeros(m,num_labels);
for i=1:m
	y_matrix(i,y(i)) = 1;
end

in_brackets = - y_matrix.*log(a3) - (1 - y_matrix) .* log(1 - a3);

J = sum(sum(in_brackets))/m;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

deriv2 = zeros(num_labels,hidden_layer_size+1);
deriv1 = zeros(hidden_layer_size,input_layer_size+1);

for t = 1:m
	
	delta3 = a3(t,:) - y_matrix(t,:);
	delta3 = delta3';
	
	Theta2_woZero = Theta2(:,2:end);
	delta2 = (Theta2_woZero'*delta3).*sigmoidGradient(z2(t,:)');	% make z2 only the t row and vector

%	delta2 = delta2(2:end);
	
	deriv2 = deriv2 + delta3*a2_withzero(t,:); % if a2 is vector, it has to trans, now, the t row is already trans result
	deriv1 = deriv1 + delta2*a1_withzero(t,:);
	
end

Theta2_grad = deriv2/m;
Theta1_grad = deriv1/m;


% fprintf('Size of deriv1: %d * %d \n', size(deriv1,1),size(deriv1,2));
% fprintf('Size of deriv2: %d * %d \n', size(deriv2,1),size(deriv2,2));
% fprintf('Size of grad: %d * %d \n', size(grad,1),size(grad,2));

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% remove theta0 from Theta1
Theta1_short = Theta1(:,2:end);

% remove Theta0 from Theta2
Theta2_short = Theta2(:,2:end);

%calculate penalty
sum_penal_theta1 = sum(sum(Theta1_short.*Theta1_short));
sum_penal_theta2 = sum(sum(Theta2_short.*Theta2_short));
penalty = lambda/(2*m)*(sum_penal_theta1+sum_penal_theta2);
J = J + penalty;

Theta1_zeroone = [zeros(hidden_layer_size)(:,1), Theta1_short];
Theta2_zeroone = [zeros(num_labels)(:,1), Theta2_short];
Theta1_grad = Theta1_grad + lambda/m*Theta1_zeroone;
Theta2_grad = Theta2_grad + lambda/m*Theta2_zeroone;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
