function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% find out good C and sigma from cross validation set
candidate_value = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
size_value = size(candidate_value,1);

gap_min = -1;

for c_index = 1:size_value
	for sigma_index = 1:size_value
		C_temp = candidate_value(c_index);
		sigma_temp = candidate_value(sigma_index);

		% get the model first from training set
		model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
		
		% run prediction on cross validation for C and sigma
		predictions = svmPredict(model,Xval);
		gap = mean(double(predictions ~= yval));
		
		% check if we need to optimize C and sigma
		if gap_min == -1
			gap_min = gap;
		end;
		if gap_min >= gap
			gap_min = gap;
			C = C_temp;
			sigma = sigma_temp;
		end;		
	end;
end;






% =========================================================================

end
