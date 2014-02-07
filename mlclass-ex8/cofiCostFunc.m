function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

prediction = X * Theta';
diff = prediction .- Y;
dist = diff .* R;
dist2 = dist .* dist;
part1 = lambda / 2 * sum(sum(Theta.*Theta));
part2 = lambda / 2 * sum(sum(X.*X));
J = 1 / 2 * sum(sum(dist2)) + part1 + part2;

for m_i = 1: num_movies
	X_grad(m_i,:) = dist(m_i,:) * Theta + lambda * X(m_i,:);
%	for feature_k = 1: num_features
%		X_grad(m_i,feature_k) = dist(m_i,:) * Theta(:,feature_k) + lambda*X(m_i,feature_k);
%		sum_result = 0;
%		for user_n = 1: num_users
%			dist_movie_user = dist(m_i,user_n);
%			sum_result += dist_movie_user * Theta(user_n,feature_k) ;
%		end
%		X_grad(m_i,feature_k) = sum_result + lambda*X(m_i,feature_k);
%	end
end

for user_n = 1: num_users
	Theta_grad(user_n,:) = dist(:,user_n)' * X + lambda * Theta(user_n,:);
%	for feature_k = 1: num_features
%		Theta_grad(user_n,feature_k) = dist(:,user_n)' * X(:,feature_k) + lambda * Theta(user_n,feature_k);
%		sum_result = 0;
%		for m_i = 1: num_movies
%			dist_movie_user = dist(m_i,user_n);
%			sum_result += dist_movie_user * X(m_i,feature_k) ;
%		end
%		Theta_grad(user_n,feature_k) = sum_result + lambda * Theta(user_n,feature_k);
%	end
end






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
