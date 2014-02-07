function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% graphics_toolkit ("gnuplot")

% create categories per y
positive_index = find(y==1);
negative_index = find(y==0);

% plot graph per categories
plot( X(positive_index,1),X(positive_index,2),'k+','LineWidth',2,'MarkerSize',5);
plot( X(negative_index,1),X(negative_index,2),'ko','LineWidth',2,'MarkerSize',5);




% =========================================================================



hold off;

end
