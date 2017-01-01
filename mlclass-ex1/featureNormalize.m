function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%
      
 col1mean=mean(X(:,1));
 col2mean=mean(X(:,2));

  
 mu(1)=col1mean;
 mu(2)=col2mean;
 
 
 col1std=std(X(:,1));
 col2std=std(X(:,2));


 sigma(1)= col1std;
 sigma(2)= col2std;
 
 m = length(X);
 
 for index = 1 :m
 
	X_norm(index,1) = (X(index,1) - col1mean)/col1std;
	X_norm(index,2) = (X(index,2) - col2mean)/col2std;
	
	
 end




% ============================================================

end
