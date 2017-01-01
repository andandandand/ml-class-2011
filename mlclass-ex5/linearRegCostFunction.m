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

h = X * theta; % 12 x 9 matrix

HMinusYSquared = (h-y).^2;

sumHMinusYSquared = sum(HMinusYSquared);

coef = 1/(2*m);

auxTheta = theta;
auxTheta(1)=0;

squaredThetasSum = sum(auxTheta.^2);

coefLambda = lambda/(2*m);

J = (coef * sumHMinusYSquared) + (coefLambda*squaredThetasSum);


%%% gradient

%grad = (1/m) .* sum(h'*X);

coef= (1/m);

htheta = X * theta; % it's the same as theta transpose * X 

temp = lambda .* theta;
temp(1) = 0;

grad = coef * ( X' * (htheta -y) + (temp));




% =========================================================================

grad = grad(:);

end
