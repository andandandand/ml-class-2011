function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

coef = (-1/m);

accum=0;


for i= 1: m

   accum +=  y(i)*log (sigmoid(X(i,:)*theta)) + (1-y(i)) * log (1-sigmoid(X(i,:)*theta));


end

J = accum * coef;

extra = (lambda/(2*m));

accum2=0;

for i=2 : size(grad)

    accum2 += theta(i)*theta(i);
	
end

extra = extra * accum2;

J += extra;

coef= (1/m);

htheta = X * theta; % it's the same as theta transpose * X 

temp = lambda .* theta;
temp(1) = 0;

grad = coef * ( X' * (sigmoid (htheta) -y) + (temp));

grad
       
        

% =============================================================

end
