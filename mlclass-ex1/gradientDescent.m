function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

coef = alpha*(1/m);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    hTheta =  X * theta; 

    hThetaMinusY = hTheta - y;

    summation0 = sum(hThetaMinusY);

    product0 = coef * summation0;
    
    summation1=0;
	
	for index =1:m
		
		summation1 = summation1 + (hThetaMinusY(index) * X(index,2)); 
		
	end
	
	product1 = coef*summation1;

    temp0 = theta(1)-product0; %theta(1) is theta0
	
    temp1= theta(2)-product1;  %theta(2) is theta1      
	
    theta(1) = temp0;
	theta(2) = temp1;	
		 
   
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

	%if (iter<150)
	%printf("iteration %d has theta0 is %f and theta1 is %f, and cost %f\n",iter, temp0, temp1, J_history(iter));
end

end
