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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





%%%%%%%%%%%%%%% PART 1: COST FUNCTION WITHOUT REGULARIZATION%%%%%%%%%%%%%%%%%%


a1 = [[ones(m,1)] X]; %add bias unit

z2 = Theta1 * a1';

a2 = sigmoid(z2); % g(Z2)

a2 = [[ones(m,1)] a2']; %5000 x 26 

%%%##### this ^m may not work on another network

z3 = Theta2 * a2';

a3 = sigmoid(z3)'; %5000 x10 OUTPUT 

%yy=eye(num_labels)(y,:);

for index=1: num_labels
	yy(:,index) = (y==index);
end 

%% yy is 5000 x 10 

delta3 = yy-a3; 


%delta2 = ( (Theta2(:,[2:end])' * delta3') * (sigmoidGradient(z2)') );

delta2 = (Theta2' * delta3) .* (sigmoidGradient(z2));


whos




J = (1/m) * sum(sum((-yy .* log(a3)) - ((1 - yy) .* log(1 - a3))));

%%%%%%%%%%%%%% END PART 1: COST FUNCTION WITHOUT REGULARIZATION %%%%%%%%%%%%%%%



%%%%%%%%%%%%%%% PART 2: COST FUNCTION **WITH** REGULARIZATION%%%%%%%%%%%%%%%%%%

squareTheta1 = Theta1.^2;
squareTheta2 = Theta2.^2;


J+=(lambda/(2*m)) * [(sum(sum(squareTheta1(:,2:end)))) + (sum(sum(squareTheta2(:,2:end))))];



%%%%%%%%%%%%%%% END PART 2: COST FUNCTION ***WITH** REGULARIZATION%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%% PART 3: GRADIENT COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Hi, I did the exercise with a vectorized solution, I did not use for loops... so my matrices have different sizes but I will put %%them here, maybe it can help you

%X = 5000x400 Theta1 = 25x401 Theta2 = 10x26

%a1 = 5000x401 z2 = 5000x25 a2 = 5000x26 z3 = 5000x10 a3 = 5000x10

%y1 = 5000x10

%delta3 = 5000x10 delta2 = 5000x25

%I think that your problem is that you are not eliminating the first column of delta3*Theta2 (remember that you do not have to %consider the bias element, I think that this is the name)... when I removed this first column I got a matrix that can be %%multiplied by g'(z2).

%D1 = 401x25 D2 = 26x10

%Theta_grad1 = 25x401 Theta_grad2 = 10x26

%grad = 10285x1

%As you can see, it's all vectorized but you can figure out how it works and can apply it in your exercise.

%D1 = 

%%%%%%%%%%%%%%%%%%% END PART 3: GRADIENT COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
