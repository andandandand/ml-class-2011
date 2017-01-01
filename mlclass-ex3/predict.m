function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%       [Values,Index] = max(Z,[],1) -> TO OBTAIN THE MAXES (INDEXES ARE WHAT MATTER) FOR EACH COLUMN  



% Add ones to the X data matrix


X = [ones(m, 1) X]; %adds a row or column of ones?? column

%first_hidden_layer = Theta1 .* X;

hidden_layer = sigmoid(Theta1*X'); %ok?



%first_hidden_layer = sigmoid(Theta1'*X); %try! 

temp = hidden_layer';
m = size(temp, 1); %row size, we need row size to add a ones column 

temp = [ones(m,1) temp]; %here we are adding a one Column, but should it be a one Row??

hidden_layer = temp';

%first_hidden_layer(1,:)

output_layer = sigmoid(Theta2 * hidden_layer);

%first_cols = size(first_hidden_layer, 2);

% first_cols = 5000 
%ones_row = ones(1, first_cols);
%first_hidden_layer = [ones_row first_hidden_layer];

%first_hidden_layer(:,1); %debería imprimir puros unos 

%second_hidden_layer_activations = Theta2*first_hidden_layer;

%second_rows = size(second_hidden_layer_activations, 1);

%second_hidden_layer_activations = [ones(second_rows, 1) second_hidden_layer_activations]; %add bias term 

[values,p]= max(output_layer',[],2);

whos



% =========================================================================


end
