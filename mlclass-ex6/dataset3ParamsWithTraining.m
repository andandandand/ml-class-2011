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

printf("********In dataset3Params..**************");

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

bestValueForSigma = -1;
bestValueForC = -1; 
minimumMeanError = 99999.0; 


for index1=1 : 8

	for index2=1: 8
       		
		C = values(index1); 
		sigma = values(index2);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

                predictions = svmPredict(model, Xval);
		
		predictionError = mean(double(predictions ~= yval));

                printf("predictionError with sigma= %d and C= %d is %f \n", sigma, C, predictionError )
		
		if(predictionError<minimumMeanError)
			
			bestValueForC     = C;
			bestValueForSigma = sigma;		
                        
			minimumMeanError = predictionError; 
	
		end %end if


	end %end inner for

end %end outer for 

C = bestValueForC;
sigma = bestValueForSigma; 

printf("bestValueForC is %d, bestValueForSigma is %d\n",C, sigma);

% =========================================================================

end
