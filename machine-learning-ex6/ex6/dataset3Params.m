function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.0;
sigma = 0.1;

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
%
%steps=[0.01,0.03,0.1,0.3,1,3,10,30];
%%steps=[0.01,1];
%store=zeros(length(steps)^2,3);
%track=0;
%for i=1:length(steps); #this is sigma  
% for j=1:length(steps); #this is C
%   model=svmTrain(X, y, steps(j), @(x1, x2) gaussianKernel(x1, x2,steps(i)));
%   predict=svmPredict(model, Xval);
%   error=mean(double(predict~= yval));
%   store(j+track,1)=steps(i);  #first col is sigma
%   store(j+track,2)=steps(j);  #sec 
%   store(j+track,3)=error;     #third col is error
% end
% track+=length(steps);
%end
%[val pos]=min(store,[],1);
%C = store(pos(3),2);
%sigma =store(pos(3),1);

% =========================================================================

end