function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;
retC = 0.01;
retSigma = 0.01;

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

for i = 1:8 
   sigma = 0.01;
   for j = 1:8
      model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
      pred = svmPredict(model, Xval);
      err = mean(double(pred ~= yval));
      if (i == 1)
         min = err;
         retC = C;
         retSigma = sigma;
      elseif (err < min)
         min = err;
         retC = C;
         retSigma = sigma;
      endif
      sigma = sigma * 3;
   endfor
   C = C * 3;
endfor

C = retC
sigma = retSigma



% =========================================================================

end
