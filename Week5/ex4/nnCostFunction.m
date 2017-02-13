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

X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
H = sigmoid(z3);

Y = zeros(m, num_labels);
for i = 1:m
   Y(i, y(i)) = 1;
endfor

%for i = 1:m
%   Jtot = 0;
%   for k = 1:num_labels
%      Jk = 0;
%      J1 = Y(i,k) * log(H(i,k));
%      J2 = (1 - Y(i,k)) * log(1 - H(i,k)); 
%      Jk = J1 + J2; 
%      Jtot = Jtot + Jk; 
%   endfor
%   J = J + Jtot;
%endfor

J = -(1/m) * J;

for i = 1:num_labels
   J1 = Y(:,i)' * log(H(:,i));
   J2 = (1 - Y(:,i))' * log(1 - H(:,i));
   Jk = J1 + J2;
   J = J + Jk;
endfor

J = -(1/m) * J;

Theta1new = Theta1(:, 2: size(Theta1, 2));
Theta2new = Theta2(:, 2: size(Theta2, 2));

Theta1sum = sum(sum(Theta1new .^ 2));
Theta2sum = sum(sum(Theta2new .^ 2));

reg = (lambda / (2 * m)) * (Theta1sum + Theta2sum);

J = J + reg;

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

Delta2 = zeros(num_labels, hidden_layer_size + 1);
Delta1 = zeros(hidden_layer_size, input_layer_size + 1); 
for i=1:m
   x = X(i,:);
   z2 = x * Theta1';
   a2 = sigmoid(z2);
   a2 = [1 a2];
   z3 = a2 * Theta2';
   a3 = sigmoid(z3);

   d3 = a3 - Y(i,:);
   d2 = (d3 * Theta2) .* (a2 .* (1 - a2));
   d2 = d2(2:end);

   Delta2 = Delta2 + (d3' * a2); 
   Delta1 = Delta1 + (d2' * x);
endfor

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (1/m) * Delta1 + ((lambda/m) * Theta1);
Theta2_grad = (1/m) * Delta2 + ((lambda/m) * Theta2);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
