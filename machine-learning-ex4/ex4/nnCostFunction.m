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
%X = [ones(m, 1) X];  %X is 5000x401 theta1=25x401
%Z_2=X*Theta1';  %Z2 is 5000x25
%a_2=sigmoid(Z_2);
%a_2 = [ones(m, 1) a_2];  #a2 is 5000x26  and Theta2 is 10x26
%Z_3=a_2*Theta2';  # Z_3 is 5000x26
%a_3=sigmoid(Z_3);  #a_3 is 10x26
%
%
X = [ones(m, 1) X];
for c = 1:m
    aa=1:num_labels;
    aa=(aa==y(c));
    Z_2=X(c,:)*Theta1'; #Z_2 is 1x25
    a_2=sigmoid(Z_2);
    a_2 = [ones(1, 1) a_2];
    Z_3=a_2*Theta2';  #Z_2 is 1x10
    J=J+((-aa*(log(sigmoid(Z_3)))')-((1-aa)*(log(1-sigmoid(Z_3)))'));
end

J=(J/m)+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

[a b]=size(Theta1);
Delta_1 = zeros(a,b);
[a b]=size(Theta2);
Delta_2 = zeros(a,b);

for c = 1:m
    aa=1:num_labels;
    aa=(aa==y(c));
    a_1=X(c,:);
    Z_2=Theta1*a_1'; #Z_2 is 25*1  Theta1 is 25*401 a_1 is 1*401
    a_2=sigmoid(Z_2);
    a_2 = [1; a_2];  #a_2 is 26*1
    Z_3=Theta2*a_2;  #Z_2 is 1x10 row vector  Theta2 is 10x26  
    a_3=sigmoid(Z_3);  #a_3 is 10*1
    delta_3=a_3-aa'; #delta_3 is 10*1
    delta_2=(Theta2')*delta_3.*[1;sigmoidGradient(Z_2)];  %Theta2 is 10x26  Theta2' is 26*10  delta_3 is 10*1  delta_2 is 26*1
    delta_2=delta_2(2:end);
    Delta_1=Delta_1+(delta_2*a_1 );  #a_1 is already a row vector
    Delta_2=Delta_2+(delta_3*a_2');   #delta_3 is 1*10  and a_2 is 1*26  Delta_1 10x26


Theta1_grad = Delta_1/m ;
Theta2_grad = Delta_2/m ;

Theta1_grad(:,2:end) =Theta1_grad(:,2:end)+ (lambda/m)*Theta1(:,2:end) ;
Theta2_grad(:,2:end) =Theta2_grad(:,2:end)+ (lambda/m)*Theta2(:,2:end) ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
