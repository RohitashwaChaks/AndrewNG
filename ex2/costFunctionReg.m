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

h_X = sigmoid(X*theta);

cost = -(sum(y'*log(h_X) + (1-y)'*log(1-h_X)));
cost = cost + (lambda/(2))*sum(theta(2:end).^2);
J = cost/m;

deltheta = X'*(h_X - y);
grad = (1/m)*deltheta;

grad = grad + [0; (lambda/m)*theta(2:end,1)];




% =============================================================

end
