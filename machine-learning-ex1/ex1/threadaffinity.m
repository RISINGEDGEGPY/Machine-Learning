

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('JestaTest2.txt');
%data = load('TF8T_HPC8T.txt');
FeatureSize = length(data(1,:))-1;
X = data(:, 2: length(data(1,:)));
y = data(:, 1);
m = length(y);



fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];



fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta = zeros(FeatureSize+1, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Solving with normal equations...\n');


% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
%fprintf('Each iters cost: %f\n',((X*theta-y).^2)/2/m);
fprintf('Total cost: %f.\n',computeCost(X, y, theta));
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('Begin PCA\n');
pause;


FeatureSize = length(data(1,:))-1;
X = data(:, 2: length(data(1,:)));
y = data(:, 1);
m = length(y);
[m, n] = size(X);
[X, mu, sigmaAA] = featureNormalize(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);
V = zeros(n);
K=2;
sigma = X'*X/m;
[U,S,V] = svd(sigma)
Z = zeros(size(X, 1), K);
Z = X*U(:,1:K);
[Z, mu, sigmaAA] = featureNormalize(Z);
Z_new = [ones(m,1) Z];
theta = normalEqn(Z_new, y);

fprintf('Total cost: %f.\n',computeCost(Z_new, y, theta));
fprintf('Theata: %f\n',theta);

% Surface plot
figure;
if (K==2)
  theta0_vals = linspace(-3, 3, 10);
  theta1_vals = linspace(-3, 3, 10);

% initialize J_vals to a matrix of 0's
  y_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
  for i = 1:length(theta0_vals)
      for j = 1:length(theta1_vals)
	    t = [1 theta0_vals(i)  theta1_vals(j)];    
	    y_vals(i,j) = t*theta;
      end
  end
  y_vals=y_vals';
  surf(theta0_vals, theta1_vals, y_vals);
  xlabel('\theta_0'); ylabel('\theta_1');
  hold on;
  scatter3(Z(:,1),Z(:,2),y);
elseif(K=1)
  a= (-3:+3)';
  plot(a,a.*theta(2)+theta(1),'-');
  hold on;
  plot(Z,y,'rx','MarkerSize',10);
endif


%plot(Z(1), Z(2),y, 'rx', 'MarkerSize', 10, 'LineWidth', 2);

