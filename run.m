clear;
addpath('Datasets/');
load('bbcsport_2view.mat');
alpha = 1; 
beta = 50;
omega = [1,10];  %1*V

result = LTBPL(X,alpha,beta,omega,gt);
