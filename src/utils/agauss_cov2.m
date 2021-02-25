function [covm]=agauss_cov2(x,dy,A,B,C,D,E)
%
%  [covm]=agauss_cov(x,dy,A,B,C,D,E);
%
%  Calculates the covariance matrix from the nonlinear fit to an asymmetric
%  gaussian of the form:
%
%  yfit=A+B*exp(-0.5*((x-C)/(D*(1+sign(x-C)*E)))^2)
%
%  Naturally must be called AFTER a call to agauss_fit, which is a non-linear
%  fitting routine to fit an asymmetric exponential curve to the data in
%  vectors "x" and "y", where "x" contains the independent variable data and
%  "y" contains the dependent data.
%
%
%  INPUTS:   x:      A vector (row or column) of independent variable data.
%            dy:     A vector (row or column) of the errors on the dependent
%                    variable data.
%            A:      DC offset (not used).
%            B:      Fitted amplitude scaling factor.
%            C:      Fitted horizontal offset (<x>).
%            D:      Fitted standard deviation (sigma).
%            E:      Fitted asymmetry parameter.
%
%  OUTPUTS:  covm:   Covariance matrix (5x5) containing the errors on each of
%                    the fitted parameters.

%===============================================================================

% For each value of x, evaluate partial derivatives of asymmetric gaussian
% function wrt each fitted variable

v1=x-C;
v1s=v1.^2;
v2=sign(v1);
v3=1+v2*E;
Dv3s=(D*v3).^2;
v3c=v3.^3;
vexp=exp(-0.5*(v1s./Dv3s));

dA=1;
dB=vexp;
dC=(B*(v1./Dv3s)).*vexp;
dD=((B/D)*(v1s./Dv3s)).*vexp;
dE=((B/D^2)*(v2.*v1s./v3c)).*vexp;

% Load upper triangle of inverse covariance matrix h

h=zeros(5,5);
dysq=dy.^2;

h(1,2)=sum((dA.*dB)./dysq);
h(1,3)=sum((dA.*dC)./dysq);
h(1,4)=sum((dA.*dD)./dysq);
h(1,5)=sum((dA.*dE)./dysq);
h(2,3)=sum((dB.*dC)./dysq);
h(2,4)=sum((dB.*dD)./dysq);
h(2,5)=sum((dB.*dE)./dysq);
h(3,4)=sum((dC.*dD)./dysq);
h(3,5)=sum((dC.*dE)./dysq);
h(4,5)=sum((dD.*dE)./dysq);

% Sum h with its transpose to make h symmetric

h=h+h';

% Load diagonal elements of h

h(1,1)=sum(1./dysq);
h(2,2)=sum((dB.^2)./dysq);
h(3,3)=sum((dC.^2)./dysq);
h(4,4)=sum((dD.^2)./dysq);
h(5,5)=sum((dE.^2)./dysq);

% Load covariance matrix

covm =inv(h);
