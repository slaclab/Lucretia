function f=agauss_min2(p, arg1)

%  f=agauss_min(p, arg1);
%
%  Returns the error between the data and the values computed by the current
%  function of p.  Assumes a function of the form:
%
%  y=c(1)+c(2)*(1-exp(-0.5*((x-p(1))/(p(2)*(1+sign(x-p(1))*p(3))))^2))
%
%  with 2 linear parameters and 3 nonlinear parameters (see also agauss).
%
%  INPUTS:   p:      A vector of 3 scalar asymmetric gaussian parameters:
%
%                    p(1) => Horizontal offset (<x>).
%                    p(4) => Standard deviation (sigma) of the asymmetric
%                            gaussian.
%                    p(3) => Asymmetry parameter (=0 for symmetric gaussian).
%
%            arg1:   x (arg1(:,1)), y (arg1(:,2)), and dy (arg1(:,3)) to fit.
%
%  OUTPUTS:  f:      Computed error value.

%===============================================================================

x=arg1(:,1);
y=arg1(:,2);
dy=arg1(:,3);

A=[ones(size(x)) agauss2(x,p(1),p(2),p(3),p(4),p(5),p(6))];
c=A\y;
z=A*c;
f=norm((z-y)./dy);
