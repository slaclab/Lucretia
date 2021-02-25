function y=agauss(x,x_bar,sig,t)

%  y=agauss(x[,x_bar,sig,t]);
%
%  Asymmetric gaussian function to create an asymmetric "bell" curve
%  from the independent variable "x" having left-half width equal to
%  "sig*(1-t)", right-half width equal to "sig*(1+t)", and center at
%  "x_bar".  It is normalized already so that the area under the full
%  curve will be 1.
%
%  y=exp(-0.5*((x-<x>)/(sig*(1+sign(x-<x>))*t))^2)/(sig*sqrt(2pi))
%
%  INPUTS:   x:      The independent variable in the exponential of the
%                    gaussian (column or row vector)
%            x_bar:  (Optional,DEF=0) The center of the gaussian on the
%                    "x" axis (mean) which defaults to 0 if not given
%                    (scalar)
%            sig:    (Optional,DEF=1) The gaussian standard deviation
%                    ("width") which defaults to 1 if not given (scalar)
%            t:      (Optional,DEF=0) The gaussian asymmetry parameter
%                    which defaults to 0 if not given (scalar)
%
%  OUTPUTS:  y:      The values of the asymmetric gaussian at each "x"
%                    (vector the same size as "x").
%
%  EXAMPLE:  >> x=[-2.5:.5:2.5];
%            >> y=agauss(x);
%            >> y
%             y =
%              Columns 1 through 6
%                0.0175    0.0540    0.1295    0.2420    0.3521    0.3989
%              Columns 7 through 11
%                0.3521    0.2420    0.1295    0.0540    0.0175

%===============================================================================

if nargin==1
  x_bar=0;
  sig=1;
  t=0;
elseif nargin==2
  sig=1;
  t=0;
elseif nargin==3
  t=0;
end

if sig==0
   sigx=1e-3;
else
   sigx=sig;
end

y=exp(-0.5*((x-x_bar)./(sigx*(1+sign(x-x_bar)*t))).^2)/(sigx*sqrt(2*pi));
