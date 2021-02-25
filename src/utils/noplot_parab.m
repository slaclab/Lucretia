function [q,dq]=noplot_parab(x,y,dy);

%   NOPLOT_PARAB
%
%      [q,dq]=noplot_parab(x,y,dy);
%
%      Function to fit X data vs Y data to a parabola of the form:
%
%         Y = A*(X-B)**2 + C
%
%      (with error bars if given).
%
%   INPUTS:
%
%      x:    The X-axis data vector (column or row vector)
%      y:    The Y-axis data vector (column or row vector)
%      dy:   (Optional) The Y-axis data vector error (or =1 for no errors
%            given; in this case the fit coefficient errors are rescaled such
%            that CHISQ/NDF = 1)
%
%   OUTPUTS:
%
%      q:    If function is written as "q = plot_parab(..." then
%            "q" will be a column vector of fit ciefficients with
%
%               q(1) = A,   q(2) = B,   q(3) = C
%
%            If function is written as "plot_parab(...", then no output is
%            echoed (plot only)
%      dq:   Error on "p" if function is written as "[q,dq] = plot_parab(..."
%
% P.Emma 02/27/92
% M. Woodley 02/28/98 ... Matlab5

%===============================================================================

x=x(:);
y=y(:);
nx=length(x);
ny=length(y);
if (nx~=ny)
   error('   X-data and Y-data are unequal length vectors')
end
if (ny<3)
   error('   Not enough data points to fit to this order')
end
if (exist('dy'))
   dy=dy(:);
   if ((length(dy)>1)&(length(dy)~=ny))
      error('   Y-data and Y-error-data are unequal length vectors')
   end
else
   dy=1;
end
bars=(length(dy)>1);

Q=[];
for j=0:2
   Q=[Q x.^j];
end

if (bars)
   [yfit,dyfit,p,dp,chisq,Cv]=fit(Q,y,dy);
else
   [yfit,dyfit,p,dp,chisq,Cv]=fit(Q,y);
end

if (p(3)~=0)
   A=p(3);
   B=-p(2)/(2*p(3));
   C=p(1)-p(2)^2/(4*p(3));
else
   error('   The quadratic coefficient = 0 ...  cannot calc A,B,C')
end

grad_A=[0 0 1]';
grad_B=[0 -1/(2*p(3)) p(2)/(2*p(3)^2)]';
grad_C=[1 -p(2)/(2*p(3)) p(2)^2/(4*p(3)^2)]';
dA=sqrt(grad_A'*Cv*grad_A);
dB=sqrt(grad_B'*Cv*grad_B);
dC=sqrt(grad_C'*Cv*grad_C);

if (nargout==1)
   q=[A  B  C];
elseif (nargout==2)
   q=[A  B  C];
   dq=[dA dB dC];
end
