function [q,dq]=plot_parab(x,y,dy,xtxt,ytxt,xunt,yunt);

%   PLOT_PARAB
%
%      [q,dq]=plot_parab(x,y,dy,[xtxt,ytxt[,xunt,yunt]]);
%
%      Function to plot X data vs Y data, and fit to a parabola of the form:
%
%         Y = A*(X-B)**2 + C
%
%      (with error bars if given).  Will plot AND return fit coefficients and
%      their errors if output variables are provided.  Otherwise only the plot
%      is generated.
%
%
%   INPUTS:
%
%      x:    The X-axis data vector (column or row vector)
%      y:    The Y-axis data vector (column or row vector)
%      dy:   (Optional) The Y-axis data vector error (or =1 for no errors
%            given; in this case the fit coefficient errors are rescaled such
%            that CHISQ/NDF = 1)
%      xtxt: (Optional) The text string describing the X-axis data
%      ytxt: (Optional) The text string describing the Y-axis data
%      xunt: (Optional) The text string of X-axis units
%      yunt: (Optional) The text string of Y-axis units
%
%   OUTPUTS:
%
%      q:    (Optional) If function is written as "q = plot_parab(..." then
%            "q" will be a column vector of fit ciefficients with
%
%               q(1) = A,   q(2) = B,   q(3) = C
%
%            If function is written as "plot_parab(...", then no output is
%            echoed (plot only)
%      dq:   (Optional) Error on "p" if function is written as
%            "[q,dq] = plot_parab(..."
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
if (~exist('xtxt'))
  xtxt= ' ';
end
if (~exist('ytxt'))
  ytxt= ' ';
end
if (~exist('xunt'))
  xunt= ' ';
end
if (~exist('yunt'))
  yunt= ' ';
end

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

[Qr,Qc]=size(Q);
NDF=Qr-Qc;

difs=(yfit-y).^2;
difbar=sqrt(mean(difs));
xsig=std(x);
ysig=std(y);

if (bars)
   plot_bars(x,y,dy,'o')
else
   plot(x,y,'o')
end
title([ytxt ' VS ' xtxt])
xlabel([xtxt ' /' xunt])
ylabel([ytxt ' /' yunt])

v=axis;
xp=[v(1):diff(v(1:2))/250:v(2)];
Q=[];
for j=0:2
   Q=[Q xp(:).^j];
end
p=p(:);
yp=Q*p;
hold on
plot(xp(:),yp(:),'-')
hold off
v=axis;

xt=v(1)+0.5*diff(v(1:2));
if (A>0)
   yf=0.85;
else
   if (bars)
      yf=0.45;
   else
      yf=0.40;
   end
end

txt='Y = A*(X-B)^2+C';
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

yf=yf-0.05;
txt=[sprintf('A = %8.5g+-%5.3g',A,dA)];
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

yf=yf-0.05;
txt=[sprintf('B = %8.5g+-%5.3g',B,dB)];
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

yf=yf-0.05;
txt=[sprintf('C = %8.5g+-%5.3g',C,dC)];
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

yf=yf-0.05;
txt=[sprintf('RMS = %8.5g ',difbar) yunt];
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

yf=yf-0.05;
txt=['N_{DOF} = ' int2str(NDF)];
yt=v(3)+yf*diff(v(3:4));
text(xt,yt,txt,'HorizontalAlignment','center')

if (bars)
   yf=yf-0.05;
   txt=['\chi^{2}/N_{DOF} = ' sprintf('%5.3f',chisq)];
   yt=v(3)+yf*diff(v(3:4));
   text(xt,yt,txt,'HorizontalAlignment','center')
end

if (nargout==1)
   q=[A  B  C];
elseif (nargout==2)
   q=[A  B  C];
   dq=[dA dB dC];
end
