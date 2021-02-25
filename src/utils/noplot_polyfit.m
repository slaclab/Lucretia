function [q,dq,chisq,rms]=noplot_polyfit(x,y,dy,n)
%===============================================================================
%
% [q,dq,chisq,rms]=noplot_polyfit(x,y,dy,n);
%                                   
% Do a "plot_polyfit" without the plot ... see plot_polyfit for details
%                                
% INPUTS:
%
%    x     = X-axis data vector
%    y     = Y-axis data vector
%    dy    = Y-axis error vector (set dy=1 when no errors are given; the fit
%            coefficient errors will be rescaled such that CHISQ/NDF=1)
%    n     = polynomial order to fit (e.g. n=4 implies 0th, 1st, 2nd, 3rd, and
%            4th order, while n=[1 0 1 1 0] implies only 0th, 2nd, & 3rd order
%            fit)
%
% OUTPUTS:
%
%    q     = fitted polynomial coefficients (lowest order first)
%    dq    = errors on fitted polynomial coefficients
%    chisq = chisquare of fit
%    rms   = rms of fit residuals
%
%===============================================================================
% 09-SEP-1997, M. Woodley
%    From P. Emma's toolbox:[slc]plot_polyfit.m
%===============================================================================

% check input arguments

x=x(:);                     
y=y(:);
dy=dy(:);
nx=length(x);
ny=length(y);
if (length(dy)==1)
   bars=0;
else
   bars=1;
   if (ny~=length(dy))
      error('Y-data and Y-error vectors are unequal length')
   end                    
end
if (nx~=ny)
   error('X-data and Y-data vectors are unequal length')
end
if (ny<(n+1))
   error('Not enough data points to fit to this order')
end

% set up vector of exponents

nn=length(n);
if (nn==1)
   mm=n+1;
   m=[0:1:mm];
else
   i=find(n);
   m=i-1;
   mm=length(m);
end

% construct fit matrix

f=mean(abs(x));
x1=x/f;
Q=[];
for j=1:mm
   Q=[Q,x1.^m(j)];
end

% do the fit

if (bars)
   [yfit,dyfit,p1,dp1,chi2]=fit(Q,y,dy);
else                        
   [yfit,dyfit,p1,dp1]=fit(Q,y);
end
for j=1:mm
   p(j)=p1(j)/(f^m(j));
   dp(j)=dp1(j)/(f^m(j));
end

% set up output args

q=p;
dq=dp;
if (bars)
   chisq=chi2;
else
   chisq=0;
end
rms=sqrt(mean((yfit-y).^2));
