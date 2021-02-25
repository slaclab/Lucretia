function [q,dq,chi2,chi2p]=linfitexy(x,dx,y,dy)

%  [q,dq,chi2,chi2p]=linfitexy(x,dx,y,dy);
%
%  Straight-line fit y=a+b*x to data with errors on both x and y values
%  (from Numerical Recipes)
%
%  INPUTs:
%
%    x  = x data (vector, max 1000 points)
%    dx = x errors (same size as x)
%    y  = y data (same size as x)
%    dy = y errors (same size as x)
%
%  OUTPUTs:
%
%    q     = [a;b]
%    dq    = [error on a;error on b]
%    chi2  = chisquare
%    chi2p = chisquare probability

% check the input/output arguments

if (nargin~=4)
  error('Four input arguments required')
end
if (nargout~=4)
  error('Four output arguments required')
end
[nrow,ncol]=size(x);
if (min([nrow,ncol])>1)
  error('Input arguments cannot be arrays')
end

% use the mex-file to do the fit

[q,dq,chi2,chi2p]=fitexy_mex(x(:),dx(:),y(:),dy(:));

% if mex-file returns NaNs, fit without the x errors

if (any(isnan(q)))
  [q,dq,chi2,rms]=noplot_polyfit(x(:),y(:),dy(:),1);
  chi2p=0;
end
