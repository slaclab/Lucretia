function [q,dq,xf,yf]=plot_polyfit(x,y,dy,n,xtxt,ytxt,xunt,yunt,axisv,ll,res)

%   PLOT_POLYFIT
%
%   [q,dq,xf,yf]=plot_polyfit(x,y,dy,n,xtxt,ytxt[,xunt,yunt,axisv,ll,res]);
%
%   Function to plot x data vs y data, and fit to a polynomial of order "n"
%   (with error bars if given), and order skipping available (see below).
%   Will plot AND return fit coefficients and their errors if output variables
%   are provided.  Otherwise only the plot is generated.
%
%   INPUTS:
%
%      x     = the X-axis data vector (column or row vector)
%      y     = the Y-axis data vector (column or row vector)
%      dy    = the Y-axis data vector error (or =1 for no errors given; in
%               this case the fit coefficient errors are rescaled such that
%               CHISQ/NDF = 1)
%      n     = polynomial order to fit; e.g. n=4 implies 0th, 1st, 2nd, 3rd,
%               & 4th order, while n=[1 0 1 1 0] implies only 0th, 2nd, & 3rd
%               order fit
%      xtxt  = (optional) the text string describing the X-axis data
%      ytxt  = (optional) the text string describing the Y-axis data
%      xunt  = (optional) the text string of X-axis units
%      yunt  = (optional) the text string of Y-axis units
%      axisv = (optional) 4-element vector of forced plot scale
%               [xmin xmax ymin ymax], or not used if 1-element
%      ll    = (optional) specifies location of fit coefficients on plot:
%               "ll"==0 -> upper right corner (the default)
%               "ll"==1 -> lower left  corner (historical)
%               "ll"==2 -> upper left  corner
%               "ll"==3 -> lower right corner
%      res   = (optional) if "res"==1, plot residuals of fit
%
%   OUTPUTS:
%
%      q     = (optional) if function is written as "q=polyfit(..." then
%               "q" will be a column vector of fit coefficients with row-1
%                as lowest order; if written as "plot_polyfit(...", then no
%                output is echoed (plot only)
%      dq    = (optional) error on "q" if function is written
%               "[q,dq]=polyfit(..."
%      xf    = (optional) more densely sampled abscissa if written as
%               "[q,dq,xf,yf]=polyfit(..."
%      yf    = (optional) fitted function if function is written as
%               "[q,dq,xf,yf]=polyfit(..."
%
%   plot_polyfit(axhan)
%      axhan = axis handle, pass this command before main request and plots
%      will be directed to given axis handle.
%==========================================================================
%   P.Emma 11/13/91
%   G. White 2/10/2010: add axhan capability and small changes to keep green
%                       box happy
%==========================================================================

persistent axhan

% If requesting an axis to plot in, check that it exists first, otherwise
% behaviour defaults to current axis
if nargin==1
  if ishandle(x)
    axhan=x;
  else
    axhan=[];
  end
  return
elseif ~isempty(axhan)
  if ~ishandle(axhan)
    axhan=[];
  end
end

% make column vectors out of everything

x=x(:);
y=y(:);
dy=dy(:);

% check input args for validity

nx=length(x);
ny=length(y);
if (length(dy)==1)
   bars=0;
else
   bars=1;
   if (ny~=length(dy))
      error('   Y-data and Y-error-data are unequal length vectors')
   end
end
if (nx~=ny)
   error('   X-data and Y-data are unequal length vectors')
end
if (ny<n+1)
   error('   Not enough data points to fit to this order')
end
if (~exist('xtxt','var'))
   xtxt=' ';
end
if (~exist('ytxt','var'))
   ytxt=' ';
end
if (~exist('xunt','var'))
   xunt=' ';
end
if (~exist('yunt','var'))
   yunt=' ';
end
if (~exist('ll','var'))
   ll=0;
end
if (~exist('res','var'))
   res=0;
end

% set up the fit

nn=length(n);
if (nn==1)
   mm=n+1;
   m=0:1:mm;
else
   i=find(n);
   m=i-1;
   mm=length(m);
end
Q=[];
f=mean(abs(x));
x1=x/f;
for j=1:mm
   Q=[Q x1.^m(j)];
end

% do the fit

if (bars)
   [yfit,dyfit,p1,dp1,chisq]=fit(Q,y,dy);
else
   [yfit,dyfit,p1,dp1]=fit(Q,y);
end
for j=1:mm
   p(j)=p1(j)/(f^m(j));
   dp(j)=dp1(j)/(f^m(j));
end
if (mm==2)
   if ((m(1)==0)&&(m(2)==1))
      r=p(2)*std(x)/std(y);  % linear correlation coefficient
   end
end
[Qr,Qc]=size(Q);
NDF=Qr-Qc;
difs=(yfit-y).^2;
difbar=sqrt(mean(difs));
% xsig=std(x);
% ysig=std(y);

% do the plot

if (res==1)
   y=y-yfit;
   ytxt=[ytxt ' (FIT RESIDUALS)'];
end
if ~isempty(axhan)
  if (bars)
     plot_bars(x,y,dy,'o','k',axhan)
  else
     plot(axhan,x,y,'o')
  end
else
  if (bars)
     plot_bars(x,y,dy,'o')
  else
     plot(x,y,'o')
  end
end
if (exist('axisv','var'))
   if (length(axisv)==4)
      axis(axisv)
   end
end
if (res==0)
   lims=get(gca,'XLim');
   hold on
   xp=lims(1):(lims(2)-lims(1))/250:lims(2);
   Q=[];
   for j=1:mm
      Q=[Q xp(:).^m(j)];
   end
   p=p(:);
   yp=Q*p;
   if ~isempty(axhan)
     plot(axhan,xp(:),yp(:),'-')
   else
     plot(xp(:),yp(:),'-')
   end
   hold off
end
if ~isempty(axhan)
  title(axhan,[ytxt ' VS ' xtxt])
  xlabel(axhan,[xtxt ' /' xunt])
  ylabel(axhan,[ytxt ' /' yunt])
else
  title([ytxt ' VS ' xtxt])
  xlabel([xtxt ' /' xunt])
  ylabel([ytxt ' /' yunt])
end

% put polynomial coefficients on the plot

if (ll==1)
  xpn=0.20;  % lower left
  ypn=0.41;
elseif (ll==2)
  xpn=0.20;  % upper left
  ypn=0.91;
elseif (ll==3)
  xpn=0.60;  % lower right
  ypn=0.41;
else
  xpn=0.60;  % upper right
  ypn=0.91;
end
for j=1:mm
   text(scx(xpn),scy(ypn-j*0.025), ...
      sprintf('p%1.0f=%7.4g+-%6.4g',m(j),p(j),dp(j)))
end
text(scx(0.70),scy(0.035),[sprintf('RMS=%5.3g ',difbar) yunt])
text(scx(0.70),scy(0.055),sprintf('NDF=%5.0f ',NDF))
if (bars)
   text(scx(0.02),scy(0.035),sprintf('chisq/N = %5.3f',chisq))
end
if (exist('r','var'))
   text(scx(0.02),scy(0.055),sprintf('rho     = %6.3f',r))
end

% download the results

if (nargout==1)
   q=p;
end
if (nargout==2)
   q=p;
   dq=dp;
end
if (nargout>=3)
   q=p;
   dq=dp;
   xf=xp;
   yf=yp;
end
