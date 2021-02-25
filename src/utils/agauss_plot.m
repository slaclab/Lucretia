function [q,dq,xf,yf] = agauss_plot(x,y,dy,y_off,no_txt,ax,fcol,pcol)

%	[q,dq,xf,yf] = agauss_plot(x,y[,dy,y_off,no_txt])
%
%	Fits and plots asymmetruc gaussian curve data vectors "x" and "y" with
%   optional error bars "dy".  Fit results are summarized on the plot.  The fit
%	form is:
%	
%       y = A + B*exp(-(x-C)^2/(2*(D*(1+sign(x-C)*E))^2))
%
%    INPUTS:
%
%       x:      The independent variable data vector (column or row)
%       y:      The dependent variable data vector (column or row)
%       dy:     (Optional,DEF=not used) The data vector of error bars on the
%               dependent variable data (column or row)
%       y_off:  (Optional,DEF=1) If "y_off" .NE. 0, then the gaussian fit
%               includes a "y" offset parameter in the fit.  If "y_off" = 0,
%               this forces the fit to include no "y" offset and therefore q(1)
%               is set exactly to zero.
%       no_txt: (Optional,DEF=1) If =0, no text will be printed on the plot
%
%    OUTPUTS:
%
%       q:      (Optional) [A B C D E] fit coefficients
%       dq:     (Optional) errors on above fit coefficients
%       xf:     (Optional) returns dense array of X-axis for re-plot
%       yf:     (Optional) returns dense array of Y-axis for re-plot

%===============================================================================

% Which axes to plot on
if ~exist('ax','var')
  figure
  ax=gca;
end

if ~exist('fcol','var')
  fcol='r';
end

if ~exist('pcol','var')
  pcol='b';
end

x = x(:);
y = y(:);
nx = length(x);
ny = length(y);
if nx ~= ny,error('X and Y data vectors must be the same length'),end
if ~exist('y_off','var') || isempty(y_off)
  y_off = 1;
end
if ~exist('no_txt','var') || isempty(no_txt)
  no_txt = 1;
end

minx = min(x);
maxx = max(x);
widx = maxx - minx;
stpx = widx/100;
xx = minx:stpx:maxx;
xx = xx(:);

if ~exist('dy','var') || isempty(dy)
  [~,p,dp,chisq] = agauss_fit(x,y);
  yyf = p(1)*ones(size(xx)) + p(2)*sqrt(2*pi)*p(4)*agauss(xx,p(3),p(4),p(5)); 
  semilogy(ax,x,y,[pcol '.'],xx,yyf,[fcol '-'],'LineWidth',3)
else
  dy = dy(:);
  ndy = length(dy);
  if ndy == 1
    dy = dy*ones(size(x));
    ndy = length(dy);
  end
  if ndy ~= nx
    error('dY error data vector must be the same length as X and Y')
  end
  [~,p,dp,chisq] = agauss_fit(x,y,dy,y_off);
  yyf = p(1)*ones(size(xx)) + p(2)*sqrt(2*pi)*p(4)*agauss(xx,p(3),p(4),p(5)); 
  plot(ax,x,y,',',xx,yyf,[fcol '-'])
  hold(ax,'on');
  plot_bars(x,y,dy,'.','k',ax)
  hold(ax,'off');
end

if no_txt==1
  title('A + B*exp( -(X-C)^2/2*(D*(1+sign(X-C)*E))^2 )')

  [tx,ty]=text_locator(2,-2,'t');
  text(tx,ty,sprintf('A = %7.3g +- %7.3g',p(1),dp(1)),'FontSize',10)
  [tx,ty]=text_locator(2,-3,'t');
  text(tx,ty,sprintf('B = %7.3g +- %7.3g',p(2),dp(2)),'FontSize',10)
  [tx,ty]=text_locator(2,-4,'t');
  text(tx,ty,sprintf('C = %7.3g +- %7.3g',p(3),dp(3)),'FontSize',10)
  [tx,ty]=text_locator(2,-5,'t');
  text(tx,ty,sprintf('D = %7.3g +- %7.3g',p(4),dp(4)),'FontSize',10)
  [tx,ty]=text_locator(2,-6,'t');
  text(tx,ty,sprintf('E = %7.3g +- %7.3g',p(5),dp(5)),'FontSize',10)
  [tx,ty]=text_locator(2,-8,'t');
  text(tx,ty,sprintf('CHISQ/NDF = %8.3g',chisq),'FontSize',10)
end

q = p;
dq = dp;

xf = xx;
yf = yyf;
