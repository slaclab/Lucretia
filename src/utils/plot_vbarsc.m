function h=plot_vbarsc(x,y,dy,c,s)
%
% h=plot_vbarsc(x,y,dy,c,s)
%
% Function to plot data with vertical error bars (y +/- dy)
%
% INPUTS:
%
%   x  = horizontal axis data vector (column or row)
%   y  = vertical axis data vector (column or row)
%   dy = half length of the error bar on "y" (column, row, or scalar)
%   c  = color ... defaults to black ('k' ... see plot)
%   s  = symbol ... defaults to circle ('o' ... see plot)
%
% OUTPUT:
%
%   h  = (optional) handles to plotted data and error bars

  [rx,cx]=size(x);
  [ry,cy]=size(y);
  [rdy,cdy]=size(dy);
  if ((min([rx,cx])>1)|(min([ry,cy])>1)|(min([rdy,cdy])>1))
    error('plot_barsc only plots vectors')
  end

% convert data to columns

  x=x(:);
  y=y(:);
  dy=dy(:);

% if the error on y is a scalar, apply it to all y values

  if (rdy==1)
    dy=dy*ones(size(y));
  end

% generate coordinates for error bars

  x_barv=[x,x];
  y_barv=[y+dy,y-dy];

% handle defaults for color and symbol

  if (exist('c')~=1),c='k';end
  if (exist('s')~=1),s='o';end

% make the plot

  hold_state=get(gca,'NextPlot');
  h1=plot(x,y,s);
  set(h1,'Color',c)
  hold on
  h2=plot(x_barv(:,1:2)',y_barv(:,1:2)','-');
  set(h2,'Color',c)
  set(gca,'NextPlot',hold_state)

  if (nargout),h=[h1;h2];end
  