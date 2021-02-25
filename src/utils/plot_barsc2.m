function plot_barsc2(x,dx,y,dy,c,s)
%
% plot_barsc2(x,dx,y,dy,c,s)
%
% Function to plot data with horizontal and vertical error bars
% (x +/- dx and y +/- dy)
%
% INPUTS:
%
%   x  = horizontal axis data vector (column or row)
%   dx = half length of the error bar on "x" (column, row, or scalar)
%   y  = vertical axis data vector (column or row)
%   dy = half length of the error bar on "y" (column, row, or scalar)
%   c  = color ... defaults to black ('k' ... see plot)
%   s  = symbol ... defaults to circle ('o' ... see plot)

% convert data to columns

  x=x(:);
  dx=dx(:);
  y=y(:);
  dy=dy(:);

% check that data is vector, not array

  [rx,cx]=size(x);
  [rdx,cdx]=size(dx);
  [ry,cy]=size(y);
  [rdy,cdy]=size(dy);
  if ((cx>1)|(cdx>1)|(cy>1)|(cdy>1))
    error('plot_barsc2 only plots vectors')
  end

% if the error on x/y is a scalar, apply it to all x/y values

  if (rdx==1)
    dx=dx*ones(size(x));
  end
  if (rdy==1)
    dy=dy*ones(size(y));
  end

% generate coordinates for error bars

  vt=(max(y)-min(y))/200;
  x_barh=[x+dx,x-dx,x-dx,x-dx,x+dx,x+dx];
  y_barh=[y,y,y+vt,y-vt,y+vt,y-vt];
  ht=(max(x)-min(x))/200;
  x_barv=[x,x,x-ht,x+ht,x-ht,x+ht];
  y_barv=[y+dy,y-dy,y-dy,y-dy,y+dy,y+dy];

% handle defaults for color and symbol

  if (exist('c')~=1),c='k';end
  if (exist('s')~=1),s='o';end

% make the plot

  hold_state=get(gca,'NextPlot');
  plot(x,y,[c,s])
  hold on
  plot(x_barh(:,1:4)',y_barh(:,1:4)',[c,'-']);
  plot(x_barh(:,5:6)',y_barh(:,5:6)',[c,'-']);
  plot(x_barv(:,1:4)',y_barv(:,1:4)',[c,'-']);
  plot(x_barv(:,5:6)',y_barv(:,5:6)',[c,'-']);
  set(gca,'NextPlot',hold_state)
