function plot_2bars(x,y,dx,dy,char);

%               plot_2bars(x,y,dx,dy[,char])
%
%               Function to plot with 2 dimensional error bars of x +/- dx,
%               and y +/- dy.
%
%     INPUTS:   x:      The horizontal axis data vector (column or row)
%               y:      The vertical axis data vector (column or row)
%               dx:     The half length of the error bar on "x" (column, row,
%                       or scalar)
%               dy:     The half length of the error bar on "y" (column, row,
%                       or scalar)
%               char:   (Optional) The plot character at the point (x,y) 
%                       (see plot)

%=============================================================================

x  = x(:)';
y  = y(:)';
dx = dx(:)';
dy = dy(:)';

[rx,cx] = size(x);
[ry,cy] = size(y);
[rdx,cdx] = size(dx);
[rdy,cdy] = size(dy);

if (rx~=1) | (ry~=1) | (rdx~=1) | (rdy~=1)
  disp(' ')
  disp('*** PLOT_BARS only plots vectors ***')
  disp(' ')
  return
end

n = cx;

if cdx==1
  dx = dx*ones(1,n);
end

if cdy==1
  dy = dy*ones(1,n);
end

x_barv = [x; x];
y_barv = [y+dy; y-dy];

x_barh = [x-dx; x+dx];
y_barh = [y; y];

if exist('char')==0,
  plot(x_barv,y_barv,'-',...
       x_barh,y_barh,'-')
else
  plot(x_barv,y_barv,'-',...
       x_barh,y_barh,'-',...
       x,y,char);
end
