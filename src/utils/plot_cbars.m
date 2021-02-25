function plot_bars_c(x,y,dy,s1,s2);

%               plot_bars_c(x,y,dy,s1,s2)
%
%               Function to plot vertical error bars of y +/- dy.
%
%     INPUTS:   x:      The horizontal axis data vector (column or row)
%               y:      The vertical axis data vector (column or row)
%               dy:     The half length of the error bar on "y" (column, row,
%                        or scalar)
%               s1:     The color/symbol for the data (see plot)
%               s2:     The color for the error bar (see plot)

%=============================================================================

x  = x(:);
y  = y(:);
dy = dy(:);

[rx,cx] = size(x);
[ry,cy] = size(y);
[rdy,cdy] = size(dy);

if (cx~=1) | (cy~=1) | (cdy~=1)
  error('*** PLOT_BARS only plots vectors ***')
end

n = rx;

if rdy==1
  dy = dy*ones(size(y));
end

tee = (max(x) - min(x))/200;

x_barv = [x x x-tee x+tee x-tee x+tee];
y_barv = [y+dy y-dy y-dy y-dy y+dy y+dy];

plot(x_barv(:,1:4)',y_barv(:,1:4)',s1);

hold_state = get(gca,'NextPlot');
hold on;
plot(x_barv(:,5:6)',y_barv(:,5:6)',s2);
set(gca,'NextPlot',hold_state);
