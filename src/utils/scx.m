function [xd] = scx(xs);

daxis = axis;
saxis = get(gca,'position');

xd = [saxis(3)+saxis(1)-xs 1;saxis(1)-xs 1]\[daxis(2);daxis(1)];
xd = xd(2);
