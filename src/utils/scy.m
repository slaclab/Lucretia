function [yd] = scy(ys);

daxis = axis;
saxis = get(gca,'position');

yd = [saxis(4)+saxis(2)-ys 1;saxis(2)-ys 1]\[daxis(4);daxis(3)];
yd = yd(2);
