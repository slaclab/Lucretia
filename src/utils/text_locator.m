function [x,y] = text_locator(col, line, flags)
%
% [x,y] = text_locator(col, line, flags);
%                                 ----- (optional)
% function returns x and y to correspond to best guess location of the
% location of the indicated place.  Note that there are fewer cols and lines
% in a subplotted plot.
%
% flags:  t or T => wrt top   margin
%         r or R => wrt right margin
%
% examples:
%
% [x,y]=text_locator(3,3)
% text(x,y,'Lower Left Corner')
%
% [x,y]=text_locator(-3, -3, 'tr')
% text(x,y,'Right top corner','HorizontalAlignment','right','FontSize',10)


Pos=get(gca,'Position');
axisx = axis;

if nargin < 3,  flags=' '; end

if ~any([find(flags=='r') find(flags=='R')])
  % this line reads as follows:
  % left-axis + col-num / (72 cols for a "full screen") * axis-range
  x = axisx(1) + col* ((1/72)*(Pos(3)/0.77))*(axisx(2)-axisx(1));
else
  x = axisx(2) + col* ((1/72)*(Pos(3)/0.77))*(axisx(2)-axisx(1));
end

if ~any([find(flags=='t') find(flags=='T')])
  y = axisx(3) + line*((1/30)*(Pos(4)/0.82))*(axisx(4)-axisx(3));
else
  y = axisx(4) + line*((1/30)*(Pos(4)/0.82))*(axisx(4)-axisx(3));
end
