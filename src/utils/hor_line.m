function hor_line(y,s);

%HOR_LINE       hor_line(y,s);
%
%               Draws a horizonal on the current plot line at y
%               and leaves the plot in the "hold" state it was in
%
%     INPUTS:   y:      (Optional) The value of the vertical axis to draw a
%                       line at (default = 0), or an array of values
%               s:      (Optional) The plot rendition (default = 'k:')
%     OUTPUTS:          Plots line on current plot
%
%Woodley 04may01: from ver_line.m
%
% modified 12apr02 JLN: allow passing of arrays of horizontal lines to draw
%===========================================================================

if (nargin<2)
  s = 'k:';                        % default to black dotted line
  if (nargin<1)
    y = 0;                         % default to line at 0
  end
end

hold_state = get(gca,'NextPlot');  % get present hold state
XLim = get(gca,'XLim');            % get present axis limits

hold on                            % hold current plot
[r,c]=size(y);                     % get size of input array
for n=1:r*c
 plot(XLim,y(n)*ones(size(XLim)),s)    % draw line
end
hold off                           % remove hold

set(gca,'NextPlot',hold_state);    % restore original hold state
