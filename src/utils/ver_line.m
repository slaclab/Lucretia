function H=ver_line(x,s,axhan)

% H=ver_line(x,s);
%
% Draws one or more vertical lines along "x" on the current plot and
% leaves the plot in whatever "hold" state it was in.
%
% INPUTs:
%
%   x : (Optional) one or more ordinate values (default = 0)
%   s : (Optional) line color/rendition (default = 'k:')
%   axhan : (Optional) axis handle to use
%
% OUTPUT:
%
%   H : (Optional) handle(s) of plotted lines

if (nargin<2)
  s='k:'; % default to black dotted line
  if (nargin<1)
    x=0;  % default to line at 0
  end
end

hold_state=get(gca,'NextPlot');             % get present hold state
YLim=get(gca,'YLim');                       % get present axis limits
hold on                                     % hold current plot
[r,c]=size(x);                              % get size of ordinate array
h=[];
for n=1:r*c
  if exist('axhan','var')
    h=[h;plot(axhan,x(n)*ones(size(YLim)),YLim,s)]; % draw line
  else
    h=[h;plot(x(n)*ones(size(YLim)),YLim,s)]; % draw line
  end
end
hold off                                    % remove hold
set(gca,'NextPlot',hold_state);             % restore original hold state
if (nargout>0),H=h;end                      % optionally return handles
