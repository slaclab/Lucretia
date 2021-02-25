function h = title2(str)
% TITLE2(STR) places a [possible multi-line] title above the plot, 
% resizing the axis if necessary to make room for a possibly multiline
% title.
%
% STR is the string matrix or cell array of lines, 
% or a string array with each line separated by a newline character.
%
% examples:
%    title2(['one potato'; 'two potato'])
%    title2({'fish'; 'cod'})
%    title2(sprintf('Hello\nGoodbye'))

% Resizing the Figure Window will effect the location
% of the titles.

% Written by John L. Galenski III
% All Rights Reserved  05-11-94
% LDM051194jlg
% modified by jae H. Roh, 1-20-97

% Determine the default title location
ht = get(gca,'Title');

% save Unit settings
aUnits = get(gca,'Units');
htUnits = get(ht,'Units');

% work in pixels
set(ht,'Units','pixels');
pt = get(ht, 'Position');

set(ht,'String', 'one line');
oneLineExtent = get(ht,'Extent');
lineHeight = oneLineExtent(4);
set(ht,'VerticalAlignment','Top');
newPos = [pt(1) oneLineExtent(2)+oneLineExtent(4)];

set(ht,'Position', newPos);
set(ht,'String', str);
titleExtent = get(ht,'Extent');

% shrink the axis to make room for the title, if necessary
set(gca,'Units','pixels');
axisPos = get(gca,'Position');
if axisPos(4)>titleExtent(2)
     axisPos(4) = axisPos(4)-(titleExtent(4)-lineHeight);
        set(gca,'Position',axisPos);
      else  % or shrink to close up space...  doesn't work great
        while (axisPos(4)+lineHeight <= titleExtent(2))
              axisPos(4) = axisPos(4)+lineHeight;
            end
               set(gca,'Position',axisPos);
             end
             
             % restore units settings
             set(ht,'Units',htUnits);
             set(gca,'Units',aUnits);
             
             % Return the handles if requested
             if nargout == 1
                 h = ht;
               end