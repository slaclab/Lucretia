function enhance_plot_h(axh,fnam,fsiz,fwgt,lwid,msiz)

%===============================================================================
%
%  enhance_plot_h(axh,fnam,fsiz,fwgt,lwid,msiz)
%
%  Function to enhance MATLAB's lousy line/text choices on plots.  Sets the
%  FontName, FontSize, and FontWeight properties for Xlabel, Ylabel, Title, and
%  all text (including tick labels) for the specified axes.  Also sets LineWidth
%  and MarkerSize of all plotted data.
%
%  INPUTS:
%
%    Set any input to 0 to get the default; set any input to -1 to get
%    MATLAB's default.
%
%    fnam = FontName ['times']  ... MATLAB's ugly default is 'helvetica'
%    fsiz = FontSize [16]       ... MATLAB's tiny default is 10
%    fwgt = FontWeight ['bold'] ... MATLAB's wimpy default is 'normal'
%    lwid = LineWidth [2]       ... MATLAB's spidery default is 0.5
%    msiz = MarkerSize [8]      ... MATLAB's squinty default is 6
%
%===============================================================================
%  Mods:
%    27-MAY-2002, M. Woodley
%       From J. Nelson's enhance_plot.m
%===============================================================================

if (nargin~=6)
  error('enhance_plot_h requires 6 input arguments')
end

% set up defaults

if (fnam==0)
  fontname='times';
elseif (fnam==-1)
  fontname='helvetica';
else
  fontname=fnam;
end
if (fsiz==0)
  fontsize=16;
elseif (fsiz==-1)
  fontsize=10;
else
  fontsize=fsiz;
end
if (fwgt==0)
  fontweight='bold';
elseif (fwgt==-1)
  fontweight='normal';
else
  fontweight=fwgt;
end
if (lwid==0)
  linewidth=2;
elseif (lwid==-1)
  linewidth=0.5;
else
  linewidth=lwid;
end
if (msiz==0)
  markersize=8;
elseif (msiz==-1)
  markersize=6;
else
  markersize=msiz;
end

% axis, ticks, and tick labels

set(axh,'LineWidth',linewidth);
set(axh,'FontName',fontname, ...
        'FontSize',fontsize, ...
        'FontWeight',fontweight)

% title and labels

h=get(axh,'XLabel');
set(h,'FontName',fontname, ...
      'FontSize',fontsize, ...
      'FontWeight',fontweight, ...
      'VerticalAlignment','cap')
h=get(axh,'YLabel');
set(h,'FontName',fontname, ...
      'FontSize',fontsize, ...
      'FontWeight',fontweight, ...
      'VerticalAlignment','bottom')
h=get(axh,'ZLabel');
set(h,'FontName',fontname, ...
      'FontSize',fontsize, ...
      'FontWeight',fontweight, ...
      'VerticalAlignment','bottom')
h=get(axh,'Title');
set(h,'FontName',fontname, ...
      'FontSize',fontsize, ...
      'FontWeight',fontweight, ...
      'VerticalAlignment','baseline')

% plotted data and text

hc=get(axh,'Children');
for n=1:length(hc)
  type=get(hc(n),'Type');
  if (strcmp(deblank(type),'text'))
    set(hc(n),'FontName',fontname, ...
              'FontSize',fontsize, ...
              'FontWeight',fontweight);
  elseif (strcmp(deblank(type),'line'))
    set(hc(n),'LineWidth',linewidth, ...
              'MarkerSize',markersize);
  elseif (strcmp(deblank(type),'rectangle'))
    set(hc(n),'LineWidth',linewidth);
  end
end
