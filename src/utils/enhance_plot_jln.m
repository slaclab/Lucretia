function enhance_plot_jln(fontname,fontsize,linewid,markersiz,lgd,ax)

%  enhance_plot_jln([fontname,fontsize]);
%
%  Function to enhance MATLAB's lousy text choices on plots.  Sets the
%  current figure's Xlabel, Ylabel, Title, and all Text on plots, plus
%  the axes-labels to the "fontname" and "fontsize" input here where
%  the defaults have been set to 'times' and 16.
%  Also sets all plotted lines to "linewid" and all markers to size
%  "markersiz".  The defaults are 2 and 8.
%
%  INPUTS:  fontname:   (Optional,DEF='TIMES') FontName string to use
%                       MATLAB's ugly default is 'Helvetica'
%           fontsize:   (Optional,DEF=16) FontSize integer to use
%                       MATLAB's tiny default is 10
%           linewid:    (Optional,DEF=2) LineWidth integer to use
%                       MATLAB's skinny default is 0.5
%           markersiz:  (Optional,DEF=8) MarkerSize integer to use
%                       MATLAB's squinty default is 6
%           lgd:     (Optional, DEF=0) if is 0, doesn't change the legend
%                       if is 1, changes only the lines on the legend
%                       if is 2, changes both the lines and the text
%                       if is 3, changes only the text
%           ax:       (Optional, DEF=current axis)
%                      pass handles to current plot axes
%  for all inputs, if pass 0, use default
%                 if pass -1, use MATLAB's default
%
% Modifications
%  19-Feb-2002 J. Nelson
%       added linewid and markersiz to help squinting readers
%  20-Feb-2002 J. Nelson
%       added check for legend.  If legend exists, increase the 
%           line and marker size, also increase the font to 
%           fontsize-2 (2 points smaller than title and labels)
%  25-Feb-2002 J. Nelson
%       added lgd (legend) input check to fix legend problems.
%======================================================================

if (~exist('fontname')|(fontname==0))
  fontname = 'times';
elseif (fontname==-1)
  fontname = 'helvetica';
end
if (~exist('fontsize')|(fontsize==0))
  fontsize = 16;
elseif (fontsize==-1)
  fontsize=10;
end
if (~exist('linewid')|(linewid==0))
  linewid=2;
elseif (linewid==-1)
  linewid=0.5;
end
if (~exist('markersiz')|(markersiz==0))
  markersiz = 8;
elseif (markersiz==-1)
  markersiz = 6;
end
if (~exist('lgd')|(lgd==0)|(lgd==-1))
  lgd=0;
end
fontweight='bold';

Hf=gcf;
if (~exist('ax'))
  ax=gca;
end
for nn=1:length(ax)
  Ha=ax(nn);
  Hx=get(Ha,'XLabel');
  Hy=get(Ha,'YLabel');
  Hz=get(Ha,'ZLabel');
  Ht=get(Ha,'Title');
  set(Ha,'LineWidth',.75); %these are the axis and grid lines
  set(Hx,'fontname',fontname);
  set(Hx,'fontsize',fontsize);
  set(Hx,'fontweight',fontweight);
  
  set(Hy,'fontname',fontname);
  set(Hy,'fontsize',fontsize);
  set(Hy,'fontweight',fontweight);

  set(Hz,'fontname',fontname);
  set(Hz,'fontsize',fontsize);
  set(Hz,'fontweight',fontweight);

  set(Ha,'fontname',fontname);
  set(Ha,'fontsize',fontsize);
  set(Ha,'fontweight',fontweight);
  set(Ht,'fontname',fontname);
  set(Ht,'fontsize',fontsize);
  set(Ht,'fontweight',fontweight);
  
  set(Hy,'VerticalAlignment','bottom');
  set(Hz,'VerticalAlignment','bottom');
  set(Hx,'VerticalAlignment','cap');
  set(Ht,'VerticalAlignment','baseline');
  Hn = get(Ha,'Children');
  n = length(Hn);
  if n > 0
    typ = get(Hn,'Type');
    for j = 1:n
      if strcmp('text',typ(j,:))
        set(Hn(j),'fontname',fontname);
        set(Hn(j),'fontsize',fontsize);
        set(Hn(j),'fontweight',fontweight);
      end
      if strcmp('line',typ(j,:))
        set(Hn(j),'LineWidth',linewid);
        set(Hn(j),'MarkerSize',markersiz);
      end
      if strcmp('patch',typ(j,:))
        set(Hn(j),'LineWidth',linewid);
        set(Hn(j),'MarkerSize',markersiz);
      end
      
    end
  end
end
%           legend:     (Optional, DEF=0) if is 0, doesn't change the legend
%                       if is 1, changes only the lines on the legend
%                       if is 2, changes both the lines and the text
%                       if is 3, changes only the text
if (lgd~=0)
  legh=legend;
  Hn=get(legh,'Children');
  n = length(Hn);
  if n > 0
    typ = get(Hn,'Type');
    for j = 1:n
      if (strcmp('text',typ(j,:)) & ((lgd==2)|(lgd==3)))
        set(Hn(j),'fontname',fontname);
        set(Hn(j),'fontsize',fontsize-2);
        set(Hn(j),'fontweight',fontweight);
    end
      if (strcmp('line',typ(j,:)) & ((lgd==1)|(lgd==2)))
        set(Hn(j),'LineWidth',linewid);
        set(Hn(j),'MarkerSize',markersiz);
      end
    end
  end
end

  
  
  
figure(Hf);