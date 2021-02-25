function [h0,h1] = AddMagnetPlot( istart, iend, han, opt )
%
% ADDMAGNETPLOT -- add a MAD-style bar schematic of magnets to a plot.
%
% [h0,h1] = AddMagnetPlot( istart, iend [,han,opt]) adds a bar over the current plot
%    showing, schematically, the magnets which are in the plot in a style
%    familiar to users of MAD and other such accelerator simulations.
%    Arguments istart and iend are the index numbers of the first and last
%    elements in the plot; h0 and h1 are Matlab handles to the original
%    plot and the magnet schematic plot, respectively.  If element slices
%    are defined, AddMagnetPlot will use the slice data to unsplit the
%    elements in the schematic.  AddMagnetPlot is based on plot_magnets,
%    written by Mark Woodley.
%  han (optional) - use given axis hangle instead of current one
%  opt (optional) - pass additional options
%      'replace' - replace current or given figure handle instead of
%                  plotting on top
%
% Version date:  11-Apr-2006.
%
% Modifications:
% 10/24/2008, GRW, added ability to replace current axis
%
%==========================================================================

global BEAMLINE PS HLINK

if ~exist('han','var') || isempty(han) ,han=gcf;end

% if first arg a handle, assume this is a callback
if ishandle(istart) && isempty(iend)
  fhan=get((get(istart,'Parent')),'Parent');
  if isprop(fhan,'ButtonDownFcn') && ~isempty(get(fhan,'ButtonDownfcn'))
    hObject=fhan; set(fhan,'UserData',str2double(get(istart,'Tag')));
    callbackfun=get(fhan,'ButtonDownfcn');
    callbackfun(hObject,[]);
  end
  return
end

% box heights:

ha=0.5;    % half-height of RF rectangle
hb=1;      % full height of bend rectangle
hq=4;      % full height of quadrupole rectangle
hs=3;      % full height of sextupole rectangle
ho=2;      % full height of octupole rectangle
hr=1;      % half-height of solenoid rectangle

% box colors

ca = 'y' ;
cb = 'b' ;
cq = 'r' ;
cs = 'g' ;
co = 'c' ;
cr = 'm' ;

% master coordinate definitions

barX = zeros(iend-istart+1,1) ;
barWidth = barX ;
barHeight = barX ;
barPolarity = barX ;
barColor = [] ;

nElem = 0 ; count = istart-1 ;

% loop over elements

while (count < iend)

  count = count + 1 ;
  height = 0 ;
  switch BEAMLINE{count}.Class

    case {'LCAV','TCAV'}
      height = ha ;
      color = ca ;
    case 'SBEN'
      height = hb ;
      color = cb ;
    case 'PLENS'
      height = hb ;
      color = cq ;
    case 'QUAD'
      height = hq ;
      color = cq ;
      strength = BEAMLINE{count}.B ;
      if (isfield(BEAMLINE{count},'PS'))
        ps = BEAMLINE{count}.PS ;
        if (ps > 0)
          strength = strength * PS(BEAMLINE{count}.PS).Ampl ;
        end
      end
      if (strength<0)
        barPolarity(nElem+1) = -1 ;
      else
        barPolarity(nElem+1) = 1 ;
      end
    case 'SEXT'
      height = hs ;
      color = cs ;
    case 'OCTU'
      height = ho ;
      color = co ;
    case 'SOLENOID'
      height = hr ;
      color = cr ;

  end
  if (height == 0)
    continue ;
  end
  nElem = nElem + 1 ;

  % look for a slice definition, if found use it

  if (isfield(BEAMLINE{count},'Slices'))
    LastSlice = BEAMLINE{count}.Slices(end) ;
  else
    LastSlice = count ;
  end
  LastSlice = min([LastSlice iend]) ;

  % fill in the data boxes

  barX(nElem) = BEAMLINE{count}.S ;
  if isfield(BEAMLINE{LastSlice},'L')
    barWidth(nElem) = BEAMLINE{LastSlice}.S + BEAMLINE{LastSlice}.L ;
  else
    barWidth(nElem) = BEAMLINE{LastSlice}.S;
  end % if L field
  barHeight(nElem) = height ;
  barColor = [barColor ; color] ;
  
  eletag(nElem) = count;

  % move the counter to the last slice and loop

  count = LastSlice ;

end

% truncate the data vectors

if (nElem == 0)
  warning('Lucretia:AddMagnetPlot:nomags','No magnets to plot') ;
  return
end

barX = barX(1:nElem) ;
barWidth = barWidth(1:nElem) - barX ;
barHeight = barHeight(1:nElem) ;
%   barPolaritry = barPolarity(1:nElem) ;

% squeeze the existing plot(s) to make room for the magnet schematic

bottom=[];
top=[];
frac=1;
if exist('opt','var') && isequal(lower(opt),'replace')
  h0=get(han,'Children');
  if isprop(han,'XLim'); h0=han; end
  v=get(h0(1),'Position');
  bottom=min([bottom,v(2)]);
  top=max([top,v(2)+v(4)]);
  height=top-bottom;
else
  frac = 0.15 ;
  hc=get(han,'Children');
  h0=[];
  for n=1:length(hc)
    if (strcmp(get(hc(n),'Type'),'axes'))
      v=get(hc(n),'Position');
      bottom=min([bottom,v(2)]);
      top=max([top,v(2)+v(4)]);
      h0=[h0;hc(n)];
    end
  end
  for n=1:length(h0)
    v=get(h0(n),'Position');
    if (v(2)>bottom),v(2)=v(2)*(1-frac);end
    v(4)=v(4)*(1-frac);
    set(h0(n),'Position',v)
  end
  height=top-bottom;
end % if ~han

% get horizontal position and width, horizontal axis limits, and title
found=0;
p0=get(h0(1),'Position');
v0=axis(h0(1));
n=0;
while ((n<length(h0))&&(~found))
  n=n+1;
  ht=get(h0(n),'Title');
  tt=get(ht,'String');
  found=(~isempty(tt));
end
if (found)
  set(ht,'String',[])
end
v1=[v0(1),v0(2),-6,6];
p1 = [p0(1),top-height*frac,p0(3),height*frac] ;
if exist('opt','var') && isequal(lower(opt),'replace')
  h1=h0(1);
else
  h1=axes('Position',p1);
end % if replace
plot(h1,v0(1:2),[0,0],'k-')
axis(h1,v1)
set(h1,'Visible','off');

% plot the magnet schematic

hold(h1,'on');

for count = 1:nElem

  %      x = barX(count) - barWidth(count)/2 ;
  x = barX(count)  ;
  w = barWidth(count) ;
  switch barPolarity(count)
    case 0
      y = -barHeight(count) ; h = 2*barHeight(count) ;
    case 1
      y = 0 ; h = barHeight(count) ;
    case -1
      y = -barHeight(count) ; h = barHeight(count) ;
  end
  color = barColor(count) ;
  rhan=rectangle('Position',[x,y,w,h],'FaceColor',color,'Parent',h1,'LineStyle','none') ;
%   rhan=rectangle('Position',[x,y,w,h],'FaceColor',color,'Parent',h1) ;
  % Set internal data identifying this object if clicked on
  set(rhan,'ButtonDownFcn',@AddMagnetPlot)
  set(rhan,'Tag',num2str(eletag(count)));

end

hold(h1,'off');

% Link x-axis for original and magnet bar plots
hl=linkprop([h0(1) h1],'XLim');
if ~isempty(HLINK)
  dhl=[];
  for iH=1:length(HLINK)
    if ~any(ishandle(HLINK(iH).Targets))
      dhl(end+1)=iH;
    end
  end
  HLINK(dhl)=[];
  HLINK(end+1)=hl;
else
  HLINK=hl;
end

% stick the title over everything

if (found)
  title(tt)
  ht=get(h1,'Title');
  set(ht,'Visible','on')
end

if isprop(han,'NextPlot')
  set(han,'NextPlot','replace')
end % if NextPlot property (is a figure window)


