function [varargout] = uisetdate(arg)
%
% uisetdate is designed to select any date among the past current and future years.
%
% uisetdate by itself uses the current date and year as a starting point and returns a string
%           containing the selected date in 'dd-mmm-yyyy' format.
%
% uisetdate(date) uses date as a starting point. date must be in 'dd-mmm-yyyy' format.
%
% [datet,Y,M,D,nD]=uisetdate returns a string containing the date plus the year, the month, the day
%                            and the number of days between the selected date and the 1st January.
%
%   example:
%      if you select the 5th of August of year 2004, the returned values would be
%
%     '05-Aug-2004'
%              2004
%                 8
%                 5
%               218  (218 days between 5th of August and 1st January)
%
% To change the year, just type + or - key while pointer is above the figure (unit step change) or
% type y to select a given year. If you close the figure, all the outputs will be empty. Figure appearance
% may be changed in the "init" function. uisetdate uses the european calendar style (starting on Monday)
% and the modified calendar2.m function written by M. Hendrix (search for "calendar2" in Matlab Central).
% If you prefer the US calendar style, just replace "calendar2" by the original Matlab function "calendar"
% and modify the variable nammed "listJ" in uisetdate.
%
%  Luc Masset (2004)  e-mail: luc.masset@ulg.ac.be
%

%arguments
switch nargin,
case 0,
  [datet,Y,M,D,nD]=init;
  varargout{1}=datet;
  varargout{2}=Y;
  varargout{3}=M;
  varargout{4}=D;
  varargout{5}=nD;
case 1,
 switch arg,
 case 'update',
  update
 case 'validate',
  validate
 case 'changeday',
  changeday
 case 'changemonth',
  changemonth
 case 'changeyear',
  changeyear
 otherwise
  [datet,Y,M,D,nD]=init(arg);
  varargout{1}=datet;
  varargout{2}=Y;
  varargout{3}=M;
  varargout{4}=D;
  varargout{5}=nD;
 end
end

%------------------------------------------------------------------------------
function [datet,Y,M,D,nD] = init(datet)

%arguments
if ~nargin,
 datet=date;
end

%day list
listJ={'Mo','Tu','We','Th','Fr','Sa','Su'};     % uncomment for European calendar style
%listJ={'Su','Mo','Tu','We','Th','Fr','Sa'};    % uncomment for US calendar style

%month list
listM={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};

%year, month and day
Y=str2num(datet(end-3:end));
M=datet(end-7:end-5);
M=strmatch(M,listM);
D=str2num(datet(1:2));

%figure
LF=290;
HF=330;
hfig=figure('units','pixels','position',[0 0 LF HF],'menubar','none', ...
            'numbertitle','off','name','Calendar','resize','off', ...
            'keypressfcn','uisetdate(''changeyear'')','color',[0.7882 0.7843 0.7216], ...
            'paperpositionmode','auto','tag','uisetdate');
set(hfig,'DefaultUIControlHitTest','off')
set(hfig,'DefaultUIControlFontName','arial')
set(hfig,'DefaultUIControlFontSize',10)
set(hfig,'DefaultUIControlFontAngle','normal')
set(hfig,'DefaultUIControlFontWeight','light')
set(hfig,'DefaultUIControlForeGroundColor','k')
set(hfig,'DefaultUIControlBackGroundColor',get(gcf,'color'))
set(hfig,'DefaultUIControlInterruptible','off')
set(hfig,'DefaultUIControlBusyAction','queue')
movegui(hfig,'center');

%frame buttons
h=uicontrol('style','frame','units','pixels','position',[2 2 286 215]);
h=uicontrol('style','frame','units','pixels','position',[2 2 286 185]);
h=uicontrol('style','frame','units','pixels','position',[2 222 286 66]);
h=uicontrol('style','frame','units','pixels','position',[2 292 286 36]);

%current date button
tts='Use +/- keys to change year by unit step. Use y key to set the year';
h=uicontrol('style','text','units','pixels','position',[10 298 270 20],'string',datet, ...
            'horizontalalignment','center','fontsize',12,'tag','date', ...
            'tooltipstring',tts);

%validate button
h=uicontrol('style','pushbutton','units','pixels','position',[255 300 20 20], ...
            'string','>','tooltipstring','Validate current date', ...
            'callback','uisetdate(''validate'')');

%static text buttons for day name
for i=1:7,
 pos=[10+40*(i-1) 190 30 20];
 st=listJ{i};
 h=uicontrol('style','text','units','pixels','position',pos,'string',st, ...
             'horizontalalignment','center');
end

%figure appdata
setappdata(gcf,'year',Y)
setappdata(gcf,'month',M)
setappdata(gcf,'day',D)
%setappdata(gcf,'SelectColor',[0 0 0])
setappdata(gcf,'SelectColor','b')

%update buttons and text
update

%temp button
htemp=uicontrol('style','text','tag','temp','visible','off');

%wait for temp button to be deleted
waitfor(htemp)
if ~ishandle(hfig),
 datet=[];
 Y=[];
 M=[];
 D=[];
 nD=[];
 return
end

%compute outputs
Y=getappdata(gcf,'year');
M=getappdata(gcf,'month');
D=getappdata(gcf,'day');
datet=datestr([Y M D 0 0 0],'dd-mmm-yyyy');
nD=0;  %indice du jour
for i=1:M-1,
 nD=nD+eomday(Y,i);
end
nD=nD+D;
close(hfig)

return

%------------------------------------------------------------------------------
function [] = validate

%delete temp button
delete(findobj('tag','temp','type','uicontrol','parent',gcf))

return

%------------------------------------------------------------------------------
function [] = update
%
% Update buttons and text when changing year, month or day
%

%delete old buttons
delete(findobj('tag','day','type','uicontrol','parent',gcf))
delete(findobj('tag','month','type','uicontrol','parent',gcf))

%year, month, day
Y=getappdata(gcf,'year');
M=getappdata(gcf,'month');
D=getappdata(gcf,'day');
Dmax=eomday(Y,M);
D=min([D Dmax]);
setappdata(gcf,'day',D)

%current month calendar
% C=calendar2(Y,M);     % uncomment for European calendar style
C=calendar(Y,M);     % uncomment for US calendar style

%month buttons
listM={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};
for i=1:2,
 for j=1:6,
  pos=[15+45*(j-1) 260-(i-1)*30 35 20];
  st=listM{6*(i-1)+j};
  h=uicontrol('style','togglebutton','units','pixels','position',pos,'string',st, ...
              'tag','month','callback','uisetdate(''changemonth'')');
 end
end

%day buttons
for i=1:size(C,1),
 for j=1:7,
  if C(i,j),
   pos=[10+40*(j-1) 160-(i-1)*30 30 20];
   st=num2str(C(i,j));
   h=uicontrol('style','togglebutton','units','pixels','position',pos,'string',st, ...
               'tag','day','callback','uisetdate(''changeday'')');
  end
 end
end

%selected month
scolor=getappdata(gcf,'SelectColor');
set(findobj('tag','month','type','uicontrol','parent',gcf),'value',0,'foregroundcolor','k')
st=listM{M};
h=findobj('tag','month','string',st,'type','uicontrol','parent',gcf);
set(h,'value',1,'foregroundcolor',scolor)

%selected day
st=num2str(D);
h=findobj('tag','day','string',st,'type','uicontrol','parent',gcf);
set(h,'value',1,'foregroundcolor',scolor)

%update current date text
st=datestr([Y M D 0 0 0],'dd-mmm-yyyy');
h=findobj('tag','date','type','uicontrol','parent',gcf);
set(h,'string',st)

return

%------------------------------------------------------------------------------
function [] = changeday

st=get(gcbo,'string');
D=str2num(st);
setappdata(gcf,'day',D)
update

return

%------------------------------------------------------------------------------
function [] = changemonth

listM={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};
M=get(gcbo,'string');
M=strmatch(M,listM);
setappdata(gcf,'month',M)
update

return

%------------------------------------------------------------------------------
function [] = changeyear

Y=getappdata(gcf,'year');
cc=get(gcf,'currentcharacter');
switch cc,
case '+',
 Y=Y+1;
case '-',
 Y=Y-1;
case 'y',
 prompt={'Year:'};
 title='Set current year';
 def={sprintf('%i',Y)};
 answer=inputdlg(prompt,title,1,def);
 if isempty(answer),
  return
 end
 Y=str2num(answer{1});
 if isempty(Y),
  return
 end
 Y=round(Y);
otherwise
 return
end
setappdata(gcf,'year',Y)
update

return
