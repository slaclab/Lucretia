function output=qm7bump(varargin)
persistent knob maxvals firstCall

output=[];

% Add bumpgui to path if first call
if isempty(firstCall)
  useApp('bumpgui')
  firstCall=false;
end

% Update model
FlHwUpdate;

% Command options
if nargin==0 % constructor is to generate knob
  varargin{1}='gen'; 
elseif isempty(knob)
  [knob maxvals] = createKnob;
end
switch lower(varargin{1})
  case('gen')
    [knob maxvals] = createKnob;
  case('set')
    if nargin~=2 || length(varargin{2})~=4; error('Wrong input arguments'); end;
    if any(abs(varargin{2})>maxvals); error('Max knob values are: %g %g %g %g',maxvals); end;
    setKnob(knob,'abs',varargin{2});
  case('reset')
    createKnob('resetcor');
  case('location')
    output=setKnob(knob,'location');
  case('setrel')
    error('this not implemented yet')
  case('setinitial')
    createKnob('setinitial');
  case('getknob')
    output=knob;
  case('provideknob')
    knob=varargin{1};
  case('setorigin')
    setKnob(knob,'setorigin');
  otherwise
    error('Unknown command')
end

function output=setKnob(knob,cmd,val)

kstr={'x' 'xp' 'y' 'yp'};
switch cmd
  case 'location'
    output=[knob.r.x.Value knob.r.xp.Value knob.r.y.Value knob.r.yp.Value];
  case 'abs'
    for ival=1:4
      stat=SetMultiKnob(['knob.r.',kstr{ival}],val(ival),1);
      if stat{1}~=1; error(stat{2}); end;
      stat=SetMultiKnob(['knob.x.',kstr{ival}],val(ival),1);
      if stat{1}~=1; error(stat{2}); end;
    end
end
    

function [knob maxvals] = createKnob(varargin)
persistent reqID origin
global BEAMLINE PS FL

if nargin>0 && isequal(varargin{1},'getorigin')
  knob=origin; return
end

% Create knob
bumploc.r=median(findcells(BEAMLINE,'Name','QM7R1'));
bumploc.x=median(findcells(BEAMLINE,'Name','QM7RX'));
cors_x.r=[findcells(BEAMLINE,'Name','ZH101R') findcells(BEAMLINE,'Name','ZH102R') findcells(BEAMLINE,'Name','ZH10R')];
cors_y.r=[findcells(BEAMLINE,'Name','ZV100R') findcells(BEAMLINE,'Name','ZV10R') findcells(BEAMLINE,'Name','ZV11R')];
cors_x.x=[findcells(BEAMLINE,'Name','ZH101RX') findcells(BEAMLINE,'Name','ZX1X') findcells(BEAMLINE,'Name','ZH1X')];
cors_y.x=[findcells(BEAMLINE,'Name','ZV100RX') findcells(BEAMLINE,'Name','ZV1X') findcells(BEAMLINE,'Name','ZV2X')];
knob.r.x = bumpgui(1,3,bumploc.r,cors_x.r);
knob.r.xp = bumpgui(2,3,bumploc.r,cors_x.r);
knob.r.y = bumpgui(3,3,bumploc.r,cors_y.r);
knob.r.yp = bumpgui(4,3,bumploc.r,cors_y.r);
knob.x.x = bumpgui(1,3,bumploc.x,cors_x.x);
knob.x.xp = bumpgui(2,3,bumploc.x,cors_x.x);
knob.x.y = bumpgui(3,3,bumploc.x,cors_y.x);
knob.x.yp = bumpgui(4,3,bumploc.x,cors_y.x);

% First knob coefficient should be same
c1=knob.r.x.Channel(1).Coefficient;
c2=knob.r.xp.Channel(1).Coefficient;
c3=knob.r.y.Channel(1).Coefficient;
c4=knob.r.yp.Channel(1).Coefficient;
for ichan=1:3
  knob.x.x.Channel(ichan).Coefficient=knob.x.x.Channel(ichan).Coefficient/(knob.x.x.Channel(ichan).Coefficient/c1);
  knob.x.xp.Channel(ichan).Coefficient=knob.x.xp.Channel(ichan).Coefficient/(knob.x.xp.Channel(ichan).Coefficient/c2);
  knob.x.y.Channel(ichan).Coefficient=knob.x.y.Channel(ichan).Coefficient/(knob.x.y.Channel(ichan).Coefficient/c3);
  knob.x.yp.Channel(ichan).Coefficient=knob.x.yp.Channel(ichan).Coefficient/(knob.x.yp.Channel(ichan).Coefficient/c4);
end

% half first PS in bump as really is 2 correctors (bump will add in each
% case)
knob.r.x.Channel(1).Coefficient=knob.r.x.Channel(1).Coefficient/2;
knob.r.xp.Channel(1).Coefficient=knob.r.xp.Channel(1).Coefficient/2;
knob.r.y.Channel(1).Coefficient=knob.r.y.Channel(1).Coefficient/2;
knob.r.yp.Channel(1).Coefficient=knob.r.yp.Channel(1).Coefficient/2;
knob.x.x.Channel(1).Coefficient=knob.x.x.Channel(1).Coefficient/2;
knob.x.xp.Channel(1).Coefficient=knob.x.xp.Channel(1).Coefficient/2;
knob.x.y.Channel(1).Coefficient=knob.x.y.Channel(1).Coefficient/2;
knob.x.yp.Channel(1).Coefficient=knob.x.yp.Channel(1).Coefficient/2;

% Power supply list
pslist=arrayfun(@(x) BEAMLINE{x}.PS,[cors_x.r cors_y.r cors_x.x cors_y.x]);

% Get max bump (based on 5A max correction)
kstr={'x' 'xp' 'y' 'yp'};
for iknob=1:4
  [max_r ind_r]=max(abs([knob.r.(kstr{iknob}).Channel(:).Coefficient]));
  [max_x ind_x]=max(abs([knob.r.(kstr{iknob}).Channel(:).Coefficient]));
  if iknob<3
    if length(FL.HwInfo.PS(BEAMLINE{cors_x.r(ind_r)}.PS).conv)>1
      maxkick=max(abs(FL.HwInfo.PS(BEAMLINE{cors_x.r(ind_r)}.PS).conv(2,:)));
    else
      maxkick=5*FL.HwInfo.PS(BEAMLINE{cors_x.r(ind_r)}.PS).conv;
    end
    maxknob_r(iknob)=abs(maxkick)/abs(max_r);
    if length(FL.HwInfo.PS(BEAMLINE{cors_x.x(ind_x)}.PS).conv)>1
      maxkick=max(abs(FL.HwInfo.PS(BEAMLINE{cors_x.x(ind_x)}.PS).conv(2,:)));
    else
      maxkick=5*FL.HwInfo.PS(BEAMLINE{cors_x.x(ind_x)}.PS).conv;
    end
    maxknob_x(iknob)=abs(maxkick)/abs(max_x);
  else
    if length(FL.HwInfo.PS(BEAMLINE{cors_y.r(ind_r)}.PS).conv)>1
      maxkick=max(abs(FL.HwInfo.PS(BEAMLINE{cors_y.r(ind_r)}.PS).conv(2,:)));
    else
      maxkick=5*FL.HwInfo.PS(BEAMLINE{cors_y.r(ind_r)}.PS).conv;
    end
    maxknob_r(iknob)=abs(maxkick)/abs(max_r);
    if length(FL.HwInfo.PS(BEAMLINE{cors_y.x(ind_x)}.PS).conv)>1
      maxkick=max(abs(FL.HwInfo.PS(BEAMLINE{cors_y.x(ind_x)}.PS).conv(2,:)));
    else
      maxkick=5*FL.HwInfo.PS(BEAMLINE{cors_y.x(ind_x)}.PS).conv;
    end
    maxknob_x(iknob)=abs(maxkick)/abs(max_x);
  end
  maxvals=min([maxknob_r; maxknob_x]);
end
if nargin==0
  fprintf('Knob Generated - Max settings based on 5A max corrector currents [x/x''/y/y''] (mm/mrad): %.2g %.2g %.2g %.2g\n',maxvals.*1e3);
  disp('Correctors in use:')
  for ips=[cors_x.r cors_y.r cors_x.x cors_y.x]
    fprintf('%s ',BEAMLINE{ips}.Name)
  end
  fprintf('\n')
end

% set initial corrector readings for reset
if isempty(origin) || (nargin>0 && isequal(varargin{1},'setinitial'))
  origin=[PS(pslist).Ampl];
elseif nargin>0 && isequal(varargin{1},'resetcor')
  for ips=1:length(pslist)
    PS(pslist(ips)).SetPt=origin(ips);
  end
  stat=PSTrim(pslist,1); if stat{1}~=1; error(stat{2}); end;
  return
end

% Get access rights
if isempty(reqID)
  [stat reqID] = AccessRequest({[] pslist []});
  if stat{1}~=1; error(stat{2}); end;
  if ~reqID; error('No access granted'); end;
else
  [stat reqID_stat] = AccessRequest('status',reqID);
  if stat{1}~=1 || ~reqID_stat
    reqID=[]; %#ok<NASGU>
    error('No access granted- re-request required')
  end
end