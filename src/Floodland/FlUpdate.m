function stat=FlUpdate(archive,dataServer,varargin)
% Update Lucretia data structures from EPICS CA
% If archive date given, restore from Channel Archiver date if possible
% instead, must provide archive date (archive) in Matlab datenum format
% and data server url (dataServer)
% FlUpdate('archinit',dataServer) initialises archive parameters and get date-time
% info

global FL
persistent firstCall archInit

stat{1}=1;

% Force Update?
if nargin==1 && exist('archive','var') && archive
  forceReq=true;
  archive=false;
else
  forceReq=false;
end

% Additional archive related args
if nargin>2
  % use progressbar
  if ~isempty(varargin{1}) && varargin{1}
    rephan=true;
  else
    rephan=[];
  end
  if nargin>3
    updateList=varargin{2};
  else
    updateList=[];
  end
else
  rephan=[];
  updateList=[];
end

% Archiver Initialisation
if ~exist('archive','var') || (islogical(archive) && ~archive)
  archive=false;
elseif isempty(archInit) || strcmp(archive,'archInit')
  stat=chanArchInit(dataServer); if stat{1}~=1; return; end;
  archInit=true;
elseif ~exist('dataServer','var') || ~ischar(dataServer)
  error('Incorrect archive request parameters')
end

if strcmp(archive,'archInit')
  return
end

% Types of hardware access possible - depends upon expt
if ~isfield(FL,'HwInfo'); error('No FL.HwInfo set'); end;
fnames=fieldnames(FL.HwInfo);
types={};
for ifn=1:length(fnames)
  types{end+1}=fnames{ifn};
end

% Form command lists to get data from EPICS and data structure lists to
% return the values to
[comstack_get comstack_put comstack_putVals comstack_monitor comdes] = getPut(types);
comstack_get=regexprep(comstack_get,'\.VAL','');
comstack_monitor=regexprep(comstack_monitor,'\.VAL','');
comdes=regexprep(comdes,'\.VAL','');

% Any put commands needed
if ~isempty(comstack_put) && ~archive
  FlCA('lcaPut',comstack_put,comstack_putVals);
end % if ~empty put cell
if isempty(firstCall)
  % Setup trigger PV monitor
  FlCA('lcaSetMonitor',FL.triggerPV,1);
  % Setup monitor PVs for GIRDER/PS elements
  FlCA('lcaSetMonitor',comstack_monitor,1);
  FlCA('lcaSetMonitor',comdes,1);
end % if firstcall, setup monitor

% Wait for PV trigger
try
  if ~archive && ~isempty(firstCall) && ~forceReq
    for imoni=1:length(FL.triggerPV)
      try
        lcaNewMonitorWait(FL.triggerPV{imoni});
      catch ME
        warning(ME.message);
      end
    end
  end
  
  % Fill Lucretia database values
  if archive % fill database from archiver
    [retVals tsVals]=chanArchGet(archive,comstack_get,dataServer,rephan,updateList);
    desVals=nan(size(retVals));
    if isequal(retVals,-1)
      stat{1}=-1; stat{2}='Archive Update Abort Requested'; return
    end
  else % fill database from live control system
    % Get changed data
    if forceReq || isempty(firstCall) % make sure update from all PVs on first call or force request
      moniUpdate=true(size(comstack_monitor));
      moniUpdateDes=true(size(comdes));
    else
      moniUpdate=FlCA('lcaNewMonitorValue',comstack_monitor); moniUpdate=logical(moniUpdate(~isnan(moniUpdate)&~isinf(moniUpdate)));
      moniUpdateDes=FlCA('lcaNewMonitorValue',comdes); moniUpdateDes=logical(moniUpdateDes(~isnan(moniUpdateDes)&~isinf(moniUpdateDes)));
    end
    moniOverlap=ismember(comstack_get,{comstack_monitor{moniUpdate}}); %#ok<*CCAT1>
    % Get PV data
    [newRetVals epicsTimeStamp]=FlCA('lcaGet',{comstack_get{moniOverlap}}',1);
    if (sum(moniUpdateDes)>0)
      newDesVals=FlCA('lcaGet',{comdes{moniUpdateDes}}',1);
    end
    tsVals=nan(size(comstack_get));
    retVals=nan(size(comstack_get));
    retVals(moniOverlap)=newRetVals;
    tsVals(moniOverlap)=epicsTimeStamp;
    desVals=nan(size(comdes));
    % DES values constantly updating in sim mode- don't update in this
    % case otherwise can't set anything
    if ~FL.SimMode && sum(moniUpdateDes)>0
      desVals(moniUpdateDes)=newDesVals;
    end
  end
catch ME
  stat{1}=-1; stat{2}=ME.message;
  warning('Floodland:FlUpdate:lca_err',['Error getting new EPICS values: ',ME.message]);
  firstCall=true;
  return
end % try/catch

% Update Lucretia data structures
getPut(types,retVals,tsVals,desVals);

firstCall=true;

% ----------------------------------------------------------------
% Internal functions
% ----------------------------------------------------------------
function [retVals epicsTimeStamp] = chanArchGet(archive,pvs,dataServer,rephan,updateList)
global FL

% Use progress bar?
pbh=[];
if exist('rephan','var') && ~isempty(rephan)
  gui_active(1);
  pbh=progressbar([],0,'Archive Restore Progress');
  drawnow('expose')
end

% Use update list?
ul=[];
if exist('updateList','var')
  ul=updateList;
end

% Time period to search around given time
ts=FL.chanArch.timeSearch;

if ~isempty(ul); pvs={pvs{ul}}; end;
for ipv=1:length(pvs)
  % Update progressbar
  if ~isempty(pbh)
    progressbar(pbh,1/length(pvs));
  end
  % Check for abort request
  if ~gui_active
    retVals=-1;
    return
  end
  try
    % get value closest in time to that requested (search +/- 10 mins)
    [times,micros,values]=ml_arch_get(dataServer,1,regexprep(pvs{ipv},'\.VAL',''),addtodate(archive,-ts,'minute'),addtodate(archive,ts,'minute'),1,100000);
    if isempty(values) || sum(sum(isnan(values)))==numel(values)
      error('No values')
    end
    [timeVal,timeInd]=min(abs(archive-times));
    retVals(ipv,1)=values(timeInd,1);
    % Form epics timestamp (seconds since jan 1 1990) (imag part = ns)
    epicsTimeStamp(ipv)=complex(etime(datevec(timeVal),datevec(datenum(1990,1,1))),micros(timeInd)*1e-3);
  catch % fail if pv not founnd or no values in time range
    retVals(ipv,1)=0;
    epicsTimeStamp(ipv)=complex(0,0);
  end
end
% delete progress bar
if ~isempty(pbh)
  progressbar(pbh,-1);
end
function stat=chanArchInit(dataServer)
global FL

stat{1}=1;

if ~isfield(FL,'chanArch') || ~isfield(FL.chanArch,'info')
  % Check extensions have been set up
  extDir=getenv('EPICS_EXTENSIONS');
  if isempty(extDir)
    stat{1}=-1;
    stat{2}='No EPICS_EXTENSIONS environment variable set';
    return
  elseif ~exist(fullfile(extDir,'src','ChannelArchiver','Matlab',['O.',getenv('EPICS_HOST_ARCH')]),'dir')
    stat{1}=-1;
    stat{2}='Channel Archiver Matlab software not found';
    return
  end
  addpath(fullfile(extDir,'src','ChannelArchiver','Matlab',['O.',getenv('EPICS_HOST_ARCH')]))
  addpath(fullfile(extDir,'src','ChannelArchiver','Matlab','util'))
  % Default time window to check around given restore point
  if ~isfield(FL,'chanArch') || ~isfield(FL.chanArch,'timeSearch')
    FL.chanArch.timeSearch=10; % minutes
  end
end
% Check out archiver data server is running and store info and available
% PV names and time vals
try
  [FL.chanArch.info.ver,FL.chanArch.info.desc, FL.chanArch.info.hows]=ArchiveData(dataServer,'info');
  [FL.chanArch.names, FL.chanArch.starts, FL.chanArch.ends]=ml_arch_names(dataServer, 1);
catch ME
  stat{1}=-1;
  stat{2}=['Error accessing EPICS Channel Archiver Data Server: ',ME.message];
  return
end

function [comstack_get comstack_put comstack_putVals comstack_monitor comdes] = getPut(types,retVals,ts,desVals)
global FL
persistent mver retValsLast csp cspv csget csmoni cdes tsLast

% Initialise retValsLast
if isempty(retValsLast) && exist('retVals','var')
  retValsLast=nan(size(retVals));
end

% get matlab version
if isempty(mver)
  mver=ver('MATLAB');
end

comstack_get={};cind=0; comstack_monitor={}; icg=0; ides=0;
comdes={}; comstack_put={}; comstack_putVals=[];
for itype=1:length(types)
  for ilist=1:length(FL.HwInfo.(types{itype}))
    hw=FL.HwInfo.(types{itype})(ilist);
    if ~isfield(hw,'pvname'); continue; end; % no need to fill data if no h/w
    % Get new value
    if isfield(hw,'preCommand') && ~isempty(hw.preCommand) && ~isempty(hw.preCommand{1})
      [nrw ncol]=size(hw.preCommand);
      if isempty(csp)
        for icmd=1:ncol
          cind=cind+1;
          comstack_put{cind,1}=hw.preCommand{1,icmd}{1}; %#ok<AGROW>
          comstack_putVals(cind,1)=hw.preCommand{1,icmd}{2}; %#ok<AGROW>
        end % for icmd
        csp=comstack_put;
        cspv=comstack_putVals;
      else
        comstack_put=csp;
        comstack_putVals=cspv;
      end
    end % if precommand
    thisDesVal=nan;
    if nargin>1 && iscell(retVals)
      thisVal={};
    else
      thisVal=[];
    end % if retVal cell array
    if isempty(hw.pvname) || isequal(hw.conv,0)
      try
        hw.pvname=hw.badpv;
      catch
        continue
      end
    end % skip if no EPICS variable for this or if want to ignore (conv set to 0)
    if ~isempty(csget)
      comstack_get=csget;
      comstack_monitor=csmoni;
      comdes=cdes;
      newcsget=false;
    else
      newcsget=true;
    end
    if ~isempty(hw.pvname{1})
      for ipv=1:length(hw.pvname{1})
        if newcsget
          comstack_get{end+1,1}=hw.pvname{1}{ipv}; %#ok<AGROW>
        end
        % Read in DES values to setpt?
        if length(hw.pvname)>2 && ~isempty(hw.pvname{3})
          if newcsget
            comdes{end+1,1}=hw.pvname{3}{ipv};
          end
          ides=ides+1;
        end
        icg=icg+1;
        % Choose what to monitor
        if newcsget
          comstack_monitor{end+1,1}=hw.pvname{1}{ipv};
        end
        if nargin>1
          if iscell(retVals)
            if isnumeric(retVals{icg,1})
              thisVal(ipv)=retVals{icg,1};
              if length(hw.pvname)>2 && ~isempty(hw.pvname{3})
                thisDesVal(ipv)=desVals{ides,1};
              else
                thisDesVal(ipv)=nan;
              end
            else
              thisVal(ipv)=str2double(retVals{icg,1});
              if length(hw.pvname)>2 && ~isempty(hw.pvname{3})
                thisDesVal(ipv)=str2double(desVals{ides,1});
              else
                thisDesVal(ipv)=nan;
              end
            end % if numeric
          else
            thisVal(ipv)=retVals(icg,1);
            if length(hw.pvname)>2 && ~isempty(hw.pvname{3})
              thisDesVal(ipv)=desVals(ides,1);
            else
              thisDesVal(ipv)=nan;
            end
          end % if retVals cell array
          valInd(ipv)=icg;
          % Keep vector of last good (non-nan) updates
          if ~isnan(thisVal(ipv)) && ~isinf(thisVal(ipv))
            retValsLast(icg,1)=thisVal(ipv);
          elseif length(retValsLast)<icg
            retValsLast(icg,1)=0;
          end
          if ~isnan(ts(icg)) && ~isinf(ts(icg))
            tsLast(icg,1)=ts(icg);
          elseif length(tsLast)<icg
            tsLast(icg,1)=0;
          else
            ts(icg)=tsLast(icg);
          end
        end % if nargin>1
      end % for ipv
      % Don't update any NaN values- means bad reading from EPICS, or EPICS
      % value not changed since last read
      if all(isnan(thisVal)) && all(isnan(thisDesVal))
        continue
      end
      % If this device has mixture of updated and not-updated values, use
      % values from last time for non-updated values
      if any(isnan(thisVal)) || any(isinf(thisVal))
        thisVal(isnan(thisVal)|isinf(thisVal))=retValsLast(valInd(isnan(thisVal)|isinf(thisVal)));
      end
      % Update value in Lucretia Globals
      if nargin>1
        if isempty(hw.conv); error('Empty conv field whilst trying to get control value in FlUpdate: %s %d', types{itype}, ilist); end;
        putVal(types{itype},ilist,thisVal,hw.conv,thisDesVal);
        % Put epics time stamp as Matlab datenum format in gui requested
        % local time
        FL.HwInfo.(types{itype})(ilist).TimeStamp=epicsts2mat(ts(icg));
      end % if putting
    end
  end % for ilist
end % for itype
csget=comstack_get;
csmoni=comstack_monitor;
cdes=comdes;

function putVal(type,hwPtr,val,conv,desVal)
global INSTR PS GIRDER BEAMLINE FL KLYSTRON
persistent lastval
if isempty(lastval)
  lastval=cell(1,length(GIRDER));
end
switch type
  case 'KLYSTRON'
    KLYSTRON(hwPtr).Ampl=val*conv;
  case 'GIRDER'
    if length(val)<length(GIRDER{hwPtr}.MoverPos)
      error('Error assigning GIRDER val- length mismatch!')
    end % arg length check
    for idim=1:length(GIRDER{hwPtr}.MoverPos)
      GIRDER{hwPtr}.MoverPos(idim)=val(idim).*conv(idim);
      FL.HwInfo.GIRDER(hwPtr).controlVal(idim)=val(idim);
      % Update setpt
      if length(desVal)>=idim && ~isnan(desVal(idim)) && ~isinf(desVal(idim))
        GIRDER{hwPtr}.MoverSetPt(idim)=desVal(idim).*conv(idim);
      end
    end % for idim
    % Set GIRDER range for current position
    if isempty(lastval{hwPtr})|| any(abs(lastval{hwPtr}-val)>10)
      FlSetMoverLimits(hwPtr);
    end
    lastval{hwPtr}=val;
  case 'PS'
    % If corrector-> converted val is radians, else B units
    % If corrector is really bend trim winding, need to find total kick of
    % bend I + trim I and subtract bend only I kick
    % (flip lookup table sign if bipolar and only given positive
    % lookup values)
    if FL.HwInfo.PS(hwPtr).unipolar && length(conv)>1 && length(val)==2 && val(2)<0 && ~(min(conv(1,:))<0)
      if sign(val(1))~=sign(sum(val))
        val=[0 0];
        warndlg(['Magnitude of trim > main current for PS: ',num2str(hwPtr),' Setting -> 0'],'FlUpdate Warning','modal')
      end
    elseif ~FL.HwInfo.PS(hwPtr).unipolar && length(conv)>1 && sum(val)<0 && conv(1,end)>0
      conv=-conv;
    elseif ~FL.HwInfo.PS(hwPtr).unipolar && length(conv)>1 && sum(val)>0 && conv(1,end)<0
    end
    % If want to keep design bend angles in EXT & FF don't set them here
    if strcmp(BEAMLINE{PS(hwPtr).Element(1)}.Class,'SBEN') && PS(hwPtr).Element(1)>FL.SimModel.extStart && FL.EScaleATF2Bends
      return
    end
    if ~isempty(strfind(BEAMLINE{PS(hwPtr).Element(1)}.Class,'COR'))
      if length(val)==2
        if ~isfield(FL.HwInfo.PS(hwPtr),'nt_ratio')
          error('No ratio of primary:secondary turns provided for PS: %d',hwPtr)
        elseif isempty(FL.HwInfo.PS(hwPtr).nt_ratio)
            FL.HwInfo.PS(hwPtr).nt_ratio=1;
        end % if no FL.HwInfo.PS.nt_ratio
        if length(conv)==1
          PS(hwPtr).Ampl=(val(1)+val(2)*FL.HwInfo.PS(hwPtr).nt_ratio)*conv-val(1)*conv;
          FL.HwInfo.PS(hwPtr).controlVal=val(2);
          if isequal(size(desVal),size(val)) && any(~isnan(desVal)) && ~all(isnan(desVal))
            desVal(isnan(desVal))=val(isnan(desVal));
          end
          if ~any(isnan(desVal)) && isequal(size(desVal),size(val))
            PS(hwPtr).SetPt=(desVal(1)+desVal(2)*FL.HwInfo.PS(hwPtr).nt_ratio)*conv-desVal(1)*conv;
          end
        else
          PS(hwPtr).Ampl=interp1(conv(1,:),conv(2,:),val(1)+val(2)*FL.HwInfo.PS(hwPtr).nt_ratio,'linear')-...
            interp1(conv(1,:),conv(2,:),val(1),'linear');
          FL.HwInfo.PS(hwPtr).controlVal=val(2);
          if isequal(size(desVal),size(val)) && any(~isnan(desVal)) && ~all(isnan(desVal))
            desVal(isnan(desVal))=val(isnan(desVal));
          end
          if ~any(isnan(desVal)) && isequal(size(desVal),size(val))
            PS(hwPtr).SetPt=interp1(conv(1,:),conv(2,:),desVal(1)+desVal(2)*FL.HwInfo.PS(hwPtr).nt_ratio,'linear')-...
              interp1(conv(1,:),conv(2,:),desVal(1),'linear');
            FL.HwInfo.PS(hwPtr).controlValDes=desVal(1);
          end
        end % if conv length 1
      else
        if length(conv)==1
          PS(hwPtr).Ampl=val.*conv;
          FL.HwInfo.PS(hwPtr).controlVal=val;
          if ~isnan(desVal) && ~isinf(desVal)
            PS(hwPtr).SetPt=desVal.*conv;
            FL.HwInfo.PS(hwPtr).controlValDes=desVal;
          end
        else
          PS(hwPtr).Ampl=interp1(conv(1,:),conv(2,:),val,'linear');
          FL.HwInfo.PS(hwPtr).controlVal=val;
          if ~isnan(desVal) && ~isinf(desVal)
            PS(hwPtr).SetPt=interp1(conv(1,:),conv(2,:),desVal,'linear');
            FL.HwInfo.PS(hwPtr).controlValDes=desVal;
          end
        end % simple conv factor or lookup table?
      end % if length val 2
      % If there is an off status, force PS -> 0
      if ~isempty(strfind(FL.mode,'trusted')) && isfield(FL.HwInfo.PS(hwPtr),'status') && ~isempty(FL.HwInfo.PS(hwPtr).status)
        psStat=int8(lcaGet(FL.HwInfo.PS(hwPtr).status));
        if bitget(psStat,3)
          PS(hwPtr).Ampl=0;
          PS(hwPtr).SetPt=0;
        end
      end
    else % not COR
      if length(val)>1 % ring quads have main and trim coils
        if ~isfield(FL.HwInfo.PS(hwPtr),'nt_ratio')
          error('No ratio of primary:secondary turns provided for PS: %d',hwPtr)
        elseif isempty(FL.HwInfo.PS(hwPtr).nt_ratio)
            FL.HwInfo.PS(hwPtr).nt_ratio=1;
        end % if no FL.HwInfo.PS.nt_ratio
        if isequal(size(desVal),size(val)) && any(~isnan(desVal)) && ~all(isnan(desVal))
          desVal(isnan(desVal))=val(isnan(desVal));
        else
          desVal=nan;
        end
        val=val(1)+FL.HwInfo.PS(hwPtr).nt_ratio*val(2);
        if ~all(isnan(desVal))
          desVal=desVal(1)+FL.HwInfo.PS(hwPtr).nt_ratio*desVal(2);
        else
          desVal=nan;
        end
      end % if ring quad
      if length(conv)==1
        PS(hwPtr).Ampl=(val*conv)/abs(BEAMLINE{PS(hwPtr).Element(1)}.B(1));
        FL.HwInfo.PS(hwPtr).controlVal=val;
        if ~isnan(desVal) && ~isinf(desVal)
          PS(hwPtr).SetPt=(desVal*conv)/abs(BEAMLINE{PS(hwPtr).Element(1)}.B(1));
          FL.HwInfo.PS(hwPtr).controlValDes=desVal;
        end
      else
        if (isfield(FL.SimModel,'XQuadPSList') && ...
            ~isempty(FL.SimModel.XQuadPSList) && ...
            ismember(hwPtr,FL.SimModel.XQuadPSList))
          PS(hwPtr).Ampl=interp1(conv(1,:),conv(2,:),val,'linear')/ ...
            abs(BEAMLINE{PS(hwPtr).Element(1)}.B(2));
          FL.HwInfo.PS(hwPtr).controlVal=val;
          if ~isnan(desVal) && ~isinf(desVal)
            PS(hwPtr).SetPt=interp1(conv(1,:),conv(2,:),desVal,'linear')/ ...
              abs(BEAMLINE{PS(hwPtr).Element(1)}.B(2));
            FL.HwInfo.PS(hwPtr).controlValDes=desVal;
          end
        else
          PS(hwPtr).Ampl=interp1(conv(1,:),conv(2,:),val,'linear')/ ...
            abs(BEAMLINE{PS(hwPtr).Element(1)}.B(1));
          FL.HwInfo.PS(hwPtr).controlVal=val;
          if ~isnan(desVal) && ~isinf(desVal)
            PS(hwPtr).SetPt=interp1(conv(1,:),conv(2,:),desVal,'linear')/ ...
              abs(BEAMLINE{PS(hwPtr).Element(1)}.B(1));
            FL.HwInfo.PS(hwPtr).controlValDes=desVal;
          end
        end
      end % simple conv factor or lookup table?
      % apply fudge factor if one exists and it isn't empty
      if (isfield(FL.HwInfo.PS(hwPtr),'fudge')&& ...
         (~isempty(FL.HwInfo.PS(hwPtr).fudge)))
        fudge=FL.HwInfo.PS(hwPtr).fudge;
        if (length(BEAMLINE{PS(hwPtr).Element(1)}.B)>1)
        % special case: if length(B)>1, apply fudge factor to B(2) only
          for nf=1:length(PS(hwPtr).Element)
            BEAMLINE{PS(hwPtr).Element(nf)}.B(2)= ...
              BEAMLINE{PS(hwPtr).Element(nf)}.B(2)*fudge;
          end
        else
          PS(hwPtr).Ampl=PS(hwPtr).Ampl*fudge;
        end
      end
      if isnan(PS(hwPtr).Ampl) || isinf(PS(hwPtr).Ampl); PS(hwPtr).Ampl=0; end;
    end % if corrector
    if strcmp(FL.currentReadMethod,'DES')
      PS(hwPtr).Ampl=PS(hwPtr).SetPt;
    end
  case 'INSTR'
    if length(conv)>1 && length(conv)~=length(val)
      error('INSTR data length and conv value mismatch!');
    elseif length(conv)==1
      conv=ones(size(val)).*conv;
    end % if conv not match with val
    if any(isnan(val)); val(isnan(val))=0; end;
    if length(val)>1
      for idim=1:length(val)
        INSTR{hwPtr}.Data(idim)=val(idim).*conv(idim);
        FL.HwInfo.INSTR(hwPtr).controlVal(idim)=val(idim);
        % make sure tmits always absolute nums
        if idim==3
          INSTR{hwPtr}.Data(idim)=abs(INSTR{hwPtr}.Data(idim));
          FL.HwInfo.INSTR(hwPtr).controlVal(idim)=val(idim);
        end
      end % for idim
    else
      INSTR{hwPtr}.Data(3)=abs(val.*conv);
      FL.HwInfo.INSTR(hwPtr).controlVal(3)=abs(val);
    end
end % switch type
