function varargout=FlCA(varargin)
% FLCA
% lcaSetMonitor / lcaNewMonitorValue / lcaGet / lcaPut / lcaPutNoWait with error catching for missing PVs
% If any PVs not found, they are removed from FL.HwInfo hardware list,
% e.g. [vals timestamps] = FlCA('lcaGet','pvname',1)
% Missing values are replaced with NaN's
% Missing PV's are remembered and excluded from future calls
%
% FlCA('aidaGet',pvlist) / FlCA('aidaPut',pvlist,data,trimType)
%   if wanting BPM vector (pvname ending with '//BPMS'):
%      FlCA('aidaGet',aidapv,aidaNames,chNames,bpmd)
%   for aidaPut, trimtype='PTRB' | 'TRIM'
%
%   FlCA('restore',type,ind) or FlCA('restoreAll')
% - restore some or all of pv's previously excluded
%   e.g. FlCA('restore','PS',33) or FlCA('restore','INSTR',133) or
%   FlCA('restore','GIRDER',11)
%
%   missingPVs = FlCA('pvlist')
%     - Get list of any missing PV's
persistent removedPV removedPVinfo da
import edu.stanford.slac.err.*;
import edu.stanford.slac.aida.lib.da.*;
import edu.stanford.slac.aida.lib.util.common.*;

% init
if isempty(removedPV); removedPV={}; end;
if isempty(removedPVinfo); removedPVinfo={}; end;

% Return bad pv list if requested
if strcmp(varargin{1},'pvlist')
  varargout{1}=removedPV;
  return
end

% arg check
allowedCMD={'lcaSetMonitor' 'lcaPut' 'lcaPutNoWait' 'lcaNewMonitorValue' 'lcaGet' 'restore' 'restoreAll' 'lcaNewMonitorWait' ...
  'aidaGet' 'aidaPut'};
if ~ismember(varargin{1},allowedCMD)
  error('Invalid command');
end

% Deal with channel access request
ismissing=true;
% remove any removedPV entries from list
nanout=nan(size(varargin{2}));
newlist=~ismember(varargin{2},removedPV);
pvlist=varargin{2}(newlist)';
varargout{1}=nanout; varargout{2}=nanout;
if ~any(newlist); return; end;
while ismissing && ~isempty(pvlist)
  if strcmp(varargin{1},'lcaSetMonitor') || strcmp(varargin{1},'lcaPut') || strcmp(varargin{1},'lcaPutNoWait')
    if nargin>2
      evalc([varargin{1},'(pvlist(:),varargin{3});']);
    else
      evalc([varargin{1},'(pvlist(:));']);
    end
  elseif strcmp(varargin{1},'lcaNewMonitorValue')
    if nargin>2
      output=eval([varargin{1},'(pvlist(:),varargin{3});']);
    else
      output=eval([varargin{1},'(pvlist(:));']);
    end
    nanout(newlist)=output;
    varargout{1}=nanout;
  elseif ~isempty(regexp(varargin{1},'^aida','once'))
    % Initialise aida if not yet done
    if isempty(da)
      da = DaObject();
    end
    if strcmp(varargin{1},'aidaGet')
      % any repeated PV names, assume one for all
      [uniPV uniI]=unique(pvlist);
      dataout=NaN(length(pvlist),1);
      for ipv=1:length(uniPV)
        % Request vector of BPM readings
        if ~isempty(regexp(uniPV{ipv},'//BPMS$', 'once'))
          if nargin<5
            error('Need to pass name list, channel list and control ID for use with getting BPMS')
          end
          aidaNames=varargin{3}; chNames=varargin{4}; bpmd=varargin{5}; if ~iscell(bpmd); bpmd={bpmd}; end;
          [names X Y Q]=aidaBPM(da,uniPV{ipv},bpmd{uniI(ipv)});
          if isempty(names); return; end; % AIDA data acquition failed?
          for ipv2=find(ismember(pvlist,uniPV))
            iname=find(ismember(names,aidaNames{ipv2}),1);
            if ~isempty(iname) && strcmp(chNames{ipv2},'x')
              dataout(ipv2)=X(iname);
            elseif ~isempty(iname) && strcmp(chNames{ipv2},'y')
              dataout(ipv2)=Y(iname);
            elseif ~isempty(iname) && strcmp(chNames{ipv2},'Q')
              dataout(ipv2)=Q(iname);
            end
          end
          % Request single data channel (scalar double)
        else
          ntries=0;
          while ntries<10
            try
              v=da.getDaValue(uniPV{ipv});
              break
            catch
              ntries=ntries+1;
              if ntries>10
                error('aidaGet failed after 10 tries')
              end
              pause(1)
            end
          end
          for ipv2=find(ismember(pvlist,uniPV{ipv}))
            dataout(ipv2)=double(v.get(0));
          end
        end
      end
      nanout2=nanout;
      nanout(newlist)=dataout;
      varargout{1}=nanout;
      nanout2(newlist)=now;
      varargout{2}=nanout2;
    elseif strcmp(varargin{1},'aidaPut')
      import edu.stanford.slac.aida.lib.util.common.*
      if nargin<4
        error('Must supply trim type')
      end
      trimStyle=varargin{4};
      if ~strcmp(trimStyle,'PTRB') && ~strcmp(trimStyle,'TRIM')
        error('trimStyle must be ''PTRB'' or ''TRIM''')
      end
      type=regexp(pvlist,'//(.+)$','match','once');
      pvname=regexprep(pvlist,'(//.+$)','');
      uType=unique(type);
      uType=uType(~cellfun(@(x) isempty(x),uType));
      for itype=1:length(uType)
%         daV=DaValue;
%         daV.type=0;
        stringsParam = pvname(ismember(type,uType));
        setValues = varargin{3}(ismember(type,uType));
%         daV.addElement(DaValue(stringsParam{1}));
%         daV.addElement(DaValue(single(setValues(1))));
%         da.reset;
%         query = sprintf('MAGNETSET%s',uType{itype});
%         da.setParam('MAGFUNC', trimStyle);
%         da.setParam('LIMITCHECK','SOME');
        ntries=0;
        while ntries<3
          try
            control_magnetSet(stringsParam, setValues);
            break
          catch
            ntries=ntries+1;
            if ntries>10
              error('aidaPut failed after 10 tries')
            end
            pause(1)
          end
        end
      end
    end
    da.reset();
  else
    if nargin>2
      [output1 output2]=eval([varargin{1},'(pvlist(:),varargin{3});']);
    else
      [output1 output2]=eval([varargin{1},'(pvlist(:));']);
    end
    nanout2=nanout;
    nanout(newlist)=output1;
    nanout2(newlist)=output2;
    varargout{1}=nanout; varargout{2}=nanout2;
  end
  ismissing=false;
end

% --- AIDA BPM Acqisition function ---
function [names X Y Q]=aidaBPM(da,str,bpmd)

import edu.stanford.slac.aida.lib.util.common.*
import java.util.Vector

% AIDA readback status bits
% Check STAT_GOOD & ~(STAT_BAD | STAT_OFF)
HSTA_XONLY = 64;    % 0x00000040
HSTA_YONLY = 128;   % 0x00000080
STAT_GOOD  = 1;     % 0x00000001
% STAT_OK    = 2;     % 0x00000002
STAT_OFF   = 8;     % 0x00000008
STAT_BAD   = 256;   % 0x00000100

da.setParam(sprintf('N=%d',4));
da.setParam(sprintf('BPMD=%d',bpmd));
try
  v = da.getDaValue(str);
catch
  names=[]; X=[]; Y=[]; Q=[];
  return
end
anames = Vector(v.get(0));
tmits = Vector(v.get(4));
hstas = Vector(v.get(5));
stats = Vector(v.get(6));
xvals = Vector(v.get(1));
yvals = Vector(v.get(2));

nsize=anames.size;
names=cell(nsize,1);
X=NaN(nsize,1);
Y=X;
Q=X;
for iele = 1:nsize
  names{iele} = char(anames.elementAt(iele-1));
  X(iele) = double(xvals.elementAt(iele-1));
  Y(iele) = double(yvals.elementAt(iele-1));
  Q(iele) = double(tmits.elementAt(iele-1));
  % flag bpm readings as bad by setting to NaN
  if ~(bitand( uint32(stats.elementAt(iele-1)),uint32(STAT_GOOD) ) > 0) && ...
      ( bitand( uint32(stats.elementAt(iele-1)),uint32(STAT_OFF) ) > 0 || ...
      bitand( uint32(stats.elementAt(iele-1)),uint32(STAT_BAD) ) > 0 )
    X(iele)=NaN;
    Y(iele)=NaN;
    Q(iele)=NaN;
  end % Good bpm check
  % Check for y-only or x-only bpms
  if (bitand(uint32(hstas.elementAt(iele-1)),uint32(HSTA_YONLY)) ~= 0)
    X(iele)=NaN;
  end % if y-only
  if (bitand(uint32(hstas.elementAt(iele-1)),uint32(HSTA_XONLY)) ~= 0)
    Y(iele)=NaN;
  end % if y-only
end % for iele

