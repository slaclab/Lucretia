function stat = corbump3(amp,n1,n2,n3,finetune)
% stat = corbump3(amp,n1,n2,n3,doiter)
% Apply a 3 corrector bump
% The bump aims to produce a bump size amp at n2
% n1,n2 and n3 are the BEAMLINE element names of the 3 correctors to use
% (must all be of the same class, XCOR or YCOR)
% if finetune=true, then fine-tune the strength of the final corrector to
% fully close the bump (minimise downstream RMS orbit)
% Set finetune='UseModel' to model the effect of creating the bump instead
% of actually doing it
global BEAMLINE FL PS
stat{1}=1;

% Get corrector indicies and form response matrices
c1=findcells(BEAMLINE,'Name',n1);
if isempty(c1)
  stat{1}=-1; stat{2}=['Unknown beamline element name: ',n1];
  return
end
t1=BEAMLINE{c1(1)}.Class;
if isfield(BEAMLINE{c1(1)},'PS')
  c1ps=BEAMLINE{c1(1)}.PS;
else
  stat{1}=-1; stat{2}=['No PS element for beamline element name: ',n1];
  return
end
c2=findcells(BEAMLINE,'Name',n2);
t2=BEAMLINE{c2(1)}.Class;
if isempty(c2)
  stat{1}=-1; stat{2}=['Unknown beamline element name: ',n2];
  return
end
if isfield(BEAMLINE{c2(1)},'PS')
  c2ps=BEAMLINE{c2(1)}.PS;
else
  stat{1}=-1; stat{2}=['No PS element for beamline element name: ',n2];
  return
end
c3=findcells(BEAMLINE,'Name',n3);
t3=BEAMLINE{c3(1)}.Class;
if isempty(c3)
  stat{1}=-1; stat{2}=['Unknown beamline element name: ',n3];
  return
end
if isfield(BEAMLINE{c3(1)},'PS')
  c3ps=BEAMLINE{c3(1)}.PS;
else
  stat{1}=-1; stat{2}=['No PS element for beamline element name: ',n1];
  return
end
if ~isequal(t1,t2,t3)
  stat{1}=-1;
  stat{2}='All 3 beamline classes must be the same (YCOR or XCOR)';
  return
elseif ~isequal(t1,'XCOR') && ~isequal(t1,'YCOR')
  stat{1}=-1;
  stat{2}='Must choose XCOR or YCOR beamline class';
  return
end
L=BEAMLINE{c1}.L;
drif=[1 L 0 0 0 0; 0 1 0 0 0 0; 0 0 1 L 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; 
[stat, R12]=RmatAtoB(c1+1,c2); if stat{1}~=1; stat{2}=['Rmat error: ',stat{2},' ','Make sure correctors are given in S order']; return; end;
R12=R12*drif;
[stat, R13]=RmatAtoB(c1+1,c3); if stat{1}~=1; stat{2}=['Rmat error: ',stat{2},' ','Make sure correctors are given in S order']; return; end;
R13=R13*drif;
[stat, R23]=RmatAtoB(c2+1,c3); if stat{1}~=1; stat{2}=['Rmat error: ',stat{2},' ','Make sure correctors are given in S order']; return; end;
L=BEAMLINE{c2}.L;
drif=[1 L 0 0 0 0; 0 1 0 0 0 0; 0 0 1 L 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; 
R23=R23*drif;

% Get required kick strengths (rad) for requested amplitude at c2
if strcmp(t1,'XCOR')
  dc1=amp/R12(1,2);
  dc2=-(amp*R13(1,2))/(R12(1,2)*R23(1,2));
  dc3=-(amp/R12(1,2))*((R13(1,2)*R23(2,2))/R23(1,2) + R13(2,2));
  dim='x';
else
  dc1=amp/R12(3,4);
  dc2=-(amp*R13(3,4))/(R12(3,4)*R23(3,4));
  dc3=-(amp/R12(3,4))*((R13(3,4)*R23(4,4))/R23(3,4) + R13(4,4));
  dim='y';
end

% Set PS's
if strcmpi(finetune,'usemodel')
  PS(c1ps).Ampl=PS(c1ps).Ampl+dc1;
  PS(c2ps).Ampl=PS(c2ps).Ampl+dc2;
  PS(c3ps).Ampl=PS(c3ps).Ampl+dc3;
  online=0;
else
  PS(c1ps).SetPt=PS(c1ps).Ampl+dc1;
  PS(c2ps).SetPt=PS(c2ps).Ampl+dc2;
  PS(c3ps).SetPt=PS(c3ps).Ampl+dc3;
  if isempty(FL) || ~isfield(FL,'SimMode') || FL.SimMode==2
    stat=PSTrim([c1ps c2ps c3ps]);
    online=0;
  else
    stat=PSTrim([c1ps c2ps c3ps],1);
    online=1;
  end
  if stat{1}~=1; return; end;
end

% Fine Tune final corrector to fully close bump
if exist('finetune','var') && logical(any(double(finetune)))
  FL.SimModel.ip_ind=findcells(BEAMLINE,'Name','IP'); FL.simBeam=2;
  if strcmpi(finetune,'usemodel')
    [stat,B,instdata]=TrackThru(1,FL.SimModel.ip_ind,FL.SimBeam{2},1,1);
  else
    FlHwUpdate;
    [stat,instdata]=FlTrackThru(FL.SimModel.extStart,FL.SimModel.ip_ind);
  end
  if stat{1}~=1; return; end;
  bpmind=[instdata{1}.Index]>c3;
  bpmres=cellfun(@(x) x.Resolution,{BEAMLINE{[instdata{1}(bpmind).Index]}});
  psval=PS(c3ps).Ampl;
  copt=fminbnd(@(x) minrmsorbit(x,c3ps,psval,online,bpmind,bpmres,dim,finetune),-1e-5,1e-5,optimset('MaxIter',100,'TolX',1e-6,'TolFun',...
    mean(var(randn(length(bpmres),10000).*repmat(bpmres',1,10000),bpmres))));
  if strcmpi(finetune,'usemodel')
    PS(c3ps).Ampl=psval+copt;
  else
    PS(c3ps).SetPt=psval+copt;
    PSTrim(c3ps,online);
  end
end

% ========================================
function chi2=minrmsorbit(x,ips,psval,online,bpmind,bpmres,dim,modelsel)
global PS FL

if strcmpi(modelsel,'usemodel')
  PS(ips).Ampl=psval+x;
  [stat,B,instdata]=TrackThru(1,FL.SimModel.ip_ind,FL.SimBeam{2},1,1);
else
  PS(ips).SetPt=psval+x;
  PSTrim(ips,online);
  FlHwUpdate;
  [stat,instdata]=FlTrackThru(FL.SimModel.extStart,FL.SimModel.ip_ind);
end
chi2=var([instdata{1}(bpmind).(dim)],bpmres);