function [INSTR FLI]=assignHW(INSTR,FLI)
% Assign Hardware details to Floodland data structures

% ------------------------------------------------------------------------------
% BPMs
% ------------------------------------------------------------------------------
global BEAMLINE PS
for ibpm=find(getIndex(INSTR,'Class','MONI'))
  pvname={'TAXXALL//BPMS' 'TAXXALL//BPMS' 'TAXXALL//BPMS'};
  INSTR.defineInstrHW(ibpm,{'x' 'y' 'Q'},pvname,'AIDA',[1e-3 1e-3 1],'AIDA_BPMD',41,'AIDA_NAMES',...
    sprintf('BPMS:%s:%d',...
    Floodland.bitid2micr(BEAMLINE{INSTR.Index(ibpm)}.Name(2:3)),...
    str2double(BEAMLINE{INSTR.Index(ibpm)}.Name(4:7))));
end % for ibpmHan

% ------------------------------------------------------------------------------
% Correctors
% ------------------------------------------------------------------------------
xcors=find(getIndex(FLI,'PS','Class','XCOR'));
ycors=find(getIndex(FLI,'PS','Class','YCOR'));
psind=FLI.PS;
for icor=xcors
  pvname=sprintf('XCOR:%s:%d//BDES',Floodland.bitid2micr(BEAMLINE{PS(psind(icor)).Element(1)}.Name(2:3)),...
    str2double(BEAMLINE{PS(psind(icor)).Element(1)}.Name(4:7)));
  pvname={pvname;pvname};
  FLI.defineIndxHW('PS',psind(icor),pvname,'AIDA',{0.1;0.1},[-0.1;-0.1],[0.1;0.1]);
end
for icor=ycors
  pvname=sprintf('YCOR:%s:%d//BDES',Floodland.bitid2micr(BEAMLINE{PS(psind(icor)).Element(1)}.Name(2:3)),...
    str2double(BEAMLINE{PS(psind(icor)).Element(1)}.Name(4:7)));
  pvname={pvname;pvname};
  FLI.defineIndxHW('PS',psind(icor),pvname,'AIDA',{0.1;0.1},[-0.1;-0.1],[0.1;0.1]);
end

% ------------------------------------------------------------------------------
% KLYSTRONS
% ------------------------------------------------------------------------------
pvname=cell(2,2);
conv={1 1;1 1};
pvname{1,1}='AMPL:TA02:38//VDES';
pvname{2,1}='AMPL:TA02:38//VDES';
FLI.defineIndxHW('KLYSTRON',1,pvname,'AIDA',conv,[0 -360;0 -360],[100 360; 100 360]);