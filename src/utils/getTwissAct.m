function Model = getTwissAct(Model)
% Calculate actual beta functions based on lattice errors
global BEAMLINE

origB=cell(length(BEAMLINE),1);
origT=cell(length(BEAMLINE),1);
for iEle=1:length(BEAMLINE)
  if isfield(BEAMLINE{iEle},'B')
    origB{iEle}=BEAMLINE{iEle}.B;
    origT{iEle}=BEAMLINE{iEle}.Tilt;
    BEAMLINE{iEle}.B=BEAMLINE{iEle}.B.*(1+BEAMLINE{iEle}.dB);
    BEAMLINE{iEle}.Tilt=BEAMLINE{iEle}.Tilt+BEAMLINE{iEle}.Offset(6);
  end
end
[stat,Model.twissActual] = GetTwiss(1,length(BEAMLINE),Model.Initial.x.Twiss,Model.Initial.y.Twiss) ;
for iEle=1:length(BEAMLINE)
  if isfield(BEAMLINE{iEle},'B')
    BEAMLINE{iEle}.B=origB{iEle};
    BEAMLINE{iEle}.Tilt=origT{iEle};
  end
end