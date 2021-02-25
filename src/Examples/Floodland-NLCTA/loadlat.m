function [FL INSTR FLI Inds]=loadlat()
% Load in the ECHO lattice and setup controls links etc
global BEAMLINE PS KLYSTRON GIRDER %#ok<NUSED>
BEAMLINE={}; PS=[]; KLYSTRON=[];

% Load in xsif deck to generate BEAMLINE array
evalc('[stat Initial]=XSIFToLucretia( ''ECHO.saveline'', ''ECHO7'');');

% Set initial twiss and bunch parameters
Initial.Momentum=0.005;
Initial.Q=1e-9;
Initial.x.NEmit=8e-6;
Initial.y.NEmit=8e-6;
Initial.sigz=2.4e-4;
Initial.SigPUncorrel=1e-3;
Initial.x.Twiss.beta = 5.466603230022 ;
Initial.x.Twiss.alpha = -4.954448737138 ;
Initial.y.Twiss.beta = 5.466603230022 ;
Initial.y.Twiss.alpha = -4.954448737138 ;

% Make TCAV's
tcav=findcells(BEAMLINE,'Name','TCAV*');
if isempty(tcav); error('No TCAVs found'); end;
for ib=tcav
  BEAMLINE{ib}.Class='TCAV';
  BEAMLINE{ib}.Tilt=0;
end

% element indexing
Inds.bpm=findcells(BEAMLINE,'Class','MONI');
if isempty(Inds.bpm); error('No BPMs found'); end;
Inds.xcor=findcells(BEAMLINE,'Class','XCOR');
if isempty(Inds.xcor); error('No XCORs found'); end;
Inds.ycor=findcells(BEAMLINE,'Class','YCOR');
if isempty(Inds.xcor); error('No YCORs found'); end;
Inds.lcav{1}=findcells(BEAMLINE,'Name','ACCL380*');
Inds.lcav{2}=findcells(BEAMLINE,'Name','ACCL430*');
Inds.lcav{3}=findcells(BEAMLINE,'Name','K821260T');
if any(cellfun(@(x) isempty(x),Inds.lcav)); error('Not all accelerating cavities found'); end;
Inds.tcav{1}=findcells(BEAMLINE,'Name','TCAVD11H');
Inds.tcav{2}=findcells(BEAMLINE,'Name','TCAVD27H');
if any(cellfun(@(x) isempty(x),Inds.tcav)); error('Not all transverse cavities found'); end;

% Floodland HW indexing object
FLI=FlIndex;

% Assign Power Supplies
for ib=[Inds.xcor Inds.ycor]
  FLI.addPS(ib);
end

% Assign Klystrons
for ilc=1:length(Inds.lcav)
  FLI.addKlystron(Inds.lcav{ilc});
end
for itc=1:length(Inds.tcav)
  FLI.addKlystron(Inds.lcav{itc});
end

% Setup momentum profile
stat = SetDesignMomentumProfile( 1, length(BEAMLINE), Initial.Q, Initial.Momentum, 0.12 );
if stat{1}~=1; error('Momentum profile setting error:\n%s',stat{2}); end;

% Create Floodland object
FL = Floodland('nlcta','ECHO7',now,Initial) ;

% Machine rep rate / Hz
FL.repRate = 10 ;

% Create INSTR object
INSTR = FlInstr;

% Specify INSTR data (perfect res initially for test orbit)
ind=INSTR.getIndex(INSTR,'Class','MONI');
INSTR.setResolution(ind,{'x' 'y'},zeros(sum(ind),2));
INSTR.Type(ind)={'stripline'};

% Define INSTR hardware connections
[INSTR FLI]=assignHW(INSTR,FLI);

% Allow no gaps longer than 2s in requested data sets
INSTR.maxDataGap=2;

% Put COR PS's to zero
for ib=[Inds.xcor Inds.ycor]
  PS(BEAMLINE{ib}.PS).SetPt = 0 ;
  PS(BEAMLINE{ib}.PS).Ampl = 0 ;
end

% Test basic tracking
[stat ~]=TrackThru(1,length(BEAMLINE),FL.BeamMacro,1,1,0);
if stat{1}~=1; error('Beam does not track through lattice:\n%s\n',stat{2}); end;

% Acquire beam data (INSTR.ndata pulses)
INSTR.ndata=10;
INSTR.acquire(FL);

% plot orbit
%plot(INSTR,{'x' 'y'},'data');

% Define gold orbit
INSTR.setRef;

% Specify INSTR resolutions
indx=findcells(BEAMLINE,'Class','MONI');
for imoni=indx
  BEAMLINE{imoni}.Resolution=20e-6;
end
INSTR.setResolution;

% Save data
save nlctaLat.mat