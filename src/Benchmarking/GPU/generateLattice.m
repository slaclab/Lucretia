function [Initial,nuxdeg,nuydeg] = generateLattice()
% GENERATELATTICE - make a lattice for benchmarking parallel Lucretia
% tracking on the GPU utilising all Lucretia element types

global BEAMLINE WF

%% Unsupported elements (not yet converted to CUDA code)
% {'LCAV' 'TCAV' 'XCOR' 'YCOR' 'XYCOR' 'COLL' 'COORD' 'MARK'};

%% First make a linac section taking beam from an initial 5 GeV to 10 GeV

% Accelerating cavity 
cavgradient=31.5; % MV/m
cavL=1.0362; % m
% cav=DrifStruc(cavL,'CAV') ;
cav=RFStruc( cavL, cavgradient*cavL, 0, 2.8560e+09, 1, 1, 0, 0.02, 'CAV' );
load srwf_long_sband.mat wf wfz
WF.ZSR(1).z=wfz; WF.ZSR(1).K=wf; WF.ZSR(1).BinWidth=0.02;
load srwf_trans_sband.mat wf wfz
WF.TSR(1).z=wfz; WF.TSR(1).K=wf; WF.TSR(1).BinWidth=0.02;

% Generate single accelerator FODO cell
drif30cm = DrifStruc(0.3,'Drift30cm') ;
drif30cm2 = DrifStruc(0.3,'Drift30cm2') ;
bpm10cm = BPMStruc(0.1,'BPM10cm') ;
lcellov2 = 8*cav.L + 8 * bpm10cm.L + drif30cm.L + 0.2 + drif30cm.L ;
Bq = sqrt(3) * (5/0.299792458) / lcellov2 ;
QF = QuadStruc(0.1,Bq/2,0,0.035,'QF')  ;
QD = QuadStruc(0.1,-Bq/2,0,0.035,'QD') ;
for count = 1:8
  BEAMLINE{2*count-1} = cav ; BEAMLINE{2*count} = bpm10cm ;
end
BEAMLINE{17}=drif30cm;
BEAMLINE{18}=QF; BEAMLINE{19}=QF;
BEAMLINE{20}=drif30cm2;
for count = 1:17
  BEAMLINE{count+20} = BEAMLINE{count} ;
end
BEAMLINE{end+1} = QD ; BEAMLINE{end+1} = QD ;
BEAMLINE{end+1}=drif30cm2;
for count = 1:length(BEAMLINE)
  BEAMLINE{count}.P = 5 ;
end
icell1end = length(BEAMLINE) ;

% compute the # of FODO cells needed to reach 10 GeV, starting from 5 GeV
voltage = cavgradient*cavL;
ncell = ceil(5000/(16*voltage)) ;

% generate the remaining cells needed for full energy
for ccount = 1:ncell-1
  for ecount = 1:icell1end
    BEAMLINE{ecount+ccount*icell1end} = BEAMLINE{ecount} ;
  end
end

%% Make Initial structure
% fill a data structure of initial conditions, assuming a charge of 3.2 nC
% (about 2e10 particles) per bunch, 1 bunch, and a 400 bucket spacing

[Initial,nuxdeg,nuydeg] = generateInitial(3.2e-9, 1, 400 / 1.3e9 ) ;
SetSPositions(1,length(BEAMLINE),0) ;
for ibl=1:length(BEAMLINE); BEAMLINE{ibl}.P=Initial.Momentum; end


%% Add matching section linac -> chicane
M={};
M{end+1}=DrifStruc(1,'MD1') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ1')  ;
M{end+1}=DrifStruc(1,'MD2') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ2')  ;
M{end+1}=DrifStruc(1,'MD3') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ3')  ;
M{end+1}=DrifStruc(1,'MD4') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ4')  ;
M{end+1}=DrifStruc(1,'MD5') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ5')  ;
M{end+1}=DrifStruc(1,'MD6') ;
M{end+1}=QuadStruc(0.5,0,0,0.1,'MQ6')  ;
M{end+1}=DrifStruc(1,'MD7') ;
% Match to chicane required input twiss:
% Tx.beta=11.768; Tx.alpha=-0.627; Tx.eta=0; Tx.etap=0; Tx.nu=0;
% Ty.beta=34.744; Ty.alpha=1.738; Ty.eta=0; Ty.etap=0; Ty.nu=0;
[~,T] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
Tx.beta=T.betax(end); Tx.alpha=T.alphax(end); Tx.eta=T.etax(end); Tx.etap=T.etapx(end); Tx.nu=T.nux(end);
Ty.beta=T.betay(end); Ty.alpha=T.alphay(end); Ty.eta=T.etay(end); Ty.etap=T.etapy(end); Ty.nu=T.nuy(end);
bl=BEAMLINE;
BEAMLINE=M';
SetSPositions(1,length(BEAMLINE),0) ;
for ibl=1:length(BEAMLINE); BEAMLINE{ibl}.P=Initial.Momentum; end

mvals=lsqnonlin(@(x) l2cmatch(x,Tx,Ty),[1 0 1 0 1 0 1 0 1 0 1 0 1],[],[],optimset('Display','off'));
% Set drift lengths / quad strengths
for ibl=1:length(mvals)
  if mod(ibl,2) % drift
    M{ibl}.L=mvals(ibl);
  else % quad
    M{ibl}.B=mvals(ibl);
  end
end
BEAMLINE=[bl M];
for ibl=1:length(BEAMLINE); BEAMLINE{ibl}.P=Initial.Momentum; end

%% Add W-chicane for bunch compression
P=BEAMLINE{end}.P;
W={};
W{end+1}=SBendStruc( 1.063, 0.3769*(P/5), 0.0226, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB1' );
W{end+1}=DrifStruc(4.9634,'WD1') ;
W{end+1}=QuadStruc(0.5962,7.3108*(P/5),0,0.1,'WQF1')  ;
W{end+1}=DrifStruc(2.0008,'WD2') ;
W{end+1}=QuadStruc(0.3813,0*(P/5),pi/4,0.1,'WSQ1')  ;
W{end+1}=DrifStruc(1.4092,'WD3') ;
W{end+1}=SBendStruc( 1.8249, -0.5472*(P/5), -0.0328, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB2' );
W{end+1}=DrifStruc(0.1936,'WD4') ;
W{end+1}=QuadStruc(1.0,-7.841*(P/5),0,0.1,'WQD2')  ;
W{end+1}=DrifStruc(0.19366,'WD5') ;
W{end+1}=SextStruc( 0.25, 62.344, 0, 0.1, 'WSF1' ) ;
W{end+1}=DrifStruc(0.4076,'WD6') ;
W{end+1}=QuadStruc(0.7142,5.7971*(P/5),0,0.1,'WQF3')  ;
W{end+1}=DrifStruc(0.2294,'WD7') ;
W{end+1}=QuadStruc(0.7142,5.7971*(P/5),0,0.1,'WQF4')  ;
W{end+1}=DrifStruc(0.24087,'WD8') ;
W{end+1}=SextStruc( 0.762, 0*-218.9407, 0, 0.1, 'WSD2' ) ;
W{end+1}=DrifStruc(1.9535,'WD9') ;
W{end+1}=SextStruc( 0.25, 0*-62.344, 0, 0.1, 'WSD3' ) ;
W{end+1}=DrifStruc(0.2397,'WD10') ;
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF5')  ;
W{end+1}=DrifStruc(0.2294,'WD11') ;
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF6')  ;
W{end+1}=DrifStruc(0.2294,'WD12') ;
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF7')  ;
W{end+1}=DrifStruc(0.2397,'WD13') ;
% W{end+1}=OctuStruc( 0.25, -600, 0, 0.1, 'WOCT1' ) ;
W{end+1}=DrifStruc(0.48,'WD14') ;
W{end+1}=QuadStruc(0.4284,-2.2116*(P/5),0,0.1,'WQD8')  ;
W{end+1}=DrifStruc(0.2293,'WD15') ;
W{end+1}=SBendStruc( 0.5287, 0.1705*(P/5), 0.0102, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB3' );
W{end+1}=DrifStruc(0.2418,'WD16') ;
W{end+1}=QuadStruc(0.155,-6.1319*0.5*(P/5),0,0.1,'WQD9A')  ;
W{end+1}=QuadStruc(0.155,-6.1319*0.5*(P/5),0,0.1,'WQD9B')  ;
W{end+1}=DrifStruc(0.2418,'WD16') ;
W{end+1}=SBendStruc( 0.5287, 0.1705*(P/5), 0.0102, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB4' );
W{end+1}=DrifStruc(0.2293,'WD19') ;
W{end+1}=QuadStruc(0.4284,-2.2116*(P/5),0,0.1,'WQD11')  ;
W{end+1}=DrifStruc(0.48,'WD20') ;
W{end+1}=MultStruc( 0.25, [1000 10000], [0.1; 0.2], [4; 5], [0 0], 0.1, 'WMULT1' ) ;
% W{end+1}=DrifStruc(0.2397,'WD21') ;
W{end+1}=CollStruc(0.2397,5e-2,1,'Rectangle',0,'WCOLL');
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF12')  ;
W{end+1}=DrifStruc(0.2294,'WD22') ;
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF13')  ;
W{end+1}=DrifStruc(0.2294,'WD23') ;
W{end+1}=QuadStruc(0.7142,5.9988*(P/5),0,0.1,'WQF14')  ;
W{end+1}=DrifStruc(0.2397,'WD24') ;
W{end+1}=SextStruc( 0.25, 0*-62.6771, 0, 0.1, 'WSD6' ) ;
W{end+1}=DrifStruc(1.9566,'WD25') ;
W{end+1}=SextStruc( 0.762, 0*-218.9407, 0, 0.1, 'WSD7' ) ;
W{end+1}=DrifStruc(0.2377,'WD26') ;
W{end+1}=QuadStruc(0.7142,5.7971*(P/5),0,0.1,'WQF15')  ;
W{end+1}=DrifStruc(0.2294,'WD27') ;
W{end+1}=QuadStruc(0.7142,5.7971*(P/5),0,0.1,'WQF16')  ;
W{end+1}=DrifStruc(0.3984,'WD28') ;
W{end+1}=SextStruc( 0.25, 0*62.344, 0, 0.1, 'WSF8' ) ;
W{end+1}=DrifStruc(0.2028,'WD29') ;
W{end+1}=QuadStruc(1,-7.8409*(P/5),0,0.1,'WQD17')  ;
W{end+1}=DrifStruc(0.1936,'WD29') ;
W{end+1}=SBendStruc( 1.8249, -0.5472*(P/5), -0.0328, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB6' );
W{end+1}=DrifStruc(0.7454,'WD30') ;
W{end+1}=DrifStruc(1.0334,'WTCAV') ; % V=1.9132, Freq=11424, tilt=pi/2
W{end+1}=DrifStruc(0.30337,'WD31') ;
W{end+1}=SBendStruc( 0.244, 0*-0.0417*(P/5), -0.0025, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WWIG1' );
W{end+1}=DrifStruc(0.12653,'WDWIG1') ;
W{end+1}=SBendStruc( 0.244, 0*2*0.0417*(P/5), 2*0.0025, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WWIG2' );
W{end+1}=DrifStruc(0.12653,'WDWIG2') ;
W{end+1}=SBendStruc( 0.244, 0*-0.0417*(P/5), -0.0025, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WWIG3' );
W{end+1}=DrifStruc(0.4801,'WD32') ;
W{end+1}=QuadStruc(0.5962,7.3099*(P/5),0,0.1,'WQF18')  ;
W{end+1}=DrifStruc(5.208,'WD33') ;
W{end+1}=SBendStruc( 1.063, 0.3768*(P/5), 0.0226, [0 0], [0 0], [0.1 0.1], [0.5 0.5], 0, 'WB7' );

% Match chicane requirements
[~,T] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
Tx.beta=T.betax(end); Tx.alpha=T.alphax(end); Tx.eta=T.etax(end); Tx.etap=T.etapx(end); Tx.nu=T.nux(end);
Ty.beta=T.betay(end); Ty.alpha=T.alphay(end); Ty.eta=T.etay(end); Ty.etap=T.etapy(end); Ty.nu=T.nuy(end);
bl=BEAMLINE;
BEAMLINE=W';
SetSPositions(1,length(BEAMLINE),0) ;
for ibl=1:length(BEAMLINE); BEAMLINE{ibl}.P=Initial.Momentum; end
iB=findcells(BEAMLINE,'B'); imid=findcells(BEAMLINE,'Name','WQD9A');
% mvals=lsqnonlin(@(x) cmatch(x,Tx,Ty,iB,imid),[arrayfun(@(x) BEAMLINE{x}.B(1),iB) Tx.beta Tx.alpha Ty.beta Ty.alpha],[],[],optimset('Display','iter'));
mvals=lsqnonlin(@(x) cmatch(x,Tx,Ty,iB,imid),arrayfun(@(x) BEAMLINE{x}.B(1),iB),[],[],optimset('Display','off'));

% Set quad strengths
for ibl=1:length(mvals)
  W{iB(ibl)}.B=mvals(ibl);
end

W{findcells(W,'Name','WMULT1')}=MultStruc( 0.25, [1000 10000], [0.1; 0.2], [4; 5], [0 0], 0.1, 'WMULT1' ) ;
% W{findcells(W,'Name','WMULT1')}=MultStruc( 0.25, 1000, 0.1, 4, [0 0], 0.1, 'WMULT1' ) ;

BEAMLINE=[bl W];

%% Add FFS
FF={};
FF{end+1}=DrifStruc(0.38382,'FFD1') ;
FF{end+1}=QuadStruc(0.4609,0.711*(P/5),0,0.1,'FFQF1') ;
FF{end+1}=DrifStruc(0.5375,'FFD2') ;
FF{end+1}=QuadStruc(0.3813,0.2*(P/5),pi/4,0.1,'FFSQ1') ;
FF{end+1}=DrifStruc(0.9406,'FFD3') ;
FF{end+1}=QuadStruc(0.4609,-2.3888*(P/5),0,0.1,'FFQD2') ;
FF{end+1}=DrifStruc(0.2394,'FFD4') ;
FF{end+1}=QuadStruc(0.4609,-2.3888*(P/5),0,0.1,'FFQD3') ;
FF{end+1}=DrifStruc(0.2396,'FFD5') ;
FF{end+1}=QuadStruc(0.4609,-2.3888*(P/5),0,0.1,'FFQD4') ;
FF{end+1}=DrifStruc(2.5359,'FFD6') ;
FF{end+1}=QuadStruc(0.7142,4.3336*(P/5),0,0.1,'FFQF5') ;
FF{end+1}=DrifStruc(0.5051,'FFD7') ;
FF{end+1}=QuadStruc(0.7142,4.3336*(P/5),0,0.1,'FFQF6') ;
FF{end+1}=DrifStruc(0.5167,'FFD8') ;
FF{end+1}=QuadStruc(2.026,-14.2347*(P/5),0,0.1,'FFQD7') ;
FF{end+1}=DrifStruc(0.5563,'FFD9') ;
FF{end+1}=QuadStruc(0.7142,10.5574*(P/5),0,0.1,'FFQF8') ;
FF{end+1}=SolenoidStruc( 2.7086*(P/5), 5, 0.2, 'FFSOL' ) ;
% Kick beam to test corrector
FF{end+1}=CorrectorStruc( 0.1, 0.5, 0, 2, 'YKICK' ) ;
% Co-ordinate transform test
FF{end+1}=CoordStruc(0.1, 1e-3, -0.2, -11e-3, 0,0, 'CROT');
% FF{end+1}=CoordStruc(0, 0, 0, 0, 0, 0, 'CROT');
% Set momentum and S coords
% stat = SetDesignMomentumProfile( 1, length(BEAMLINE), Initial.Q, Initial.Momentum );
% if stat{1}~=1; error(stat{2}); end;
for iele=1:length(BEAMLINE)
  BEAMLINE{iele}.P=Initial.Momentum;
end
SetSPositions(1,length(BEAMLINE),0) ;
% Match ffs requirements
[stat,T] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
if stat{1}~=1; error(stat{2}); end
Tx.beta=T.betax(end); Tx.alpha=T.alphax(end); Tx.eta=T.etax(end); Tx.etap=T.etapx(end); Tx.nu=T.nux(end);
Ty.beta=T.betay(end); Ty.alpha=T.alphay(end); Ty.eta=T.etay(end); Ty.etap=T.etapy(end); Ty.nu=T.nuy(end);
bl=BEAMLINE;
BEAMLINE=FF';
SetSPositions(1,length(BEAMLINE),0) ;
for ibl=1:length(BEAMLINE); BEAMLINE{ibl}.P=Initial.Momentum; end
iB=findcells(BEAMLINE,'B');
mvals=lsqnonlin(@(x) ffmatch(x,Tx,Ty,iB),arrayfun(@(x) BEAMLINE{x}.B(1),iB),[],[],optimset('Display','off'));
% Set quad strengths
for ibl=1:length(mvals)
  FF{iB(ibl)}.B(1)=mvals(ibl);
end
% Put instrument at IP
IP=InstStruc( 0, 'INST', 'IP' ) ;
BEAMLINE=[bl FF IP];



%% Other BEAMLINE setup tasks

% now set the correct no-load momentum profile and scale the magnets to it,
% and set the S positions as well
SetSPositions(1,length(BEAMLINE),0) ;

% find the blocks and slices and set the data on them into the lattice
SetElementBlocks(1,length(BEAMLINE)) ;
SetElementSlices(1,length(BEAMLINE)) ;


%% matching functions

% matching objective function for linac->chicane
function ret=l2cmatch(x,Tx,Ty)
global BEAMLINE

% Set drift lengths / quad strengths (odd x are drifts)
for ibl=1:length(x)
  if mod(ibl,2) % drift
    BEAMLINE{ibl}.L=x(ibl);
  else % quad
    BEAMLINE{ibl}.B=x(ibl);
  end
end

% Get Twiss parameters
[~,T]=GetTwiss(1,length(BEAMLINE),Tx,Ty);

% Match specific alpha, beta values at chicane entrance
% ret=[T.alphax(end)-(-0.627) T.betax(end)-11.768 T.alphay(end)-7.5 T.betay(end)-35];
ret=[T.alphax(end)-(-0.627) T.betax(end)-11.768 T.alphay(end)-1.738 T.betay(end)-34.744];


% matching objective function for chicane
function ret=cmatch(x,Tx,Ty,iB,imid)
global BEAMLINE

% Set quad strengths
for ibl=1:length(iB)
  BEAMLINE{iB(ibl)}.B=x(ibl);
end

% Tx.beta=x(end-3); Tx.alpha=x(end-2);
% Ty.beta=x(end-1); Ty.alpha=x(end); 

% Get Twiss parameters at chicane center and end
[~,T_mid]=GetTwiss(1,imid,Tx,Ty);
[~,T_end]=GetTwiss(1,length(BEAMLINE),Tx,Ty);

% Alpha x & y zero and dispersion' zero in chicane center and dispersion
% closed at chicane exit
ret=[T_mid.alphax(end) T_mid.alphay(end) T_mid.etapx(end) T_end.etax(end) T_end.etapx(end) ...
   (sum(T_end.betay)./length(T_end.betay))-200];

% matching objective function for FFS
function ret=ffmatch(x,Tx,Ty,iB)
global BEAMLINE

% Set quad strengths
for ibl=1:length(iB)
  BEAMLINE{iB(ibl)}.B=x(ibl);
end

% Get Twiss parameters at end
[~,T_end]=GetTwiss(1,length(BEAMLINE),Tx,Ty);

% Alpha x & y zero and dispersion' zero in chicane center and dispersion
% closed at chicane exit
ret=[T_end.alphax(end) T_end.alphay(end) T_end.betax(end)-0.1 T_end.betay(end)-1];