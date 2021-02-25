%
% script for loading and assembling the Lucretia model of Wolski's
% single-stage BC + PT's 3-stage main linac
%
%==================================================================

% clear existing lattice, if any 

  clear global BEAMLINE GIRDER PS KLYSTRON WF ;
  
% clear the base workspace  
  
  clear
  
% reset Gaussian random number generator  
  
  randn('state',0) ;
  statall = InitializeMessageStack( ) ;
  
% parse the deck and the SRWFs

  disp('...parsing the deck...') ;
  [stat,Initial] = XSIFToLucretia('BUNCHCOMPRESSOR.XSIF','let') ;
  Initial1sigy = Initial ;
  Initial1sigy.y.pos = -8.39e-6 ;
  Initial1sigy.y.ang = 0.583e-6 ;
  statall = AddStackToMasterStack(statall,stat,'XSIFToLucretia') ;
  global BEAMLINE GIRDER PS KLYSTRON WF ;
  WF.TSR(1).BinWidth = 0.3 ;
  WF.ZSR(1).BinWidth = 0.3 ;
  WF.TSR(2).BinWidth = 0.3 ;
  WF.ZSR(2).BinWidth = 0.3 ;
  [stat,W] = ParseFrequencyDomainTLR('LBand_Freq_Wake_XY_0.dat',0) ;
  statall = AddStackToMasterStack(statall,stat,'ParseFrequencyDomainTLR') ;
  if (stat{1} == 1)
      WF.TLR{1} = W ;
  end
  clear W
  rflist = findcells(BEAMLINE,'Class','LCAV') ;
  for count = rflist
      BEAMLINE{count}.Wakes(3) = 1 ;
  end
  clear count rflist
  endbc = findcells(BEAMLINE,'Name','BC1WIGGEND') ;
  
% set slices and blocks

  disp('...setting element slices...')
  stat = SetElementSlices( 1, length(BEAMLINE) ) ;
  statall = AddStackToMasterStack(statall,stat,'SetElementSlices') ;
  disp('...setting alignment blocks...')
  stat = SetElementBlocks( 1, length(BEAMLINE) ) ;
  statall = AddStackToMasterStack(statall,stat,'SetElementBlocks') ;

% set girders:  in SC regions girder == cryomodule, 
%               elsewhere set 1 block = 1 girder,
%               bend girders are long, quad girders are short 

  disp('...setting all girders...')
  stat = SetOneStageGirders( ) ;
  statall = AddStackToMasterStack(statall,stat,'SetOneStageGirders') ;

% every magnet has an independent power supply

  disp('...setting independent power supplies...')
  stat = SetIndependentPS( 1, length(BEAMLINE) ) ;
  statall = AddStackToMasterStack(statall,stat,'SetIndependentPS') ;

% since the present optics uses 6 modules for the BC RF,
% go ahead and assign 3 modules per klystron, or 24
% cavities per klystron.  By staggering coincidence, this
% is what the Three Kings recommend

  disp('...setting 24 structures per klystron...') 
  stat = SetKlystrons( 1, length(BEAMLINE), 24  ) ;
  statall = AddStackToMasterStack(statall,stat,'SetKlystrons') ;

% move dimensioned magnet units to the PS array 

  disp('...moving physics units from magnets to power supplies...')
  stat = MovePhysicsVarsToPS(1:length(PS)) ;
  statall = AddStackToMasterStack(statall,stat,'MovePhysicsVarsToPS') ;
  
% move dimensioned RF units to the KLYSTRON array  

  disp('...moving physics units from structures to klystrons...')
  stat = MovePhysicsVarsToKlystron(1:length(KLYSTRON)) ;
  statall = AddStackToMasterStack(statall,stat,'MovePhysicsVarsToKlystron') ;
  
  disp('...generating "sparse" 21 x 21 beam...')
  beamsparse2121 = MakeBeam6DSparse(Initial,3,21,21) ;
  disp('...generating "sparse" beam with 1 sigy offset...')
  beam1sigy = MakeBeam6DSparse(Initial1sigy,3,21,21) ;
  disp('...generating PZ grid 11 x 11 beam...')
  beamflat1111 = MakeBeamPZGrid(Initial,3,11,11) ;
  disp('...generating 10 k rays in 6D Gaussian distribution...')
  beamgauss10k = MakeBeam6DGauss(Initial,10000,3,0) ;
  disp('...generating 1000 bunches of 1 ray per bunch and 1 sigy offset ...')
  beamMB = CreateBlankBeam(1000,1,Initial.Momentum,438/1.3e9) ;
  for count = 1:1000
      beamMB.Bunch(count).x(3) = Initial1sigy.y.pos ;
      beamMB.Bunch(count).x(4) = Initial1sigy.y.ang ;
      beamMB.Bunch(count).Q = beamMB.Bunch(count).Q * 2e10 ;
  end
  clear count

% verify the lattice   
  
%   disp('...Verifying the lattice...')
%   [stat,err,warn,info] = VerifyLattice( ) ;
%   statall = AddStackToMasterStack(statall,stat,'VerifyLattice') ;
%   disp(['...Number of lattice errors:     ',num2str(err{1})]) ;
%   disp(['...Number of lattice warnings:   ',num2str(warn{1})]) ;
%   disp(['...Number of lattice info msgs:  ',num2str(info{1})]) ;
  
% set tracking flags

  disp('...Setting tracking flags...')
  blist = SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('LRWF_T',1,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('GetBPMData',1,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('GetSBPMData',0,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('GetBPMBeamPars',1,1,length(BEAMLINE)) ;
  blist = SetTrackFlags('ZMotion',1,1,length(BEAMLINE)) ;

% set BPM resolution to zero
  disp('...Setting BPM resolution to zero...')
  bpmlist = findcells(BEAMLINE,'Class','MONI') ;
  for count = 1:length(bpmlist)
    BEAMLINE{bpmlist(count)}.Resolution = 0 ;
  end

  disp('...generating misalignment error groups...')
  
% prepare the all-structure error group

  [stat,RFMisalignGroup] = MakeErrorGroup( ...
    {'BEAMLINE','LCAV'},[1 20981],'Offset',0,...
    'All RF cavity misalignments') ;
  statall = AddStackToMasterStack(statall,stat,'MakeErrorGroup') ;
  
% prepare the all-BPM error group

  [stat,BPMMisalignGroup] = MakeErrorGroup( ...
    {'BEAMLINE','MONI'},[1 20981],'Offset',0,...
    'All BPM misalignments') ;
  statall = AddStackToMasterStack(statall,stat,'MakeErrorGroup') ;

% prepare the all-girder error group

  [stat,GirderMisalignGroup] = MakeErrorGroup( ...
      'GIRDER',[1 length(GIRDER)],'Offset',0, ...
      'All Girder Misalignments') ;
  statall = AddStackToMasterStack(statall,stat,'MakeErrorGroup') ;
  
% prepare the all-quad error group, using blocks.  For quads which have
% captive BPMs, this will also move the BPMs

  [stat,QuadMisalignGroup] = MakeErrorGroup( ...
      {'BEAMLINE','QUAD'},[1 20981],'Offset',1, ...
      'All Quadrupole Misalignments') ;
  statall = AddStackToMasterStack(statall,stat,'MakeErrorGroup') ;
  
% summary display

  disp('...Return status of all called functions:  ');
  disp('   should be a row of 1''s followed by some ''OK'' messages...')
  DisplayMessageStack(statall) ;
  
