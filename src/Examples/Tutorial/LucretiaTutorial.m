% LucretiaTutorial -- Lucretia simple example which illustrates some
% (most?) of Lucretia's most interesting features.  This script
% (LucretiaTutorial.m) is the "master" script, which calls everything else.
%  Hopefully this is a useful starting point for people who want to master
%  Lucretia's tricks and features.
%
% Version date:  11-Mar-2008.

% Author:  PT
%
% Modification History:
%
%==========================================================================
%==========================================================================

% Short abstract:  in this file, we will:
%
%    Generate a linac lattice which accelerates from 5 GeV to 10 GeV, and
%    which includes a 4-bend chicane at the end to measure momentum and
%    momentum spread.  All necessary correctors and instrumentation will be
%    included.
%
%    Apply klystrons, power supplies, and girder definitions to the
%    resulting lattice.
%
%    Demonstrate a variety of Lucretia features on the generated lattice,
%    including:
%        Twiss function calculation and plotting
%        Floor coordinate calculation
%        Tracking, including generation of instrument readings
%        Application of errors 
%        Adjustment of the momentum profile

%==========================================================================
%==========================================================================

  global BEAMLINE GIRDER KLYSTRON WF %#ok<NUSED>
  
% step 1:  generate the lattice and load the wakefields  
  
  statall = GenerateTutorialLattice() ;  
  if (sum(statall{1}) ~= length(statall)-1)
      error('Problem encountered in GenerateTutorialLattice') ;
  end

% fill a data structure of initial conditions, assuming a charge of 3.2 nC
% (about 2e10 particles) per bunch, 1000 bunches, and a 400 bucket spacing

  [stat,Initial,nuxdeg,nuydeg] = GenerateTutorialInitial(3.2e-9, 1000, 400 / 1.3e9 ) ;
  statall = AddStackToMasterStack(statall,stat,'GenerateTutorialInitial') ;
  if (stat{1} ~= 1)
      error('Problem encountered in GenerateTutorialInitial') ;
  end
  
% display the phase advance per cell, which is going to be slightly
% different from 120 degrees since the calculation of quad strengths was
% performed in the thin-lens approximation
  
  disp(['Phase advance per cell, x:  ',num2str(nuxdeg),' degrees.']) ;    
  disp(['Phase advance per cell, y:  ',num2str(nuydeg),' degrees.']) ; 
  
% set the tracking flags as follows:
%
%    longitudinal motion simulation in quads and drifts OFF;
%    wakefields ON;
%    BPMs and instruments set to return readout data
%    BPMs and instruments set to produce bunch-by-bunch data

  SetTrackFlags('ZMotion',0,1,length(BEAMLINE)) ;
  SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
  SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;
  SetTrackFlags('LRWF_T',1,1,length(BEAMLINE)) ;
  SetTrackFlags('GetBPMData',1,1,length(BEAMLINE)) ;
  SetTrackFlags('GetInstData',1,1,length(BEAMLINE)) ;
  SetTrackFlags('MultiBunch',1,1,length(BEAMLINE)) ;
  
% Get and plot the Twiss functions

  [stat,T] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
  statall = AddStackToMasterStack(statall,stat,'GetTwiss') ;
  PlotTwiss(1,length(BEAMLINE),T,'Twiss of Tutorial Linac',[1 1 0]) ;

% get and plot the floor coordinates, using [0 0 0 0 0 0] as the final (not
% the initial) coordinates

  [stat,Coords] = SetFloorCoordinates(length(BEAMLINE),1,[0 0 0 0 0 0]) ;
  statall = AddStackToMasterStack(statall,stat,'SetFloorCoordinates') ;
  figure ; plot(Coords(:,3),Coords(:,1)) ;
  title('Floor Coordinates of Tutorial Linac') ;
  xlabel('Z Position [m]') ; ylabel('X Position [m]') ;
  
% Use tracking to calculate and apply the loss factor for the cavities

  stat = CalcAndApplyLossFactor( Initial ) ;
  statall = AddStackToMasterStack(statall,stat,'CalcAndApplyLossFactor') ;
  
% apply power supplies, girders, and klystrons to the beamline

  stat = SetPS_G_K ;
  statall = AddStackToMasterStack(statall,stat,'SetPS_G_K') ;
  
  
% In the current configuration, all the RF stations are on and at full
% gradient.  It's convenient to have one spare tube, and to set the final
% energy to be some reasonably convenient number.  So scale the RF to get
% to a convenient value with 19 tubes on the beam, set the last klystron to
% be a spare (put it in "standby" state), and rescale the optics to the new
% momentum profile.

  stat = SetStandbyKlystron( Initial.Q ) ;
  statall = AddStackToMasterStack(statall,stat,'SetStandbyKlystron') ;
  
% generate some beams and track them through the lattice

  beamsparse = MakeBeam6DSparse(Initial,3,11,11) ;
  beamdense = MakeBeam6DGauss(Initial,10000,4,1) ;
  
  disp('...tracking one bunch of sparse beam through the lattice...')
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch sparse') ;
  disp('...done!')
  [nx,ny] = GetNEmitFromBeam(beamout,1) ;
  disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
  disp(['Vertical norm emittance = ',num2str(ny*1e9),' nm']) ;
  disp(['x Beam size at wire scanner = ',...
      num2str(1e6*sqrt(instdata{2}.sig11)),' um']) ;
  disp(' ') ;
  BPMZplot(instdata{1}) ;  
  subplot(2,1,1) ; title('Initial orbit') ;
     
% Demonstrate some klystron complement fun and games -- trip off the first
% klystron and observe the resulting change in emittance (from mismatch)
% and orbit in the chicane

  KLYSTRON(1).Stat = 'TRIPPED' ;
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch sparse') ;
  [nx,ny] = GetNEmitFromBeam(beamout,1) ;
  disp('Trip off first RF station: ') ;
  disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
  disp(['Vertical norm emittance = ',num2str(ny*1e9),' nm']) ;
  disp(' ') ;
  BPMZplot(instdata{1}) ; 
  subplot(2,1,1) ; title('Energy mismatch orbit')
  
% now turn on the spare to make up the energy, but don't scale the linac

  KLYSTRON(end).Stat = 'MAKEUP' ;
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch sparse') ;
  [nx,ny] = GetNEmitFromBeam(beamout,1) ;
  disp('Turn on makeup station but don''t adjust the optics: ') ;
  disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
  disp(['Vertical norm emittance = ',num2str(ny*1e9),' nm']) ;
  disp(' ') ;
  BPMZplot(instdata{1}) ; 
  subplot(2,1,1) ; title('Orbit with corrected energy')
  
% finally, scale the linac and update the tube status

  stat = UpdateMomentumProfile(1,length(BEAMLINE),Initial.Q,BEAMLINE{1}.P,1) ;
  statall = AddStackToMasterStack(statall,stat,'UpdateMomentumProfile') ;
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch sparse') ;
  [nx,ny] = GetNEmitFromBeam(beamout,1) ;
  disp('Now scale the optics to the new momentum profile: ') ;
  disp(['Horizontal norm emittance = ',num2str(nx*1e6),' um']) ;
  disp(['Vertical norm emittance = ',num2str(ny*1e9),' nm']) ;
  disp(' ') ;
  [x,y,s] = BPMZplot(instdata{1}) ;  %#ok<ASGLU>
  subplot(2,1,1) ; title('Orbit with corrected energy and scaled')
  
% Track the dense beam and look at the y-py distribution at the end

  disp('...tracking 1 bunch of dense beam (10k rays) thru the linac...')
  [stat,beamout] = TrackThru(1,length(BEAMLINE),beamdense,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch dense') ;
  disp('...done!')
  x = GetBeamPars(beamout,1) ;
  figure ; plot( (beamout.Bunch.x(3,:)-x(3))*1e6, ...
                 (beamout.Bunch.x(4,:)-x(4))*1e6,'k.') ;
  xlabel('Y position [{\mu}m]') ; ylabel('Y momentum [{\mu}rad]') ;
  hold on ;
  
% now generate some RF cavity position errors and apply them

  [stat,CavityMisalign] = MakeErrorGroup({'BEAMLINE','LCAV'},[1 length(BEAMLINE)], ...
      'Offset', 2, 'Cavity Misalignments') ;
  statall = AddStackToMasterStack(statall,stat,'MakeErrorGroup') ;
  
  [stat,ErrorData] = ErrorGroupGaussErrors(CavityMisalign,[0 0 0 0 0 0],...
      [0 0 1e-3 0 0 0],[0 0 0 0 0 0]) ;
  statall = AddStackToMasterStack(statall,stat,'ErrorGroupGaussErrors') ;
  
  disp('...tracking 1 bunch of dense beam (10k rays) thru misaligned linac...')
  [stat,beamout] = TrackThru(1,length(BEAMLINE),beamdense,1,1,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1 bunch dense') ;
  disp('...done!')
  [x,sig] = GetBeamPars(beamout,1) ;
  plot( (beamout.Bunch.x(3,:)-x(3))*1e6, ...
        (beamout.Bunch.x(4,:)-x(4))*1e6,'m.') ;
  [nx0,ny0,nt0] = GetNEmitFromBeam(beamdense,1) ;
  [nx,ny,nt]    = GetNEmitFromBeam(beamout,1) ;
  disp(['Vertical emittance increased from ',num2str(ny0*1e9),' nm to ', ...
       num2str(ny*1e9),' nm.']) ;
  
  disp('...tracking 1000 bunches of sparse beam to see effect of LRWFs (takes a couple minutes)...')
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse,1,1000,0) ;
  statall = AddStackToMasterStack(statall,stat,'TrackThru 1000 bunches sparse') ;
  disp('...done!') 
  
% get the position and angle of each bunch, subtract the centroid of bunch
% 1 (from the previous track), and plot on same axes to see whether the
% effect of LRWFs is large or small compared to the size of a single bunch
% (ie, whether LRWFs are comparable in effect to SRWFs or not)
  
  [xt,sigt] = GetBeamPars(beamout,-1) ;
  plot( (xt(3,:)-x(3))*1e6, (xt(4,:)-x(4))*1e6,'g.' ) ;
  legend('Single bunch, no misalignments','Single bunch, misalignments',...
      'Multibunch centroids, misalignments') ;
  
% and last but not least, take a look at the master status array.  The
% first cell should be a vector of 1's (all good returns), and all the
% messages in the rest of the cells should end in , 'OK'

  DisplayMessageStack(statall) ;
  