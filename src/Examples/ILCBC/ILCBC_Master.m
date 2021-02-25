%
% master script for the ILC-BC Lucretia example:
%
%==========================================================================

%
% first, instantiate the beamline
%
  MakeOneStage ;
%
% compute the Twiss functions and plot
%
  disp('...Generating Twiss parameters...')
  [stat,Twiss] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
  if (stat{1} == 1)
    figure ;
    subplot(3,1,1) ;
    plot(Twiss.S/1000,Twiss.betax) ;
    ylabel('\beta_x [m]') ;
    title('Twiss parameters of Single Stage BC + Linac')
    subplot(3,1,2) ;
    plot(Twiss.S/1000,Twiss.betay) ;
    ylabel('\beta_y [m]') ;
    subplot(3,1,3)
    plot(Twiss.S/1000,Twiss.etax) ;
    ylabel('\eta_x [m]') ;
    xlabel('S position [km]') ;
  else
    disp('Problem in GetTwiss!  Halting execution!') ;
    return ;
  end
  pause(5) ;
%
% Track the beam on-axis and look at its emittances
%
  disp('...tracking on-axis single-bunch sparse beam...')
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse2121,...
                                      1,1,0) ;
  if (stat{1} == 1)
    [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1} ) ;
    figure ; 
    subplot(2,1,1)
    plot(S/1000,nx*1e6) ;
    axis([0 14 8 8.8]) ;
    hold on
    ylabel('\gamma\epsilon_x [{\mu}m]') ;
    title('Emittance for On-axis beam, One-Stage BC+Linac') ;
    subplot(2,1,2) ;
    plot(S/1000,ny*1e9) ;
    hold on
    ylabel('\gamma\epsilon_y [nm]') ;
    xlabel('S position [km]') ;
    [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1}, 'normalmode' ) ;
    subplot(2,1,1)
    plot(S/1000,nx*1e6,'r--') ;
%    legend('Projected','Normal Mode') ;
    subplot(2,1,2) ;
    plot(S/1000,ny*1e9,'r--') ;
  else
    disp('Problem in TrackThru!  Halting Execution!') ;
    return ;
  end
  pause(5) ;
%
% now for the 1 sigy beam
%
  disp('...tracking single-bunch sparse beam with 1 sigy offset...')
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beam1sigy,...
                                      1,1,0) ;
  if (stat{1} == 1)
    [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1} ) ;
    figure ; 
    subplot(2,1,1)
    plot(S/1000,nx*1e6) ;
    axis([0 14 8 8.8]) ;
    hold on
    ylabel('\gamma\epsilon_x [{\mu}m]') ;
    title('Emittance for beam with 1 \sigma_y offset, One-Stage BC+Linac') ;
    subplot(2,1,2) ;
    plot(S/1000,ny*1e9) ;
    hold on
    ylabel('\gamma\epsilon_y [nm]') ;
    xlabel('S position [km]') ;
    [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1}, 'normalmode' ) ;
    subplot(2,1,1)
    plot(S/1000,nx*1e6,'r--') ;
%    legend('Projected','Normal Mode') ;
    subplot(2,1,2) ;
    plot(S/1000,ny*1e9,'r--') ;
  else
    disp('Problem in TrackThru!  Halting Execution!') ;
    return ;
  end
  pause(5) ;
%
% track the Gaussian beam and examine its phase-space distribution at the
% end of the BC and end of the linac
%
  disp('...Tracking dense beam with 10,000 rays to end BC...')
  [stat,beamout] = TrackThru(1,endbc,beamgauss10k,1,1,0) ;
  if (stat{1} == 1)
    [x,sigma] = GetBeamPars(beamout,1) ;
    sigz = sqrt(sigma(5,5)) ;
    sigdelta = sqrt(sigma(6,6))/x(6) ;
    figure ;
    subplot(2,2,3) ;
    plot(beamout.Bunch(1).x(5,:)*1000,...
         beamout.Bunch(1).x(6,:),'.'      ) ;
    axis([-2 2 3.5 5.5]) ;
    xlabel('z position, mm') ;
    ylabel('Momentum, GeV/c') ;
    subplot(2,2,1)
    [center,height] = BeamHistogram( beamout, 1, 5, 0.25 ) ;
    bar(center*1000,height*1e9) ;
    axis([-2 2 0 0.4]) ;
    title(['z RMS = ',num2str(1000*sigz),' mm']) ;
    subplot(2,2,4) ;
    [center,height] = BeamHistogram( beamout, 1, 6, 0.25 ) ;
    barh(center,height*1e9) ;
    axis([0 0.4 3.5 5.5]) ;
    title(['\delta RMS = ',num2str(100*sigdelta),'%']) ;
  else
    disp('Problem in TrackThru!  Halting Execution!') ;
    return ;
  end
  pause(5) ;
%  
  disp('...Tracking dense beam with 10,000 rays to end of linac...')
  [stat,beamout] = TrackThru(endbc+1,length(BEAMLINE),beamout,1,1,0) ;
  if (stat{1} == 1)
    [x,sigma] = GetBeamPars(beamout,1) ;
    sigz = sqrt(sigma(5,5)) ;
    sigdelta = sqrt(sigma(6,6))/x(6) ;
    figure ;
    subplot(2,2,3) ;
    plot(beamout.Bunch(1).x(5,:)*1000,...
         beamout.Bunch(1).x(6,:),'.'      ) ;
    axis([-2 2 249 252]) ;
    xlabel('z position, mm') ;
    ylabel('Momentum, GeV/c') ;
    subplot(2,2,1)
    [center,height] = BeamHistogram( beamout, 1, 5, 0.25 ) ;
    bar(center*1000,height*1e9) ;
    axis([-2 2 0 0.4]) ;
    title(['z RMS = ',num2str(1000*sigz),' mm']) ;
    subplot(2,2,4) ;
    [center,height] = BeamHistogram( beamout, 1, 6, 0.25 ) ;
    barh(center,height*1e9) ;
    axis([0 0.4 249 252]) ;
    title(['\delta RMS = ',num2str(100*sigdelta),'%']) ;
  else
    disp('Problem in TrackThru!  Halting Execution!') ;
    return ;
  end
  pause(5) ;
%  
% vary the BC RF phase and look at longitudinal phase space
%
  disp('...Plotting end-linac Z parameters vs BC Phase Error...') 
  bcphase = linspace(-1,1,21) ;
  beamsuperstructure = [] ;
  for count = 1:length(bcphase)
    KLYSTRON(1).dPhase = bcphase(count) ;
    KLYSTRON(2).dPhase = bcphase(count) ;
    [stat,beamout] = TrackThru(1,length(BEAMLINE),beamflat1111,1,1,0) ;
    if (stat{1} ~= 1)
      disp('Problem in TrackThru!  Halting Execution!') ;
      return ;
    end
    beamsuperstructure = [beamsuperstructure beamout] ;
  end
  KLYSTRON(1).dPhase = 0 ;
  KLYSTRON(2).dPhase = 0 ;
  PlotPZVariation( bcphase, beamsuperstructure, beamsuperstructure(11) ) ;
  title('BC phase') ;
  pause(5) ;
%
% Generate steering matrices for 1:1 steering
%
  disp('...Generating 1:1 steering matrices...')
  steerstruc = Make121YSteerStruc(1,length(BEAMLINE),50) ;
%
% introduce misalignments in structures, girders, BPMs, quads
%
  disp('...introducing misalignments...')
%
% 300 um RMS structure misalignments, and 300 urad pitches
%
  AllMisyMean = [0 0 0 0 0 0] ;
  RFMisyRMS = [0 0 300e-6 300e-6 0 0] ;
  keepyes = [1 1 1 1 1 1] ; keepno = [0 0 0 0 0 0] ;
  [stat,values] = ErrorGroupGaussErrors( RFMisalignGroup,...
                                         AllMisyMean,RFMisyRMS,...
                                         keepno ) ;
  if (stat{1} ~= 1)
    disp('Problem in ErrorGroupGaussErrors!  Halting Execution!') ;
    return ;
  end
%
% 200 um girder misalignments
%
  GirdMisyRMS = [0 0 200e-6 0 0 0] ;
  [stat,values] = ErrorGroupGaussErrors( GirderMisalignGroup,...
                                         AllMisyMean,GirdMisyRMS,...
                                         keepno ) ;
  if (stat{1} ~= 1)
    disp('Problem in ErrorGroupGaussErrors!  Halting Execution!') ;
    return ;
  end
%
% 200 um RMS BPM misalignments 
%
  BPMMisyRMS = [0 0 200e-6 0 0 0] ;
  [stat,values] = ErrorGroupGaussErrors( BPMMisalignGroup,...
                                         AllMisyMean,BPMMisyRMS,...
                                         keepno ) ;
  if (stat{1} ~= 1)
    disp('Problem in ErrorGroupGaussErrors!  Halting Execution!') ;
    return ;
  end
%
% 300 um RMS quad misalignments, plus 300 urad tilts; here we keep the
% existing errors since some BPMs are captive, and we want the captive BPMs
% to have the errors applied above (bpm-to-quad) plus the ones below
% (quad-to-survey-line)
% 
  QuadMisyRMS = [0 0 300e-6 0 0 300e-6] ;
  [stat,values] = ErrorGroupGaussErrors( QuadMisalignGroup,...
                                         AllMisyMean,QuadMisyRMS,...
                                         keepyes ) ;
  if (stat{1} ~= 1)
    disp('Problem in ErrorGroupGaussErrors!  Halting Execution!') ;
    return ;
  end
%
% set BPM resolution to 5 um and steer flat
%
  disp('...setting BPM resolution to 5 um...')
  list = findcells(BEAMLINE,'Class','MONI') ;
  for count = list
    BEAMLINE{count}.Resolution = 5e-6 ;
  end
%
  disp('...steering flat in segments with BPM resolution limit...')
  SteerY121(steerstruc,beamflat1111,3) ;
  disp('...steering complete, plotting final orbit + emittances...')
  [stat,beamout,instdata] = TrackThru(1,length(BEAMLINE),beamsparse2121,...
                                      1,1,0) ;
  if (stat{1} ~= 1)
    disp('Problem in TrackThru!  Halting Execution!') ;
    return ;
  end
  [x,y,S] = BPMZplot( instdata{1},'um','um','km' ) ;
  subplot(2,1,1)
  ylabel('X BPM Reading [{\mu}m]') ;
  subplot(2,1,2)
  ylabel('Y BPM Reading [{\mu}m]') ;
  xlabel('S Position [km]') ;
  [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1} ) ;
  figure ; 
  subplot(2,1,1)
  plot(S/1000,nx*1e6) ;
  axis([0 14 8 8.8]) ;
  hold on
  ylabel('\gamma\epsilon_x [{\mu}m]') ;
  title('Emittance after misaligning + steering One-Stage BC+Linac') ;
  subplot(2,1,2) ;
  plot(S/1000,ny*1e9) ;
  hold on
  ylabel('\gamma\epsilon_y [nm]') ;
  xlabel('S position [km]') ;
  [S,nx,ny,nt] = GetNEmitFromBPMData( instdata{1}, 'normalmode' ) ;
  subplot(2,1,1)
  plot(S/1000,nx*1e6,'r--') ;
%  legend('Projected','Normal Mode') ;
  subplot(2,1,2) ;
  plot(S/1000,ny*1e9,'r--') ;
  pause(5) ;
%
% setup the long-range wakefields use Roger Jones' well-damped XY_0 wake
% for all structures
%
  disp('...Setting up frequency-domain long range wakefields...')
  [stat,W] = ParseFrequencyDomainTLR('LBand_Freq_Wake_XY_0.dat',0) ;
  if (stat{1} ~= 1)
    disp('Unable to parse LRWF!  Halting execution!') ;
    return ;
  end
  WF.TLR{1} = W ;
  list = findcells(BEAMLINE,'Class','LCAV') ;
  for count = list
    BEAMLINE{count}.Wakes(3) = 1 ;
  end
  list = SetTrackFlags('LRWF_T',1,1,length(BEAMLINE)) ;
  disp('...Tracking multi-bunch beam thru steered misaligned beamline...')
  [stat,beamout] = TrackThru(1,length(BEAMLINE),beamMB,1,1000,0) ;
  if (stat{1} ~= 1)
    disp('Problems in TrackThru!  Halting execution!') ;
    return
  end
  [xMB,sigMB] = GetBeamPars(beamout,-1) ;
  figure
  plot(xMB(3,:)*1e6)
  xlabel('Bunch Number') ;
  ylabel('Y position [{\mu}m]') ;
%
