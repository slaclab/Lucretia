function SteerY121( steerstruc, beam, niter )

  global PS BEAMLINE ;

% perform simple 1:1 steering in the vertical plane, in segments, with some
% number of iterations per segment, using a selected beam definition

% start by figuring out the region count

  nregion = length(steerstruc) ;

% initialize the start-tracking index

  start_track = 1 ;
  beamin = beam ;
  
% loop over regions and over iterations

  for rcount = 1:nregion
    end_track = steerstruc(rcount).lastelem ;
    for icount = 1:niter
      [stat,beamout,instdata] = TrackThru(start_track,end_track,beamin,1,1,0) ;
      [S,x,y] = GetBPMvsS( instdata{1} ) ;
      y = y(:) ;
      theta = steerstruc(rcount).xfermat \ y ;
      for ecount = 1:length(theta)
        psno = steerstruc(rcount).ycorps(ecount) ;
        elno = steerstruc(rcount).ycorelem(ecount) ;
        brho = BEAMLINE{elno}.P / 0.299792458 ;
        PS(psno).SetPt = PS(psno).Ampl - theta(ecount)*brho ;
      end
      stat = PSTrim(steerstruc(rcount).ycorps) ;
    end
    [stat,beamout] = TrackThru(start_track,end_track,beamin,1,1,0) ;
    beamin = beamout ;
    start_track = end_track + 1 ;
  end
  
      