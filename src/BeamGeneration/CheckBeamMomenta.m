function beamout = CheckBeamMomenta( beamin )
%
% CHECKBEAMMOMENTA Check the transverse and longitudinal momenta of a beam.
%
% BeamOut = CheckBeamMomenta( BeamIn ) checks to make sure that the total
%    momentum of each ray in a beam is > 0, and that the total transverse
%    momentum of each ray is < 1.  If any ray violates these conditions,
%    its "stop" parameter is set and a warning is issued.  The returned
%    beam data structure is a copy of the beam data structure in the
%    argument with appropriate "stop" parameters set.
%
% Version date:  24-may-2007.

%==========================================================================

% copy the input to the output

  beamout = beamin ;
  
  P0flag = 0 ; Pperpflag = 0 ;
  
% loop over bunches and over rays

  for buncount = 1:length(beamout.Bunch)
    P0 = beamout.Bunch(buncount).x(6,:) ;
    badP0 = find(P0 <= 0) ;
    for rcount = badP0 
      if (beamout.Bunch(buncount).stop(rcount) == 0)
        beamout.Bunch(buncount).stop(rcount) = 1 ;
        if (P0flag==0)
          warning('Un-stopped rays with non-positive total momentum found') ;
          P0flag = 1 ;
        end
      end
    end
  
    
    Px = beamout.Bunch(buncount).x(2,:) ;
    Py = beamout.Bunch(buncount).x(4,:) ;
    Pperp = Px .* Px + Py .* Py ;
    badPperp = find(Pperp >= 1) ;
    for rcount = badPperp
      if (beamout.Bunch(buncount).stop(rcount) == 0)
        beamout.Bunch(buncount).stop(rcount) = 1 ;
        if (Pperpflag==0)
          warning('Un-stopped rays with transverse momentum >= total momentum found') ;
          Pperpflag = 1 ;
        end
      end
    end
    
  end
        