function DualPSGradientBend( istart, iend )
%
% DUALPSGRADIENTBEND Configure gradient bend magnets to be powered by two
%    power supplies.
%
% DualPSGradientBend( start, end ) converts all gradient bend
%    magnets (bend magnets with a quadrupole term) within the range from
%    start to end from single power supply to dual power supply (ie, the
%    bending field and the quad field are excited by separate power
%    supplies), and at the same time converts the dB (field error) data
%    entry from a scalar to a vector.  For all converted magnets the second
%    PS entry and the second dB entry will be set to zero.  

%=========================================================================

  global BEAMLINE
  
% find the sector bends in the desired range

  sblist = findcells(BEAMLINE,'Class','SBEN',istart,iend) ;
  
% loop over sector bends  

  for count = sblist
    
    if (size(BEAMLINE{count}.B) == [1 1])
      continue ;
    end
    
% if there are multiple entries in B then it's a gradient bend so
% if necessary expand its PS and dB entries

    if (isfield(BEAMLINE{count},'PS'))
      if (size(BEAMLINE{count}.PS) == [1 1])
        BEAMLINE{count}.PS = [BEAMLINE{count}.PS 0] ;
      end
    end
    if (isfield(BEAMLINE{count},'dB'))
      if (size(BEAMLINE{count}.dB) == [1 1])
        BEAMLINE{count}.dB = [BEAMLINE{count}.dB 0] ;
      end
    end
    
  end
  
% and that's it!  