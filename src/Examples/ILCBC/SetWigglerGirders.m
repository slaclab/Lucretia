function stat = SetWigglerGirders( istart, iend )

  global BEAMLINE 

% assign wiggler magnets to girders

  stat = {1} ;

  statcall = SetGirderByBlock(istart,iend,0) ;
  if (statcall{1} ~= 1)
    stat = statcall ;
    return ;
  end
  
  count = istart;
  while (count<=iend) 
      
      if (strcmp(BEAMLINE{count}.Class,'SBEN'))
        g = BEAMLINE{count}.Girder ;
        stat = ConvertGirderToLong(g) ;
      end
      count = count + 1 ;
      
  end