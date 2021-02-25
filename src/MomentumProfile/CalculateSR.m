function [stat,umean,varargout] = CalculateSR( P, BL, L )
%
% CALCULATESR Calculate synchrotron radiation parameters 
%
% [stat,umean] = CalculateSR( P, BL, L ) computes the mean energy loss
%    experienced by a particle with initial momentum P [GeV/c] which passes
%    through an element of effective length L [m] with integrated field BL
%    [T.m].  The returned loss is in GeV.  Return stat is a Lucretia status
%    cell array (type help LucretiaStatus for more information) with a
%    return value stat{1} == 1 indicating success, stat{1} == 0 indicating
%    L <= 0 bad argument, stat{1} == -2 indicating P <= 0 bad argument.
%
% [stat,umean,urms] = CalculateSR( P, BL, L ) also returns the RMS energy
%    loss (in GeV) for particles with the given parameters.
%
% [stat,umean,urms,uc] = CalculateSR( P, BL, L ) also returns the critical
%    energy (in GeV) for particles with the given parameters.
%
% [stat,umean,urms,uc,nphot] = CalculateSR( P, BL, L ) also returns the
%    mean number of emitted photons for particles with the given parameters.
%
% Version date:  06-Dec-2005.
%

%==========================================================================

  stat = InitializeMessageStack( ) ;
  umean = 0 ;
  if (nargout > 2)
    varargout{1} = 0 ;
  end
  if (nargout > 3)
    varargout{2} = 0 ;
  end
  if (nargout > 4)
    varargout{3} = 0 ;
  end
  if (nargout > 5)
    stat{1} = 0 ;
    stat  = AddMessageToStack(stat,...
       'Invalid output argument list for CalculateSR') ;
     for count = 6:nargout-5
       varargout{nargout} = 0 ;
     end
  end
  
  if (L<=0) 
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
        'Negative or zero element length encountered in CalculateSR') ;
  end
  if (P<=0) 
    stat{1} = -2 ;
    stat = AddMessageToStack(stat,...
        'Negative or zero momentum encountered in CalculateSR') ;
  end
  
  if stat{1} ~= 1
      return ;
  end
  
% now we can do some calculating...

  BL = abs(BL) ; B = BL / L ;
  umean = 1.2654e-6 * P^2 * BL * B ;         % mean energy loss
  if (nargout>2)                             % RMS energy loss
      varargout{1} = sqrt(1.1133e-12*P^4 * BL * B^2 ) ;
  end
  if (nargout>3)                             % critical energy in GeV
    varargout{2} = 6.6501369e-7 * P^2 * B ;    
  end
  if (nargout > 4)                           % total number of photons
    varargout{3} = 6.1793319 * BL ;
  end
  

   