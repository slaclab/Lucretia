function [nx,ny,nt] = GetNEmitFromBeam( beam, bunchno, varargin )
%
% GETNEMITFROMBEAM Get the normalized emittances from a beam data
%    structure.
%
% [nx,ny,nt] = GetNEmitFromBeam( beam, bunchno ) computes and returns the
%    projected emittances for bunch number bunchno in beam data structure
%    beam.  If the bunch number is zero, all the bunches are considered as
%    an ensemble.  If the bunch number is -1, the emittances of each bunch
%    are returned.
%
% [nx,ny,nt] = GetNEmitFromBeam( beam, bunchno, 'normalmode' ) returns the
%    normal-mode emittances rather than the projected emittances.
%
% See also:  GetBeamPars, GetNEmitFromSigmaMatrix.
%
% Version date:  16-Jan-2007.
%

% MOD:
%
%==========================================================================

  nx = [] ; ny = [] ; nt = [] ;

  if (nargin > 2)
      textstring = varargin{1} ;
  end

% start with the beam pars

  [x,sig] = GetBeamPars( beam, bunchno ) ;
  
% now do the emittance calculation

  if (bunchno >= 0)
      bunchstart = 1 ; bunchend = 1 ;
  else
      bunchstart = 1 ; bunchend = length(beam.Bunch) ; 
  end
  
  for count = bunchstart:bunchend
    if (nargin == 2)
     [nx0,ny0,nt0] = GetNEmitFromSigmaMatrix(x(6,count), ...
                                             sig(:,:,count)) ;
    else
     [nx0,ny0,nt0] = GetNEmitFromSigmaMatrix(x(6,count), ...
                                             sig(:,:,count), ...
                                             textstring) ;
    end
    nx = [nx ; nx0] ; ny = [ny ; ny0] ; nt = [nt ; nt0] ;
  end