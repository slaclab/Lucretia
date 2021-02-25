function elist = GetIndicesBySPosition(smin,smax,varargin)
%
% GETINDICESBYSPOSITION Find the indices of Lucretia elements within a
% specified range of S positions.
%
% list = GetIndicesBySPosition(smin,smax) finds all elements in BEAMLINE
% such that BEAMLINE{}.S >= smin AND BEAMLINE{}.S <= smax.  
%
% list = GetIndicesBySPosition(smin,smax,imin,imax) finds all elements in
% BEAMLINE which are within the desired S range and are also within the
% limits of indices imin and imax.  This is useful in finding elements by
% S position when more than one distinct line is represented in BEAMLINE.
%
% Version date:  21-Mar-2007.
%

%==========================================================================

  global BEAMLINE

% argument checking

  if ( (nargin ~= 2) & (nargin ~= 4) )
      error('Bad arguments to GetIndicesBySPosition!') ;
  end
  if (nargin == 4)
      i1 = varargin{1} ; i2 = varargin{2} ;
  else
      i1 = 1 ; i2 = length(BEAMLINE) ;
  end
  
  elist = [] ;
  
% do the work

  for count = i1:i2
      if ( (BEAMLINE{count}.S >= smin) & (BEAMLINE{count}.S <= smax) )
          elist = [elist count] ;
      end
  end
  