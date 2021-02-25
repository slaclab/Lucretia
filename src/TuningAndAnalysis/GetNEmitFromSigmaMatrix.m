function [nx,ny,nt] = GetNEmitFromSigmaMatrix( P, sigma, varargin )
%
% GETNEMITFROMSIGMAMATRIX Compute normalized emittances from the matrix
% of second moments of a beam.
%
% [nx,ny,nt] = GetNEmitFromSigmaMatrix( P, sigma ) computes the normalized
%    emittances in the 3 degrees of freedom from the given sigma matrix and
%    central momentum.  All 3 emittances are returned in m.rad.  The
%    calculation is a projected emittance using the determinants of the
%    appropriate submatrices of the 6 x 6 sigma.
%
% [nx,ny,nt] = GetNEmitFromSigmaMatrix( P, sigma, 'normalmode' ) computes
%    normal-mode emittances by extracting the eigenvalues of sigma*S, where
%    S is the 6-dimensional anti-symmetric unit matrix.
%
% SeeAlso:  GetNEmitFromBPMData.
%

% MOD:  21-jun-2005, PT:
%          Improved logic for finding the normal mode emittances.

%==========================

persistent S ;
if (isempty(S))
  S = [ 0  1  0  0  0  0 ;
       -1  0  0  0  0  0 ;
        0  0  0  1  0  0 ;
        0  0 -1  0  0  0 ;
        0  0  0  0  0  1 ;
        0  0  0  0 -1  0  ] ;
end

normalmode = 0 ;
if (nargin > 3)
    error('Invalid number of arguments')
end
if (nargin == 3)
  if (ischar(varargin{1}))
    if (strcmpi(varargin{1},'normalmode'))
      normalmode = 1 ;
    end
  end
end

numbunches = length(P);

nx = zeros(1,numbunches) ; ny = zeros(1,numbunches) ; nt = zeros(1,numbunches) ;

% loop over bunches
for bunch=1:numbunches

% compute projected emittances...

  if (normalmode == 0)
    nx(bunch) = sqrt( det( squeeze( sigma(1:2,1:2,bunch) ) ) ) ; 
    ny(bunch) = sqrt( det( squeeze( sigma(3:4,3:4,bunch) ) ) ) ; 
    nt(bunch) = sqrt( det( squeeze( sigma(5:6,5:6,bunch) ) ) ) ;
  
% if normal-mode emittances are desired compute them now:

  else 
    [d,v] = eig(squeeze(sigma(:,:,bunch)) * S) ;
    
% identify eigenmodes

    dvec = max(d(1:2,:)) ;
    [~,b] = max(dvec) ;
    nx(bunch) = imag(v(b,b)) ;
    dvec = max(d(3:4,:)) ;
    [~,b] = max(dvec) ;
    ny(bunch) = imag(v(b,b)) ;
    dvec = max(d(5:6,:)) ;
    [~,b] = max(dvec) ;
    nt(bunch) = imag(v(b,b)) ;
    
  
  end

% At the moment the nx and ny are not normalized to gamma, and nt is in
% GeV.m.  Convert to correct units and normalize.
  gamma = sqrt(1+(P(bunch)/0.5109989461e-3)^2) ; beta=sqrt(1-gamma^-2);
  nx(bunch) = abs(nx(bunch) * gamma * beta); ny(bunch) = abs(ny(bunch) * gamma * beta);
  nt(bunch) = abs(nt(bunch) / 0.5109989461e-3) ;
  
end
