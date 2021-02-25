function [S,nx,ny,nt] = GetNEmitFromBPMData( datastruc, varargin ) 
%
% GETNEMITFROMBPMDATA Compute normalized emittances at BPMs 
%
% [S,nx,ny,nt] = GetNEmitFromBPMData( bpmstruc ) computes the normalized
%    projected emittances at the BPMs for which the GetBPMBeamPars tracking
%    flag is set to 1.  Argument bpmstruc is the first cell of the third
%    return argument from TrackThru (type help TrackThru for more
%    information). Return values S, nx, ny, and nt are vectors of BPM S
%    positions and normalized emittances, respectively.
%
% [S,nx,ny,nt] = GetNEmitFromBPMData( bpmstruc, 'normalmode' ) computes the
%    normalized normal-mode emittances at each BPM.
%
% See also:  GetNEmitFromSigmaMatrix.
%

if (length(varargin) == 1)
  mode = varargin{1} ;
else
  mode = 'projected' ;
end

numbunches = length(datastruc(1).x);
n = length(datastruc) ; gx = zeros(n,numbunches) ; gy = gx ; S = gx ; gt = gx ;
S = zeros(1,n) ;
ngood = 0 ;
for count = 1:n
    if (~isempty(datastruc(count).sigma))
      ngood = ngood + 1 ;
      S(ngood) = datastruc(count).S ;
      [gx(ngood,:),gy(ngood,:),gt(ngood,:)] = GetNEmitFromSigmaMatrix( ...
                                          datastruc(count).P,...
                                          datastruc(count).sigma,...
                                          mode) ;
    end
end
S = S(1:ngood) ; nx = gx(1:ngood) ; ny = gy(1:ngood) ; nt = gt(1:ngood) ;