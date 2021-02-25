function twiss = ReturnMatchedTwiss( istart, iend, plane )
%
% RETURNMATCHEDTWISS Return the matched Twiss parameters for a cell.
%
% twiss = ReturnMatchedTwiss(istart, iend, plane) will compute the matched
%    Twiss parameters for a cell defined by istart and istop, in a plane
%    given by plane (1=x, 2=y).  The results are returned in a format
%    suitable for use in a GetTwiss operation.  Uncoupled optics is
%    assumed.
%
% See also:  GetTwiss, RmatAtoB.
%
% Version Date:  10-Jan-2007.
%

% MOD:
%      PT, 10-jan-2007:
%          add support for dispersion calculations.
%
%=================================================================

if (plane == 1)           % horizontal
    pos = 1 ; ang = 2 ;
else                      % vertical
    pos = 3 ; ang = 4 ;  
end


[~,R] = RmatAtoB(istart,iend) ;

cosphi = 0.5 * trace(R(pos:ang,pos:ang)) ;
if (abs(cosphi)>1)
  disp(cosphi);
   error('Unstable motion detected') ;
end

phi = acos(cosphi) ;

twiss.beta = abs(R(pos,ang) / sin(phi)) ;
twiss.alpha = (R(pos,pos) - cosphi)/sin(phi) ;
%twiss.eta = 0 ; twiss.etap = 0 ; twiss.nu = phi/2/pi ;
twiss.eta = ( R(pos,ang)*R(ang,6) - (R(ang,ang)-1)*R(pos,6)  ) / ...
    ( (R(pos,pos)-1)*(R(ang,ang)-1) - R(ang,pos)*R(pos,ang) ) ;
twiss.etap = ( -R(ang,6) - R(ang,pos)*twiss.eta ) / (R(ang,ang)-1) ;
twiss.nu = phi/2/pi ;
