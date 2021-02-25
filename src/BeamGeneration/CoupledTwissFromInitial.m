function C = CoupledTwissFromInitial( Initial )
%
% COUPLEDTWISSFROMINITIAL Compute coupled Twiss parameters in Wolski
%    formulation from data in Initial structure.
%
% C = CoupledTwissFromInitial( Initial ) will use a Lucretia initial
%    conditions data structure to compute the 6 x 6 x 3 coupled initial
%    Twiss functions, which are returned in array C.
%
% See also:  InitCondStruc.
%
% Version date:  09-mar-2006.

%==========================================================================

% initialize to zero

  C = zeros(6,6,3) ; 
  
% set the easy parameters

  C(1,1,1) =  Initial.x.Twiss.beta ;
  C(1,2,1) = -Initial.x.Twiss.alpha ;
  C(2,1,1) = -Initial.x.Twiss.alpha ;
  C(2,2,1) = (1+(Initial.x.Twiss.alpha)^2)/Initial.x.Twiss.beta ;
  
  C(3,3,2) =  Initial.y.Twiss.beta ;
  C(3,4,2) = -Initial.y.Twiss.alpha ;
  C(4,3,2) = -Initial.y.Twiss.alpha ;
  C(4,4,2) = (1+(Initial.y.Twiss.alpha)^2)/Initial.y.Twiss.beta ;
  
% now for somewhat more difficult things:  compute the longitudinal emittance
% and the total energy spread

  eps3 = Initial.SigPUncorrel * Initial.sigz ;
  sigP = sqrt(Initial.SigPUncorrel^2 + Initial.sigz^2 * Initial.PZCorrel^2) ;
  sigz = Initial.sigz ;
  sig56 = Initial.PZCorrel * sigz^2 ;
  
% purely longitudinal parameters 

  C(5,5,3) = sigz^2 / eps3 ;
  C(6,6,3) = sigP^2 / eps3 ;
  C(5,6,3) = sig56 / eps3 ;
  
% dispersion terms

  C(1,6,3) = Initial.x.Twiss.eta * C(6,6,3) ;
  C(2,6,3) = Initial.x.Twiss.etap * C(6,6,3) ;
  C(3,6,3) = Initial.y.Twiss.eta * C(6,6,3) ;
  C(4,6,3) = Initial.y.Twiss.etap * C(6,6,3) ;
  