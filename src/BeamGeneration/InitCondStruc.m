function x = InitCondStruc( )
%
% INITCONDSTRUC Create blank data structure for beam initial conditions.
%
% x = InitCondStruc( ) returns a data structure of beam initial conditions,
%    which has the proper fields but is otherwise blank.
%
%===============================================================

x.Momentum = 1 ;                          % default to 1 GeV/c
%
% horizontal plane parameters 
%
x.x.NEmit = 0 ;
x.x.pos = 0 ;
x.x.ang = 0 ;
x.x.Twiss.beta = 1 ;
x.x.Twiss.alpha = 0 ;
x.x.Twiss.eta = 0 ;
x.x.Twiss.etap = 0 ;
x.x.Twiss.nu = 0 ;
%
% vertical plane parameters
%
x.y = x.x ;
%
% longitudinal parameters 
%
x.zpos = 0 ;
x.sigz = 0 ;
x.SigPUncorrel = 0 ;
x.PZCorrel = 0 ;
x.NBunch = 1;
x.BunchInterval = 1; 
x.Q = 1.60217653e-19 ;
%
% coupling parameters
%
x.CoupCoef.xy = 0 ;
x.CoupCoef.xpy = 0 ;
x.CoupCoef.xyp = 0 ;
x.CoupCoef.xpyp = 0 ;
x.CoupCoef.xz = 0 ;