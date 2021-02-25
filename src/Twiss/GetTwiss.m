% GETTWISS Compute twiss parameters over a range.
%
% [STAT,T] = GetTwiss( START, FINISH, XTWISSI, YTWISSI ) Returns a data
%    structure containing the linear optics ("twiss") functions of the
%    BEAMLINE from the upstream face of the element indexed by START to the
%    downstream face of the element indexed by FINISH.  XTWISSI and YTWISSI
%    are data structures with the initial conditions:  each is a 1x1
%    structure with fields beta, alpha, eta, etap, nu. If START > FINISH,
%    GetTwiss will use XTWISSI and YTWISSI as the target Twiss parameters
%    for the end of the line. Data structure T contains the following
%    vectors, all dimensioned 1 x (finish-start + 2):
%
%    T.S = s positions [m]
%    T.E = design momenta [GeV/c]
%    T.betax  = horizontal betatron functions [m]
%    T.alphax = horizontal alpha [-]
%    T.etax   = horizontal dispersion [? either m or m/(GeV/C)]
%    T.etapx  = horizontal momentum dispersion [? either - or 1/(GeV/c)]
%    T.nux    = horizontal phase advance [rad/2pi]
%    T.betay  = vertical betatron functions [m]
%    T.alphay = vertical alpha [-]
%    T.etay   = vertical dispersion [? either m or m/(GeV/C)]
%    T.etapy  = vertical momentum dispersion [? either - or 1/(GeV/c)]
%    T.nuy    = vertical phase advance [rad/2pi]
%
% STAT is a Lucretia status and message cell array (type help
%    LucretiaStatus for more information).  Return values in STAT{1} can be 
%    +1 (completed without incident) or 0 (unable to complete due to
%    errors).  
%
% [STAT,T] = GetTwiss( START, FINISH, CTWISS ) propagates the fully-coupled
%    Twiss functions in the Wolski notation.  CTWISS is a 6 x 6 x n array,
%    where n is between 1 and 3.  Returned data structure T has the form:
%
%    T.S = s positions [m]
%    T.P = design momenta [GeV/c]
%    T.beta = 6 x 6 x n x |FINISH - START + 2| matrix of coupled Twiss
%             parameters.
%
% For more information on the Wolski notation for coupled Twiss parameters,
% see PRST-AB 9, 024001 (2006).
%
% V = GetTwiss( "version" ) Returns the version dates of all key components
%    in Lucretia.
%
% See also:  CoupledTwissFromInitial
