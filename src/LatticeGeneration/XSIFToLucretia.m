function [stat,Initial] = XSIFToLucretia( filename, varargin )
% XSIFTOLUCRETIA Parse an XSIF deck and generate Lucretia data arrays
%
%   [stat,Initial] = XSIFToLucretia( filename ) parses the named XSIF file
%      and generates the Lucretia BEAMLINE and WF global data structures.
%      Existing WF and BEAMLINE structures are deleted. Return variable
%      stat is a cell array, with an integer status value followed by
%      specific text messages:  stat == 1 indicates success, stat == 0
%      indicates failure of the XSIF parser or no beamline selected by USE
%      statement, stat == -1 indicates that one or more wakefield could not
%      be parsed, and/or that a problem was encountered in setting the
%      momentum and a default value of 1 GeV/c has been used, and/or that
%      an element with an unknown or unsupported class was detected, and/or
%      that requested beta0 / beam elements could not be found. Return
%      variable Initial is the initial beam conditions captured from BEAM
%      and BETA0 statements, if any, in the XSIF deck.
%
% [stat,Initial] = XSIFToLucretia( filename, linename ) expands the named
%    line, which supercedes any lines expanded by USE statements in the
%    deck.
%
% [stat,Initial] = XSIFToLucretia( filename, linename, beta0, beam ) uses
%    the named beta0 and beam entries in the deck, which supersede any
%    beta0 or beam selected in the deck.
%
% See also XSIFParse, GetXSIFDictionary.
%
% Version Date:  22-may-2007.

% MOD:
%       22-may-2007, PT [SLAC]:
%           support for XYCORs.
%       22-jun-2006, PT [SLAC]:
%           bugfix to UnpackXSIFMultPars change below.
%       25-apr-2006, M. Woodley [SLAC]:
%           Handle zero-strength MULTs (change is in UnpackXSIFMultPars).
%       31-mar-2006, PT:
%           coord elements have names, too!
%       30-mar-2006, PT:
%           bugfix -- BEAMLINE is a column-vector, but algorithm for
%           adding elements to an existing beamline added them as though
%           BEAMLINE was a row-vector.
%       09-mar-2006, PT:
%          support for solenoids.
%       11-feb-2006, PT:
%           support for GKICK->Coord element.
%       18-oct-2005, PT:
%          If no BEAM or BETA0 is present, and momentum defaults to
%          1 GeV/c, set return status -> -1 and add a warning message.
%       28-sep-2005, PT:
%          Bugfix in conversion of K-to-B for sector bends, and in the
%          parsing of multipoles with all field components set to zero.

%=========================================================================

Initial = [] ; beta0name = [] ; beamname = [] ; linename = [] ;
beta0indx = 0  ; beamindx = 0 ;
stat = InitializeMessageStack( ) ;

% check arguments

if (~ischar(filename))
  error('First argument to XSIFToLucretia must be a char string') ;
end
if (nargin > 4)
  error('Too many arguments for XSIFToLucretia') ;
end
if (nargin >= 2)
  linename = varargin{1} ;
  if ( (~ischar(linename)) && (~isempty(linename)) )
    error('Second argument to XSIFToLucretia must be a char string') ;
  end
end
if (nargin >= 3)
  beta0name = varargin{2} ;
  if ( (~ischar(beta0name)) && (~isempty(beta0name)) )
    error('Third argument to XSIFToLucretia must be a char string') ;
  end
end
if (nargin == 4)
  beamname = varargin{3} ;
  if ( (~ischar(beamname)) && (~isempty(beamname)) )
    error('Third argument to XSIFToLucretia must be a char string') ;
  end
end

%
% parse the beamline and get data structures
%
if (isempty(linename))
  [S,E,P,L,B,W] = XSIFParse(filename) ;
else
  [S,E,P,L,B,W] = XSIFParse(filename,linename) ;
end

% handle any error messages from XSIF:  some values are unrecoverable
% errors, others are just warnings

switch S
  case -191
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  No line expanded') ;
  case -193
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  Syntax error') ;
  case -195
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  Couldn''t open a file') ;
  case -199
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  Fatal read error') ;
  case -201
    stat{1} = 0 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  Internal allocation error') ;
  case {-197,197}
    stat{1} = -1 ;
    stat = AddMessageToStack(stat,...
      'XSIFToLucretia:  Undefined parameters found') ;
end
if (stat{1} == 0)
  return ;
end

global BEAMLINE WF ;
%
% initialize badelm to a blank vector
%
badelm = [] ;
%
% unpack and instantiate the entries in the ELEMENT table
%
etable = cell(size(E)) ;
for count = 1:length(etable)
  
  ename = E(count).name ;
  etype = E(count).type ;
  edp   = E(count).data(1) ;
  
  switch( etype )
    
    case 1  % drift
      etable{count} = DrifStruc( P(edp).value, ename ) ;
    case {2,3} % bend magnets
      etable{count} = SBendStruc( P(edp).value, ...
        [P(edp+1).value P(edp+2).value],...
        P(edp+1).value, ...
        [P(edp+3).value P(edp+4).value],...
        [P(edp+7).value P(edp+8).value],...
        [P(edp+9).value P(edp+11).value],...
        [P(edp+10).value P(edp+12).value],...
        P(edp+5).value, ename ) ;
    case 5  % quad
      etable{count} = QuadStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, P(edp+3).value, ...
        ename ) ;
    case 6  % sextupole
      etable{count} = SextStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, P(edp+3).value, ...
        ename ) ;
    case 7  % octupole
      etable{count} = OctuStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, P(edp+3).value, ...
        ename ) ;
    case 8  % multipole
      [mB,mT,mPI,mA] = UnpackXSIFMultPars( P, edp ) ;
      etable{count} = MultStruc( P(edp).value, mB, mT, mPI, mA,...
        P(edp+44).value, ename ) ;
    case 9 % solenoid
      etable{count} = SolenoidStruc( P(edp).value, P(edp+1).value, ...
        P(edp+4).value, ename ) ;
    case { 12, 36 } % roll
      etable{count} = CoordStruc( 0, 0, 0, 0, 0, P(edp).value, ename ) ;
    case { 13, 35 } % rotation about y axis
      etable{count} = CoordStruc( 0, P(edp).value, 0, 0, 0, 0, ename ) ;
    case 14  % xcor
      etable{count} = CorrectorStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, 1, ename ) ;
    case 15  % ycor
      etable{count} = CorrectorStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, 2, ename ) ;
    case 39  % xycor
      etable{count} = CorrectorStruc( P(edp).value, ...
        [P(edp+1).value P(edp+2).value], ...
        P(edp+3).value, 3, ename ) ;
    case{ 16, 17, 18 } % BPM
      etable{count} = BPMStruc( P(edp).value, ename ) ;
    case 19  % marker
      etable{count} = MarkerStruc( ename ) ;
    case 20  % elliptical collimator
      etable{count} = CollStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, 'Ellipse', 0, ...
        ename ) ;
    case 21  % rectangular collimator
      etable{count} = CollStruc( P(edp).value, P(edp+1).value, ...
        P(edp+2).value, 'Rectangle', 0, ...
        ename ) ;
    case 23 % gkick
      if ( P(edp+10).value ~= 0 )
        stat{1} = 0 ;
        stat = AddMessageToStack(stat, ...
          'GKICK with T != 0 encountered in lattice') ;
        return ;
      end
      if ( P(edp).value ~= 0 )
        stat{1} = 0 ;
        stat = AddMessageToStack(stat, ...
          'GKICK with nonzero length encountered in lattice') ;
        return ;
      end
      % nb:  XSIF GKICK has two parameters related to the
      % longitudinal DOF, DL and DZ.  DL is used by Lucretia, DZ is
      % ignored.
      etable{count} = CoordStruc( -P(edp+1).value, -P(edp+2).value, ...
        -P(edp+3).value, -P(edp+4).value, ...
        -P(edp+5).value, -P(edp+7).value, ename ) ;
    case 26 % MATRIX (TMAP)
      etable{count} = TMapStruc(ename, P(edp).value ) ;
      pars=[11 12 13 14 15 16 111 112 113 114 115 116 122 123 124 125 126 ...    
            133 134 135 136 144 145 146 155 156 166 21 22 23 24 25 26 211 212 ...
            213 214 215 216 222 223 224 225 226 233 234 235 236 244 245 246 255 ...
            256 266 31 32 33 34 35 36 311 312 313 314 315 316 322 323 324 325 326 ...    
            333 334 335 336 344 345 346 355 356 366 41 42 43 44 45 46 411 412 413 ...
            414 415 416 422 423 424 425 426 433 434 435 436 444 445 446 455 456 ...
            466 51 52 53 54 55 56 511 512 513 514 515 516 522 523 524 525 526 533 ...
            534 535 536 544 545 546 555 556 566 61 62 63 64 65 66 611 612 613 614 ...
            615 616 622 623 624 625 626 633 634 635 636 644 645 646 655 656 666] ;
      vals=[P(edp+1:edp+length(pars)).value];
      Tinds=find(vals & pars>100);
      Rinds=find(vals & pars<100);
      if ~isempty(Tinds)
        etable{count}.T=zeros(length(Tinds),1);
        etable{count}.Tinds=zeros(length(Tinds),1);
      end
      for ipar=Rinds
        etable{count}.R(floor(pars(ipar)/10),mod(pars(ipar),10))=vals(ipar);
      end
      npar=1;
      for ipar=Tinds
        etable{count}.Tinds(npar,1)=pars(ipar);
        etable{count}.T(npar,1)=vals(ipar);
        npar=npar+1;
      end
    case 27  % lcav
      etable{count} = RFStruc( P(edp).value, P(edp+2).value, ...
        P(edp+3).value, P(edp+4).value, ...
        P(edp+7).value, P(edp+8).value, ...
        P(edp+9).value, P(edp+12).value, ...
        ename ) ;
      
      % there may be valid wakefields already resident in the WF structure, so
      % offset the wake indices in the new structure in etable
      
      
      
    case 28
      etable{count} = InstStruc( P(edp).value, 'INST', ename ) ;
    case 29
      etable{count} = InstStruc( P(edp).value, 'BLMO', ename ) ;
    case 30
      etable{count} = InstStruc( P(edp).value, 'PROF', ename ) ;
    case 31
      etable{count} = InstStruc( P(edp).value, 'WIRE', ename ) ;
    case 32
      etable{count} = InstStruc( P(edp).value, 'SLMO', ename ) ;
    case 33
      etable{count} = InstStruc( P(edp).value, 'IMON', ename ) ;
      
  end ; % SWITCH statement
  
end ; % loop over entries in E
%
% loop over L and count valid elements
%
nelm = 0 ;
for count = 1:length(L)
  
  if (L(count).element == 0)
    continue ;
  end
  elem = L(count).element ;
  if (isempty(etable{elem}))
    badelm = [badelm elem] ;
  else
    nelm = nelm + 1 ;
  end
  
end
nelmold = length(BEAMLINE) ;
BEAMLINE = [BEAMLINE ; cell(nelm,1)] ;
%
% repeat the loop and populate BEAMLINE
%
nelm = 0 ;
for count = 1:length(L)
  
  if (L(count).element == 0)
    continue ;
  end
  elem = L(count).element ;
  if (~isempty(etable{elem}))
    nelm = nelm + 1 ;
    BEAMLINE{nelmold+nelm} = etable{elem} ;
  end
  
end
%
% set S positions
%
SetSPositions( nelmold+1, nelmold+nelm, 0 );

% locate the requested BEAM and BETA0 elements, if possible

if (~isempty(beta0name))
  for betacount = 1:length(E)
    if ( (strcmpi(beta0name,E(betacount).name)) && ...
        (E(betacount).type == 37)                    )
      beta0indx = betacount ;
      break ;
    end
  end
  if (beta0indx == 0)
    stat = AddMessageToStack(stat, ...
      ['Requested Beta0 element ',beta0name,' not found']) ;
    stat{1} = -1 ;
  else
    B.betapointer = beta0indx ;
  end
end

if (~isempty(beamname))
  for betacount = 1:length(E)
    if ( (strcmpi(beamname,E(betacount).name)) && ...
        (E(betacount).type == 38)                    )
      beamindx = betacount ;
      break ;
    end
  end
  if (beamindx == 0)
    stat = AddMessageToStack(stat, ...
      ['Requested Beam element ',beta0name,' not found']) ;
    stat{1} = -1 ;
  else
    B.beampointer = beamindx ;
  end
end

% if there is / are BEAM / BETA0 pointers, set values into the Initial
% data structure.  If not, issue a warning and add a message.

if ( (B.betapointer == 0) && (B.beampointer==0) )
  stat = AddMessageToStack(stat, ...
    'No BETA0 or BEAM element found, initial momentum defaulting to 1 GeV/c') ;
  stat{1} = -1 ;
end

Initial = InitCondStruc ;
MomentumError = 0 ;
Momentum = 0 ; %#ok<NASGU>
Etot = 0 ;
if (B.betapointer ~= 0)
  edp = E(B.betapointer).data(1) ;
  Initial.x.pos = P(edp+10).value ;
  Initial.x.ang = P(edp+11).value ;
  Initial.x.Twiss.beta = P(edp).value ;
  Initial.x.Twiss.alpha = P(edp+1).value ;
  Initial.x.Twiss.eta = P(edp+6).value ;
  Initial.x.Twiss.etap = P(edp+7).value ;
  Initial.y.pos = P(edp+12).value ;
  Initial.y.ang = P(edp+13).value ;
  Initial.y.Twiss.beta = P(edp+3).value ;
  Initial.y.Twiss.alpha = P(edp+4).value ;
  Initial.y.Twiss.eta = P(edp+8).value ;
  Initial.y.Twiss.etap = P(edp+9).value ;
  Initial.zpos = -P(edp+14).value ;
  MomentumError = P(edp+15).value ;
  Energy = P(edp+26).value ; %#ok<NASGU>
end
if (B.beampointer ~= 0)
  edp = E(B.beampointer).data(1) ;
  Energy = P(edp+3).value ;
  PC = P(edp+4).value ;
  RelGamma = P(edp+5).value ;
  
  % set momentum, with the following priority:
  %  if P is nonzero, use it; otherwise, if Energy is nonzero, use it ;
  %   otherwise, if RelGamma is > 1, use it
  
  if (RelGamma > 1)
    Etot = RelGamma * 0.000510998918 ;
  end
  if (Energy > 0)
    Etot = Energy ;
  end
  if (PC>0)
    Momentum = PC ;
  else
    Momentum = Etot^2 - 0.000510998918^2 ;
    if (Momentum <=0)
      Momentum = 0 ;
    else
      Momentum = sqrt(Momentum) ;
    end
  end
  if (Momentum == 0)
    stat{1} = -1 ;
    stat = AddMessageToStack(stat,...
      'Problem in setting momentum, PC = 1 GeV/c used') ;
    Momentum = 1 ;
  end
  Initial.Momentum = Momentum * (1+MomentumError) ;
  
  % perform a similar operation with the emittances:  geometric emittances
  % take priority over normalized ones
  
  RelGamma = sqrt(Initial.Momentum^2 + 0.000510998918^2) / 0.000510998918 ;
  Initial.x.NEmit = P(edp+7).value / 4 ;
  Initial.y.NEmit = P(edp+9).value / 4 ;
  Initial.x.NEmit = P(edp+6).value * RelGamma ;
  Initial.y.NEmit = P(edp+8).value * RelGamma ;
  
  Initial.sigz = P(edp+11).value ;
  Initial.SigPUncorrel = Initial.Momentum * P(edp+12).value ;
  Initial.NBunch = P(edp+13).value ;
  Initial.Q = P(edp+14).value * 1.6e-19 ;
  if (P(edp+15).value == 0)
    if (Initial.NBunch > 1)
      stat{1} = -1 ;
      stat = AddMessageToStack(stat,...
        'Bunch interval not specified, setting to zero') ;
    end
    Initial.BunchInterval = 0 ;
  else
    Initial.BunchInterval = Initial.Q / P(edp+15).value ;
  end
  
end
%
% set the energy profile
%
UpdateMomentumProfile( nelmold+1, nelmold+nelm, ...
  Initial.Q, Initial.Momentum, 0 ) ;
%
% convert from MAD K values to Lucretia B values
%
KtoB( nelmold+1, nelmold+nelm ) ;
%
% parse the wakefields
%
if (isempty(WF))
  WF.ZSR = [] ;
  WF.TSR = [] ;
  WF.TLR = {} ;
  WF.TLRErr = {} ;
end
nzsr = length(W.longit) ;
ntsr = length(W.transv) ;
if ( (nzsr>0) || (ntsr>0) )
  for count = 1:nzsr
    [retstat,parsedwf] = ParseSRWF(W.longit{count},0.1) ;
    if (retstat{1} == 1)
      WF.ZSR = [WF.ZSR parsedwf] ;
    else
      stat{1} = -1 ;
      stat = AddMessageToStack(stat,retstat{2}) ;
    end
  end
  for count = 1:ntsr
    [retstat,parsedwf] = ParseSRWF(W.transv{count},0.1) ;
    if (retstat{1} == 1)
      WF.TSR = [WF.TSR parsedwf] ;
    else
      stat{1} = -1 ;
      stat = AddMessageToStack(stat,retstat{2}) ;
    end
  end
end

% if any bad elements were found, make a message about them

if ~isempty(badelm)
  stat{1} = -1 ;
  for count = 1:length(badelm)
    stat = AddMessageToStack(stat,...
      ['Element # ',num2str(badelm(count)),...
      ', name = "',E(badelm(count)).name,'":',...
      ' class is not supported by Lucretia']) ;
  end
end
%
%==========================================================================
%
function KtoB( istart, iend )
%
% KTOB convert MAD-style K values to Lucretia-style B values.
%
%   KtoB( istart, iend ) takes MAD-style K values (assumed to be stored in
%      the element B data fields) and converts to Lucretia's native B
%      format.  This permits a Lucretia BEAMLINE to be built from a MAD
%      deck, temporarily storing K values in the BEAMLINE B fields, and
%      then converting to B once the momentum profile has been computed.

%========================================================================

global BEAMLINE ;

brhofact = 1/0.299792458 ; % convert between GeV/c and T.m

for count = min(istart,iend):max(istart,iend)
  
  switch BEAMLINE{count}.Class
    
    case{ 'XCOR' , 'YCOR' , 'RBEN', 'MULT', 'XYCOR' }
      
      BEAMLINE{count}.B = BEAMLINE{count}.B * ...
        BEAMLINE{count}.P * brhofact ;
    case{ 'QUAD' , 'SEXT' , 'OCTU', 'SOLENOID' }
      BEAMLINE{count}.B = BEAMLINE{count}.B * ...
        BEAMLINE{count}.L * ...
        BEAMLINE{count}.P * brhofact ;
    case{ 'SBEN' }
      BEAMLINE{count}.B = BEAMLINE{count}.B * ...
        BEAMLINE{count}.P * brhofact ;
      if (length(BEAMLINE{count}.B) > 1)
        BEAMLINE{count}.B(2) = BEAMLINE{count}.B(2) * ...
          BEAMLINE{count}.L ;
      end
      
  end
  
end
%
%==========================================================================
%
function [mB,mT,mPI,mA] = UnpackXSIFMultPars( P, edp )
%
% UNPACKXSIFMULTPARS Unpack the multipole field parameters from an XSIF
%    multipole.
%
% [mB, mT, mPI, mA] = UnpackXSIFMultPars( ParDB, Eptr ) finds and returns
%    the non-zero KnL components of a multipole magnet which has been
%    parsed by XSIF, along with its component tilt values, indices to
%    indicate the pole number, and the design horizontal and vertical bend
%    angles.  ParDB is the parameters data structure returned by XSIFParse,
%    and Eptr is the pointer into ParDB for the first parameter of the
%    desired multipole.  Only nonzero multipole components are returned.
%
% See also:  XSIFParse.
%

% MOD:
%       22-jun-2006, PT [SLAC]:
%          bugfix to zero-strength MULT code.
%       25-apr-2006, M. Woodley [SLAC]:
%          Handle zero-strength MULTs

%==========================================================================

mB = [] ; mT = [] ; mPI = [] ; mA = [] ; %#ok<NASGU>

for count = 0:20
  c2 = 2*count + 1 ;
  if (P(edp+c2).value ~= 0)
    mB = [mB P(edp+c2).value] ;
    mT = [mT P(edp+c2+1).value] ;
    mPI = [mPI count] ;
  end
end
mT = mT + P(edp+49).value ;
mB = mB * P(edp+43).value ;

% Now, the multipole can be completely blank, no strength values at all.
% handle that case now

if (isempty(mPI))
  mT = 0 ; mB = 0 ; mPI = 0 ;
end
if (mPI(1) == 0)
  %  if (~isempty(mPI) && (mPI(1) == 0))
  mA = [mB(1) * cos(mT(1)) mB(1) * sin(mT(1))] ;
else
  mA = [0 0] ;
end