function S = SextStruc( length, ISdL, Tilt, aper, Name )
%
% SEXTSTRUC Return a Lucretia data structure for a sextupole magnet
%
% S = SextStruc( L, B, Tilt, aper, Name ) returns a fully-formed Lucretia
%    data structure for a quadrupole, with the fields specified in the 5
%    calling arguments (length, integrated B'', xy rotation, name)
%    filled in and other fields in a default state.  The magnetic field
%    convention is:
%
%       S.B = d^2B/dr^2 = K2[MAD]L * brho, or B(r) = S.B * r^2 / 2 / S.L.
%

%================================================================

% since the fields are the same as for a quadrupole:

  S = QuadStruc( length, ISdL, Tilt, aper, Name ) ;
  S.Class = 'SEXT' ;
  
 % 