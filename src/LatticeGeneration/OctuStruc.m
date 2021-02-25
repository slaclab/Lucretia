function O = OctuStruc( length, ITdL, Tilt, aper, Name )
%
% OCTUSTRUC Generate a Lucretia data structure for an octupole magnet.
%
% O = OctuStruc( L, ITDL, Tilt, aper, Name ) returns a Lucretia data
%    structure for an octupole magnet with the desired length, integrated
%    B''', xy rotation angle, aperture, and name.  The strength uses the
%    convention:
%
%       O.B = d^3B/dr^3 = K3[MAD]L * brho, B(r) = O.B * r^3 / 6 / O.L.
%

%================================================================

% since the fields are the same as for a quadrupole:

  O = QuadStruc( length, ITdL, Tilt, aper, Name ) ;
  O.Class = 'OCTU' ;
  
 % 