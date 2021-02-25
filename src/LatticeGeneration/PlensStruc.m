function P = PlensStruc( length, IGdL, aper, Name )
%
% PLENSSTRUC Return a Lucretia data structure for a plasma lens
%
% P = PlensStruc( L, B, aper, Name ) returns a fully-formed Lucretia
%    data structure for a plasma lens, with the fields specified in the 4
%    calling arguments (length, integrated equivalent gradient, name)
%    filled in and other fields in a default state.  The magnetic field
%    convention is:
%
%       P.B = B'L = K1L * brho, B(r) = P.B * r / P.L.
% PLENS is basically a quad with focusing (or defocusing) the same in
%       both planes

%================================================================

P.Name = Name ;
P.S = 0 ; P.P = 0 ;
P.Class = 'PLENS' ;
P.L = length ;
P.B = IGdL ; P.dB = 0 ;
P.aper = aper ;
P.PS = 0 ; P.Offset = [0 0 0 0 0 0] ;
P.Girder = 0 ;

TrackFlag.SynRad = 0 ;
P.TrackFlag = TrackFlag ;
P.TrackFlag.ZMotion = 0 ;
P.TrackFlag.LorentzDelay = 0 ;
P.TrackFlag.Aper = 0 ;

P.Tilt = 0; % this parameter ignored for plasma lens but must be present