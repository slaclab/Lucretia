function Q = SolenoidStruc( length, IBdL, aper, Name )
%
% SOLENOIDSTRUC Return a Lucretia data structure for a solenoid
%
% S = SolenoidStruc( L, B, aper, Name ) returns a fully-formed Lucretia
%    data structure for a solenoid, with the fields specified in the 4
%    calling arguments (length, integrated longitudinal Bfield, name)
%    filled in and other fields in a default state.  The magnetic field
%    convention is:
%
%       Q.B = B_0,z L.
%
% Version date:  09-Mar-2006.

%================================================================

Q.Name = Name ;
Q.S = 0 ; Q.P = 0 ;
Q.Class = 'SOLENOID' ;
Q.L = length ;
Q.B = IBdL ; Q.dB = 0 ;
Q.aper = aper ;
Q.PS = 0 ; Q.Offset = [0 0 0 0 0 0] ;
Q.Girder = 0 ;

TrackFlag.SynRad = 0 ;
Q.TrackFlag = TrackFlag ;
Q.TrackFlag.ZMotion = 0 ;
Q.TrackFlag.LorentzDelay = 0 ;
Q.TrackFlag.Aper = 0 ;
