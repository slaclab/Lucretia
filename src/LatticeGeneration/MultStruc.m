function Q = MultStruc( len, IGdL, Tilt, PIndx, Angle, aper, Name )
%
% MULTSTRUC Lucretia data structure for an arbitrary multipole magnet.
%
% M = MultStruc( L, B, Tilt, PIndx, Angle, aper, Name ) returns a
%     Lucretia data structure for a multipole magnet.  Arguments are
%     defined as follows:
%
%     L:      magnet length in meters.  By default, the magnet's length
%             for synchrotron radiation calculations, Lrad, is set equal to
%             L.
%     B:      row vector containing the integrated multipole field
%             components.
%     Tilt:   row vector containing the xy rotation angles of the multipole
%             field components.
%     PIndx:  Multipole index of field components (0 == dipole, 1 ==
%             quadrupole, 2 == sextupole, etc.).
%     Angle:  1 x 2 vector of design bending angles in x and y.  If Angle
%             == [0 0], dipole fields are treated as orbit correctors.
%     aper:   Magnet aperture.
%     Name:   char string, element name.
%
%     Vectors BnL, Tilt, and PIndx must be of equal length, but multipole
%     components can be entered in any order.  The field expansion in the
%     midplane for untilted components is given by:
%
%       B(x,y=0) = sum_j { M.B(j) * x^(M.PIndx(j)) / M.L / (M.PIndx(j))! }.
%
% Version date:  16-Dec-2005.
%

%================================================================

Q.Name = Name ;
Q.S = 0 ; Q.P = 0 ;
Q.Class = 'MULT' ;
Q.L = len ; Q.Lrad = len ;
IGdL = IGdL(:) ; IGdL = IGdL' ;
Tilt = Tilt(:) ; Tilt = Tilt' ;
PIndx = PIndx(:) ; PIndx = PIndx' ;
if ( (length(PIndx) ~= length(Tilt)) | ...
     (length(PIndx) ~= length(IGdL))       )
   error('Multipole field, tilt, and field index vectors must have equal length') ;
end
Q.Angle = Angle ;
Q.B = IGdL ; 
Q.Tilt = Tilt ;
Q.PoleIndex = PIndx ;
Q.dB = 0 ;
Q.aper = 1e-2 ;
Q.PS = 0 ; Q.Offset = [0 0 0 0 0 0] ;
Q.Girder = 0 ;

TrackFlag.SynRad = 0 ;
Q.TrackFlag = TrackFlag ;
Q.TrackFlag.ZMotion = 0 ;
Q.TrackFlag.LorentzDelay = 0 ;
Q.TrackFlag.Aper = 0 ;
