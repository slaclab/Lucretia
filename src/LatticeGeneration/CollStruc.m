function C = CollStruc( length, xgap, ygap, shape, Tilt, Name )
%
% COLLSTRUC create Lucretia data structure for a collimator
%
% C = CollStruc( length, xgap, ygap, shape, Tilt, Name ) returns a Lucretia
%    data structure for a collimator.  The xgap and ygap are the half-gaps
%    of the collimator, and the shape is a string indicating its geometry
%    (suppored values are 'Ellipse' and 'Rectangle').

%================================================================

C.Name = Name ;
C.S = 0 ; C.P = 0 ;
C.Class = 'COLL' ;
C.L = length ;
C.Lrad = 0. ;
C.Geometry = shape ;
C.aper = [xgap ygap] ;
C.Offset = [0 0 0 0 0 0] ;
C.Girder = 0 ;
C.Tilt = Tilt ; 

C.TrackFlag.ZMotion = 0 ;
C.TrackFlag.LorentzDelay = 0 ;
C.TrackFlag.Aper = 0 ;
