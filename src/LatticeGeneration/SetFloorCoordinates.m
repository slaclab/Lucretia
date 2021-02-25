function [stat,varargout] = SetFloorCoordinates(istart,iend,initial)
%
% SETFLOORCOORDINATES Compute the floor positions of elements in an
%    absolute coordinate system and add them to the lattice.
%
% stat = SetFloorCoordinates(istart,iend,initial) adds the floor
%    coordinates of all elements between istart and iend, inclusive, to the
%    BEAMLINE global array.  Both upstream face and downstream face
%    coordinates are added.  Argument initial is a 1 x 6 vector containing
%    the initial values of X, Y, Z, Theta, Phi, Psi, in that order,
%    where:
%
%       X,Y,Z = absolute coordinates of start point [m],
%       Theta = initial rotation in xz plane [radians],
%       Phi = initial rotation in yz plane [radians],
%       Psi = initial rotation in xy plane [radians].
%
%    When iend >= istart, argument initial is used as the coordinates of
%    the upstream face of the first element in the global coordinate
%    system.  When iend < istart, argument initial is used as the
%    coordinates of the downstream face of the last element in the global
%    coordinate system (ie, the coordinates where the beamline should end).
%
%    Return argument stat is a Lucretia status cell array where stat{1} == 1
%    indicates success.
%
% [stat,coords] = SetFloorCoordinates(istart,iend,initial) returns the
%    floor coordinates which are assigned to the BEAMLINE elements.  Return
%    argument coords has dimensions of (|iend-istart|+2,6).  The 6 columns
%    are X,Y,Z,Theta,Phi,Psi.  The first row is the start coordinates of
%    the first element in the list (ie, the lesser of istart and iend), and
%    the subsequent rows are the final coordinates of each element.
%
% Version date:  05-Dec-2007.

% MOD:
%      05-Dec-2007, PT:
%         add optional return of assigned coordinates.
%      11-Jan-2007, PT:
%         added capacity to apply coordinates backwards from the end rather
%         than forwards from the beginning.
%      01-Aug-2006, PT:
%         correct logic for systems with both horizontal and vertical
%         bending.

%==========================================================================

  global BEAMLINE

%==========================================================================
%  Initialization
%==========================================================================

  stat = InitializeMessageStack( ) ;
  ReturnCoord = zeros(2+abs(iend-istart),6) ;

% unpack initial coordinates

  X = initial(1) ; Y = initial(2) ; Z = initial(3) ;
  Theta = initial(4) ; Phi = initial(5) ; Psi = initial(6) ;
  
% construct initial vector [Z,X,Y] and initial rotation matrix between
% beamline local coordinates and global coordinates 

  R = [Z ; X ; Y] ;
  
  U = Umatrix(Theta,Phi,Psi) ;

%==========================================================================
%  Master Loop
%==========================================================================

  if (istart > iend)
      step = -1 ;
  else
      step = 1 ;
  end

  for count = istart:step:iend

   if (step==1)   
    BEAMLINE{count}.Coordi = [R(2) R(3) R(1)] ;
    BEAMLINE{count}.Anglei = [Theta Phi Psi] ;
   else
    BEAMLINE{count}.Coordf = [R(2) R(3) R(1)] ;
    BEAMLINE{count}.Anglef = [Theta Phi Psi] ;
   end
   
% get the element length if nonzero

    if (isfield(BEAMLINE{count},'L'))
      L = BEAMLINE{count}.L ;
    else
      L = 0 ;
    end
      
% Initialize element angle-changes to zero

    ThetaNew = 0 ; PhiNew = 0 ; PsiNew = 0 ; BendAngle = 0 ; Tilt = 0 ;
    ZNew = 0 ; XNew = 0 ; YNew = 0 ; 
    BendElem = 0 ; CoordElem = 0 ;
  
% see if this is a Coord element, a bend, or a MULT, and if so get its
% relevant coordinates

    if (strcmp(BEAMLINE{count}.Class,'SBEN'))
      BendAngle = BEAMLINE{count}.Angle ;
      Tilt = BEAMLINE{count}.Tilt ;
    end
    if (strcmp(BEAMLINE{count}.Class,'MULT'))
      BendAngle = sqrt(BEAMLINE{count}.Angle(1)^2 + ...
                       BEAMLINE{count}.Angle(2)^2       ) ;
      Tilt = atan2(BEAMLINE{count}.Angle(2),BEAMLINE{count}.Angle(1)) ;
    end
    if (BendAngle ~= 0)
      BendElem = 1 ;
    end
    if (strcmp(BEAMLINE{count}.Class,'COORD'))
      XNew     = BEAMLINE{count}.Change(1) ;
      YNew     = BEAMLINE{count}.Change(3) ;
      ZNew     = BEAMLINE{count}.Change(5) ;
      ThetaNew = BEAMLINE{count}.Change(2) ;
      PhiNew   = BEAMLINE{count}.Change(4) ;
      PsiNew   = BEAMLINE{count}.Change(6) ;
      CoordElem = 1 ;
    end
  
% if this is neither a bend nor a Coord type element, the transformation is
% simple, since U(:,1) is the unit vector of local Z in the global coords

    if ( (CoordElem == 0) & (BendElem==0) )
      R = R + U(:,1)*L*step ;
    end
  
% if this is a Coord type element, the transformation is almost as simple

    if (CoordElem == 1)
      R = R + step * [ZNew ; XNew ; YNew] ;
      if (step==1)
        U = U * Umatrix(ThetaNew,PhiNew,PsiNew)  ;
      else
        U = U * inv(Umatrix(ThetaNew,PhiNew,PsiNew)) ;
      end
    end
  
% for a bending type of element things are a bit trickier  

    if (BendElem == 1)
    
% Compute some temporary matrices

      UT  = Umatrix(0,0, Tilt) ;    

% compute the change to the coordinates in the local coordinates of the
% bend magnet (ie, the location of the exit point wrt the entry point in
% a coordinate system where the bend magnet is purely horizontal and the
% entry face is at R = [0 0 0])
      
      rho = L / BendAngle ;
      RBendPlane = - rho * [-sin(BendAngle) ; 1-cos(BendAngle) ; 0] ;

% Now:  the RBendPlane is the change in coordinates at the end of the bend
% magnet in the local reference frame of the magnet's upstream face.  We
% can now convert to the global reference frame:

      if (step==1)
        R = R + U * UT * RBendPlane ;
      end
      
% now find the new U-matrix:

      if (step==1)
        U = U * Umatrix(-BendAngle*cos(Tilt),-BendAngle*sin(Tilt),0) ;
      else
        U = U * inv(Umatrix(-BendAngle*cos(Tilt),-BendAngle*sin(Tilt),0)) ;
      end
      
      if (step==-1)
        R = R - U * UT * RBendPlane ;
      end
    
    end
  
    denom = sqrt(U(2,1)^2 + U(1,1)^2) ;
    Theta = atan2(U(2,1),U(1,1)) ;
    Phi = atan2(U(3,1),denom) ;
    Psi = atan2(U(3,2),U(3,3)) ;
    if (step==1)
     BEAMLINE{count}.Coordf = [R(2) R(3) R(1)] ;
     BEAMLINE{count}.Anglef = [Theta Phi Psi] ;
    else
     BEAMLINE{count}.Coordi = [R(2) R(3) R(1)] ;
     BEAMLINE{count}.Anglei = [Theta Phi Psi] ;
    end
    
  end
  
% second loop to get the coordinates out of the BEAMLINE

  i1 = min([istart iend]) ; i2 = max([istart iend]) ;
  ReturnCoord(1,1:3) = BEAMLINE{i1}.Coordi ;
  ReturnCoord(1,4:6) = BEAMLINE{i1}.Anglei ;
  for count = i1:i2
     ccount = count - i1 + 2 ;
     ReturnCoord(ccount,1:3) = BEAMLINE{count}.Coordf ;
     ReturnCoord(ccount,4:6) = BEAMLINE{count}.Anglef ;
  end
  if (nargout > 1)
      varargout{1} = ReturnCoord ;
  end

%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================

% Auxiliary function Umatrix:  generates rotation matrix from local
% coordinates to global coordinates (ie, U(i,j) == projection of beamline
% coordinate j in global coordinate i; or, if you prefer, U(i,1) is the
% unit vector of local Z in the global coordinates; U(i,2) is the unit
% vector of local X in the global coordinates; U(i,3) is the unit vector of
% local Y in the global coordinates).

function U = Umatrix( Theta, Phi, Psi )

  U = zeros(3) ;
  
  CT  = cos(Theta) ; ST  = sin(Theta) ;
  CF  = cos(Phi)   ; SF  = sin(Phi)   ;
  CPS = cos(Psi)   ; SPS = sin(Psi)   ;
  
  U(1,1) =  CT*CF ;
  U(1,2) = -CT*SF*SPS - ST*CPS ;
  U(1,3) = -CT*SF*CPS + ST*SPS ;
  
  U(2,1) =  ST*CF ;
  U(2,2) = -ST*SF*SPS + CT*CPS ;
  U(2,3) = -ST*SF*CPS - CT*SPS ;
  
  U(3,1) =  SF ;
  U(3,2) =  SPS*CF ;
  U(3,3) =  CPS*CF ;

