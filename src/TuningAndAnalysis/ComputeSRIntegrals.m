function [stat,Ix,Iy] = ComputeSRIntegrals(istart,iend,Twiss)
%
% COMPUTESRINTEGRALS Compute Sands' Synchrotrion Radiation integrals using
%    Helms' method.
%
% [stat,Ix,Iy] = ComputeSRIntegrals(istart,iend,Twiss) computes the
%    synchrotron radiation integrals I1 through I5 in both the horizontal
%    and vertical planes [1].  The computation uses Helms' method [2] and
%    integrates over the region from the start of element number istart to
%    the end of element number iend.  Argument Twiss is a data structure of
%    Twiss parameters, in the format returned by GetTwiss, which includes
%    the parameters for the range from istart to iend.  
%
% Return arguments Ix and Iy are 1 x 5 vectors containing the synchrotron
%    radiation integrals in the horizontal and vertical planes,
%    respectively.  Return argument stat is a Lucretia status and message
%    cell array (type help LucretiaStatus for more information).  In the
%    event of success, stat{1} == +1.  If coupling elements are detected in
%    the region of interest, stat{1} == -1, indicating that results may not
%    be correct.  If the region from istart to iend is not located in
%    Twiss, stat{1} == 0.
%
% Notes:  
% [1] M. Sands, SLAC-Report-121 (1971).
% [2] R. Helm, SLAC-Pub-1193 (1973).
%
% See also:  GetTwiss, LucretiaStatus.
%
% Version date:  22-jan-2008.

% MOD:
%       22-jan-2008, PT:
%          handle bends where EdgeAngle or B are scalar.
%          Corrected bend plane selection again...
%       08-Mar-2007, PT:
%          correct bug in selection of the bend plane (how embarrasing).

%==========================================================================

  global BEAMLINE ;

% initialize the message stack and integral vectors

  stat = InitializeMessageStack( ) ;
  Ix = zeros(1,5) ; Iy = Ix ;
  CouplingMessage = 0 ; RollBendMessage = 0 ;
  CouplingTolerance = 1e-6 ;
  
% find the bends in the area of interest  

  blist = findcells(BEAMLINE,'Class','SBEN',istart,iend) ;
  
  if (isempty(blist))
      return ;
  end
  
% make sure that the Twiss vectors contain values for the integration
% region

  tstart = 0 ; Sstart = BEAMLINE{istart}.S ;
  for count = 1:length(Twiss.S) 
      if (Twiss.S(count) == Sstart)
          tstart = count ;
          break ;
      end
  end
  tend = 0 ; Send = BEAMLINE{iend}.S ;
  for count = tstart+1:length(Twiss.S) 
      if (Twiss.S(count) == Send)
          tend = count ;
          break ;
      end
  end
  
  if ( (tstart == 0) | (tend == 0) )
      stat{1} = 0 ;
      stat = AddMessageToStack(stat,...
          'Twiss data does not fully overlap integration region') ;
      return ;
  end
  
% find the quads in the area of interest and look for coupling

  qlist = findcells(BEAMLINE,'Class','QUAD',istart,iend) ;
  
  for count = qlist
      cfat = abs(sin(2*BEAMLINE{count}.Tilt)*BEAMLINE{count}.B) ;
      brho = BEAMLINE{count}.P / 0.299792458 ;
      if (cfat * brho > CouplingTolerance)
          stat{1} = -1 ;
          stat = AddMessageToStack(stat,...
              'Nontrivial coupling detected in integration region') ;
          CouplingMessage = 1 ;
          break ;
      end
  end
  
% if no coupling message yet, look for solenoids

  slist = findcells(BEAMLINE,'Class','SOLENOID',istart,iend) ;
  
  if (CouplingMessage == 0)
    for count = slist
      cfat = abs(BEAMLINE{count}.B) ;
      brho = BEAMLINE{count}.P / 0.299792458 ;
      if (cfat * brho > CouplingTolerance)
          stat{1} = -1 ;
          stat = AddMessageToStack(stat,...
              'Nontrivial coupling detected in integration region') ;
          CouplingMessage = 1 ;
          break ;
      end
    end
  end  
      
% loop over sector bends and do the computations of interest

  for count = blist
      
      Ivec = zeros(1,5) ;
      tpoint = tstart + count - istart ;
      brho = BEAMLINE{count}.P / 0.299792458 ; 
      L = BEAMLINE{count}.L ; Tilt = BEAMLINE{count}.Tilt ;
      Angle = BEAMLINE{count}.Angle ; 
      E1 = BEAMLINE{count}.EdgeAngle(1) ; 
      if (length(BEAMLINE{count}.EdgeAngle) > 1)
        E2 = BEAMLINE{count}.EdgeAngle(2) ;
      else
        E2 = E1 ;
      end
      if (length(BEAMLINE{count}.B) > 1)
        Kquad = BEAMLINE{count}.B(2) / L / brho ;
      else
        Kquad = 0 ;
      end
      rhoinv = Angle/L ;
      k2 = rhoinv^2 + Kquad ; 
%
% here I'm going to cheat so I don't need a case where k2 == 0:
%
      if (k2==0)
          k2 = 1e-9 ;
      end
      k = sqrt(abs(k2)) ;
% 
% which powers of k have the same sign as k?  I don't know.  DIMAD claims
% to know, and I'll tentatively follow its lead here...
%
      k3 = k2*k ; k4 = k2*k2 ; k5 = k3*k2;
      kl = k*L ; k2l = k2*L ; k3l = k3*L ;
      k2l2 = k2*L^2 ; k3l2 = k3*L^2 ; k3l3 = k3*L^3 ;
      k4l3 = k4*L^3 ; k5l3 = k5*L^3 ;
      if (k2 > 0)
          sinkl = sin(k*L) ;
          coskl = cos(k*L) ;
      else
          sinkl = sinh(k*L) ;
          coskl = cosh(k*L) ;
      end
      
%
% is the magnet upright, 90 degree rotated, or other?
%
      if (abs(sin(Tilt)) < 1e-6)
          BendPlane = 1; 
%          BendPlane = 2; 
      elseif (abs(cos(Tilt)) < 1e-6)
          BendPlane = 2;
%          BendPlane = 1;
      else
          BendPlane = 1 ;
          if (RollBendMessage==0)
            stat{1} = -1 ;
            stat = AddMessageToStack(stat,...
              'Rotated bend detected in integration region') ;
            RollBendMessage = 1 ;
          end
      end
%
% now for the serious math
%
      if (BendPlane == 1)
          beta = Twiss.betax(tpoint) ;
          alpha = Twiss.alphax(tpoint) - beta*rhoinv*tan(E1) ;
          eta = Twiss.etax(tpoint) ;
          etap = Twiss.etapx(tpoint) + eta*rhoinv*tan(E1) ;
      else
          beta = Twiss.betay(tpoint) ;
          alpha = Twiss.alphay(tpoint) - beta*rhoinv*tan(E1) ;
          eta = Twiss.etay(tpoint) ;
          etap = Twiss.etapy(tpoint) + eta*rhoinv*tan(E1) ;
      end
      tgamma = (1+alpha^2)/beta ;
      eta2 = eta*coskl+etap*sinkl/k + (1-coskl)*rhoinv/k2 ;
      meaneta = eta*sinkl/kl + etap*(1-coskl)/k2l + ...
          rhoinv * (kl-sinkl)/k3l ;
      Ivec(1) = rhoinv * meaneta ;
      Ivec(2) = rhoinv^2 ;
      Ivec(3) = abs(rhoinv^3) ;
      Ivec(4) = meaneta * rhoinv^3 - 2*( ...
          -Kquad*rhoinv*meaneta + ...
          (eta*tan(E1)+eta2*tan(E2))*rhoinv^2/2/L ...
                                     ) ;
      Ivec(5) = tgamma*eta^2 + 2*alpha*eta*etap+beta*etap^2  ...
        + 2*L*rhoinv*( ...
             - (tgamma*eta+alpha*etap)*(kl-sinkl)/k3l2 ...
             + (alpha*eta+beta*etap)*(1-coskl)/k2l2 ...
                      ) ...
        + (L*rhoinv)^2*( ...
               tgamma*(3*kl-4*sinkl+sinkl*coskl)/2/k5l3 ...
             - alpha*(1-coskl)^2/k4l3 ...
             + beta*(kl-coskl*sinkl)/2/k3l3) ;
      Ivec(5) = Ivec(5) * abs(rhoinv^3) ;
      Ivec = Ivec * L ;
%
% now attach the vector to the appropriate sum
%
      if (BendPlane==1)
          Ix = Ix + Ivec ;
      else
          Iy = Iy + Ivec ;
      end
%      
% end of the loop
%
  end

% and that's it.  
         
      