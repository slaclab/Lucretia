function stat = MovePhysicsVarsToPS( klyslist )

% MOVEPHYSICSVARSTOPS exchange the dimensionless PS Ampl and
%    dimensioned BEAMLINE B parameters
%
%    stat = MovePhysicsVarsToPS( PSlist ) exchanges power supply
%       Ampl parameters (nominally dimensionless) with element B
%       parameters, which nominally has dimensions of T.m, T, etc.  In the
%       process the PS Ampl becomes the sum of all strengths of all magnets
%       supported by the PS, and the magnet B parameters become
%       dimensionless scale factors (ie, if a PS supports 10 equal magnets,
%       then after the exchange each magnet will have a B of 0.1).  
%       All power supplies in the PSlist will be exchanged in this way. The
%       SetPt and Step parameters are also rescaled.  Return variable stat
%       is a Lucretia status and message stack (type help LucretiaStatus
%       for more information).
%
% Return status value == 1 for success, == 0 if PSlist includes power
% supplies which are out of range.
%
% Version date:  12-June-2006.
%

% MOD:
%       12-jun-2006, PT:
%          bugfix to allow multiple devices with opposite signs to be
%          assigned to a single power supply without risk of a /0 error.
%       28-sep-2005, PT:
%          Improved handling of power supplies with zero amplitude.
%          Support for devices with multiple power supplies.

%==========================================================================

global BEAMLINE ;
global PS ;
stat = InitializeMessageStack( ) ;  

for count = 1:length(klyslist)    
  klysno = klyslist(count) ;  
  
% range check

  if ( klysno > length(PS) )
      stat = AddMessageToStack(stat,['PS # ',num2str(klysno),...
          ' out of range in MovePhysicsVarsToPS']) ;
      stat{1} = 0 ;
  end
  
end
if (stat{1} ~= 1)
    return ;
end
  
for count = 1:length(klyslist)    
  klysno = klyslist(count) ;  
  
% if the PS is at zero amplitude, then there's nothing to do here...

%  if PS(klysno).Ampl == 0
%      continue ;
%  end
  
% compute the total strength

  V = [] ; BSignMaster = 1 ;
  for count = 1:length(PS(klysno).Element)
      elemno = PS(klysno).Element(count) ;
      PS_index = find(BEAMLINE{elemno}.PS == klysno) ;
      PS_index = PS_index(1) ;
      B = BEAMLINE{elemno}.B(PS_index) ;
      if ( (B==0) & (length(BEAMLINE{elemno}.B)>1) )
          B = BEAMLINE{elemno}.B(2) ;
      end
      if ( (B<0) & (count==1) )
          BSignMaster = -1 ; 
      end
      if (sign(B) ~= BSignMaster)
          B = -B ;
      end
      V = [V B] ;
  end
  
  [a,b] = max(abs(V)) ;
  Vsign = sign(V(b)) ;
  V = Vsign * abs(sum(V)) ;
  
% compute the scale factor; we need to do different things if the PS
% amplitude is nonzero versus zero, since in either case we want the 
% magnet B to be scaled by the total # of slices and their relative
% weights

  if (PS(klysno).Ampl ~= 0)
    scale = V / PS(klysno).Ampl ;
  else
    scale = V ;
  end
  
% apply the scale factor to the PS

  PS(klysno).Ampl = PS(klysno).Ampl * scale ;
  PS(klysno).Step = PS(klysno).Step * scale ;
  PS(klysno).SetPt = PS(klysno).SetPt * scale ;
  
% now apply the reverse transformation on elements. Note that if V==0 then
% scale == 0 and a div0 error will occur.  In theory, though, the only way
% that the scale could be == 0 is if all magnets are at zero and Ampl ~= 0.
% This was taken care of in AssignToPS (ie, it prevents a situation where
% all magnets are zero and the PS amplitude is zero), so this can only
% happen if something unauthorized has been done to the beamline.

  for count = 1:length(PS(klysno).Element)
      elemno = PS(klysno).Element(count) ;
      if (length(BEAMLINE{elemno}.PS) < 2)
         BEAMLINE{elemno}.B = BEAMLINE{elemno}.B / scale ;
      else
         PS_index = find(BEAMLINE{elemno}.PS == klysno) ;
         BEAMLINE{elemno}.B(PS_index) = BEAMLINE{elemno}.B(PS_index) / scale ;
      end
  end
  
end
  
      
      