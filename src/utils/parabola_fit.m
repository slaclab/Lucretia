function [A,B,C,rms,chisq] = parabola_fit(xvalues,yvalues,yerrors)
%
% performs a parabolic fit to data:
%
%   y = A(x-B)^2 + C
%
% returns A, B, C (each a vector with value and error), rms fit error of
% the points, and chisq of the fit (if weighted).
%
% In point of fact, the fit is a fit to a quadratic, which is then re-
% parameterized into a parabolic fit.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  xvalues = xvalues(:) ; yvalues = yvalues(:) ; yerrors = yerrors(:) ;

%
% how many values?
%
  numval = length(xvalues) ;
%
% weighted or unweighted fit?
%
  if (min(yerrors)<=0)
    yerrors = ones(numval,1) ;
    calc_chisq = 0 ;
  else
    calc_chisq = 1 ;
  end
%
% prepare a weighted fit...
%
  x2 = xvalues .* xvalues ; x0 = ones(numval,1) ;
  lqA = [x2 xvalues x0] ;
%  lqV = yerrors * eye(numval) * yerrors'
  lqV = eye(numval) ;
  for countval = 1:numval
    lqV(countval,countval) = yerrors(countval)^2 ;
  end
%
% get the coefficients
%
  lqx = lscov(lqA,yvalues,lqV) ;
%
% compute error matrix
%
  if (numval > 3)
   lqB = yvalues ;
   mse = lqB' * ...
    (inv(lqV) - inv(lqV)*lqA*inv(lqA'*inv(lqV)*lqA)*lqA'*inv(lqV))*lqB./(numval-3) ;
   errmat = inv(lqA'*inv(lqV)*lqA)*mse ;
  else
   errmat = zeros(3) ;
  end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% compute the rms fit error now
%
  ypoly = polyval(lqx,xvalues) ;
  rms = std(ypoly-yvalues) ;
%
% if desired, compute chisq of the fit
%
  if (calc_chisq ==1) 
    chisq = (ypoly-yvalues) .* (ypoly-yvalues) ./ ...
      ( yerrors .* yerrors ) ;
    chisq = sum(chisq) ;
  else
    chisq = 0 ;
  end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% convert to A,B,C coefficients
%
  A = lqx(1) ; B = -lqx(2)/lqx(1) * 0.5 ;
  C = lqx(3) - lqx(2)^2 /lqx(1) * 0.25 ;
  A = [A ; 0] ; B = [B ; 0] ; C = [C ; 0] ;
  a1_err = sqrt(errmat(3,3)) ;
  a2_err = sqrt(errmat(2,2)) ;
  a3_err = sqrt(errmat(1,1)) ;
  a1_vec = lqx(3) ; a2_vec = lqx(2) ; a3_vec = lqx(1) ;
%
% convert the errors
%
  A(2)  = sqrt(errmat(1,1)) ;

%  B(2)  = sqrt(   0.25*A(1)^2 * ...
%                ( errmat(2,2) + errmat(2,1) * B(1) + ...
%                  errmat(1,1) * B(1)^2              )   ) ;
%
%  C(2)  = sqrt(   errmat(3,3) + errmat(3,2) * B(1) + ...
%                ( errmat(1,3) + errmat(2,2) ) * B(1)^2 + ...
%                  errmat(2,1) * B(1)^3 + errmat(1,1)*B(1)^4 ) ;

  B(2) = (a2_err / (-2*a3_vec))^2 + ...
         (a3_err * a2_vec / (2*a3_vec^2))^2 +...
         2 * errmat(2,1) ...
       * (1/(-2*a3_vec)) ...
       * a2_vec / (2*a3_vec^2) ;
  if (B(2)>0)
    B(2) = sqrt(B(2)) ;
  else
    B(2) = 0 ;
  end
%
  C(2) = a1_err^2 + ...
         (-a2_err * 2 * a2_vec / (4*a3_vec) )^2 + ...
         (a3_err * a2_vec^2 / (4*a3_vec^2) )^2 + ...
         2*errmat(1,3) ...
       * a2_vec^2 / (4*a3_vec^2) + ...
         2*errmat(2,1) ...
       * (-2*a2_vec / (4*a3_vec)) ...
       * (a2_vec^2 / (4*a3_vec^2)) ;
  if (C(2)>0)
    C(2) = sqrt(C(2)) ;
  else
    C(2) = 0 ;
  end
%
% and that's it.