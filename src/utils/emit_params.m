function [p,dp] = emit_params(sig11,sig12,sig22,C,b0,a0)

%       [p,dp] = emit_params(sig11,sig12,sig22,C,b0,a0);
%
%       Returns emittance, bmag, emit*bmag, beta, alpha, and errors on
%       all these given fitted sigma11(or 33), sigma12(or 34),
%       sigma22(or 44) and the 3X3 covariance matrix of this fit.
%
%     INPUTS:   sig11:  The 1,1 (3,3) sigma matrix element (in m^2-rad)
%               sig12:  The 1,2 (3,4) sigma matrix element (in m-rad)
%               sig22:  The 2,2 (4,4) sigma matrix element (in rad^2)
%               C:      The 3X3 covariance matrix of the above fitted
%                       sig11,12,22 (33,34,44) (in squared units of
%                       the 3 above sigij's)
%               b0:     The matched (design) beta function (in meters)
%               a0:     The matched (design) alpha function (unitless)
%     OUTPUTS:  p:      p(1) = unnormalized emittance in (meter-radians)
%                       p(2) = bmag (unitless beta beat magnitude)
%                       p(3) = unnormalized emit*bmag (meter-radians)
%                       p(4) = beta function (in meters)
%                       p(5) = alpha function (unitless)
%                       p(6) = (BMAG^2-1)*cos(phi)/BMAG
%                       p(7) = (BMAG^2-1)*sin(phi)/BMAG
%               dp:     Measurement errors on above p(1),p(2),...

%===============================================================================

sig0 = [ b0         -a0
        -a0 (1+a0^2)/b0];

sig  = [sig11 sig12
        sig12 sig22];

ebm = 0.5*trace(inv(sig0)*sig);

e2  =  sig11*sig22 - sig12^2;
if e2 < 0
  e  = -sqrt(abs(e2));
  b  =  sig11/(-e);
  a  = -sig12/(-e);
else
  e =  sqrt(e2);
  b =  sig11/e;
  a = -sig12/e;
end

bm  = ebm/e;
g0  = (1+a0^2)/b0;

bm_cos = (b/b0 - abs(bm))/abs(bm);
bm_sin = (a - a0*b/b0)/abs(bm);

grad_e   = [ sig22/(2*e)
            -sig12/e
             sig11/(2*e)];

grad_bm  = [(g0/(2*e) - bm*sig22/(2*e^2))
            (a0/e     + bm*sig12/(e^2))
            (b0/(2*e) - bm*sig11/(2*e^2))];

grad_ebm = [g0/2
            a0
            b0/2];

grad_b   = [-0.5*sig11*sig22/(e^3)+(1/e)
                 sig11*sig12/(e^3)
            -0.5*sig11*sig11/(e^3)      ];

grad_a   = [ 0.5*sig12*sig22/(e^3)
                -sig12*sig12/(e^3)-(1/e)
             0.5*sig11*sig12/(e^3)      ];

de       = sqrt(grad_e'*C*grad_e);
dbm      = sqrt(grad_bm'*C*grad_bm);
debm     = sqrt(grad_ebm'*C*grad_ebm);
db       = sqrt(grad_b'*C*grad_b);
da       = sqrt(grad_a'*C*grad_a);

p  = [ e  bm  ebm  b  a bm_cos bm_sin];
dp = [de dbm debm db da 0 0];
