function bmag=bmag_sig(sig,sig0)
%
% bmag=bmag_sig(sig,sig0)
%
% Compute bmag from two 2x2 beam sigma matrices (bmag is the mismatch
% of sig with respect to sig0).
%
emit0=sqrt(det(sig0));
beta0=sig0(1,1)/emit0;
alpha0=-sig0(2,1)/emit0;
gamma0=(1+alpha0^2)/beta0;

emit=sqrt(det(sig));
beta=sig(1,1)/emit;
alpha=-sig(2,1)/emit;
gamma=(1+alpha^2)/beta;

bmag=0.5*(beta*gamma0-2*alpha*alpha0+gamma*beta0);