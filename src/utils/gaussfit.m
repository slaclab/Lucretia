function chi2 = gaussfit(x,datfx)
% Return chi2 fit of gaussian to data provided
[ fx , bc ] = hist(datfx,100);
fxCalc=gaussfun(x(1),x(2),bc);
chi2 = sum( ( fx/max(abs(fx)) - fxCalc/max(abs(fxCalc)) ).^2);
function fx=gaussfun(mu,sigma,x)
fx=(1/(sigma*sqrt(2*pi)))*exp( -((x-mu).^2)./(2.*sigma.^2) );
fx=fx+1e-20;