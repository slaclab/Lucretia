function	P = erfn(x);

%	P = erfn(x);
%
%	Function to calculate normal probability distribution in the standard way by
%	fixing Matlab's screwy "erf(x)".  This "erfn(x)" function is defined as:
%
%	erfn(x) = integral from 0 to x of { (1/sqrt(2*pi))*exp(-t.^2/2) }
%
%	whereas MATLAB's function uses exp(-t.^2).
%
%	INPUTS:	x:	The 2nd integration limit(s) {-inf < x < +inf}
%					NOTE: erfn(-x) = -erfn(x)
%	OUTPUTS:	P:	The probability that a value between 0 and x will turn up in a
%					gaussian ditribution.
%
%		e.g.:		erfn(0)   => 0
%					erfn(1)   => 0.3413
%					erfn(inf) => 0.5
%
%=============================================================================

P = erf(x/sqrt(2))/2;