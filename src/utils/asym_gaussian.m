function x = asym_gaussian(N,xsig,xbar,asym,cut,tail,halo,halo_pop);

%	x = asym_gaussian(N,xsig,xbar,asym,cut,tail,halo,halo_pop);
%
%	Function to generate an asymmetric gaussian truncated at
%	+/- cut*xsig and with added tails described by "tail".
%
%    INPTS:
%			N:			Total number of particles, even after cuts.
%			xsig:		rms of distribution after asymmetry and cuts
%			xbar:		mean of distribution after all cust, etc.
%			asym:		asymmetry parameter (-1 < asym < 1) - [asym
%						of +-0.2 is weak and +-0.8 is strong (0 is none)
%						[asym>0 gives slope>0; i.e. larger sigma @ x<0]
%			cut:		Number of "xsig" to truncate dist. at
%						(0.5 <= cut < inf) - still get length(x)=N.
%			tail:		Like a rise/fall (0 <= tail <1) - [0.05 is
%						fast rise while 0.4 starts to look gaussian]
%			halo:		(Opt,DEF=0)Add a "hal_pop" halo which has halo*xsig
%						rms spread
%			halo_pop:	(Opt,DEF=0.01) Halo relative population
%
%    OUTPUTS:
%			x:			Array of random numbers with length N

%===================================================================

if abs(asym) >= 1
  error('asymmetry parameter must be -1 < asym < 1')
end
if cut < 0.5
  error('cut must be 0.5 <= cut < inf')
end
if tail < 0 | tail >=1
  error('tail must be 0 <= tail < 1')
end
if ~exist('halo')
  halo = 0;
end
if halo < 0
  error('halo must be >= 0')
end

f1 = 1 + asym;
f2 = 1 - asym;

N1 = round(N*f1/4/erfn(cut));		% boost N1 to accomodate cuts
N2 = round(N*f2/4/erfn(cut));		% boost N2 to accomodate cuts

x1 = f1*xsig*randn(2*N1,1);		% generate a gaussian distribution + offset
i  = find(x1<0 & x1>(-cut*xsig*f1));	% eliminate positive values and cuts
x  = x1(i);
x1 = f2*xsig*randn(2*N2,1);		% generate a gaussian distribution + offset
i = find(x1>0 & x1<(cut*xsig*f2));	% eliminate positive values
x = [x; x1(i)];
clear x1, i;
NN = length(x);
dN = N - NN;
if dN > 0
  x1 = min([f1 f2])*xsig*(rand(dN,1)-0.5);	% generate a gaussian distribution + offset
  x = [x; x1];
elseif dN < 0
  adN = abs(dN);
  i = round((NN/adN)*(1:adN));
  x(i) = [];
end
x = x - mean(x);
x = x + randn(N,1)*tail*xsig;
x = x*xsig/std(x);
if halo > 0
  if halo_pop > 0
    i = 1:round(1/halo_pop):N;
    Nf = length(i);
    x(i) = x(i) + randn(Nf,1)*halo*xsig;
  end
end
x = x + xbar;
