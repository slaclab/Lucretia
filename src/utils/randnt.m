function R = randnt(nsig,varargin)
% Extension to standard Matlab randn function with truncation at nsig
% sigma- the following input args are the same as randn

if nargin==0; error('Need at least 1 input argument!'); end;
if nargin==1
  randArg=1;
else
  randArg=[];
  for iArg=1:nargin-1
    randArg = [randArg varargin{iArg}]; %#ok<AGROW>
  end % for iArg
end % if nargin==1

R = randn(randArg);
if nsig
  while any(abs(R(:))>nsig)
     R(abs(R)>nsig) = randn(1,sum(abs(R(:))>nsig)) ;
  end % while R element > nsig
end % if ~nsig, randnt reduces to randn