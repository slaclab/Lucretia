function R = randnt(nsig,varargin)
% R = randnt(nsig,[...])
% Extension to standard Matlab randn function with truncation at nsig
% sigma- the following input args are the same as randn

% Form randn argument list
if nargin==0; error('Need at least 1 input argument!'); end;
if nargin==1
  randArg=1;
else
  randArg=[];
  for iArg=1:nargin-1
    randArg = [randArg varargin{iArg}]; %#ok<AGROW>
  end % for iArg
end % if nargin==1

% nsig should be a scalar or have the same dimensions as randArg
if length(nsig)>1
  if ~isequal(size(nsig),size(ones(randArg)))
    error('nsig should either be scalar or have the same dimensions as the randn arguments!');
  end % if ~equal
end % if nsig a vector

% Generate random numbers, throwing away ones greater than nsig and
% regenerating
R = randn(randArg);
if nsig
  while any(abs(R(:))>nsig(:))
     R(abs(R)>nsig) = randn(1,sum(abs(R(:))>nsig(:))) ;
  end % while R element > nsig
end % if ~nsig, randnt reduces to randn