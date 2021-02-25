function index = findcells(CELLARRAY, field, varargin)
%FINDCELLS performs a search on MATLAB cell arrays of structures
%   
% INDEX = FINDCELLS(CELLARRAY, 'field') 
%   returns indexes of elements that have a field named 'field'   
%
% INDEX = FINDCELLS(CELLARRAY, 'field', VALUE) 
%   returns indexes of elements whose field 'field'
%   is equal to VALUE, a wildcard is supported for the VALUE field- use a
%   single * in the VALUE field
%
% INDEX = FINDCELLS(CELLARRAY, 'field', [], START, END) and
% INDEX = FINDCELLS(CELLARRAY, 'field', VALUE, START, END)
%   do the same things as the forms listed above, except that they
%   limit their search range to the selected cell ranges.
%
% FINDCELLS was written by A. Terebilo for the Accelerator
% Toolbox (AT) and is used in Lucretia by permission of the 
% author.
%
% See also GETCELLSTRUCT, SETCELLSTRUCT.
%
% Version date:  26-May-2009.


% 10/27/06, G.White: Modified to use cellfun and added wildcard
% functionality

persistent verInfo % only get version first time routine is called (is slow)

% Check if the first argument is the cell arrray of tstructures 
if(~iscell(CELLARRAY) || isempty(CELLARRAY)) || ~isstruct(CELLARRAY{1}) 
   error('The first argument must be a non-empty cell array of structures') 
end
NE = length(CELLARRAY);
% Chechk if the second argument is a string
if(~ischar(field))
      error('The second argument must be a character string')
end
% check the number of arguments
if( (nargin > 5) || (nargin == 4) )
     error('Incorrect number of inputs')
end
% check that if args 4 and 5 exist, they are numbers
if (nargin == 5)
  if ( (~isnumeric(varargin{2}) ) || (~isnumeric(varargin{3}) ) )
      error('Arguments 4 and 5 must be numeric')
  end
  istart = varargin{2} ;
  iend   = varargin{3} ;
  if ( (istart < 1) || (iend < 1) )
      error('Arguments 4 and 5 must be >= 1') 
  end
  if ( (istart > NE) || (iend > NE) )
      error('Arguments 4 and 5 must be <= length of argument 1') 
  end
else
  istart = 1 ;
  iend = NE ;
end
if nargin>2
  if length(strfind(varargin{1},'*')) > 1
    error('Use of one wildcard ''*'' only supported');
  end
end

% If using Matlab >7 use cellfun command
if isdeployed
  verInfo(1).Version=7.7;
else
  if isempty(verInfo); verInfo=ver; end;
end
if verInfo(1).Version > 7
  if nargin==2 || isempty(varargin{1})
    index=find(cellfun(@(x) isfield(x,field), {CELLARRAY{istart:iend}}, 'UniformOutput', true)); index=index+istart-1;
  else
    if isempty(strfind(varargin{1},'*')) || isnumeric(varargin{1})
      index=find(cellfun(@(x) isfield(x,field) && isequal(x.(field),varargin{1}), ...
        {CELLARRAY{istart:iend}}, 'UniformOutput', true)); index=index+istart-1;
    else
      if isnumeric(varargin{1})
        varString=num2str(varargin{1});
      else
        varString=varargin{1};
      end
      if strcmp(varString(1),'*')
        pat=strrep(varString,'*',''); pat=[pat,'$'];
      elseif strcmp(varString(end),'*')
        pat=strrep(varString,'*',''); pat=['^',pat];
      else
        pat=strrep(varString,'*','.*'); pat=['^',pat,'$'];
      end
      index=find(cellfun(@(x) isfield(x,field) && ~isempty(regexp(x.(field),pat,'ONCE')), ...
        {CELLARRAY{istart:iend}}, 'UniformOutput', true)); index=index+istart-1;  %#ok<*CCAT1>
    end
  end
else
  matchesfound = 0;
  index = zeros(istart,iend);
  for I = istart:iend
   if(isfield(CELLARRAY{I},field))
    matchesfound = matchesfound+1;
    index(matchesfound) = I; 
   end
  end
  index =  index(1:matchesfound);
  if (nargin==2) 
    return ;
  end
  if (isempty(varargin{1}))
    return ;
  end
  matchesfound = 0;
  for I = index
   if isequal(CELLARRAY{I}.(field),varargin{1})
    matchesfound = matchesfound+1;
    % since 'matchesfound' counter is <= loop number,
    % it is save to modify elements of 'index' inside the loop
    index(matchesfound) = I; 
   end
  end
  index =  index(1:matchesfound); 
end
