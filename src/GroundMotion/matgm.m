function [dx,dy] = matgm( s_coords, time, gm_model, TF_xList, TF_yList, nTF)
% [dx,dy] = matgm( s_coords, time, gm_model) 
% Generates ground motion according to the model implemented in Liar by Andrei Seryi
% Requires input data files with A,B,C,K and ATFA/B ground motion models
% Produce dx, dy (m) vectors of ground motion at given s co-ordinates (m)
% gm_model should be 'A', 'B', 'C', 'K', 'ATFA' or 'ATFB'
% Set time=0 to generate new seed and harmonics etc, also happens on change
% of gm_model from previous call and with first call
% ------------------------------------
% Also supported are TF's for elements
% ------------------------------------
% There must exist data files in the format "tfData_N.dat" in the current
% working directory where 1 <= N <= nTF (number of TF's used). These are
% Liar-format TF files (PSD data). TF_xList and TF_yList should be a
% vector of length s_coords with the TF id (file) to associate with the
% elements in s_coords (0 = nothing to be assigned)
% 
% WARNING: GENERATION OF NEW SEED CAUSES ALL MEX FUNCTION MEMORY TO CLEAR-
% WILL POSSIBLY AFFECT ANY OTHER MEMORY-RESIDENT MEX FUNCTIONS
persistent lastModel

switch upper(gm_model)
  case 'A'
    iModel=1;
  case 'B'
    iModel=2;
  case 'C'
    iModel=3;
  case 'K'
    iModel=4;
  case 'X'
    iModel=5;
  case 'Y'
    iModel=6;
  otherwise
    error('Models supported are: A,B,C,K,X or Y only!');
end

% Check time format
if length(time)>1; error('time must be a scalar'); end;
if time < 0; error('time must be >=0'); end;

% Check s_coords format
[nrow ncol] = size(s_coords);
if nrow < ncol
  error('Should pass s_coords as a row vector!')
end

% Re-calculate harmonics and re-seed if model different from the last time
% subroutine ran or requested through time=0 or first run
if time==0 || isempty(lastModel) || ~(lastModel==iModel)
  clear mex
  seed=round(rand*1e6);
else
  seed=0;
end
lastModel=iModel;

% Check TF input structure
if nargin>3
  if ~isequal(size(TF_xList),size(TF_yList)) || ~isequal(size(TF_xList),size(s_coords)) || ...
     ~isequal(size(TF_yList),size(s_coords))
    error('TF_yList, TF_xList and s_coords should all be a row vector of the same dimension!')
  end
  if max(TF_xList)>nTF || max(TF_yList)>nTF; error('Trying to access a TF that doesn''t exist!'); end;
  if ~isempty(find(TF_xList<0,1)) || ~isempty(find(TF_yList<0,1)); error('All TF list elements should be >0!'); end;
  % Check TF data files exists
  for iFile=1:nTF
    if ~exist(['tfData_',num2str(iFile),'.dat'],'file'); error(['File: tfData_',num2str(iFile),'.dat Doesn''t exist!']); end;
  end
end

% Run ground motion model
if nargin > 3
  [dx, dy]=matgm_mex(s_coords,seed,time,iModel,TF_xList,TF_yList,nTF);
else
  [dx, dy]=matgm_mex(s_coords,seed,time,iModel);
end

return
