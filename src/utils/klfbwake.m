function [res,errest]=klfbwake(x,y,s,ctype)
%
% [res,errest]=klfbwake(x,y,s,ctype);
%
% Inputs:
%
%   x      = first input argument
%   y      = second input argument
%   s      = third input argument
%   ctype  = fourth input argument
%
% Outputs:
%
%   res    = first output argument
%   errest = second output argument

% use the mex-file to do the computation

[res,errest]=klfbwake_mex(x,y,s,ctype);
