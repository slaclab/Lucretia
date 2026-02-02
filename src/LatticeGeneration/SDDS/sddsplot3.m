function [ none ] = sddsplot3(X, Y, Z, labels, symbols)
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
% SDDSPLOT3 plots lines and points in 3-D space from the data returned by SDDS3D

if nargin < 3
    error('Not enough input arguments.')
end

if nargin ~= 5
    symbols = '.';
end
axisXmin = min(min(X));
axisXmax = max(max(X));
axisYmin = min(min(Y)); 
axisYmax = max(max(Y));
axisZmin = min(min(Z));
axisZmax = max(max(Z));
plot3(X,Y,Z,symbols);
grid on
xlim([axisXmin axisXmax]);
ylim([axisYmin axisYmax]);
zlim([axisZmin axisZmax]);

if nargin == 4
    xlabel(labels(1,:));
    ylabel(labels(2,:));
    zlabel(labels(3,:));
end

