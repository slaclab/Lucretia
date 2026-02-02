function [ none ] = sddssurfc(X, Y, Z, labels)
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
% SDDSSURFC creates a 3-D shaded surface plot with a contour plot underneath
%  from the data returned by SDDS3D

if nargin < 3
    error('Not enough input arguments.')
end

axisXmin = min(min(X));
axisXmax = max(max(X));
axisYmin = min(min(Y)); 
axisYmax = max(max(Y));
axisZmin = min(min(Z));
axisZmax = max(max(Z));
surfc(X,Y,Z);
shading interp;
axis([axisXmin axisXmax axisYmin axisYmax axisZmin axisZmax])
if nargin == 4
    xlabel(labels(1,:));
    ylabel(labels(2,:));
    zlabel(labels(3,:));
end
