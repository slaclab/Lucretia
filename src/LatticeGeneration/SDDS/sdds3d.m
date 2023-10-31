function [ X,Y,Z,labels ] = sdds3d(filename, page, zcolumn, xcolumn, ycolumn)
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
% SDDS3D reads an sdds data file and converts it into a MATLAB 3D mesh
%  There are three valid file structures for this command
%  #1:
%  SDDS3D(FILENAME,PAGE)
%  SDDS3D(FILENAME,PAGE,ZCOLUMN)
%   parameters: Variable1Name, [Varible1Name]Minimum, [Varible1Name]Dimension, [Varible1Name]Interval,
%               Variable2Name, [Varible2Name]Minimum, [Varible2Name]Dimension, [Varible2Name]Interval
%   column: [zcolumn]
%  #2:
%  SDDS3D(FILENAME,PAGE,ZCOLUMNS,XCOLUMN)
%   columns: [xcolumn], [zcolumns*]
%            zcolumns will end in a number which will indicate the location on the y axis
%  #3:
%  SDDS3D(FILENAME,PAGE,ZCOLUMNS,XCOLUMN,YCOLUMN)
%   columns: [xcolumn], [ycolumn], [zcolumn]

import SDDS.java.SDDS.*

if nargin < 2
    error('Not enough input arguments.')
end
if nargin == 2
    zindex = 1;
else
    zindex = 0;
end

sdds = sddsload(filename);

if isfield(sdds, 'parameter') == 0 | isfield(sdds.parameter, 'Variable1Name') == 0
    if nargin == 4
        labels = strvcat(xcolumn, ' ', ' ');
        Xname = convertSDDSname(xcolumn);
        if isfield(sdds.column, Xname) == 0
            error('X axis column does not exist.')
        end
        eval(['X = SDDSUtil.castArrayAsDouble(sdds.column.',Xname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Xname,'.type));'])

        [n_columns,tmp] = size(sdds.column_names);
        lengthvar = length(zcolumn);
        n = lengthvar+1;
        j = 1;
        for i = 1:n_columns
            if strncmp(zcolumn,sdds.column_names(i,:),lengthvar) == 1
                Y(j) = str2double(sdds.column_names(i,n:end));
                Zname = convertSDDSname(sdds.column_names(i,:));
                eval(['Z(:,j) = SDDSUtil.castArrayAsDouble(sdds.column.',Zname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Zname,'.type));'])
                X(:,j) = X(:,1);
                j = j+1;
            end
        end
        [n_rows,tmp] = size(X);
        for i = 2:n_rows
            Y(i,:) = Y(1,:);    
        end
    elseif nargin == 5
        labels = strvcat(xcolumn, ycolumn, zcolumn);
        Xname = convertSDDSname(xcolumn);
        Yname = convertSDDSname(ycolumn);
        Zname = convertSDDSname(zcolumn);
        if isfield(sdds.column, Xname) == 0
            error('X axis column does not exist.')
        end
        if isfield(sdds.column, Yname) == 0
            error('X axis column does not exist.')
        end
        if isfield(sdds.column, Zname) == 0
            error('X axis column does not exist.')
        end
        eval(['X = SDDSUtil.castArrayAsDouble(sdds.column.',Xname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Xname,'.type));'])
        eval(['Y = SDDSUtil.castArrayAsDouble(sdds.column.',Yname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Yname,'.type));'])
        eval(['Z = SDDSUtil.castArrayAsDouble(sdds.column.',Zname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Zname,'.type));'])
    else
        error('Invalid file or command parameters')
    end
else
    labels = strvcat(sdds.parameter.Variable1Name.data, sdds.parameter.Variable2Name.data, ' ');
    Xname = convertSDDSname(sdds.parameter.Variable1Name.data);
    eval(['axisXmin = sdds.parameter.',Xname,'Minimum.data;'])
    eval(['axisXdim = sdds.parameter.',Xname,'Dimension.data;'])
    eval(['axisXint = sdds.parameter.',Xname,'Interval.data;'])
    Yname = convertSDDSname(sdds.parameter.Variable2Name.data);
    eval(['axisYmin = sdds.parameter.',Yname,'Minimum.data;'])
    eval(['axisYdim = sdds.parameter.',Yname,'Dimension.data;'])
    eval(['axisYint = sdds.parameter.',Yname,'Interval.data;'])

    axisXmax = axisXmin + (axisXint * (axisXdim - 1));
    axisYmax = axisYmin + (axisYint * (axisYdim - 1));
    [X,Y] = meshgrid(axisXmin:axisXint:axisXmax,axisYmin:axisYint:axisYmax);
    
    if zindex ~= 0
        zcolumn = sdds.column_names(zindex,1:end);
    end
    Zname = convertSDDSname(zcolumn);
    if isfield(sdds.column, Zname) == 0
        error('Z axis column does not exist.')
    end
    eval(['z = SDDSUtil.castArrayAsDouble(sdds.column.',Zname,'.page',int2str(page),',SDDSUtil.identifyType(sdds.column.',Zname,'.type));'])
    for i = 1:axisXdim
        for j = 1:axisYdim
            Z(j,i) = z((j-1)+((i-1)*axisXdim)+1);
        end
    end
end
