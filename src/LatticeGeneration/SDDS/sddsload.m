function [sdds] = sddsload(filename)
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
%   SDDSLOAD loads the SDDS data file into memory.
%   SDDSLOAD(FILENAME), where filename is the input SDDS data file


if nargin < 1
    error('Not enough input arguments.')
end
 
is_octave = exist ('OCTAVE_VERSION', 'builtin');

if is_octave
    sddsData = java_new('SDDS.java.SDDS.SDDSFile', filename);
    SDDSUtil = java_new('SDDS.java.SDDS.SDDSUtil');
else
    sddsData = SDDS.java.SDDS.SDDSFile(filename);
    SDDSUtil = SDDS.java.SDDS.SDDSUtil();
end

i = sddsData.readFile;
if i == 0
    error(char(sddsData.getErrors))
end
sdds.filename = sddsData.fileName;
sdds.description.contents = sddsData.descriptionContents;
if length(sdds.description.contents) == 0
    sdds.description.contents = [];
end
sdds.description.text = sddsData.descriptionText;
if length(sdds.description.text) == 0
    sdds.description.text = [];
end
sdds.ascii = sddsData.asciiFile;
sdds.parameter_names = char(sddsData.getParameterNames);
sdds.array_names = char(sddsData.getArrayNames);
sdds.column_names = char(sddsData.getColumnNames);
sdds.pages = sddsData.pageCount;

if sdds.pages == 0
    return
end

k = size(sdds.parameter_names,1);
for j = 1:k
    name = strtrim(convertSDDSname(sdds.parameter_names(j,1:end)));
    type = sddsData.getParameterType(j-1);
    sdds.parameter.(name).type = SDDSUtil.getTypeName(type);
     
    units = sddsData.getParameterUnits(j-1);
    if length(units) == 0
        units = [];
    end
    sdds.parameter.(name).units = units;
  
    symbol = sddsData.getParameterSymbol(j-1);
    if length(symbol) == 0
        symbol = [];
    end
    sdds.parameter.(name).symbol = symbol;
  
    format_string = sddsData.getParameterFormatString(j-1);
    if length(format_string) == 0
        format_string = [];
    end
    sdds.parameter.(name).format_string = format_string;

    description = sddsData.getParameterDescription(j-1);
    if length(description) == 0
        description = [];
    end
    sdds.parameter.(name).description = description;

    if (type == SDDSUtil.SDDS_STRING) || (type == SDDSUtil.SDDS_CHARACTER)
        values = char(sddsData.getParameterValue(j-1,1,0));
        for i = 2:sdds.pages
	    values = char(values,sddsData.getParameterValue(j-1,i,0));
        end
        sdds.parameter.(name).data = values;
    else
        for i = 1:sdds.pages
	    if is_octave
                if type == SDDSUtil.SDDS_DOUBLE
		  sdds.parameter.(name).data(i) = sddsData.getParameterValue(j-1,i,0);
                else
		  sdds.parameter.(name).data(i) = sddsData.getParameterValue(j-1,i,0).doubleValue();
		end
	    else
	      sdds.parameter.(name).data(i) = sddsData.getParameterValue(j-1,i,0);
            end
        end
    end
end

k = size(sdds.column_names,1);
for j = 1:k
    name = strtrim(convertSDDSname(sdds.column_names(j,1:end)));
    type = sddsData.getColumnType(j-1);
    sdds.column.(name).type = SDDSUtil.getTypeName(type);
    
    units = sddsData.getColumnUnits(j-1);
    if length(units) == 0
        units = [];
    end
    sdds.column.(name).units = units;

    symbol = sddsData.getColumnSymbol(j-1);
    if length(symbol) == 0
        symbol = [];
    end
    sdds.column.(name).symbol = symbol;

    format_string = sddsData.getColumnFormatString(j-1);
    if length(format_string) == 0
        format_string = [];
    end
    sdds.column.(name).format_string = format_string;

    description = sddsData.getColumnDescription(j-1);
    if length(description) == 0
        description = [];
    end
    sdds.column.(name).description = description;

    for i = 1:sdds.pages
        rows = sddsData.getRowCount(i);
        if rows == 0
            continue
        end
        values = sddsData.getColumnValues(j-1,i,0);
	aaa = sprintf('page%d',i);
	if is_octave
            if (type == SDDSUtil.SDDS_STRING) || (type == SDDSUtil.SDDS_CHARACTER)
	        values = SDDSUtil.castArrayAsString(values, type);
	        sdds.column.(name).(aaa) = values;
            else
	        values = SDDSUtil.castArrayAsDouble(values, type);
	        for r = 1:rows
	            sdds.column.(name).(aaa)(r) = values(r);
                end
            end
        else
            if (type == SDDSUtil.SDDS_STRING) || (type == SDDSUtil.SDDS_CHARACTER)
                values = SDDSUtil.castArrayAsString(values, type);
            else
                values = SDDSUtil.castArrayAsDouble(values, type);
            end
            sdds.column.(name).(aaa) = values;
        end    
    end
end

k = size(sdds.array_names,1);
for j = 1:k
    name = strtrim(convertSDDSname(sdds.array_names(j,1:end)));
    dimensions = sddsData.getArrayDimensions(j-1);
    sdds.array.(name).dimensions = dimensions;
    type = sddsData.getArrayType(j-1);
    sdds.array.(name).type = SDDSUtil.getTypeName(type);
    
    units = sddsData.getArrayUnits(j-1);
    if length(units) == 0
        units = [];
    end
    sdds.array.(name).units = units;
    
    symbol = sddsData.getArraySymbol(j-1);
    if length(symbol) == 0
        symbol = [];
    end
    sdds.array.(name).symbol = symbol;

    format_string = sddsData.getArrayFormatString(j-1);
    if length(format_string) == 0
        format_string = [];
    end
    sdds.array.(name).format_string = format_string;

    group_name = sddsData.getArrayGroupName(j-1);
    if length(group_name) == 0
        group_name = [];
    end
    sdds.array.(name).group_name = group_name;

    description = sddsData.getArrayDescription(j-1);
    if length(description) == 0
        description = [];
    end
    sdds.array.(name).description = description;
    
    if dimensions == 0
        continue
    end
    for i = 1:sdds.pages
        values = sddsData.getArrayDim(i,j-1);
        clear datasize
        for n = 1:dimensions
            datasize(n) = double(values(n));
        end
        aaa = sprintf('size_page%d',i);
        sdds.array.(name).(aaa) = datasize;
        arraySize = datasize(1);
        for n = 2:dimensions
            arraySize = datasize(n) * arraySize;
        end
        if arraySize == 0
            continue
        end
        aaa = sprintf('page%d',i);
        values = sddsData.getArrayValues(j-1,i,0);
	if is_octave
            if (type == SDDSUtil.SDDS_STRING) || (type == SDDSUtil.SDDS_CHARACTER)
                values = SDDSUtil.castArrayAsString(values, type);
	        sdds.array.(name).(aaa) = values;
            else
	        values = SDDSUtil.castArrayAsDouble(values, type);
                for r = 1:arraySize
	            sdds.array.(name).(aaa)(r) = values(r);
                end
            end
        else
            sdds.array.(name).(aaa) = values;
        end
    end
end
