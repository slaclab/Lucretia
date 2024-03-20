function sddssave(sdds, filename)
% ************************************************************************
% Copyright (c) 2002 The University of Chicago, as Operator of Argonne
% National Laboratory.
% Copyright (c) 2002 The Regents of the University of California, as
% Operator of Los Alamos National Laboratory.
% This file is distributed subject to a Software License Agreement found
% in the file LICENSE that is included with this distribution. 
% ************************************************************************
%   SDDSSAVE saves the SDDS data structure to a file.
%   SDDSSAVE(SDDS,FILENAME), where fileName is the output SDDS data file
%   and sdds is the SDDS data structure.

%import SDDS.java.SDDS.*

if nargin < 1
    error('Not enough input arguments.')
end
if nargin == 1
    if isfield(sdds, 'filename') == 0
        error('filename in SDDS structure is missing')
    end
    filename = sdds.filename;
end

is_octave = exist ('OCTAVE_VERSION', 'builtin');

% Set any missing description variables
if isfield(sdds, 'description') == 0
    sdds.description = [];
end
if isvalidSDDSelement(sdds, 'sdds.description', 'contents') == 0
    sdds.description.contents = [];
end
if isvalidSDDSelement(sdds, 'sdds.description', 'text') == 0
    sdds.description.text = [];
end

% Ensure ascii value is set
if isfield(sdds, 'ascii') == 0
    sdds.ascii = 1;
else
%    if isnumeric(sdds.ascii) == 0 
%        error('sdds.ascii must be 1 (true) or 0 (false)')
%    end
    if isequal(sdds.ascii,0) == 0 && isequal(sdds.ascii,1) == 0
        error('sdds.ascii must be 1 (true) or 0 (false)')
    end
end

% Ensure the pages value is set
if isfield(sdds, 'pages') == 0
    sdds.pages = 1;
else
    if isnumeric(sdds.pages) == 0 || rem(sdds.pages,1) ~= 0
        error('sdds.pages must be an integer')
    end
end

% Set any missing parameter variables
if isfield(sdds, 'parameter_names') == 0
    n_parameters = 0;
else
    [n_parameters,tmp] = size(sdds.parameter_names);
end
if n_parameters > 0
    if isfield(sdds, 'parameter') == 0
        error('sdds.parameter is missing')
    end
    for i = 1:n_parameters
        if is_octave
            name = convertSDDSname(strtok(sdds.parameter_names{i,1:end}));
        else
	    name = convertSDDSname(strtok(sdds.parameter_names(i,1:end)));
        end
        if isfield(sdds.parameter, name) == 0
            error(strcat('sdds.parameter.',name,' is missing'))
        end
        if isfield(sdds.parameter.(name) , 'type') == 0
            error(['sdds.parameter.',name,'.type is missing'])
        end
        if isvalidSDDSelement(sdds, ['sdds.parameter.',name], 'units') == 0
	    sdds.parameter.(name).units = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.parameter.',name], 'symbol') == 0
	    sdds.parameter.(name).symbol = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.parameter.',name], 'format_string') == 0
	    sdds.parameter.(name).format_string = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.parameter.',name], 'description') == 0
	  sdds.parameter.(name).description = [];
        end
        if isfield(sdds.parameter.(name), 'data') == 0
            error(['sdds.parameter.',name,'.data is missing'])
        end
    end
end

% Set any missing array variables
if isfield(sdds, 'array_names') == 0
    n_arrays = 0;
else
    [n_arrays,tmp] = size(sdds.array_names);
end
if n_arrays > 0
    if isfield(sdds, 'array') == 0
        error('sdds.array is missing')
    end
    for i = 1:n_arrays
        if is_octave
	    name = convertSDDSname(strtok(sdds.array_names{i,1:end}));
        else
            name = convertSDDSname(strtok(sdds.array_names(i,1:end)));
        end
        if isfield(sdds.array, name) == 0
            error(strcat('sdds.array.',name,' is missing'))
        end
	if isfield(sdds.array.(name), 'dimensions') == 0
            error(['sdds.array.',name,'.dimensions is missing'])
        end
	if isfield(sdds.array.(name), 'type') == 0
            error(['sdds.array.',name,'.type is missing'])
        end
        if isvalidSDDSelement(sdds, ['sdds.array.',name], 'units') == 0
	    sdds.array.(name).units = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.array.',name], 'symbol') == 0
	    sdds.array.(name).symbol = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.array.',name], 'format_string') == 0
	    sdds.array.(name).format_string = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.array.',name], 'group_name') == 0
	    sdds.array.(name).group_name = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.array.',name], 'description') == 0
           sdds.array.(name).description = [];
        end
        for j = 1:sdds.pages
	    aaa = sprintf('size_page%d', j);
	    if isfield(sdds.array.(name), aaa) == 0
                error(['sdds.array.',name,'.size_page',int2str(j),' is missing'])
            end
            aaa = sprintf('page%d', j);
            if isfield(sdds.array.(name), aaa) == 0
                error(['sdds.array.',name,'.page',int2str(j),' is missing'])
            end
        end
    end
end

% Set any missing column variables
if isfield(sdds, 'column_names') == 0
    n_columns = 0;
else
    [n_columns,tmp] = size(sdds.column_names);
end
if n_columns > 0
    if isfield(sdds, 'column') == 0
        error('sdds.column is missing')
    end
    for i = 1:n_columns
        if is_octave
	    name = convertSDDSname(strtok(sdds.column_names{i,1:end}));
        else
            name = convertSDDSname(strtok(sdds.column_names(i,1:end)));
        end
        if isfield(sdds.column, name) == 0
            error(strcat('sdds.column.',name,' is missing'))
        end
	  if isfield(sdds.column.(name), 'type') == 0
            error(['sdds.column.',name,'.type is missing'])
        end
        if isvalidSDDSelement(sdds, ['sdds.column.',name], 'units') == 0
	    sdds.column.(name).units = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.column.',name], 'symbol') == 0
	    sdds.column.(name).symbol = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.column.',name], 'format_string') == 0
	    sdds.column.(name).format_string = [];
        end
        if isvalidSDDSelement(sdds, ['sdds.column.',name], 'description') == 0
	    sdds.column.(name).description = [];
        end
        for j = 1:sdds.pages
	    aaa = sprintf('page%d', j);
            if isfield(sdds.column.(name), aaa) == 0
                error(['sdds.column.',name,'.page',int2str(j),' is missing'])
            end
        end
    end
end

if is_octave
    sddsData = java_new('SDDS.java.SDDS.SDDSFile', filename, ~sdds.ascii);
else
    sddsData = SDDS.java.SDDS.SDDSFile(filename, ~sdds.ascii);
end

sddsData.setDescription(sdds.description.text, sdds.description.contents)
for i = 1:n_parameters
    if is_octave
	name = strtok(sdds.parameter_names{i,1:end});
    else
        name = strtok(sdds.parameter_names(i,1:end));
    end
    name2 = convertSDDSname(name);
     j = sddsData.defineParameter(name, sdds.parameter.(name2).symbol, sdds.parameter.(name2).units, sdds.parameter.(name2).description, sdds.parameter.(name2).format_string, sdds.parameter.(name2).type, []);
    if j == -1
        error(char(sddsData.getErrors))
    end
    if (strcmp(sdds.parameter.(name2).type, 'string')) || (strcmp(sdds.parameter.(name2).type, 'character'))
        [pages,string_length] = size(sdds.parameter.(name2).data);
        for k = 1:sdds.pages
            string_end = string_length;
            if string_end == 0
                sddsData.setParameter(j, ' ', k);
            else
	        while isequal(sdds.parameter.(name2).data(k,string_end), ' ')
                    string_end = string_end - 1;
                end
                sddsData.setParameter(j, sdds.parameter.(name2).data(k,1:string_end), k);
            end
        end
    else
        sddsData.setParameter(j, sdds.parameter.(name2).data, 1);
    end
end

for i = 1:n_columns
    if is_octave
	name = strtok(sdds.column_names{i,1:end});
    else
        name = strtok(sdds.column_names(i,1:end));
    end
    name2 = convertSDDSname(name);
    j = sddsData.defineColumn(name, sdds.column.(name2).symbol, sdds.column.(name2).units, sdds.column.(name2).description, sdds.column.(name2).format_string, sdds.column.(name2).type, 0);
    if j == -1
        error(char(sddsData.getErrors))
    end
    for k = 1:sdds.pages
	aaa = sprintf('page%d', k);
        sddsData.setColumn(j, sdds.column.(name2).(aaa), k);
    end
end

for i = 1:n_arrays
    if is_octave
	name = strtok(sdds.array_names{i,1:end});
    else
        name = strtok(sdds.array_names(i,1:end));
    end
    name2 = convertSDDSname(name);
    j = sddsData.defineArray(name, sdds.array.(name2).symbol, sdds.array.(name2).units, sdds.array.(name2).description, sdds.array.(name2).format_string, sdds.array.(name2).group_name, sdds.array.(name2).type, 0, sdds.array.(name2).dimensions);
    if j == -1
        error(char(sddsData.getErrors))
    end
    for k = 1:sdds.pages
        aaa = sprintf('size_page%d', k);
        if sdds.array.(name2).dimensions == 1
	    bbb(1) = sdds.array.(name2).(aaa);
	    bbb(2) = 0;
            sddsData.setArrayDim(k, j, bbb);
        else
            sddsData.setArrayDim(k, j, sdds.array.(name2).(aaa));
        end
        aaa = sprintf('page%d', k);
        sddsData.setArray(j, sdds.array.(name2).(aaa), k);
    end
end

if sddsData.writeFile == 0
    error(char(sddsData.getErrors))
end
return



function [valid] = isvalidSDDSelement(sdds, structure, element)
%   ISVALIDSDDSELEMENT checks for a valid SDDS data structure element.

valid = 1;
if eval(['isfield(',structure,', element)']) == 0
    valid = 0;
else
    if eval(['isequal(',structure,'.',element,',[])']) == 0
        if eval(['length(',structure,'.',element,')']) == 0
            valid = 0;
        end
    end 
end

