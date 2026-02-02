function mat2sdds(input, output, variable)
%   mat2sdds saves the MATLAB data from a MAT file to an SDDS file.
%   MAT2SDDS(inputfile, outputfile, variable)

javaclasspath('/usr/local/oag/apps/bin/solaris/SDDS.jar');
import SDDS.java.SDDS.*

if nargin < 3
    error('Not enough input arguments.')
end

load(input);
a1 = 0;
a2 = 0;
[a1, a2] = eval(['size(',variable,')']);
if (a1 ~= 1) & (a2 ~= 1)
     error('Currently unable to convert variable with multiple dimensions.')
end
sdds.ascii = 0;
sdds.column_names = char(variable);
sdds.pages = 1;

eval(['sdds.column.',variable,'.type = ''double'';']);
eval(['sdds.column.',variable,'.page1 = ',variable,';']);

sddssave(sdds, output);

return

