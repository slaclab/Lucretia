% 
% i)  Lucretia2AML  -- Assumes BEAMLINE is in the global workspace, and
% defaults to no XSIF conversion.  Writes the output to testfile.aml.
% 
% ii) Lucretia2AML(bool)  -- Assumes BEAMLINE is in the global workspace
% and will perform the XSIF conversion if bool==true.  Writes the output
% to testfile.aml (and testfile.xsif if appropriate).
% 
% iii) Lucretia2AML('input','inputfile.mat') -- Grabs BEAMLINE, etc. from
% inputfile.mat, writes to testfile.aml.  No XSIF conversion.
% 
% iv)  Lucretia2AML('output','outputfile.aml') -- Assumes BEAMLINE is in
% the global workspace, and defaults to no XSIF conversion.  Writes the
% output to outputfile.aml.
% 
% v)  Lucretia2AML('input','inputfile.mat',bool) -- As for (iii), but will
% perform XSIF conversion if bool==true.
% 
% vi)  Lucretia2AML('output','outputfile.aml',bool) -- I'm sure you can
% guess ;)
% 
% vii)  Lucretia2AML('input','inputfile.mat','output','outputfile.aml') --
% An exercise for the student.
% 
% viii)
% Lucretia2AML('input','inputfile.mat','output','outputfile.aml',bool) --
% Do I need to spell it out?