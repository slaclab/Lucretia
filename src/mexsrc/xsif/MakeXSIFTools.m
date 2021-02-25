% ==============================================================================
% script for mex-ing XSIF tools XSIFParse and GetXSIFDictionary
% ==============================================================================

% first get the OS environment variable

ostype=getenv('OSTYPE');if (isempty(ostype)),ostype=computer;end

% do things differently for Windows, Linux, or Solaris

if (strcmp(ostype,'PCWIN'))
  cmd='mex -ID:/xsif/Release -LD:/xsif/Release -lxsif XSIFParse.f';
  eval(cmd) ;
  cmd=['mex -ID:/xsif/Release -LD:/xsif/Release -lxsif ', ...
    'GetXSIFDictionary.f PutElemParsToMatlab.f'];
  eval(cmd) ;
else
  if (strcmp(ostype,'solaris'))
    xsifpath='/afs/slac/g/nlc/codes/matliar/bin/xsif';
    xsiflib='xsif_sun';
  elseif (strcmp(ostype,'linux'))
    xsifpath='/afs/slac/g/nlc/codes/xsif/binlinux';
    xsiflib='xsif_linux';
  end
  
  % build command strings
  
  mexXSIFParse=['mex -I. -L. -l',xsiflib,' XSIFParse.f'];
  mexXSIFDict=['mex -I. -L. -l',xsiflib,' GetXSIFDictionary.f'];
  
  % execute command strings
  
  eval(mexXSIFParse)
  eval(mexXSIFDict)
end
