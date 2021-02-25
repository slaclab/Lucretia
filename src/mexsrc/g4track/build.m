function hasbuilt=build(varargin)
% Build GEANT4 Lucretia interface library
% build() - build libg4track.a library
% ===========
% Commands
% - can be issued as standalone arguments or added to above compile targets
% -----------
% 'clean' - clear all build object files and libraries
% ===========
% Options
% - Provide arguments below in addition to build target arguments to modify
%   the build
% -----------
% 'verbose' : echo all build command output
% =======
%
% Output:
%  hasbuilt = true if anything built, else all was up-to-date

hasbuilt=false;
% If cleaning, deal with that now
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'clean')
      delete(sprintf('*.%s',mexext));
      if ispc
        delete('*.obj');
        delete('*.lib');
      else
        delete('*.o');
        delete('*.a');
      end
    end
  end
end

% Set compiler flags and perform initialization actions
FLAGS = 'CC=c++ -largeArrayDims';
CC = 'mex -v' ;
% Verbose output?
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'verbose')
      FLAGS=[FLAGS ' -v'];
    end
  end
end
[stat,LIBS] = system('geant4-config --libs');
if stat
  error('Error getting geant4 library flags- check ''geant4-config'' on search path')
end
[stat,G4FLAGS] = system('geant4-config --cflags');
% Strip newlines
G4FLAGS=strrep(G4FLAGS,sprintf('\n'),''); %#ok<SPRINTFN>
LIBS=strrep(LIBS,sprintf('\n'),''); %#ok<SPRINTFN>
G4FLAGS=strrep(G4FLAGS,sprintf('\r'),'');
LIBS=strrep(LIBS,sprintf('\r'),'');
if stat
  error('Error getting geant4 compile flags- check ''geant4-config'' on search path')
end
FLAGS=[FLAGS sprintf(' CXXFLAGS=''$CXXFLAGS %s''',G4FLAGS)];
FLAGS=[FLAGS ' -I/home/glen/xerces-c-3.0.1/include '];
% Dependencies
dep.g4track={'../LucretiaCommon.h' 'geomConstruction.hh' 'actionInitialization.hh'};
dep.actionInitialization={'primaryGeneratorAction.hh' 'runAction.hh' 'eventAction.hh' 'trackingAction.hh' 'steppingAction.hh'};
dep.trackingAction={'TrackInformation.hh'};
dep.steppingAction={'TrackInformation.hh'};

% Build list
blist={'g4track' 'PhysicsList' 'lucretiaManager' 'geomConstruction' 'actionInitialization' 'primaryGeneratorAction' 'runAction' 'trackingAction' 'steppingAction' 'StackingAction' 'FieldSetup' 'GlobalField' 'lSession' 'TrackInformation'};

% Perform the build
for ib=1:length(blist)
  doThisBuild=false;
  % Check if any dependenices need building first
  if isfield(dep,blist{ib})
    for idep=1:length(dep.(blist{ib}))
      if isempty(regexp(dep.(blist{ib}){idep},'\.','once')) && checkBuild(dep.(blist{ib}){idep},dep)
        eval([CC '-c ' sprintf(' %s.cpp ',dep.(blist{ib}){idep}) FLAGS ' ' LIBS]);
        doThisBuild=true;
        hasbuilt=true;
      end
    end
  end
  % Build this item if needed
  if doThisBuild || checkBuild(blist{ib},dep)
    eval([CC ' -c ' sprintf(' %s.cpp ',blist{ib}) FLAGS ' ' LIBS]);
    hasbuilt=true;
  end
end

% Package library
if ispc
  delete('*.lib');
  system('lib *.obj -OUT:libg4track.lib');
else
  delete('*.a');
  system('ar rcs libg4track.a *.o');
%  system('ar rcs libg4track.a ~/u01/whitegr/xerces-c-3.1.4/obj/*.o');
end

% return true if need to build
function dobuild=checkBuild(name,dep)
dobuild=false;
% Check source and header exists and add to
% build list if file not compiled or source/header newer than compiled file
if ispc
  oname='obj';
else
  oname='o';
end
if ~exist(sprintf('%s.cpp',name),'file') || ~exist(sprintf('%s.hh',name),'file')
  error('Missing dependenices for %s',name)
end
if ~exist(sprintf('%s.%s',name,oname),'file')
  dobuild=true;
else
  f=dir(sprintf('%s.%s',name,oname));
  odate=f.datenum;
  f=dir(sprintf('%s.cpp',name)); fdate(1)=f.datenum;
  f=dir(sprintf('%s.hh',name)); fdate(2)=f.datenum;
  if isfield(dep,name)
    for idep=1:length(dep.(name))
      if ~isempty(regexp(dep.(name){idep},'\.','once'))
        try
          f=dir(sprintf('%s',dep.(name){idep})); fdate(end+1)=f.datenum;
        catch
          error('File dependency missing? (%s)',sprintf('%s',dep.(name){idep}))
        end
      end
    end
  end
  if any(fdate>odate)
    dobuild=true;
  end
end

  
