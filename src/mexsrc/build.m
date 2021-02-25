function build(varargin)
% Build Lucretia mex files
% - See below for available build targets
% build cpu - build Lucretia mex libraries for cpu target, no GEANT4
%    support (default)
% build cpu-g4  - build Lucretia mex libraries for cpu target, with GEANT4
%    support. Requires GEANT4 (v.10+) installed
%    (with "geant4-config" script on the system search path)
% build gpu - build Lucretia mex libraries for gpu target, no GEANT4
%    support. This requires an NVidia graphics card with compute capability
%    2.0+ and the latest cuda drivers installed. Also requires the Matlab
%    parallel comupting toolbox.
%
% WARNING: When changing compile targets, issue a 'clean' command before
% re-building
% ===========
% Commands
% - can be issued as standalone arguments or added to above compile targets
% -----------
% install - move built mex libraries into correct folders
% clean - clear all build object files and libraries
% cleanall - as above but also clear GEANT4 library in g4track if built
% ===========
% Options
% - Provide arguments below in addition to build target arguments to modify
%   the build
% -----------
% mlrand : use Matlabs default random number generator instead of C
% compiler standard rand() function. e.g. for synchrotron radiation
% calculation related functions. Caution: This is MUCH slower.
% verbose : echo all build command output
% =======

% If cleaning, deal with that now
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'clean')
      delete(sprintf('*.%s',mexext));
      if ispc
        delete('*.obj');
      else
        delete('*.o');
      end
    end
    if strcmp(varargin{iarg},'cleanall')
      delete(sprintf('*.%s',mexext));
      if ispc
        delete('*.obj')
        delete('g4track/*.lib');
      else
        delete('*.o');
        delete('g4track/*.a');
      end
    end
  end
end

% - Set defaults and parse input arguments
target='none';
randFunc='c';
if nargin>0
  for iarg=1:nargin
    switch lower(varargin{iarg})
      case {'cpu','gpu','cpu-g4'}
        target=lower(varargin{iarg});
      case 'mlrand'
        randFunc='matlab';
    end
  end
end

% Set compiler flags and perform initialization actions
FLAGS = '-DLUCETIA_DPREC -largeArrayDims';
LIBS = [] ;
if strcmp(randFunc,'matlab')
  FLAGS = [FLAGS ' -DLUCRETIA_MLRAND'] ;
end
if strcmp(target,'gpu')
  CC='mexcuda -dynamic';
elseif strcmp(target,'cpu-g4')
  CC='mex CC=c++';
else
  if ismac
%     CC='mex CC=clang++';
    CC='mex';
  else
    CC='mex CC=c++';
  end
end
% Verbose output?
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'verbose')
      FLAGS=[FLAGS ' -v'];
    end
  end
end
if strcmp(target,'cpu-g4')
  [stat,G4LIBS] = system('geant4-config --libs');
  %G4LIBS='-L/var/geant4/bin/../lib64 -lG4Tree -lG4FR -lG4GMocren -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling -lG4interfaces -lG4analysis -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4global -lG4clhep -lG4zlib';
  if stat
    error('Error getting geant4 library flags- check ''geant4-config'' on search path')
  end
  [stat,G4FLAGS] = system('geant4-config --cflags');
  if stat
    error('Error getting geant4 compile flags- check ''geant4-config'' on search path')
  end
  G4FLAGS=strrep(G4FLAGS,sprintf('\n'),''); %#ok<SPRINTFN>
  G4LIBS=strrep(G4LIBS,sprintf('\n'),''); %#ok<SPRINTFN>
  G4FLAGS=strrep(G4FLAGS,sprintf('\r'),'');
  G4LIBS=strrep(G4LIBS,sprintf('\r'),'');
  FLAGS=[FLAGS ' -DLUCRETIA_G4TRACK' sprintf(' CXXFLAGS=''$CXXFLAGS %s'' CFLAGS=''$CFLAGS %s''',G4FLAGS,G4FLAGS)];
  LIBS=[LIBS ' -Lg4track/ -lg4track -L/home/glen/xerces-c-3.0.1/lib -lxerces-c ' G4LIBS];
%   LIBS=[LIBS ' -Lg4track/ -lg4track ' G4LIBS];
end

% Dependencies
dep.LucretiaCommon={'LucretiaCommon.h' 'LucretiaMatlab.h' 'LucretiaDictionary.h' 'LucretiaPhysics.h' 'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h' 'LucretiaCuda.h'};
dep.LucretiaPhysics={'LucretiaCommon.h' 'LucretiaPhysics.h' 'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h'};
dep.LucretiaMatlab={'LucretiaGlobalAccess.h' 'LucretiaVersionProto.h' 'LucretiaMatlab.h' 'LucretiaCuda.h'};
dep.LucretiaMatlabErrMsg={'LucretiaGlobalAccess.h'};
dep.GetRmats={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
dep.GetTwiss={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h' 'LucretiaPhysics.h'};
dep.RmatAtoB={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
dep.TrackThru={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h' 'LucretiaCuda.h'};
dep.VerifyLattice={'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg' 'LucretiaMatlab.h' 'LucretiaCommon.h' 'LucretiaGlobalAccess.h'};
% - target dependent dependencies
if strcmp(target,'cpu-g4')
  if ispc
    dep.TrackThru{end+1}='g4track/libg4track.lib';
    dep.LucretiaCommon{end+1}='g4track/libg4track.lib';
  else
    dep.TrackThru{end+1}='g4track/libg4track.a';
    dep.LucretiaCommon{end+1}='g4track/libg4track.a';
  end
end

% Form build & install lists
if strcmp(target,'gpu')
  blist={'TrackThru' 'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg'};
  ilist.exe={'TrackThru'};
  ilist.dir={'Tracking'};
else
  blist={'GetRmats' 'GetTwiss' 'RmatAtoB' 'TrackThru' 'VerifyLattice' 'LucretiaCommon' 'LucretiaPhysics' 'LucretiaMatlab' 'LucretiaMatlabErrMsg'};
  ilist.exe={'GetRmats' 'GetTwiss' 'RmatAtoB' 'TrackThru' 'VerifyLattice'};
  ilist.dir={'RMatrix'  'Twiss'    'RMatrix'  'Tracking'  'LatticeVerification'};
end
  
% Perform the build
anybuild=false;
if ~strcmp(target,'none')
  for ib=1:length(blist)
    doThisBuild=false;
    % Check if any dependenices need building first
    for idep=1:length(dep.(blist{ib}))
      if isempty(regexp(dep.(blist{ib}){idep},'\.','once')) && checkBuild(dep.(blist{ib}){idep},dep)
        doBuild(dep.(blist{ib}){idep},CC,FLAGS,LIBS,ilist,dep,target);
        doThisBuild=true;
        anybuild=true;
      end
    end
    % Build this item if needed
    if doThisBuild || checkBuild(blist{ib},dep)
      doBuild(blist{ib},CC,FLAGS,LIBS,ilist,dep,target);
      anybuild=true;
    end
  end
  if ~anybuild
    disp('Build up to date, nothing done.')
  end
end

% Install files
if nargin>0
  for iarg=1:nargin
    if strcmp(varargin{iarg},'install')
      for il=1:length(ilist.exe)
        if exist(sprintf('%s.%s',ilist.exe{il},mexext),'file')
          if strcmp(target,'gpu') && strcmp(ilist.exe{il},'TrackThru')
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s_gpu.%s',ilist.exe{il},mexext)));
          elseif strcmp(target,'cpu-g4')
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)));
            copyfile(fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)),fullfile('..',ilist.dir{il},sprintf('%s_g4.%s',ilist.exe{il},mexext)))
          else
            movefile(sprintf('%s.%s',ilist.exe{il},mexext),fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)));
            copyfile(fullfile('..',ilist.dir{il},sprintf('%s.%s',ilist.exe{il},mexext)),fullfile('..',ilist.dir{il},sprintf('%s_cpu.%s',ilist.exe{il},mexext)))
          end
        end
      end
    end
  end
end

% perform build
function doBuild(bname,CC,FLAGS,LIBS,ilist,dep,target)
olist=[];
if ispc
  oext='obj';
else
  oext='o';
end
for idep=1:length(dep.(bname))
  if isempty(regexp(dep.(bname){idep},'\.','once'))
    olist=[olist sprintf('%s.%s',dep.(bname){idep},oext) ' '];
  end
end
fext='c';
if strcmp(target,'gpu')
  copyfile(sprintf('%s.c',bname),sprintf(sprintf('%s.cu',bname)));
  fext='cu';
end
if ismember(bname,ilist.exe)
  bcmd=[CC ' ' FLAGS sprintf(' %s.%s ',bname,fext) olist LIBS];
else
  bcmd=[CC ' ' FLAGS sprintf(' -c %s.%s ',bname,fext) olist LIBS];
end
try
  eval(bcmd);
catch ME
  fprintf('Error building: %s\n',bname)
  fprintf('(%s)\n',bcmd);
  if strcmp(target,'gpu')
    delete(sprintf('%s.cu',bname));
  end
  rethrow(ME)
end
if strcmp(target,'gpu')
  delete(sprintf('%s.cu',bname));
end

% return true if need to build
function dobuild=checkBuild(name,dep)
dobuild=false;
if ispc
  oext='obj';
else
  oext='o';
end
% Check source and header exists and add to
% build list if file not compiled or source/header newer than compiled file
if ~exist(sprintf('%s.c',name),'file')
  error('Missing dependenices for %s',name)
end
if ~exist(sprintf('%s.%s',name,oext),'file') && ~exist(fullfile('.',sprintf('%s.%s',name,mexext)),'file')
  dobuild=true;
else
  f=dir(sprintf('%s.%s',name,oext));
  if isempty(f)
    f=dir(sprintf('%s.%s',name,mexext));
  end
  odate=f.datenum;
  f=dir(sprintf('%s.c',name)); fdate(1)=f.datenum;
  for idep=1:length(dep.(name))
    if ~isempty(regexp(dep.(name){idep},'\.','once'))
      f=dir(dep.(name){idep}); fdate(end+1)=f.datenum;
    end
  end
  if any(fdate>odate)
    dobuild=true;
  end
end

  
