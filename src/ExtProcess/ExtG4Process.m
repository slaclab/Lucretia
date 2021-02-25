classdef ExtG4Process < ExtProcess & ExtGeometry & ExtG4EMField & ExtMesh & ExtTunnelGeometry & handle
  %EXTG4PROCESS - class to handle interface of Lucretia beam using GEANT4 engine
  
  properties(Constant)
    supportedPrimaryParticleTypes={'all'};
    G4TrackStatus={'Alive' 'StopButAlive' 'StopAndKill' 'KillTrackAndSecondaries' 'Suspend' 'PostponeToNextEvent'};
  end
  properties
    Verbose=0; % Verbosity level for printing status info to stdout in GEANT
    Ecut=0; % Energy cut for GEANT generated tracks / GeV
  end
  properties(Access=private)
    apercheck=true;
    envCheckResp=[];
  end
  properties(Hidden)
    envVars % structure of required environment variables
  end
  properties(Constant,Hidden)
    % Required physics process data files (to be found in src/ExtProcess/)
    dataFiles={'G4Data/G4SAIDDATA1.1' 'G4Data/G4EMLOW6.35' 'G4Data/RealSurface1.0' 'G4Data/G4NEUTRONXS1.4' 'G4Data/G4PII1.3' ...
      'G4Data/PhotonEvaporation3.0' 'G4Data/G4ABLA3.0' 'G4Data/RadioactiveDecay4.0' 'G4Data/G4NDL4.4'};
    % List of names of environment variables which shold be set to above data directory locations
    evarNames={'G4SAIDXSDATA' 'G4LEDATA' 'G4REALSURFACEDATA' 'G4NEUTRONXSDATA' 'G4PIIDATA' 'G4LEVELGAMMADATA' 'G4ABLADATA' 'G4RADIOACTIVEDATA' 'G4NEUTRONHPDATA'};
  end
  methods
    function obj = ExtG4Process(varargin)
      % Superclass initialization
      obj = obj@ExtProcess() ;
      obj = obj@ExtGeometry() ;
      obj = obj@ExtG4EMField() ;
      % Default primary particle type is e-
      obj.PrimaryType = 'e-';
      % List of required environment variables
      for ivar=1:length(obj.dataFiles)
        obj.envVars.(obj.evarNames{ivar})=[obj.extDir obj.dataFiles{ivar}];
      end
      % Get list of materials from GEANT4 database
      dbfile=which('G4MaterialsDatabase.txt');
      if isempty(dbfile)
        error('G4MaterialsDatabase.txt database file not on Matlab search path')
      end
      fid=fopen(dbfile,'r');
      while 1
        tline=fgetl(fid);
        if ~ischar(tline); break; end;
        t=regexp(tline,'(G4_\S+)','tokens','once');
        if ~isempty(t)
          obj.allowedMaterials{end+1}=t{1};
        end
      end
      fclose(fid);
      % Set other requested properties
      if ~nargin; return; end;
      if mod(nargin,2)
        error('Must supply property, value pairs as creation arguments')
      end
      for iarg=1:2:nargin
        if isprop(ExtG4Process,varargin{iarg})
          obj.(varargin{iarg})=varargin{iarg+1};
        else
          error('No settable property: %s',varargin{iarg})
        end
      end
      % Check any set properties
      obj.apercheck=false;
      checkExtGeometryProps(obj);
      obj.apercheck=true;
    end
  end
  methods
    function [resp,message]=checkEnv(obj)
      resp=1; message=[];
      try
        if obj.isparallel || isinparfor || (~isempty(obj.envCheckResp) && obj.envCheckResp==true)
          return
        end
      catch
        if  (~isempty(obj.envCheckResp) && obj.envCheckResp==true)
          return
        end
      end
      df=obj.dataFiles;
      ev=obj.envVars;
      evf=fieldnames(ev);
      obj.envCheckResp=true;
      if ~isempty(df)
        for idf=1:length(df)
          if ~exist(fullfile(obj.extDir,df{idf}),'file')
            obj.envCheckResp=false;
            message=sprintf('ExtG4Process is missing required file :%s (dir= %s)',fullfile(obj.extDir,df{idf}),evalc('!ls'));
          end
        end
      end
      if ~isempty(evf)
        for iev=1:length(evf)
          setenv(evf{iev},ev.(evf{iev}));
        end
      end
      resp=obj.envCheckResp;
    end
    function SetMaterial(obj,material)
      SetMaterial@ExtGeometry(obj,material);
      obj.Material=obj.Material;
    end
    function SetGeometryType(obj,type)
      global BEAMLINE
       if ~exist('type','var') || isempty(type)
         if ~isempty(obj.elemno) && isfield(BEAMLINE{obj.elemno},'Geometry')
           type=BEAMLINE{obj.elemno}.Geometry;
         else
           type='Ellipse';
         end
       end
       SetGeometryType@ExtGeometry(obj,type);
    end
    function SetAper(obj,val1,val2)
      global BEAMLINE
      % If this object attached to a BEAMLINE element then issue a warning
      % if the aperture does not match the BEAMLINE element or set to
      % BEAMLINE aper values if no values provided
      if (~exist('val1','var') || isempty(val1)) && ~isempty(obj.elemno) && isfield(BEAMLINE{obj.elemno},'aper')
        val1=BEAMLINE{obj.elemno}.aper(1);
      elseif (~exist('val1','var') || isempty(val1)) && ~isempty(obj.elemno) && isfield(BEAMLINE{obj.elemno},'HGAP')
        val1=mean(BEAMLINE{obj.elemno}.HGAP);
      end
      if ~exist('val2','var') || isempty(val2)
        val2=val1;
        if ~isempty(obj.elemno) && isfield(BEAMLINE{obj.elemno},'aper') && length(BEAMLINE{obj.elemno}.aper)>1
          val2=BEAMLINE{obj.elemno}.aper(2);
        elseif ~isempty(obj.elemno) && isfield(BEAMLINE{obj.elemno},'HGAP')
          val2=mean(BEAMLINE{obj.elemno}.HGAP);
        end
      end
      SetAper@ExtGeometry(obj,val1,val2);
      if isempty(obj.elemno) || ~obj.apercheck; return; end;
      if isfield(BEAMLINE{obj.elemno},'aper')
        if BEAMLINE{obj.elemno}.aper(1)~=val1 || (length(BEAMLINE{obj.elemno}.aper)>1 && BEAMLINE{obj.elemno}.aper(2)~=val2)
          warning('ExtG4Process:AperMismatch','Provided geometry apertures don''t match associated Lucretia BEAMLINE element')
        end
      end
    end
  end
end

