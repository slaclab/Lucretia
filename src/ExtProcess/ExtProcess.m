classdef ExtProcess < handle & ExtPhysics
  % EXTPROCESS - Class for handling interfaces to external code linked with Lucretia
  % Currently supported is GEANT4
  
  properties(Constant, Hidden)
    supportedProcessTypes = {'GEANT4'} ;
  end
  properties(Abstract)
    Verbose ;
  end
  properties(Abstract,Constant)
    supportedPrimaryParticleTypes ;
  end
  properties(Access=protected)
    elemno ;
    processID ;
    type ;
    processPhysicsEnabled = 'All' ;
  end
  properties
    PrimarySampleOrder
    beamlineCoord = zeros(1,4) ; % Physical location of center of beam pipe [x,xang,y,yang] in longitudinal center of element / m or deg.
  end
  properties(SetAccess=protected)
    SecondaryParticleTypes = 'All' ; % Which particle types to store
    PrimaryParticlesData
    SecondaryParticlesData
    isparallel=false; % Set true if in parallel environment
  end
  properties(Dependent)
    MaxSecondaryParticlesPerPrimary
    MaxSecondaryParticles % set >0 to store up to N secondary particles produced
    MaxPrimaryParticles
    SecondaryStorageCuts % 1 = only store hits on d/s element face, 0 = store all
    ForceProcess % Force ExtProcess regardless of aperture if == true
    TrackStoreMax % Set >0 to store tracking point data in EXT process (max # points per primary particle to store)
    PrimaryType % Type of particle to use for primaries transfered from Lucretia to ExtProcess
    TrackSecondaryAsPrimary % Track secondaries as primary particles (ensures z-location kept when passing to external process)
  end
  properties(Access=protected)
    fTrackSecondaryAsPrimary=false;
    fPrimarySampleOrder ;
    fMaxSecondaryParticlesPerPrimary = uint32(10) ;
    fMaxSecondaryParticles= uint32(0) ;
    fMaxPrimaryParticles = uint32(1e4) ;
    fNumSecondariesStored = uint32(0) ; % Number of secondary particles actually stored
    fSecondaryStorageCuts = uint8(1) ;
    fForceProcess = false ; % Force ExtProcess regardless of aperture
    extDir % absolute path to root ExtProcess diretcory
    fTrackStoreMax=0;
    fPrimaryType
  end
  properties(Abstract,Constant,Hidden)
    dataFiles % define cell array of data files associated with process type (must provide empty cell array if none) [paths relative to Lucretia/src/ExtProcess]
  end
  properties(Abstract,Hidden)
    envVars % structure of required environment variables
  end
  methods
    function obj = ExtProcess()
      td=which('TrackThru');
      if isempty(td)
        error('TrackThru mex file not on search path!')
      end
      obj.extDir=regexprep(td,sprintf('TrackThru.%s',mexext),'../ExtProcess/');
    end
    function val = get.TrackStoreMax(obj)
      val = uint32(obj.fTrackStoreMax);
    end
    function set.TrackSecondaryAsPrimary(obj,val)
      obj.fTrackSecondaryAsPrimary=logical(val);
    end
    function val=get.TrackSecondaryAsPrimary(obj)
      val=obj.fTrackSecondaryAsPrimary;
    end
    function set.PrimaryType(obj,val)
      if ~ismember('all',obj.supportedPrimaryParticleTypes) && ~ismember(val,obj.supportedPrimaryParticleTypes)
        error('Unsupported type of primary particle')
      end
      obj.fPrimaryType=val;
    end
    function val=get.PrimaryType(obj)
      val=obj.fPrimaryType;
    end
    function set.TrackStoreMax(obj,val)
      if val<0
        error('Supply positive integer')
      end
      obj.fTrackStoreMax=uint32(val);
    end
    function val = get.ForceProcess(obj)
      val=obj.fForceProcess;
    end
    function set.ForceProcess(obj,val)
      obj.fForceProcess=logical(val);
    end
    function val = get.SecondaryStorageCuts(obj)
      val=obj.fSecondaryStorageCuts;
    end
    function set.SecondaryStorageCuts(obj,val)
      obj.fSecondaryStorageCuts=uint8(val);
    end
    function val=get.MaxPrimaryParticles(obj)
      val=obj.fMaxPrimaryParticles;
    end
    function set.MaxPrimaryParticles(obj,val)
      obj.fMaxPrimaryParticles=uint32(val);
    end
    function val=get.MaxSecondaryParticlesPerPrimary(obj)
      val=obj.fMaxSecondaryParticlesPerPrimary;
    end
    function val=get.MaxSecondaryParticles(obj)
      val=obj.fMaxSecondaryParticles;
    end
    function set.MaxSecondaryParticlesPerPrimary(obj,val)
      obj.fMaxSecondaryParticlesPerPrimary=uint32(val);
    end
    function set.MaxSecondaryParticles(obj,val)
      obj.fMaxSecondaryParticles=uint32(val);
    end
    function FinalizeTrackingData(obj)
      % To be called by Track after tracking
      global BEAMLINE
      if isfield(BEAMLINE{obj.elemno},'ExtProcess_primariesData')
        BEAMLINE{obj.elemno}.ExtProcess(obj.processID).PrimaryParticlesData=BEAMLINE{obj.elemno}.ExtProcess_primariesData;
        BEAMLINE{obj.elemno}=rmfield(BEAMLINE{obj.elemno},'ExtProcess_primariesData') ;
      end
      if isfield(BEAMLINE{obj.elemno},'ExtProcess_secondariesData')
        BEAMLINE{obj.elemno}.ExtProcess(obj.processID).SecondaryParticlesData=BEAMLINE{obj.elemno}.ExtProcess_secondariesData;
        BEAMLINE{obj.elemno}=rmfield(BEAMLINE{obj.elemno},'ExtProcess_secondariesData') ;
      end
    end
    function InitializeTrackingData(obj,primaryBeam,b1,b2)
      % Set order in which primary rays are serviced, order by highest
      % charge weight first, with equal charge weights randomized
      global BEAMLINE
      if ~isfield(primaryBeam,'Bunch') || ~isfield(primaryBeam.Bunch,'Q') || length(primaryBeam.Bunch.Q)<1
        error('Badly formatted Lucretia Beam ''primaryBeam''')
      end
      randind=randperm(length(primaryBeam.Bunch(b1).Q));
      [~, sortind]=sort(primaryBeam.Bunch(b1).Q(randind));
      for ibunch=b1:b2
        obj.PrimarySampleOrder{ibunch}=uint32(randind(sortind));
      end
      % Set up data structures for returning info about primary and
      % secondary particles not codified in Lucretia Beam
      for ibunch=b1:b2
        BEAMLINE{obj.elemno}.ExtProcess_primariesData(ibunch)=ExtPrimaryParticles();
        BEAMLINE{obj.elemno}.ExtProcess_secondariesData(ibunch)=ExtSecondaryParticles();
      end
    end
  end
  methods(Abstract)
    [resp,message] = checkEnv(obj) % return true if environment for Ext Process checks out OK
  end
  methods(Static)
    function new(processType,elemno,varargin)
      global BEAMLINE
      % First parse inputs and generate new ExtProcess structure in
      % BEAMLINE
      if elemno<1 || elemno>length(BEAMLINE)
        error('No existing BEAMLINE element matching request')
      end
      if ~isfield(BEAMLINE{elemno},'ExtProcess') || isempty(BEAMLINE{elemno}.ExtProcess)
        procno=1;
      else
        procno=length(BEAMLINE{elemno}.ExtProcess);
      end
      switch processType
        case 'GEANT4'
          % Check there is some length to this element
          if ~isfield(BEAMLINE{elemno},'L')
            error('No length field to this element, GEANT4 tracking not possible!')
          end
          if BEAMLINE{elemno}.L==0
            warning('ExtProcess:noLength','Zero length for this element, GEANT4 will not attempt tracking here');
          end
          BEAMLINE{elemno}.ExtProcess(procno)=ExtG4Process(varargin{:});
          BEAMLINE{elemno}.ExtProcess(procno).elemno=elemno;
          BEAMLINE{elemno}.ExtProcess(procno).type='GEANT4';
          % If not explicitly setting apertures, set them to default values
          % (equal to BEAMLINE aper field if there is one)
          if ~any(strcmp('AperX',varargin)) && ~any(strcmp('AperY',varargin))
            BEAMLINE{elemno}.ExtProcess(procno).SetAper([],[]) ;
          end
          % If not explicitly set and the BEAMLINE element has a Geometry
          % flag then set the GeometryType as this
          if ~any(strcmp('GeometryType',varargin))
            BEAMLINE{elemno}.ExtProcess(procno).SetGeometryType();
          end
        otherwise
          error('Unsupported EXT Process Type')
      end
      % Set beamline location based on BEAMLINE Floor Coords
      if isfield(BEAMLINE{elemno},'Coordi') && length(BEAMLINE{elemno}.Coordi)==3
        BEAMLINE{elemno}.ExtProcess(procno).beamlineCoord=[mean([BEAMLINE{elemno}.Coordi(1) BEAMLINE{elemno}.Coordf(1)]) ...
          atan2d(BEAMLINE{elemno}.Coordf(1)-BEAMLINE{elemno}.Coordi(1), BEAMLINE{elemno}.Coordf(3)-BEAMLINE{elemno}.Coordi(3)) ...
          mean([BEAMLINE{elemno}.Coordi(2) BEAMLINE{elemno}.Coordf(2)]), ...
          atan2d(BEAMLINE{elemno}.Coordf(2)-BEAMLINE{elemno}.Coordi(2), BEAMLINE{elemno}.Coordf(3)-BEAMLINE{elemno}.Coordi(3))];
      end
      BEAMLINE{elemno}.ExtProcess(procno).processID=procno;
    end
  end
  
end

