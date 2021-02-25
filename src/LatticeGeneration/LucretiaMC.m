classdef LucretiaMC
  %LucretiaMC Lucretia Monte Carlo Class
  %   Functionality for using Lucretia in a (possibly heterogeneous)
  %   parallel envirnoment, geared towards multi-seed Monte Carlo style
  %   studies
  
  properties (Dependent, SetAccess = private)
    maxHosts % Number of matlab sessions available
    activeHosts % Number of currently running hosts
  end
  properties (Access = private)
    isdct=license('test','distrib_computing_toolbox');
  end
  properties
    verbosity = 1; % Verbosity level (0= silent; 1=minimum status messages)
    numHosts % Max number of Matlab hosts (labs) to use
    seedId % random seed states for each host
  end
  properties % Composite objects (these initialised with startHosts method)
    Initial
    Twiss
    rStream
  end
  
  methods
    function obj = LucretiaMC(cmd)
      if nargin==0
        cmd='none';
      end
      sched=findResource;
      mh = sched.ClusterSize;
      obj.numHosts=mh;
      obj.seedId=1:mh;
      switch cmd
        case 'startHosts'
          obj.startHosts;
      end
    end
    function obj = set.seedId(obj,value)
      if length(value) < obj.numHosts || ~isnumeric(value)
        error('Need to give vector of numeric seeds > MC.numHosts (%d)',obj.numHosts)
      end
      if length(value)<length(obj.seedId)
        obj.seedId(1:length(value))=value;
      else
        obj.seedId=value;
      end
    end
    function value = get.activeHosts(obj)
      if obj.isdct
        value=matlabpool('size');
      else
        value=0;
      end
    end
    function value = get.maxHosts(obj)
      if obj.isdct
        sched=findResource;
        value = sched.ClusterSize;
      else
        value = 1;
      end
    end
  end
  methods
    % Initialisation routine, start worker sessions on required number of
    % hosts and initialise composite Lucretia objects on all hosts hosts
    % start on hardware according to default scheduler configuration
    % Also initialise random number seed in each host
    function obj=startHosts(obj,forcepool)
      global BEAMLINE PS GIRDER WF KLYSTRON INSTR FL %#ok<NUSED>
      glNames={'BEAMLINE' 'PS' 'GIRDER' 'WF' 'KLYSTRON' 'INSTR' 'FL'};
      if obj.isdct
        % Start hosts
        while ~matlabpool('size')
          try
            if obj.verbosity
              matlabpool(obj.numHosts)
            else
              evalc(sprintf('matlabpool(%d)',obj.numHosts));
            end
          catch ME
            if ~exist('forcepool','var') || ~forcepool
              error('Error starting pool of Matlab hosts: %s',ME.message)
            end
          end
        end
        % Initialise main Lucretia data structures across hosts
        for iname=1:length(glNames)
          if eval(['isempty(' glNames{iname} ')']) || eval(['strcmp(class(' glNames{iname} '),''Composite'')'])
            eval([glNames{iname} '=Composite();']);
          else
            error('%s defined as non-composite object, LucretiaMC operation not compatible with standard Lucretia operation, first delete Lucretia globals',glNames{iname})
          end
        end
        % Initialise random number streams
        seedId=obj.seedId;
        spmd
          rStream=RandStream.create('mt19937ar','seed',seedId(labindex));
          RandStream.setDefaultStream(rStream);
        end
        obj.rStream=rStream;
        % Initialise other model composites
        obj.Initial=Composite();
        obj.Twiss=Composite();
      end
    end
    function obj=stopHosts(obj)
      if matlabpool('size')
        if obj.verbosity
          matlabpool('close')
        else
          evalc('matlabpool(''close'')');
        end
      end
    end
    function obj=resetRandomSeeds(obj)
      rStream=obj.rStream;
      spmd
        reset(rStream);
      end
      obj.rStream=rStream;
    end
    function obj=loadModel(obj,method,modelFilename,varargin)
      % LucretiaMC.loadModel Load Lucretia model data structures on all active
      % hosts
      global BEAMLINE PS GIRDER WF KLYSTRON FL INSTR
      switch lower(method)
        case 'xsif'
          spmd
            [stat,Initial] = XSIFToLucretia( modelFilename, varargin );
          end
          obj.Initial=Initial;
        case 'mat'
          spmd
            fileVars=load(modelFilename);
            if isfield(fileVars,'BEAMLINE')
              BEAMLINE=fileVars.BEAMLINE;
            end
            if isfield(fileVars,'PS')
              PS=fileVars.PS;
            end
            if isfield(fileVars,'GIRDER')
              GIRDER=fileVars.GIRDER;
            end
            if isfield(fileVars,'WF')
              WF=fileVars.WF;
            end
            if isfield(fileVars,'KLYSTRON')
              KLYSTRON=fileVars.KLYSTRON;
            end
            if isfield(fileVars,'FL')
              FL=fileVars.FL;
            end
            if isfield(fileVars,'INSTR')
              INSTR=fileVars.INSTR;
            end
          end
          clear fileVars
        case 'aml'
          error('This method not yet implemented')
        otherwise
          error('Unknown model load method')
      end
    end
  end
end
