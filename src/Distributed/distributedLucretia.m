classdef distributedLucretia < handle
  %DISTRIBUTEDLUCRETIA Distributed Lucretia operations class
  %   REQUIRES MATLAB PARALLEL COMPUTING TOOLBOX
  %   Base class for operating Lucretia functions on distributed hardware
  %   or threads. This class supports either synchronous parallel tasks or
  %   asynchronous ones through file transfer using whatever the default
  %   distributed toolbox scheduler configuration is that you have setup
  %   (this can be changed by using the setConfig method)
  %
  % Constructor:
  %  DL=distributedLucretia(issync,nworkers)
  %    Synchronous/Asynchronous behavious is tied to the object and
  %    unchangeable once created, decide with issync=true|false
  %    Supply nworkers to restrict max number of distributed worker nodes
  %    available to functions which use this object (optional, default is
  %    the max defined by your scheduler preferences)
  %
  %    ----------------------
  %    Synchronous behaviour:
  %    ----------------------
  %    The requested number of parallel worker nodes are initialised
  %    (in a 'matlabpool') and stay running as long as this object exists
  %    or the pool is shut down for external reasons. Trim methods act
  %    instantly on the worker lattice configurations and functions which
  %    use this object are instantly set running.
  %    Functions which use this object in this mode are BLOCKING- that is,
  %    they block any further serial execution on the host Matlab session
  %    until the tasks are complete.
  %    Functions utilise this mode by use of the spmd code block.
  %    Use synchronous behaviour when you want the fastest possible
  %    execution time from functions using this object and don't mind the
  %    blocking behaviour.
  %
  %    ----------------------
  %    Asynchronous behaviour:
  %    ----------------------
  %    Parallel worker nodes are not immediately launched, all lattice
  %    configurations for the different workers are kept in local memory
  %    only until a function which uses this object is executed whereupon
  %    the worker nodes are started and load their lattices through file
  %    exchange.
  %    Functions which use this object in this mode are NON-BLOCKING- that
  %    is, job execution returns immediately and the user must check
  %    manually for job completion at a later time before collecting the data.
  %    Functions using this mode do so using the methodsL createAsynTask,
  %    launchAsynJob, asynWait, asynGetData
  %    Use Asynchronous behaviour when you want to quickly generate
  %    background jobs and can perform other tasks while you wait for the
  %    results. Asynchronous mode returns faster from job submission but
  %    takes longer to complete due to setup overhead.
  %
  %  ------- Main Public methods
  %  setConfig - change distributed matlab scheduler
  %  setError - set BEAMLINE error terms (dB, Offset etc...)
  %  PSTrim - trim power supply(ies) (PS)
  %  MoverTrim - trim movers (GIRDERS)
  %  KlystronTrim - trim Klystron Ampls and Phases
  %  latticeCopy - copy in-memory lattice (PS/GIRDER/KLYSTRON) too all
  %                workers
  %  delete - delete object when done with it to free up resources
  %  ------- Asynchrous job related methods
  %  createAsynTask - create a new asyn task (with function handle)
  %  launchAsynJob - launch a job of one or more tasks
  %  clearAsynJob - clear any existing asyn jobs and associated data
  %  asynWait - block execution of any interactive tasks until job finished
  %  asynGetData - get data from finished asyn jobs
  %
  % See also:
  %  Track spmd matlabpool createJob
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc distributedLucretia">doc distributedLucretia</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  
  properties(SetAccess=private)
    schedConfigs % list of available scheduler configurations
    thisConfig % The currently inuse configuration
    maxworkers % max number of worker nodes available
    sched % scheduler resource object
    synchronous=true; % wait for parallel tracking results or track asynchronously
    asynJob % distributed job handle
  end
  properties(Dependent)
    workers % array of worker ID's to act upon (min=1, max=obj.maxworkers)
  end
  properties(Access=private)
    workersToUse
    asynJobOutputArgs % Output arguments from asynchronous jobs. Cell array {iworker,iarg}
  end
  properties
    latticeSyncVals % data values used to specify differences between lattices in different workers (PS/GIRDER/KLYSTRON_V/KLYSTRON_PHA values)
    syncMethod='Ampl'; % Method to synchronise lattice on workers, ('SetPt' or 'Ampl')
    asynDataFile % Location of data file for parameter exchange when running in asynchronous mode, default=distributedLucretiaStartupData.mat in scheduler data area. Also used for initial setup in synchronous mode.
  end
  properties(Dependent,SetAccess=private)
    asynJobStatus % Job status for asyn jobs= 'pending','runnning','queued','finished','failed','destroyed','unavailable'
  end
  
  % Get/Set Methods
  methods
    function status=get.asynJobStatus(obj)
      try
        status=obj.asynJob.State;
      catch
        status='unknown';
      end
    end
    function set.workers(obj,val)
      if any(val<1) || any(val>obj.maxworkers) || length(unique(val))~=length(val)
        error('Must pass vector of unique integers, >1 and <obj.maxworkers')
      end
      obj.workersToUse=val;
    end
    function val=get.workers(obj)
      val=obj.workersToUse;
    end
    function set.syncMethod(obj,val)
      if ~ischar(val) || (~strcmp(val,'SetPt') && ~strcmp(val,'Ampl'))
        error('syncMethod must be ''SetPt'' or ''Ampl''')
      end
      obj.syncMethod=val;
    end
  end
  
  % Main public methods
  methods
    function obj=distributedLucretia(synch,nworkers,sched)
      % distributedLucretia(synchronous,initFileName,[nworkers,sched])
      %   Create Distributed object for use with othe distributed Lucretia
      %   tools. Initially setup with default scheduler configuration.
      % 'synch' = true | false
      %   Optionally supply 'nworkers' to set maxworkers, if omitted then use
      %   the default number in the scheduler configuration
      %   argument checks
      %   Optionally supply scheduler name to use in 'sched' else default
      %   scheduler used (setup in Matlab parallel options)
      global BEAMLINE PS GIRDER KLYSTRON WF %#ok<NUSED>
      % Arg check
      if ~exist('synch','var') || ~islogical(synch)
        error('Must specify synchronous or not (synch=true|false)')
      end
      
      % check for toolbox
      verInfo=ver;
      if ~ismember('Parallel Computing Toolbox',{verInfo.Name})
        error('Parallel Computing Toolbox required to use this object')
      end
      
      % Get scheduler information and setup with currently set default
      % sceduler configuration
      [config, allconfigs] = defaultParallelConfig;
      obj.schedConfigs=allconfigs;
      if exist('sched','var') && ~isempty(sched)
        useSched=sched;
      else
        useSched=config;
      end
      if exist('nworkers','var') && ~isempty(nworkers)
        setConfig(obj,useSched,nworkers);
      else
        setConfig(obj,useSched);
      end
      
      % Initialise to use all workers
      obj.workers=1:obj.maxworkers;
      
      if synch
        % --- preload lucretia globals in workers workspaces
        % startup job pool and distribute lucretia globals 
        if matlabpool('size')
          error('There is already parallel resources allocated, maybe another distributedLucretia object is open, destroy it using the destroy method, or try closing resources manually with ''matlabpool close''')
        end
        matlabpool(obj.sched,obj.maxworkers);
        spmd
          warning off all
        end
        obj.synchronous=true;
      else
        obj.synchronous=false;
      end

      % Initialise all workers with Lucretia global data
      if isprop(obj.sched,'DataLocation')
        obj.asynDataFile=fullfile(obj.sched.DataLocation,'distributedLucretiaStartupData.mat');
      else
        obj.asynDataFile='distributedLucretiaStartupData.mat';
      end
      save(obj.asynDataFile,'BEAMLINE','PS','KLYSTRON','WF','GIRDER')
      if synch
        pctRunOnAll(sprintf('load %s',obj.asynDataFile));
        pctRunOnAll(sprintf('rng(labindex)'));
      end
      
      % Sync lattice with all workers
      latticeCopy(obj,~synch);
      
    end
    function setRandSeed(obj,randVals)
      if length(randVals)~=obj.maxworkers
        error('Must pass a random seed for each worker')
      end
      if obj.synchronous
        spmd
          rng(randVals(labindex));
        end
      end
    end
    function setConfig(obj,configName,nworkers)
      % setConfig(obj,configName,nworkers)
      %   Choose a new parallel configuration from obj.schedConfigs list
      %   Optionally set max number of workers to use
      if ~ismember(configName,obj.schedConfigs)
        error('Not an existing parallel configuration, choose from obj.schedConfigs list')
      end
      obj.sched=findResource('scheduler','configuration',configName);
      if isprop(obj.sched,'HasSharedFilesystem') && ~obj.sched.HasSharedFilesystem
        warning('This scheduler is not setup for a shared filesystem, this may cause problems and is not recommended!') %#ok<WNTAG>
      end
      obj.thisConfig=configName;
      % get max number of useable workers
      try
        if isprop(obj.sched,'IdleWorkers')
          iw=obj.sched.IdleWorkers;
          if isempty(iw)
            error('No parallel workers available');
          end
          if isprop(iw(1),'Computer') % max is max number of same kind of architctures available
            comps=get(iw,'Computer');
            obj.maxworkers=max(cellfun(@(x) sum(ismember(comps,x)),unique(comps)));
          else
            obj.maxworkers=obj.sched.ClusterSize;
          end
        else
          obj.maxworkers=obj.sched.ClusterSize;
        end
        if exist('nworkers','var') && nworkers<obj.maxworkers
          obj.maxworkers=nworkers;
        end
      catch
        obj.maxworkers=obj.sched.ClusterSize;
      end
    end
    function setError(obj,errTerm,errEleList)
      % setError(obj,errTerm,errEleList)
      %   Set an error term on all workers for selected elements
      %   errTerm (char): can be 'dB', 'dV', 'dPhase', 'dAmpl',
      %           'dScale', 'Offset', 'ElecOffset', 'BPMOffset'
      %   errEleList: list of BEAMLINE element indices
      %   valList: value to set desired error term to
      if ~ismember(errTerm,{'dB', 'dV', 'dPhase', 'dAmpl','dScale', 'Offset', 'ElecOffset', 'BPMOffset', 'B'})
        error('No support for this errTerm, see documentation for help')
      end
      sz=size(obj.latticeSyncVals(1).(errTerm));
      if sz(1)==1; for iw=obj.workers; obj.latticeSyncVals(1).(errTerm)=obj.latticeSyncVals(1).(errTerm)'; end; end;
      eUpload=NaN(length(obj.workers),length(errEleList),length(obj.latticeSyncVals(1).(errTerm)(1,:)));
      ind=0;
      for iw=obj.workers
        ind=ind+1;
        eind=0;
        for iele=errEleList
          eind=eind+1;
          eUpload(ind,eind,:)=reshape(obj.latticeSyncVals(iw).(errTerm)(iele,:),1,1,length(obj.latticeSyncVals(iw).(errTerm)(iele,:)));
        end
      end
      if obj.synchronous
        pctRunOnAll(['workers=[',num2str(obj.workers,6),'];']);
        pctRunOnAll(['eInd=[',num2str(errEleList,6),'];']);
        if length(size(eUpload))<3
          eus=[num2str(size(eUpload)) ' 1;'];
        else
          eus=num2str(size(eUpload));
        end
        pctRunOnAll(['edim=[',eus,'];']);
        pctRunOnAll(['eVals=[',num2str(eUpload(:)',6),'];eVals=reshape(eVals,length(workers),length(eInd),edim(3));']);
        pctRunOnAll(sprintf('if ismember(labindex,workers);for iele=1:length(eInd);BEAMLINE{eInd(iele)}.%s=squeeze(eVals(ismember(workers,labindex),iele,:))'';end;end;',errTerm));
      else
        dlStatus.workers=obj.workers; dlStatus.syncMethod=obj.syncMethod; %#ok<STRNU>
        save(obj.asynDataFile,'eUpload','errEleList','errTerm','dlStatus','-append')
      end
    end
    function PSTrim(obj,psList)
      % PSTrim(obj,psList)
      %  Perform PSTrim operation on all workers to trim their PS values to
      %  those held in local obj.latticeSyncVals.PS property array
      %
      % psList=list of PS indices to trim (on all workers selected in
      %        obj.workers)
      % [if obj.syncMethod=='Ampl', just copy
      %  obj.latticeSyncVals.PS values to Ampl fields of PS's and
      %  don't perform PSTrim operation on workers]
      psUpload=NaN(length(obj.workers),length(psList));
      ind=0;
      for iw=obj.workers
        ind=ind+1;
        pind=0;
        for ips=psList
          pind=pind+1;
          psUpload(ind,pind)=obj.latticeSyncVals(iw).PS(ips);
        end
      end
      if obj.synchronous
        pctRunOnAll(['workers=[',num2str(obj.workers,6),'];']);
        pctRunOnAll(['psInd=[',num2str(psList,6),'];']);
        pctRunOnAll(['psVals=[',num2str(psUpload(:)',6),'];psVals=reshape(psVals,length(workers),length(psInd));']);
        if strcmp(obj.syncMethod,'Ampl')
          pctRunOnAll('if ismember(labindex,workers);for ips=1:length(psInd);PS(psInd(ips)).Ampl=psVals(ismember(workers,labindex),ips);end;end;');
        else
          pctRunOnAll('if ismember(labindex,workers);for ips=1:length(psInd);PS(psInd(ips)).SetPt=psVals(ismember(workers,labindex),ips);end;end;');
          pctRunOnAll('if ismember(labindex,workers);PSTrim(psInd);end;');
        end
      else
        dlStatus.workers=obj.workers; dlStatus.syncMethod=obj.syncMethod; %#ok<STRNU>
        save(obj.asynDataFile,'psUpload','psList','dlStatus','-append')
      end
    end
    function MoverTrim(obj,mList)
      % MoverTrim(obj,mList)
      %  Perform MoverTrim operation on all workers to trim their mover
      %  positions those held in local obj.latticeSyncVals.GIRDER property array
      %
      % mList=list of GIRDER indices to trim (on all workers selected in
      %        obj.workers)
      % [if obj.syncMethod=='Ampl', just copy
      %  obj.latticeSyncVals.GIRDER values to MoverPos fields of GIRDER's and
      %  don't perform MoverTrim operation on workers]
      mUpload=NaN(length(obj.workers),length(mList));
      ind=0;
      for iw=obj.workers
        ind=ind+1;
        mind=0;
        for im=mList
          if ~isempty(obj.latticeSyncVals(iw).GIRDER{im})
            mind=mind+1;
            for ig=1:length(obj.latticeSyncVals(iw).GIRDER{im})
              mUpload(ind,mind,ig)=obj.latticeSyncVals(iw).GIRDER{im}(ig);
            end
          else
            mList(mList==im)=[];
          end
        end
      end
      if obj.synchronous
        pctRunOnAll(['workers=[',num2str(obj.workers,6),'];']);
        pctRunOnAll(['mInd=[',num2str(mList,6),'];']);
        pctRunOnAll(['mVals=[',num2str(mUpload(:)',6),'];mVals=reshape(mVals,length(workers),length(mInd),numel(mVals)/(length(workers)*length(mInd)));']);
        if strcmp(obj.syncMethod,'Ampl')
          pctRunOnAll('if ismember(labindex,workers);for im=1:length(mInd);for ig=1:length(GIRDER{mInd(im)}.MoverPos);GIRDER{mInd(im)}.MoverPos(ig)=mVals(ismember(workers,labindex),im,ig);end;end;end;');
        else
          pctRunOnAll('if ismember(labindex,workers);for im=1:length(mInd);for ig=1:length(GIRDER{mInd(im)}.MoverPos);GIRDER{mInd(im)}.MoverSetPt(ig)=mVals(ismember(workers,labindex),im,ig);end;end;end;');
          pctRunOnAll('if ismember(labindex,workers);MoverTrim(mInd);end;');
        end
      else
        dlStatus.workers=obj.workers; dlStatus.syncMethod=obj.syncMethod; %#ok<STRNU>
        save(obj.asynDataFile,'mUpload','mList','dlStatus','-append')
      end
    end
    function KlystronTrim(obj,kList)
      % KlystronTrim(obj,kList)
      %  Perform KlystronTrim operation on all workers to trim their KLYSTRON
      %  values to those held in local obj.latticeSyncVals.KLYSTRON_V / KLYSTRON_PHA property array
      %
      % kList=list of KLYSTRON indices to trim (on all workers selected in
      %        obj.workers)
      % [if obj.syncMethod=='Ampl', just copy
      %  obj.latticeSyncVals.KLYSTRON values to Ampl fields of KLYSTRON's and
      %  don't perform KlystronTrim operation on workers]
      kUpload_V=NaN(length(obj.workers),length(kList)); kUpload_PHA=kUpload_V;
      ind=0;
      for iw=obj.workers
        ind=ind+1;
        kind=0;
        for ik=kList
          kind=kind+1;
          kUpload_V(ind,kind)=obj.latticeSyncVals(iw).KLYSTRON_V(ik);
          kUpload_PHA(ind,kind)=obj.latticeSyncVals(iw).KLYSTRON_PHA(ik);
        end
      end
      if obj.synchronous
        pctRunOnAll(['workers=[',num2str(obj.workers,6),'];']);
        pctRunOnAll(['kInd=[',num2str(kList,6),'];']);
        pctRunOnAll(['kVals_V=[',num2str(kUpload_V(:)',6),'];kVals_V=reshape(kVals_V,length(workers),length(kInd));']);
        pctRunOnAll(['kVals_PHA=[',num2str(kUpload_PHA(:)',6),'];kVals_PHA=reshape(kVals_PHA,length(workers),length(kInd));']);
        if strcmp(obj.syncMethod,'Ampl')
          pctRunOnAll('if ismember(labindex,workers);for ik=1:length(kInd);KLYSTRON(kInd(ik)).Ampl=kVals_V(ismember(workers,labindex),ik);end;end;');
          pctRunOnAll('if ismember(labindex,workers);for ik=1:length(kInd);KLYSTRON(kInd(ik)).Phase=kVals_PHA(ismember(workers,labindex),ik);end;end;');
        else
          pctRunOnAll('if ismember(labindex,workers);for ik=1:length(kInd);KLYSTRON(kInd(ik)).AmplSetPt=kVals_V(ismember(workers,labindex),ik);end;end;');
          pctRunOnAll('if ismember(labindex,workers);for ik=1:length(kInd);KLYSTRON(kInd(ik)).PhaseSetPt=kVals_PHA(ismember(workers,labindex),ik);end;end;');
          pctRunOnAll('if ismember(labindex,workers);KlystronTrim(kInd);end;');
        end
      else
        dlStatus.workers=obj.workers; dlStatus.syncMethod=obj.syncMethod; %#ok<STRNU>
        save(obj.asynDataFile,'kUpload_V','kUpload_PHA','kList','dlStatus','-append')
      end
    end
    function latticeCopy(obj,doSync)
      %latticeCopy(obj,doSync)
      % Copy current in-local-memory Lucretia lattice to all workers
      % (Copy takes effect next sync event)
      % doSync (optional), if true then also sync all workers with Trim
      % methods. Default=true
      global PS GIRDER KLYSTRON
      for iw=obj.workers
        if ~isempty(PS)
          for ips=1:length(PS)
            if strcmp(obj.syncMethod,'SetPt')
              obj.latticeSyncVals(iw).PS(ips)=PS(ips).SetPt;
            else
              obj.latticeSyncVals(iw).PS(ips)=PS(ips).Ampl;
            end
          end
          if (~exist('doSync','var') || doSync) && iw==obj.workers(end)
            obj.PSTrim(1:length(PS));
          end
        else
          obj.latticeSyncVals(iw).PS=[];
        end
        if ~isempty(GIRDER)
          for ig=1:length(GIRDER)
            if strcmp(obj.syncMethod,'SetPt')
              if isfield(GIRDER{ig},'MoverSetPt')
                obj.latticeSyncVals(iw).GIRDER{ig}=GIRDER{ig}.MoverSetPt;
              else
                obj.latticeSyncVals(iw).GIRDER{ig}=[];
              end
            else
              if isfield(GIRDER{ig},'MoverPos')
                obj.latticeSyncVals(iw).GIRDER{ig}=GIRDER{ig}.MoverPos;
              else
                obj.latticeSyncVals(iw).GIRDER{ig}=[];
              end
            end
          end
          if (~exist('doSync','var') || doSync) && iw==obj.workers(end)
            obj.MoverTrim(1:length(GIRDER));
          end
        else
          obj.latticeSyncVals(iw).GIRDER=[];
        end
        if ~isempty(KLYSTRON)
          for ik=1:length(KLYSTRON)
            if strcmp(obj.syncMethod,'SetPt')
              obj.latticeSyncVals(iw).KLYSTRON_V(ik)=KLYSTRON(ik).AmplSetPt;
              obj.latticeSyncVals(iw).KLYSTRON_PHA(ik)=KLYSTRON(ik).PhaseSetPt;
            else
              obj.latticeSyncVals(iw).KLYSTRON_V(ik)=KLYSTRON(ik).Ampl;
              obj.latticeSyncVals(iw).KLYSTRON_PHA(ik)=KLYSTRON(ik).Phase;
            end
          end
          if (~exist('doSync','var') || doSync) && iw==obj.workers(end)
            obj.KlystronTrim(1:length(KLYSTRON));
          end
        else
          obj.latticeSyncVals(iw).KLYSTRON_V=[];
          obj.latticeSyncVals(iw).KLYSTRON_PHA=[];
        end
      end
    end
    function delete(obj)
      % Destructor function
      try
        if obj.synchronous && matlabpool('size')
          matlabpool close force
        elseif obj.synchronous
          obj.clearAsynJob;
        elseif ~obj.synchronous
          delete(obj.asynDataFile);
        end
      catch
      end
    end
    function createAsynTask(obj,fhan,nOutputArgs,inputArgs)
      % createAsynTask(obj,fhan,nOutputArgs,inputArgs)
      %  Create a new asyn job if not already existing
      %  Then add a new task:
      %   fhan: function handle to run
      %   nOutputArgs: number of output arguments from fhan
      %   inputArgs: cell array of input arguments to fhan
      %
      % Output from function is stored in obj.asynJobOutputArgs,
      %  fetch using asynGetData method
      if isempty(obj.asynJob)
        obj.asynJob=createJob(obj.sched);
      end
      obj.asynJob.createTask(fhan,nOutputArgs,inputArgs);
    end
    function launchAsynJob(obj)
      % launchAsynJob(obj)
      %  Set asynchronous job running and return immediately
      obj.asynJobOutputArgs=[];
      obj.asynJob.submit;
    end
    function clearAsynJob(obj)
      % clearAsynJob(obj)
      %  Clears data from any previous job. It is reccommended to call this
      %  before creation of every new job
      try
        if ~isempty(obj.asynJob)
          obj.asynJob.destroy;
          obj.asynJob=[];
          obj.asynJobOutputArgs=[];
        end
      catch
      end
    end
    function asynWait(obj,timeout)
      % asynWait(obj,timeout)
      %  Wait for asynchronous job to complete
      %  Optional timeout in seconds
      if obj.synchronous
        return
      end
      t0=clock;
      while ~strcmp(obj.asynJobStatus,'finished') && ~strcmp(obj.asynJobStatus,'failed') && ~strcmp(obj.asynJobStatus,'destroyed') ...
          && ~strcmp(obj.asynJobStatus,'unavailable')
        pause(1)
        if exist('timeout','var') && etime(clock,t0)>timeout
          break
        else
          t0=clock;
        end
      end
    end
    function val=asynGetData(obj,arg)
      % val=asynGetData(obj,arg)
      %  extract data from asyn job
      %  arg=argument # from asyn function
      %  val=cell output length nworkers
      switch obj.asynJobStatus
        case 'finished'
          if isempty(obj.asynJobOutputArgs)
            obj.asynJobOutputArgs=obj.asynJob.getAllOutputArguments();
          end
          if isempty(obj.asynJobOutputArgs)
            tasks=obj.asynJob.findTask;
            em=get(tasks,'ErrorMessage'); emess=[];
            for itask=1:length(em)
              emess=[emess sprintf('Worker %d: %s\n',itask,em{itask})];
            end
            error('Asynchronous Job error:\n%s',emess)
          end
          for iw=obj.workers
            val{iw}=obj.asynJobOutputArgs{iw,arg};
          end
        otherwise
          warning('Lucretia:distributedLucretia:noAsynData','Job not finished running yet or in error state (output data from tracking cannot be provided): asynJobStatus=%s',obj.asynJobStatus)
          val=[];
      end
    end
  end
  
  % Worker callable methods
  methods(Static)
    function [BEAMLINE PS GIRDER KLYSTRON WF]=asynLoadLattice(dataFile,iworker) %#ok<STOUT>
      load(dataFile)
      if exist('eUpload','var')
        for iele=1:length(errEleList)
          BEAMLINE{errEleList(iele)}.(errTerm)=squeeze(eUpload(ismember(dlStatus.workers,iworker),iele,:))'; %#ok<NODEF>
        end
      end
      if exist('psUpload','var')
        for ips=1:length(psList)
          if strcmp(dlStatus.syncMethod,'Ampl')
            PS(psList(ips)).Ampl=psUpload(ismember(dlStatus.workers,iworker),ips);
          else
            PS(psList(ips)).SetPt=psUpload(ismember(dlStatus.workers,iworker),ips);
          end
        end
        if strcmp(dlStatus.syncMethod,'SetPt')
          PSTrim(psList);
        end
      end
      if exist('mUpload','var')
        for im=1:length(mList)
          for ig=1:length(GIRDER{mList(im)}.MoverPos)
            if strcmp(dlStatus.syncMethod,'Ampl')
              GIRDER{mList(im)}.MoverPos(ig)=mUpload(ismember(dlStatus.workers,iworker),im,ig);
            else
              GIRDER{mList(im)}.MoverSetPt(ig)=mUpload(ismember(dlStatus.workers,iworker),im,ig);
            end
          end
        end
        if strcmp(dlStatus.syncMethod,'SetPt')
          MoverTrim(mList);
        end
      end
      if exist('kUpload_V','var')
        for ik=1:length(kList)
          if strcmp(dlStatus.syncMethod,'Ampl')
            KLYSTRON(kList(ik)).Ampl=kUpload_V(ismember(dlStatus.workers,iworker),ik);
            KLYSTRON(kList(ik)).Phase=kUpload_PHA(ismember(dlStatus.workers,iworker),ik);
          else
            KLYSTRON(kList(ik)).AmplSetPt=kUpload_V(ismember(dlStatus.workers,iworker),ik);
            KLYSTRON(kList(ik)).PhaseSetPt=kUpload_PHA(ismember(dlStatus.workers,iworker),ik);
          end
        end
        if strcmp(dlStatus.syncMethod,'SetPt')
          KlystronTrim(kList);
        end
      end
    end
  end
  
end

