classdef Track < handle
  % TRACK Lucretia beam tracking interface
  %   Perform tracking, either singly or on distributed Lucretia interface
  %   defined by optionally passed distributedLucretia object on construction
  %
  % Supports both asynchronous and synchronous parallel tracking (set in
  % distributedLucretia class passed to this object upon creation)
  %   - Asynchronous is slower to run than synchronous due to extra setup
  %   time but the trackThru command returns immediately which is useful if
  %   you want to process other commands serially whilst the parallel
  %   tracking is being computed
  %
  % Contructor:
  %   T=Track(InputBeam,distributedLucretiaObject)
  %     Omit distributedLucretiaObject if using this object in a non-parallel
  %     environment
  %
  % Main public methods (see doc help for details):
  %   trackThru - main tracking method
  %   trackThru('singleRay') - track single particle (mean of particle
  %                            distributions, sum of charge)
  %   getBeamData - get detailed data about output beam (Tracking must have
  %              occured) - e.g. RMS and gaussian fitted beam sizes, aberrations
  %              calculated from particle distributions etc
  %      Fetches beam data from all workers
  %
  % Example:
  %  % Create a distributedLucretia object (choose synchronicity with
  %    isasyn=true|false)
  %  DL=distributedLucretia(isasyn)
  %  % Create Track object with a Lucretia InputBeam
  %  T=Track(InputBeam,DL) % Now set track indices, T.startInd,T.finishInd etc as desired
  %  % Make any lattice changes
  %  DL.latticeSyncVals(1).PS(53)=0.85;
  %  DL.latticeSyncVals(2).PS(53)=0.85;
  %  DL.PSTrim(53);
  %  % Issue track command (sends tracking job to parallel worker nodes)
  %  T.trackThru;
  %  % Wait for results (if isasyn=true this is instantaneous and command is
  %                      not necessary)
  %  DL.asynWait
  %  % Get results (the main output arguments from Lucretia's TrackThru
  %  %  function), if there were any tracking errors trying to access these
  %  %  parameters results in an error with the error output messages from the
  %  %  workers shown.
  %  for iw=DL.workers
  %    beamOut(iw)=T.beamOut{iw};
  %    trackStatus{iw}=T.trackStatus{iw};
  %    instrumentData(iw)=T.instrData{iw};
  %  end
  %
  % See also:
  %  TrackThru distributedLucretia
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc Track">doc Track</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  properties
    startInd % Finish tracking index
    finishInd % Start tracking index
    firstBunch=1; % first bunch to track (if multibunch)
    lastBunch=1; % last bunch to track (if multibunch)
    loopFlag=0; % loop over elements (0) or over bunches (1)
    beamType=0; % 0=all input beams the same, 1=possibly different beams for each worker
    verbose=0; % verbosity level (0= don't print anything, 1=print at each CSR integration step)
    doPlasmaTrack=0; % Include tracking through plasma region (0=no)
    beamStoreInd=[]; % Store the full beam at additional points along the lattice
    centerZInd=[]; % Indices to re-center longitudinal distribution
    centerTInd=[]; % Indices to re-center transverse distribution
    zOffset=0; % Phase offset in bunch (m), used if centerZInd present
    isgpu=false; % Wishing to track with GPU-optimized mex function?
  end
  properties(SetAccess=protected)
    isDistrib=false; % Is this Track object opererating in distributed mode?
    isPTrack=false; % Track object setup to parallel track through common lattice?
    DL % distributedLucretia object reference
    plasmaData % Tracked data through plasma channel if requested
    beamStore % Beam saved at intermediate tracking locations if requested in 'beamStoreInd'
  end
  properties(Access=protected)
    dBeamIn % distributed input beam
    lBeamIn % local input beam
    instr
    beamout
    stat
    nray
    beamInSingle
    dBeamData % distributed beam data
    beamDividers_r0
    beamDividers_r1
    pTrackStopElements
    checkExtProcess=[]; % flag to see if ExtProcess envirnoment checked to be OK
  end
  properties(Dependent)
    beamIn % Lucretia beam structure to track
  end
  properties(Dependent,SetAccess=protected)
    instrData % (distributed) instrument data from tracking
    beamOut % (distributed) beam object post tracking
    trackStatus % (distributed) Lucretia status from tracking
    beamData % Processed beam data
  end
  properties(Constant)
    minPTrackDist=3; % Min number of elements to bother tracking in parallel
  end
  
  %% Get/Set methods
  methods
    function set.trackStatus(obj,val)
      obj.stat=val;
    end
    function val=get.trackStatus(obj)
      if obj.isDistrib && ~obj.DL.synchronous
        val=asynGetData(obj.DL,1);
      else
        val=obj.stat;
      end
    end
    function set.instrData(obj,val)
      obj.instr=val;
    end
    function val=get.instrData(obj)
      if obj.isDistrib && ~obj.DL.synchronous
        val=asynGetData(obj.DL,3);
      else
        val=obj.instr;
      end
    end
    function set.beamOut(obj,val)
      obj.beamout=val;
    end
    function val=get.beamOut(obj)
      if obj.isPTrack
        r0=obj.beamDividers_r0;
        r1=obj.beamDividers_r1;
        B=obj.beamout;
        val=B{obj.DL.workers(1)};
        for iB=obj.DL.workers(2:end)
          beam=B{iB};
          val.Bunch.Q(r0(iB):r1(iB))=beam.Bunch.Q;
          val.Bunch.stop(r0(iB):r1(iB))=beam.Bunch.stop;
          val.Bunch.x(:,r0(iB):r1(iB))=beam.Bunch.x;
        end
      elseif obj.isDistrib && ~obj.DL.synchronous
        val=asynGetData(obj.DL,2);
      else
        val=obj.beamout;
      end
    end
    function set.beamIn(obj,beamStruc)
      % Update single ray structure
      if length(beamStruc.Bunch.Q)>1
        obj.beamInSingle=beamStruc;
        obj.beamInSingle.Bunch.Q=sum(beamStruc.Bunch.Q);
        obj.beamInSingle.Bunch.stop=0;
        obj.beamInSingle.Bunch.x=mean(beamStruc.Bunch.x,2);
      end
      % Store given beam as local beam
      obj.lBeamIn=beamStruc;
      % Deal with new Beam
      if obj.isPTrack
        % Divide up beam particles amongst worker nodes
        np=length(obj.DL.workers);
        nmp_1=length(beamStruc.Bunch.Q);
        nmp=floor(nmp_1/np);
        n=0;
        for iw=obj.DL.workers
          n=n+1;
          if n==np
            r1(iw)=nmp_1;
          else
            r1(iw)=n*nmp;
          end
          r0(iw)=(n-1)*nmp+1;
        end
        useWorkers=obj.DL.workers;
        nbunch=length(beamStruc.Bunch);
        spmd
          if ismember(labindex,useWorkers)
            BeamIn.BunchInterval=beamStruc.BunchInterval;
            for bn=1:nbunch
              BeamIn.Bunch(bn).Q=beamStruc.Bunch(bn).Q(r0(labindex):r1(labindex));
              BeamIn.Bunch(bn).stop=beamStruc.Bunch(bn).stop(r0(labindex):r1(labindex));
              BeamIn.Bunch(bn).x=beamStruc.Bunch(bn).x(:,r0(labindex):r1(labindex));
            end
          end
        end
        obj.dBeamIn=BeamIn;
        obj.beamDividers_r0=r0;
        obj.beamDividers_r1=r1;
      elseif obj.isDistrib && obj.DL.synchronous
        useWorkers=obj.DL.workers;
        spmd
          if ismember(labindex,useWorkers)
            BeamIn=beamStruc;
          end
        end
        obj.dBeamIn=BeamIn;
      elseif obj.isDistrib % asyn
        vars=whos('-file',obj.DL.asynDataFile);
        if any(ismember({vars.name},'trackBeamIn')) && obj.beamType
          ld=load(obj.DL.asynDataFile,'trackBeamIn');
          trackBeamIn=ld.trackBeamIn;
        end
        if obj.beamType
          for iw=obj.DL.workers
            trackBeamIn(iw)=beamStruc;
          end
        else
          trackBeamIn=beamStruc; %#ok<NASGU>
        end
        save(obj.DL.asynDataFile,'trackBeamIn','-append')
      else
        if obj.beamType
          obj.lBeamIn=[];
          obj.lBeamIn(obj.DL.workers)=beamStruc;
        else
          obj.lBeamIn=beamStruc;
        end
      end
    end
    function beam=get.beamIn(obj)
      if (obj.isDistrib && obj.DL.synchronous) || obj.isPTrack
        beam=obj.dBeamIn;
      elseif obj.isDistrib
        dataFile=fullfile(obj.DL.sched.DataLocation,'distributedLucretiaStartupData.mat');
        vars=whos('-file',dataFile);
        if any(ismember({vars.name},'trackBeamIn'))
          ld=load(dataFile,'trackBeamIn');
          beam=ld.trackBeamIn;
        else
          beam=[];
        end
      else
        beam=obj.lBeamIn;
      end
    end
    function data=get.beamData(obj)
      data=obj.dBeamData;
    end
  end
  
  %% Main public methods
  methods
    function obj=Track(beamIn,distribObj,setPT)
      global BEAMLINE
      if exist('distribObj','var') && ~isempty(distribObj)
        if ~strcmp(class(distribObj),'distributedLucretia') %#ok<STISA>
          error('Can only pass a distributedLucretia object to Track')
        end
        obj.isDistrib=true;
        obj.isPTrack=false;
        obj.DL=distribObj;
      else
        obj.isDistrib=false;
        obj.isPTrack=false;
      end
      if ~exist('beamIn','var')
        error('Must supply input beam structure')
      end
      obj.startInd=1;
      obj.finishInd=length(BEAMLINE);
      obj.beamIn=beamIn;
      obj.nray=numel(beamIn.Bunch.Q);
      if exist('setPT','var') && setPT
        try
          obj.setupPTrackThru(beamIn);
        catch ME
          obj.isDistrib=true;
          obj.isPTrack=false;
          obj.beamIn=beamIn;
          error('Setup of parallel tracking failed, reverting to distributed setup:\n%s',ME.message)
        end
      end
    end
    function [stat, beamout, instdata] = dotrack(obj,startInd,finishInd,BeamIn,b1,b2,lf)
      if obj.isgpu
        [stat, beamout, instdata] = TrackThru_gpu(startInd,finishInd,BeamIn,b1,b2,lf) ;
      else
        [stat, beamout, instdata] = TrackThru(startInd,finishInd,BeamIn,b1,b2,lf) ;
      end
    end
    function trackThru(obj,cmd)
      global BEAMLINE
      % Any Ext Processes to pre-condition?
      ext=findcells(BEAMLINE,'ExtProcess',[],obj.startInd,obj.finishInd);
      pord=[];
      if ~isempty(ext)
        for iele=ext
          for iproc=1:length(BEAMLINE{iele}.ExtProcess)
            % Check validity of any EM field
            if any(isnan(BEAMLINE{iele}.ExtProcess(iproc).Bx(:))) ||...
               any(isnan(BEAMLINE{iele}.ExtProcess(iproc).By(:))) ||...
               any(isnan(BEAMLINE{iele}.ExtProcess(iproc).Bz(:))) ||...
               any(isnan(BEAMLINE{iele}.ExtProcess(iproc).Ex(:))) ||...
               any(isnan(BEAMLINE{iele}.ExtProcess(iproc).Ey(:))) ||...
               any(isnan(BEAMLINE{iele}.ExtProcess(iproc).Ez(:)))
               error('Badly formatted EM field components in element %d (NaN''s present)')
            end
            % If wanting to use an external process, check that the environment
            % has been set up correctly one time per process type
            if isempty(obj.checkExtProcess) || ~isfield(obj.checkExtProcess,class(BEAMLINE{iele}.ExtProcess(iproc)))
              [resp,message]=BEAMLINE{iele}.ExtProcess(iproc).checkEnv;
              if ~resp
                error('Error checking ExtProcess envirnment for element # %d: %s',iele,message)
              end
              obj.checkExtProcess.(class(BEAMLINE{iele}.ExtProcess(iproc)))=true;
            end
            BEAMLINE{iele}.ExtProcess(iproc).InitializeTrackingData(obj.beamIn,obj.firstBunch,obj.lastBunch);
            if ~isempty(pord)
              BEAMLINE{iele}.ExtProcess(iproc).PrimarySampleOrder=pord;
            else
              pord=BEAMLINE{iele}.ExtProcess(iproc).PrimarySampleOrder;
            end
          end
        end
      end
      % Asking for intermediate track locations?
      interele=obj.beamStoreInd;
      bsind=obj.beamStoreInd; czind=obj.centerZInd; ctind=obj.centerTInd;
      if ~isempty(obj.centerZInd) || ~isempty(obj.centerTInd)
        interele=unique([interele obj.centerZInd obj.centerTInd]);
      end
      if ~isempty(interele)
        interele=interele(interele<=obj.finishInd & interele>=obj.startInd);
      end
      % Asking for single-ray tracking?
      if exist('cmd','var') && isequal(cmd,'singleRay')
        doSingleRay=true;
      else
        doSingleRay=false;
      end
      if doSingleRay
        BeamIn=obj.beamInSingle;
      else
        BeamIn=obj.beamIn;
      end
      doplas=obj.doPlasmaTrack;
      % tracking in parallel across possibly multiple workers
      if obj.isDistrib || obj.isPTrack
        % If asynchronous, submit jobs and return
        if ~obj.DL.synchronous
          % Can't store beams or re-center z distributions in this mode
          if ~isempty(interele)
            error('Intermediate element tracking (setting of beamStoreInd or centerZInd properties) not supported in asynchronos parallel tracking mode')
          end
          % Remove last job if there is one
          obj.DL.clearAsynJob;
          % Make new asyn job
          for iw=obj.DL.workers
            obj.DL.createAsynTask(@Track.asynTrack,3,{obj.DL.asynDataFile,iw,obj.startInd,obj.finishInd,...
              obj.firstBunch,obj.lastBunch,obj.loopFlag,doSingleRay});
          end
          obj.DL.launchAsynJob();
        else % if sycnchronous, submit tracking tasks to the pool and wait for them to finish
          startInd=obj.startInd;
          finishInd=obj.finishInd;
          b1=obj.firstBunch;
          b2=obj.lastBunch;
          lf=obj.loopFlag;
          useWorkers=obj.DL.workers;
          spmd
            if ismember(labindex,useWorkers)
              bstore=[];
              if isempty(interele)
                [stat, beamout, instdata]=obj.dotrack(startInd,finishInd,BeamIn,b1,b2,lf);
              else
                if interele(end)~=finishInd; interele=[interele finishInd]; end;
                t1=startInd;
                t2=interele(1);
                B=BeamIn;
                bi=1;
                instdata={};
                for iele=1:length(interele)
                  [stat, beamout, idat]=obj.dotrack(t1,t2,B,b1,b2,lf);
                  for id=1:length(idat)
                    if length(instdata)<id; instdata{id}=[]; end;
                    instdata{id}=[instdata{id} idat{id}];
                  end
                  if iele<length(interele)
                    B=beamout;
                    t1=t2+1;
                    t2=interele(iele+1);
                  end
                  if ismember(interele(iele),bsind)
                    bstore(bi)=beamout;
                  end
                  if ismember(interele(iele),czind)
                    B.Bunch.x(5,:)=B.Bunch.x(5,~B.Bunch.stop)-median(B.Bunch.x(5,~B.Bunch.stop))+obj.zOffset;
                  end
                  if ismember(interele(iele),ctind)
                    for ind=1:4
                      B.Bunch.x(ind,:)=B.Bunch.x(ind,~B.Bunch.stop)-median(B.Bunch.x(ind,~B.Bunch.stop));
                    end
                  end
                end
              end
            end
            if doplas
              try
                [bunchProfile, plas_sx, plas_sy]=Track.plasmaTrack(beamout,4);
              catch
                bunchProfile=[]; plas_sx=1e10; plas_sy=1e10;
              end
            end
          end
          % Store results
          obj.trackStatus=stat;
          obj.beamOut=beamout;
          obj.instrData=instdata;
          if doplas
            obj.plasmaData.bunchProfile=bunchProfile;
            obj.plasmaData.sx=plas_sx;
            obj.plasmaData.sy=plas_sy;
          end
          obj.beamStore=bstore;
        end
      else % local tracking
        if isempty(interele)
          [stat, beamout, instdata]=obj.dotrack(obj.startInd,obj.finishInd,BeamIn,obj.firstBunch,obj.lastBunch,obj.loopFlag);
        else
          if interele(end)~=obj.finishInd; interele=[interele obj.finishInd]; end;
          t1=obj.startInd;
          t2=interele(1);
          instdata={};
          B=BeamIn;
          bi=1;
          for iele=1:length(interele)
            [stat, beamout, idat]=obj.dotrack(t1,t2,B,obj.firstBunch,obj.lastBunch,obj.loopFlag);
            for id=1:length(idat)
              if length(instdata)<id; instdata{id}=[]; end;
              instdata{id}=[instdata{id} idat{id}];
            end
            if iele<length(interele)
              B=beamout;
              t1=t2+1;
              t2=interele(iele+1);
            end
            if ismember(interele(iele),bsind)
              obj.beamStore(bi)=beamout;
            end
            if ismember(interele(iele),czind)
              B.Bunch.x(5,~B.Bunch.stop)=B.Bunch.x(5,~B.Bunch.stop)-median(B.Bunch.x(5,~B.Bunch.stop))+obj.zOffset;
            end
            if ismember(interele(iele),ctind)
              for ind=1:4
                B.Bunch.x(ind,~B.Bunch.stop)=B.Bunch.x(ind,~B.Bunch.stop)-median(B.Bunch.x(ind,~B.Bunch.stop));
              end
            end
          end
        end
        if doplas
          try
            [bunchProfile, plas_sx, plas_sy]=Track.plasmaTrack(beamout,4);
          catch
            bunchProfile=[]; plas_sx=1e10; plas_sy=1e10;
          end
        end
        obj.trackStatus=stat;
        obj.beamOut=beamout;
        obj.instrData=instdata;
        if doplas
          obj.plasmaData.bunchProfile=bunchProfile;
          obj.plasmaData.sx=plas_sx;
          obj.plasmaData.sy=plas_sy;
        end
      end
      % Any Ext Processes to finalize data with
      ext=findcells(BEAMLINE,'ExtProcess',[],obj.startInd,obj.finishInd);
      if ~isempty(ext)
        for iele=ext
          for iproc=1:length(BEAMLINE{iele}.ExtProcess)
            BEAMLINE{iele}.ExtProcess(iproc).FinalizeTrackingData() ;
          end
        end
      end
    end
    function pTrackThru(obj)
      % Perform parallel tracking through the BEAMLINE lattice
      global BEAMLINE
      
      % Check for correct setup
      if ~obj.isPTrack
        error('setupPTrackThru not done')
      end
      
      % - If no collective effects requested, can simply proceed with
      % tracking of already distributed particles
      se=obj.pTrackStopElements;
      se(se<obj.startInd)=[];
      se(se>obj.finishInd)=[];
      if isempty(se)
        obj.trackThru;
        return
      end
      
      % Otherwise must pause at stop points then redistribute
      mindist=obj.minPTrackDist;
      istart=obj.startInd;
      ifinish=obj.finishInd;
      ibeam=obj.beamIn;
      if se(1)<=mindist
        obj.isPTrack=false;
        obj.beamIn=obj.lBeamIn;
        obj.finishInd=se(1)-1;
        obj.trackThru;
      else
        obj.finishInd=se(1)-1;
        obj.trackThru;
      end
      beamout=obj.beamOut;
      ise=2;
      while ise<=length(se) && se(ise)<=ifinish && ise<=length(se)
        obj.startInd=obj.finishInd+1;
        obj.finishInd=se(ise)-1;
        if obj.isPTrack
          while se(ise+1)==se(ise)+1 && ise<length(se)
            ise=ise+1;
            obj.finishInd=se(ise)-1;
            continue;
          end
          obj.isPTrack=false;
          ise=ise+1;
        else
          obj.isPTrack=true;
        end
        obj.beamIn=beamout;
        obj.trackThru;
        beamout=obj.beamOut;
      end
      if se(ise)>ifinish || ise<=length(se)
        ise=ise-1;
      end
      if se(ise)==ifinish
        obj.isPTrack=true;
        obj.startInd=istart;
        obj.finishInd=ifinish;
        obj.beamIn=ibeam;
        return
      end
      % --- now track through to the end (in parallel if there is far
      % enough to go)
      obj.startInd=obj.finishInd+1;
      obj.finishInd=ifinish;
      obj.isPTrack=se(end)<(length(BEAMLINE)-mindist);
      obj.beamIn=beamout;
      obj.trackThru;
      % -- return original properties
      obj.isPTrack=true;
      obj.startInd=istart;
      obj.finishInd=ifinish;
      obj.beamIn=ibeam;
    end
    function beamData=getBeamData(obj,dims)
      if isempty(obj.beamOut); error('No tracking has taken place yet!'); end;
      if ~exist('dims','var'); dims='xyz'; end;
      bo=obj.beamOut;
      if obj.isDistrib
        if ~obj.DL.synchronous
          % Remove last job if there is one
          obj.DL.clearAsynJob;
          % Make new asyn job
          for iw=obj.DL.workers
            obj.DL.createAsynTask(@Track.procBeamData,1,{bo{iw},dims});
          end
          obj.DL.launchAsynJob();
          obj.DL.asynWait;
          bd=asynGetData(obj.DL,1);
          for ib=1:length(bd); beamData(ib)=bd{ib}; end;
        else
          spmd
            bd=Track.procBeamData(bo,dims) ;
          end
          for ib=1:length(bd); beamData(ib)=bd{ib}; end;
        end
      else
        beamData=Track.procBeamData(bo,dims);
      end
    end
    function setupPTrackThru(obj,beam)
      % setupPTrackThru - Setup parallel config for parallel tracking
      %  de-selects distributed mode, copies current lattice acorss all
      %  worker nodes and splits up beam definition across nodes for
      %  parallel tracking of beam. Must have instantiated Track object
      %  with distributedLucretia object set in synchronous mode
      %  (DL.synchronous=true)
      %
      % beam: Lucretia Beam structure
      global BEAMLINE
      if isempty(obj.DL)
        error('No distributedLucretia object attached to this Track object')
      end
      if ~obj.DL.synchronous
        error('DL object attached to Track object must be type ''synchronous''')
      end
      if ~exist('beam','var')
        error('Must supply Lucretia beam to distribute')
      end
      obj.isDistrib=false;
      obj.isPTrack=true;
      obj.beamIn=beam;
      obj.DL.latticeCopy;
      % Get locations where parallel tracking must pause and send rays back
      % for central tracking (i.e. when collective effects need to be
      % applied. i.e. CSR or Wakefield calculations)
      tf=findcells(BEAMLINE,'TrackFlag');
      stopele=[];
      stopfields={'SRWF_Z' 'SRWF_T' 'LRWF_T' 'LRWF_ERR'};
      noptfields={'CSR'};
      mindist=obj.minPTrackDist;
      for iele=tf
        fn=fieldnames(BEAMLINE{iele}.TrackFlag);
        for ifn=1:length(fn)
          if ismember(fn{ifn},noptfields) && BEAMLINE{iele}.TrackFlag.(fn{ifn})
            error('BEAMLINE contains element with Tracking Flag ''%s'' set, this is not currently supported by this parallel tracking code',fn{ifn})
          end
          if ismember(fn{ifn},stopfields) && BEAMLINE{iele}.TrackFlag.(fn{ifn})
            if ~isempty(stopele) && (iele-stopele(end))<=mindist
              stopele=[stopele (stopele(end)+1):iele];
            else
              stopele(end+1)=iele;
            end
          end
        end
      end
      obj.pTrackStopElements=stopele;
    end
  end
  
  %% Static methods (includes those needing to be called in worker environment for distributed jobs)
  methods(Static)
    function [stat, beamout, instdata]=asynTrack(dataFile,iworker,i1,i2,b1,b2,lf,doSingleParticle)
      [BEAMLINE, PS, GIRDER, KLYSTRON, WF]=distributedLucretia.asynLoadLattice(dataFile,iworker); %#ok<ASGLU>
      load(dataFile,'trackBeamIn')
      if length(trackBeamIn)>1
        beam=trackBeamIn(iworker);
      else
        beam=trackBeamIn;
      end
      if doSingleParticle
        beamInSingle=beam;
        beamInSingle.Bunch.Q=sum(beam.Bunch.Q);
        beamInSingle.Bunch.stop=0;
        beamInSingle.x=mean(beam.Bunch.x,2);
        beam=beamInSingle;
      end
      [stat, beamout, instdata]=obj.dotrack(i1,i2,beam,b1,b2,lf);
    end
    function data=procBeamData(beam,dims)
      data.qloss=sum(beam.Bunch.Q(beam.Bunch.stop>0));
      beam.Bunch.x=beam.Bunch.x(:,~beam.Bunch.stop);
      beam.Bunch.Q=beam.Bunch.Q(~beam.Bunch.stop);
      beam.Bunch.stop=beam.Bunch.stop(~beam.Bunch.stop);
      if isempty(beam.Bunch.x)
        error('All particles in provided beam stopped!')
      end
      gamma=mean(beam.Bunch.x(6,:))/0.511e-3;
      if any(ismember(dims,'x'))
        [fitTerm,fitCoef,bsize_corrected,bsize] = beamTerms(1,beam);
        [~, I]=sort(bsize,'descend');
        data.fitTerms_x=fitTerm(I,:);
        data.fitBeamSizeContrib_x=bsize(I);
        data.fitCoef_x=fitCoef(I);
        data.fitBeamSizeCorrected_x=bsize_corrected;
      end
      if any(ismember(dims,'y'))
        [fitTerm,fitCoef,bsize_corrected,bsize] = beamTerms(3,beam);
        [~, I]=sort(bsize,'descend');
        data.fitTerms_y=fitTerm(I,:);
        data.fitBeamSizeContrib_y=bsize(I);
        data.fitCoef_y=fitCoef(I);
        data.fitBeamSizeCorrected_y=bsize_corrected;
      end
      if any(ismember(dims,'z'))
        [fitTerm,fitCoef,bsize_corrected,bsize] = beamTerms(5,beam);
        [~, I]=sort(bsize,'descend');
        data.fitTerms_z=fitTerm(I,:);
        data.fitBeamSizeContrib_z=bsize(I);
        data.fitCoef_z=fitCoef(I);
        data.fitBeamSizeCorrected_z=bsize_corrected;
      end
      [nx,ny] = GetNEmitFromBeam( beam, 1 );
      data.emit_x=nx/gamma; data.emit_y=ny/gamma;
      data.xpos=mean(beam.Bunch.x(1,:));
      data.ypos=mean(beam.Bunch.x(3,:));
      data.xrms=std(beam.Bunch.x(1,:)); data.xprms=std(beam.Bunch.x(2,:));
      data.yrms=std(beam.Bunch.x(3,:)); data.yprms=std(beam.Bunch.x(4,:));
      data.zrms=std(beam.Bunch.x(5,:));
      data.erms=std(beam.Bunch.x(6,:))/mean(beam.Bunch.x(6,:));
      data.sigma=cov(beam.Bunch.x');
      R=diag(ones(1,6));L=zeros(6,6);L(1,2)=1;L(3,4)=1;
      data.xwaist=fminsearch(@(x) Track.minWaist(x,R,L,data.sigma,1),0,optimset('Tolx',1e-6,'TolFun',0.1e-6^2));
      data.ywaist=fminsearch(@(x) Track.minWaist(x,R,L,data.sigma,3),0,optimset('Tolx',1e-6,'TolFun',0.1e-6^2));
      [Tx,Ty] = GetUncoupledTwissFromBeamPars(beam,1);
      data.xdisp=Tx.eta;
      data.ydisp=Ty.eta;
      data.xdp=Tx.etap;
      data.ydp=Ty.etap;
      data.betax=Tx.beta;
      data.betay=Ty.beta;
      data.alphax=Tx.alpha;
      data.alphay=Tx.alpha;
      data.sig13=data.sigma(1,3);
      data.sig23=data.sigma(2,3);
      data.sig14=data.sigma(1,4);
      data.sig15=data.sigma(1,5);
      nbin=max([length(beam.Bunch.Q)/100 100]);
      [ fx , bc ] = hist(beam.Bunch.x(3,:),nbin);
      [~, q] = gauss_fit(bc,fx) ;
      data.yfit=abs(q(4));
      [ fx , bc ] = hist(beam.Bunch.x(1,:),nbin);
      [~, q] = gauss_fit(bc,fx) ;
      data.xfit=abs(q(4));
      [ fx , bc ] = hist(beam.Bunch.x(5,:),nbin);
      [~, q] = gauss_fit(bc,fx) ;
      data.zfit=abs(q(4));
      % Get stats at waist
%       beam.Bunch.x=(R+L.*data.ywaist)*beam.Bunch.x;
      
    end
    function [bunchProfile, sx, sy]=plasmaTrack(beamIn,decimate,doplot)
      persistent n_in s_in
      if isempty(n_in)
        ld=load('data/plasProf');
        n_in=ld.n; s_in=ld.s;
      end
      n0=1e11; % / m^3 (1e17 cm^-3)
      
      s=linspace(s_in(1),s_in(end),2048);
      n=interp1(s_in,n_in,s);
      
      gamma=mean(beamIn.Bunch.x(6,:))/0.511e-3;
      
      B=beamIn.Bunch.x;
      if ~exist('decimate','var') || isempty(decimate)
        decimate=1;
      end
      NH=100; NLAST=length(s)/decimate;
      
      sx=zeros(1,NLAST-1); sy=sx;
      
      bunchProfile=zeros(NH,NH,NLAST);
      bp=zeros(2,length(B),length(s)-1);
      nbin=max([length(beamIn.Bunch.Q)/100 100]);
      for is=2:NLAST
        K=n(is)*(n0/gamma);
        dl=s(is)-s(is-1);
        R=[1 0 0 0 0 0;
          -K*dl 1 0 0 0 0;
          0 0 1 0 0 0;
          0 0 -K*dl 1 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 1];
        %   R=diag(ones(1,6));
        RL=[1 dl/2 0 0 0 0;
          0 1 0 0 0 0;
          0 0 1 dl/2 0 0;
          0 0 0 1 0 0;
          0 0 0 0 1 0;
          0 0 0 0 0 1];
        B=RL*(R*(RL*B));
        bp(:,:,is-1)=B([1 3],:);
        [ fx , bc ] = hist(B(3,:),nbin);
        [~, q] = gauss_fit(bc,fx) ;
        sy(is-1)=abs(q(4));
        [ fx , bc ] = hist(B(1,:),nbin);
        [~, q] = gauss_fit(bc,fx) ;
        sx(is-1)=abs(q(4));
      end
      xran=linspace(min(min(bp(1,:,1:NLAST-1))),max(max(bp(1,:,1:NLAST-1))),NH);
      yran=linspace(min(min(bp(2,:,1:NLAST-1))),max(max(bp(2,:,1:NLAST-1))),NH);
      for is=1:NLAST-1
        hh=hist2(bp(1,:,is),bp(2,:,is),xran,yran);
        bunchProfile(:,:,is)=hh;%reshape(hh,NH,NH,1);
      end
      if exist('doplot','var') && ~isempty(doplot)
        % fit plot
        figure
        plot(s(2:NLAST).*1e2,sx.*1e6)
        hold on
        plot(s(2:NLAST).*1e2,sy.*1e6,'r')
        hold off
        xlabel('s / cm')
        ylabel('Fitted Transverse Beam Size / um')
        legend({'X' 'Y'})
        grid on
        % 3D plot
        figure
        data=bunchProfile;
        % data(data<100)=0;
        data = smooth3(data,'box',5);
        p = patch(isosurface(data,.5), ...
          'FaceColor', 'blue', 'EdgeColor', 'none');
        patch(isocaps(data,.5), ...
          'FaceColor', 'interp', 'EdgeColor', 'none');
        isonormals(data,p)
        view([4 -32]);
        axis vis3d tight
        % axis([1 NH 1 NH 1 NLAST])
        camlight; lighting phong
        colorbar
        axis off
      end
    end
  end
  methods(Static,Access=private)
    function chi2 = minWaist(x,R,L,sig,dir)
      newsig=(R+L.*x(1))*sig*(R+L.*x(1))';
      chi2=newsig(dir,dir)^2;
    end
    function chi2 = sinFit(x,data,error)
      chi2=sum( ( data - ( x(1) * sin((1:length(data))/x(2)+2*pi*x(3))+mean(data) ) ).^2 ./ error.^2);
    end
  end
  methods(Access=private)
    function [bininds, z, Z, ZSP]=doBinning(beamZ,nbin)
      zmin=min(-beamZ);
      zmax=max(-beamZ);
      if zmin==zmax
        error('Need some spread in z-distribution of bunch to compute CSR!')
      end
      z=linspace(zmin,zmax,nbin);
      z=z-mean(z);
      [~,bininds] = histc(-beamZ,z);
      [Z, ZSP]=meshgrid(z,z);
    end
  end
end