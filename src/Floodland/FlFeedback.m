classdef FlFeedback < handle & FlIndex & FlInstr & FlGui & matlab.mixin.Copyable & FlApp
  %FLFEEDBACK Generic feedback class for Floodland
  %  Create one or multiple feedbacks between any chosen actuator(s)
  %  (PS,GIRDER or KLYSTRON) and any BPM(s) using simple gain or PID
  %  feedback algorithms
  %
  % It is the intension that this class be used mainly through its
  % graphical user interface (GUI). To do that, create object with
  % constructor and then call 'guiMain' method.
  % It is however also possible to run the feedback class directly through
  % the object interface if required.
  %
  % Constructor:
  %  FB = FlFeedback(FL,instObj,indxObj)
  %   Generate a FB object by passing a Floodland object which describes
  %   the online environment, an FlInstr and FlIndex object which provides
  %   a full list of BPMs and correctors or other actuator devices
  %   available to this feedback class from the control system. Multiple FB
  %   objects can be created under this master object, each of which may
  %   contain a subset of this master set of control elements.
  %   The FB object inherits methods and properties from FlIndex, FlInstr
  %   and FlGui classes. Use methods of those classes to set the BPMs and
  %   actuators associated with a feedback object.
  %   Launch the main GUI figure with guiMain method. See the Lucretia
  %   documentation on the main Lucretia website for useage instructions.
  %   All FB settings and states are saved along with object, save the FB
  %   object in a matlab file and reload to continue using a previously
  %   configured feedback.
  %
  % Main public methods:
  %  guiMain - launch main GUI interface
  %  + operator available for merging Feedback objects
  %  manualCal - perform calibration of feedback according to pre-set
  %              configurations
  %  manualCalDataList - list of calibrations available
  %  manualCalDataSave/manualCalDataLoad - save/load previous calibration
  %                                        set
  %  calcR - calculate response matrix data from calibration data
  %  run - operate the feedback one time
  %  toggle - toggle between running/off states
  %  start - start feedback timer which operates feedback
  %  stop - stop the timer
  %  takeNewFbSetpoint - new set point for the feedback (sets zero point of
  %                      BPM readings to the currently measured ones)
  %  resetFbSetpoint - reset setpoint to zero
  %  newFB - add a new feedback object, list of created objects stored in
  %          fbList property.
  %  rmFB(fbObj) - remove the Feedback object passed from the list (cannot
  %                delete the last entry in the list)
  %
  % See also:
  %  Floodland FlInstr FlIndex FlGui FlApp
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc FlFeedback">doc FlFeedback</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  

  properties
    feedbackAlgorithm = 'Gain' ; % Choice of feedback algorithm to use, currently supported 'Gain' or 'PID'
    PID = [0.2 5 0] ; % feedback PID coefficients
    Gain = 0.1 ; % feedback gain factor
    BPMBufferDepth = 1000 ; % depth of BPM buffer
    Name = 'myFB' ; % User name for feedback
    manualCalStepsLow % lower bound of calibration steps
    manualCalStepsHigh % upper bound of calibration steps
    manualCalNsteps % number of calibration steps
    manualCalNbpm % unused
    manualCalName='none'; % name associated with calibration and settings for this FB
    lastStartDate % datenum of last time feedback was started
    fbResetVals % Values of feedback actuators at last start
  end
  properties(SetAccess=private)
    state % stopped;running;error;stopping;starting
    timer % timer to run feedback constantly when in running mode
    fbCounter=0; % increments each time feedback is run
    fbList={}; % pointers to all generated feedback objects
    fbID % the id of this FB object in fbList
  end
  properties(Access=private)
    manualCalDataDate % date of last calibration
    manualCalStepsUsed % calibration setup data for loaded calibration
    manualCalUseCntrlChan % control channel selection for calibration
    manualCalR % calibrated response matrix data
    manualCalR_err % error on calibrated response matrix data
    pid_t0 % t0 for PID loop
    pid_pe % last error term for PID
    pid_ppe % previous to last error term for PID
    FL % pointer to Floodland object
    lasterr % last error
    calret % data from calibration
  end
  properties(Dependent,Access=private)
    stateVal % unused
    stateCol % unused
  end
  properties(Constant)
    version=1.0; % version number for this Floodland app
    fbStates={'stopped' 'running' 'error' 'stopping' 'starting'}; % possible feedback state flags
    appName='Orbit Feedback'; % application name
  end

  %% Get/Set methods
  methods
    function val=get.stateVal(obj)
      val=ismember(obj.fbStates,obj.state);
    end
    function col=get.stateCol(obj)
      stateCols={[1 0 0] [0 1 0] [1 1 0] [0.5 0 0.01] [0 0.9 0.1]};
      col=stateCols{obj.stateVal};
    end
    function set.feedbackAlgorithm(obj, value)
      if ~ischar(value) || (~isequal(value,'PID') && ~isequal(value,'Gain'))
        error('Valid feedbackAlgorithm options are ''PID'' or ''Gain''')
      end
      obj.feedbackAlgorithm=value;
    end
    function set.state(obj,value)
      % start/stop feedback
      
      if ~ismember(value,obj.fbStates)
        error('Allowed state choices are: %s',obj.fbStates)
      end
      if strcmp(obj.state,'error') && obj.guiExists('guiMain') && ~strcmp(value,'error')
        try
          resp=questdlg(sprintf('Reset error condition?\n%s',obj.lasterr.message),'Error reset','run','stop','cancel','cancel'); %#ok<MCSUP>
        catch
          resp=questdlg('Reset error condition?','Error reset','run','stop','cancel','cancel');
        end
        if strcmp(resp,'run')
          value='starting';
        elseif strcmp(resp,'stop')
          value='stopped';
        else
          return
        end
      end
      obj.state=value;
      % Update GUI
      if obj.guiExists('guiMain')
        set(obj.gui.main_runstop,'String',obj.state)
        set(obj.gui.main_runstop,'BackgroundColor',obj.stateCol) %#ok<MCSUP>
        drawnow('expose')
      end
      obj.stateChangeEvnt;
    end
  end
  %% Main public methods
  methods
    function obj = FlFeedback(FL,instObj,indxObj)
      % obj = FlFeedback(FL,instObj,indxObj)
      % Feedback class constructor, pass Floodland object FL, FlInstr
      % object instObj and FlIndex object indxObj. These should be the
      % 'master' objects for all available resources, of which a subset are
      % selected using this object.
      
      % Need to pass FlInstr object
      if exist('instObj','var') && ~strcmp(class(instObj),'FlInstr')
        error('Must pass FlInstr object as second argument');
      end
      % Need to pass FlIndex object
      if exist('indxObj','var') && ~strcmp(class(indxObj),'FlIndex')
        error('Must pass FlInstr object as third argument');
      end
      % Need to pass Floodland object
      if exist('FL','var') && ~strcmp(class(FL),'Floodland')
        error('Must pass Floodland object as first argument');
      end
      % Incorporate INSTR object
      obj+instObj;
      % Incorporate Index Object (controllers)
      obj+indxObj;
      % additional parameter setup
      for ind=1:length(obj.MasterInd)
        obj.manualCalStepsLow{ind}=obj.Ampl{ind}.*0.9;
        obj.manualCalStepsHigh{ind}=obj.Ampl{ind}.*1.1;
        obj.manualCalNsteps{ind}=ones(1,length(obj.Ampl{ind})).*5;
        obj.manualCalNbpm{ind}=ones(1,length(obj.Ampl{ind})).*10;
      end
      obj.manualCalUseCntrlChan=obj.useCntrlChan;
      obj.manualCalR=[];
      obj.manualCalR_err=[];
      obj.pid_t0=now;
      obj.state='stopped';
      % make pointer to Floodland object passed
      obj.FL=FL;
      % add timer to deal with FB state changes
      obj.timer=timer('Tag','FB timer','TimerFcn',@(src,event)fbTimer(obj,src,event),'BusyMode','drop',...
        'ExecutionMode','fixedSpacing','Period',1/obj.FL.repRate,'StartFcn',@(src,event)fbTimer(obj,src,event,'start'),...
        'StopFcn',@(src,event)fbTimer(obj,src,event,'stop'));
      % Initialise feedback list
      obj.fbID=1;
      obj.fbList{1}=obj;
      % Use single particle tracking for sim mode
      obj.simBeam='BeamSingle';
      % warnings to switch off
      warning('off','MATLAB:lscov:RankDefDesignMat')
    end
    function obj=plus(obj,A)
      % addition operator for merging two feedback objects
      if strcmp(class(A),'FlIndex')
        plus@FlIndex(obj,A);
      elseif strcmp(class(A),'FlInstr')
        plus@FlInstr(obj,A);
      else
        error('Plus operator only supported with FlIndex or FlInstr objects')
      end
    end
    function b=loadobj(a)
      % Specifies object load behaviour
      b=a;
      % warnings to switch off
      warning('off','MATLAB:lscov:RankDefDesignMat')
      % Make sure everything switched off
      for ifile=1:length(b.fbList)
        b.fbList{ilist}.state='stopped';
      end
      disp('loading FB obj')
    end
    function manualCal(obj)
      % manualCal(obj)
      % perform manual calibration of response matrices
      
      % Make sure FB not running
      if ~strcmp(obj.state,'stopped')
        obj.stop;
      end
      % Process GUI
      if obj.guiExists('takeNewManualCal')
        set(obj.gui.unitpanel,'Visible','off')
        set(obj.gui.nbpmpanel,'Visible','off')
        set(obj.gui.takeNewManualCal_go,'String','STOP')
        set(obj.gui.takeNewManualCal_go,'Callback',@(src,event)guiStopEvent(obj,src,event))
        set(obj.gui.takeNewManualCal_changeCuts,'Callback','')
        ocb=get(obj.gui.takeNewManualCal_changeCuts,'Callback');
        ostr=get(obj.gui.takeNewManualCal_changeCuts,'String');
        set(obj.gui.takeNewManualCal_changeCuts,'Enable','off')
        set(obj.gui.takeNewManualCal_nbpm,'Enable','off')
        set(obj.gui.takeNewManualCal_namepanel,'Title','status')
        set(obj.gui.takeNewManualCal_table,'Visible','off')
        tpos=get(obj.gui.takeNewManualCal_table,'Position');
        hsize=tpos(3)*0.9;
        vsize=[tpos(4)*2/5 tpos(4)*3/5];
        obj.guiCreateAxes('takeNewManualCal_axis1','takeNewManualCal',[0.1 tpos(2)+vsize(1) hsize vsize(2)*0.9])
        obj.guiCreateAxes('takeNewManualCal_axis2','takeNewManualCal',[0.1 tpos(2) hsize vsize(1)])
      end
      % Initialise data array
      obj.manualCalDataDate=NaN(sum(obj.useCntrl),max(arrayfun(@(x) length(obj.useCntrlChan{x}),find(obj.useCntrl))));
      % Update hw vals
      obj.FL.hwGet(obj);
      initVals=obj.Ampl;
      % drive all FB actuators randomly and store response
      thiscontroller=0;
      onetime=0;
      for indx=find(obj.useCntrl)
        chans=find(obj.manualCalUseCntrlChan{indx});
        if isempty(chans); continue; end;
        for ichan=chans
          thiscontroller=thiscontroller+1;
          steps=linspace(obj.manualCalStepsLow{indx}(ichan),obj.manualCalStepsHigh{indx}(ichan),obj.manualCalNsteps{indx}(ichan));
          obj.manualCalStepsUsed{indx,ichan}=randperm(obj.manualCalNsteps{indx}(ichan));
          try
            thisstep=0;
            % Process GUI
            if obj.guiExists('takeNewManualCal')
              hold(obj.gui.takeNewManualCal_axis1,'off')
              hold(obj.gui.takeNewManualCal_axis2,'off')
            end
            % Start at initial values
            obj.SetPt=initVals;
            obj.FL.hwSet(obj);
            for istep=obj.manualCalStepsUsed{indx,ichan}
              thisstep=thisstep+1;
              % Process GUI
              if obj.guiExists('takeNewManualCal')
                set(obj.gui.takeNewManualCal_name,'String',...
                  sprintf('Contoller %d to %g (step %d of %d)',thiscontroller,steps(istep),thisstep,length(obj.manualCalStepsUsed{indx,ichan})))
                drawnow('expose')
              end
              % Set new controller value
              newVals=initVals;
              newVals{indx}(ichan)=steps(istep);
              obj.SetPt=newVals;
              % Write new controller value to control system
              obj.FL.hwSet(obj,indx);
              % Acquire new BPM data
              obj.acquire(obj.FL);
              % Process stop request
              if obj.stopReq
                obj.stopReq=false;
                % return controllers to initial values
                obj.SetPt=initVals;
                obj.FL.hwSet(obj,indx);
                delete(takeNewManualCal)
                return
              end
              % Store data date for processing later
              obj.manualCalDataDate(indx,ichan,istep)=now;
              % Process GUI
              if obj.guiExists('takeNewManualCal')
                obj.plot(obj.gui.takeNewManualCal_axis1,'x','data','r',~onetime)
                onetime=onetime+1;
                hold(obj.gui.takeNewManualCal_axis1,'on')
                obj.plot(obj.gui.takeNewManualCal_axis2,'y','data','b',false)
                hold(obj.gui.takeNewManualCal_axis2,'on')
                set(obj.gui.takeNewManualCal_axis1,'XTick',[])
                set(obj.gui.takeNewManualCal_axis1,'YTick',[])
                set(obj.gui.takeNewManualCal_axis2,'YTick',[])
                drawnow('expose')
              end
            end
            obj.manualCalUseCntrlChan{indx}(ichan)=false;
          catch ME
            warning('Lucretia:FlFeedback','error getting manual cal data: %s',ME.message)
            % Process GUI
            if obj.guiExists('takeNewManualCal')
              set(obj.gui.takeNewManualCal_name,'String',sprintf('error getting manual cal data: %s\nContinue?',ME.message))
              set(obj.gui.takeNewManualCal_go,'String','NO - STOP')
              set(obj.gui.takeNewManualCal_changeCuts,'Enable','on')
              set(obj.gui.takeNewManualCal_changeCuts,'String','YES')
              set(obj.gui.takeNewManualCal,'CurrentObject',0)
              drawnow('expose')
              while obj.guiExists('takeNewManualCal') && ...
                  ~isequal(get(obj.gui.takeNewManualCal,'CurrentObject'),obj.gui.takeNewManualCal_go) && ...
                  ~isequal(get(obj.gui.takeNewManualCal,'CurrentObject'),obj.gui.takeNewManualCal_changeCuts)
                pause(0.2)
              end
              if ~obj.guiExists('takeNewManualCal')
                % return controllers to initial values
                obj.SetPt=initVals;
                obj.FL.hwSet(obj);
                delete(obj.gui.takeNewManualCal)
                return
              end
              if obj.stopReq
                obj.stopReq=false;
                % return controllers to initial values
                obj.SetPt=initVals;
                obj.FL.hwSet(obj);
                delete(obj.gui.takeNewManualCal)
                return
              else
                set(obj.gui.takeNewManualCal_go,'String','STOP')
                set(obj.gui.takeNewManualCal_changeCuts,'Enable','off')
                set(obj.gui.takeNewManualCal_changeCuts,'String',ostr)
                set(obj.gui.takeNewManualCal_changeCuts,'Callback',ocb)
              end
            end
          end
        end
      end
      % return controllers to initial values
      obj.SetPt=initVals;
      obj.FL.hwSet(obj);
       % Process GUI
      if obj.guiExists('takeNewManualCal')
        set(obj.gui.takeNewManualCal_name,'String','Use calibration data?')
        set(obj.gui.takeNewManualCal_changeCuts,'Enable','on')
        set(obj.gui.takeNewManualCal_changeCuts,'String','YES')
        opos=get(obj.gui.takeNewManualCal_go,'Position');
        set(obj.gui.takeNewManualCal_go,'String','NO, Cancel')
        set(obj.gui.takeNewManualCal_go,'Position',opos.*[1 1 1 0.45])
        obj.guiCreatePushbutton('takeNewManualCal_showSlopes','Show Slopes','takeNewManualCal',opos.*[1 1 1 0.45]+[0 opos(4).*0.5 0 0]);
        set(obj.gui.takeNewManualCal_showSlopes,'Callback',@(src,event)guiCheckSlopes(obj,src,event))
        set(obj.gui.takeNewManualCal,'CurrentObject',0)
        drawnow('expose')
        while obj.guiExists('takeNewManualCal') && ...
            ~isequal(get(obj.gui.takeNewManualCal,'CurrentObject'),obj.gui.takeNewManualCal_go) && ...
            ~isequal(get(obj.gui.takeNewManualCal,'CurrentObject'),obj.gui.takeNewManualCal_changeCuts)
          pause(0.2)
        end
        if ~obj.guiExists('takeNewManualCal') || obj.stopReq
          obj.stopReq=false;
          delete(obj.gui.takeNewManualCal)
          return
        end
      end
      try
        % Calc response matrix data from calibration
        obj.calcR;
        % Save cal data
        manualCalDataSave(obj);
        obj.calret=true;
      catch ME
        if obj.guiExists('takeNewManualCal')
          errordlg(sprintf('Failed to create or save response matrix:\n%s',ME.message),'R calc failed')
          takeNewManualCal(obj,[],[]);
        else
          rethrow(ME)
        end
      end
      if obj.guiExists('takeNewManualCal')
        delete(obj.gui.takeNewManualCal)
      end
    end
    function dataList=manualCalDataList(obj)
      % dataList=manualCalDataList(obj)
      % list manual calibration data files (in date order) MAX=30
      caldata=dir(fullfile('FlFeedback_manualCalData',sprintf('%s_*.mat',obj.Name)));
      if ~isempty(caldata)
        cnames={caldata.name};
        cdates={caldata.date};
        [a ind]=sort(datenum(cdates),'descend');
        dataList=regexprep(cnames(ind(1:min([30,length(ind)]))),'\.mat$','');
      else
        dataList=[];
      end
    end
    function manualCalDataSave(obj)
      % save manual calibration data
      mc=?FlFeedback;
      pl={mc.PropertyList.Name};
      cf=pl(cellfun(@(x) ~isempty(x),regexp(pl,'^manualCal')));
      manualCalData=cell(1,length(cf));
      for icf=1:length(cf)
        manualCalData{icf}=obj.(cf{icf});
      end
      if ~exist('FlFeedback_manualCalData','dir')
        mkdir('FlFeedback_manualCalData');
      end
      version=obj.version; %#ok<NASGU>
      actuators={obj.useCntrl obj.useCntrlChan}; %#ok<NASGU>
      bpms=obj.useInstr; %#ok<NASGU>
      save(fullfile('FlFeedback_manualCalData',sprintf('%s_%s.mat',obj.Name,obj.manualCalName)),'manualCalData','version','actuators','bpms')
    end
    function manualCalDataLoad(obj,name)
      % load manual calibration data
      if ~exist('FlFeedback_manualCalData','dir')
        error('No saved calibration data')
      end
      if isempty(regexp(name,'\.mat$', 'once'))
        name=sprintf('%s.mat',name);
      end
      calList=dir('FlFeedback_manualCalData');
      if ~ismember(name,{calList.name})
        error('No saved calibration data of this name')
      end
      manualCalData=[];
      load(fullfile('FlFeedback_manualCalData',name),'manualCalData','version','actuators','bpms')
      if version~=obj.version
        warning('Lucretia:FlFeedback:incompcaldata','Saved file from different version of this Class, there may be problems')
      end
      mc=?FlFeedback;
      pl={mc.PropertyList.Name};
      cf=pl(cellfun(@(x) ~isempty(x),regexp(pl,'^manualCal')));
      if length(cf)~=length(manualCalData)
        error('Incompatible calibration data')
      end
      obj.useCntrl=actuators{1}; %#ok<USENS>
      obj.useCntrlChan=actuators{2};
      obj.useInstr=bpms;
      for icf=1:length(cf)
        obj.(cf{icf})=manualCalData{icf};
      end
    end
    function calcR(obj)
      % calculate response matrices from manual calibration data
      initDataDate=obj.lastdata;
      sz=size(obj.data);
      obj.manualCalR=[]; obj.manualCalR_err=[];
      icntrl=0;
      chans=obj.INDXchannels;
      for indx=find(obj.useCntrl)
        for ichan=find(obj.useCntrlChan{indx})
          if ~ismember(ichan,chans{2,indx}); continue; end;
          steps=linspace(obj.manualCalStepsLow{indx}(ichan),obj.manualCalStepsHigh{indx}(ichan),obj.manualCalNsteps{indx}(ichan));
          bpmValsX=NaN(sz(1),length(steps));
          bpmValsY=bpmValsX;
          bpmErrX=bpmValsX;
          bpmErrY=bpmValsX;
          for istep=obj.manualCalStepsUsed{indx,ichan}
            obj.lastdata=obj.manualCalDataDate(indx,ichan,istep);
            [data dataerr]=obj.meanstd;
            bpmValsX(:,istep)=data(:,1); bpmErrX(:,istep)=dataerr(:,1);
            bpmValsY(:,istep)=data(:,2); bpmErrY(:,istep)=dataerr(:,2);
          end
          icntrl=icntrl+1;
          for ibpm=1:sz(1)
            [q dq]=noplot_polyfit(steps,bpmValsX(ibpm,:),bpmErrX(ibpm,:),1);
            obj.manualCalR(ibpm,icntrl)=q(2); obj.manualCalR_err(ibpm,icntrl)=dq(2);
            [q dq]=noplot_polyfit(steps,bpmValsY(ibpm,:),bpmErrY(ibpm,:),1);
            obj.manualCalR(sz(1)+ibpm,icntrl)=q(2); obj.manualCalR_err(sz(1)+ibpm,icntrl)=dq(2);
          end
        end
      end
      obj.lastdata=initDataDate;
    end
    function run(obj)
      % Run the feedback (once)
      
      % Check we have a cal loaded
      if strcmp(obj.manualCalName,'none')
        error('No calibation loaded')
      end
      % Update BPMs
      obj.acquire(obj.FL);
      % Update cntrl readings
      obj.FL.hwGet(obj);
      cntrlSetPt=obj.Ampl;
      chans=obj.INDXchannels;
      limitLow=obj.limitLow;
      limitHigh=obj.limitHigh;
      % Get BPM readings
      [data dataerr]=obj.meanstd;
      bpmValsX=data(:,1); bpmErrX=dataerr(:,1);
      bpmValsY=data(:,2); bpmErrY=dataerr(:,2);
      % form error signal as model estimate of correction to apply to
      % selected controllers
      errX=-bpmValsX; errY=-bpmValsY;
      % Check for bad BPM data
      badbpms={}; names=obj.instrName(obj.useInstr);
      if any(isnan(errX))
        bbpm=isnan(errX);
        for ibpm=find(bbpm)
          badbpms{end+1}=[names{ibpm} ':X'];
        end
      end
      if any(isnan(errY))
        bbpm=isnan(errY);
        for ibpm=find(bbpm)
          badbpms{end+1}=[names{ibpm} ':Y'];
        end
      end
      if ~isempty(badbpms)
        warning('BAD BPM READING, not applying corrections this time')
        disp(badbpms)
        return
      end
      if obj.ndata>1
        try
          C=lscov(obj.manualCalR,[errX;errY],1./([bpmErrX.^2;bpmErrY.^2]));
        catch
          C=lscov(obj.manualCalR,[errX;errY]);
        end
      else
        C=lscov(obj.manualCalR,[errX;errY]);
      end
      dt=(now-obj.pid_t0)*3600*24;
      % Compute and apply correction
      ival=0; cval=[]; climLow=[]; climHigh=[];
      for indx=find(obj.useCntrl)
        for ichan=find(obj.useCntrlChan{indx})
          if ~ismember(ichan,chans{2,indx}); continue; end;
          ival=ival+1;
          if length(obj.pid_pe)<ival
            obj.pid_pe(ival)=0;
          end
          if length(obj.pid_ppe)<ival
            obj.pid_ppe(ival)=0;
          end
          switch obj.feedbackAlgorithm
            case 'PID'
              % use PID discrete implementation in standard form
              cntrlSetPt{indx}(ichan) = cntrlSetPt{indx}(ichan) + ...
                obj.PID(1)*( (1+dt/obj.PID(2)+obj.PID(3)/dt).*C(ival) + ...
                (-1 - ((2*obj.PID(3))/dt)).*obj.pid_pe(ival) + ...
                (obj.PID(3)/dt).*obj.pid_ppe(ival) ) ;
              obj.pid_ppe(ival)=obj.pid_pe(ival);
              obj.pid_pe(ival)=C(ival);
            case 'Gain'
              % use simple gain factor
              cntrlSetPt{indx}(ichan) = cntrlSetPt{indx}(ichan) + C(ival)*obj.Gain;
          end
          cval(end+1)=cntrlSetPt{indx}(ichan);
          climLow(end+1)=limitLow{indx}(2,ichan);
          climHigh(end+1)=limitHigh{indx}(2,ichan);
        end
      end
      obj.SetPt=cntrlSetPt;
      obj.FL.hwSet(obj);
      % Input time for next dt computation
      obj.pid_t0=now;
      % Increment counter
      obj.fbCounter=obj.fbCounter+1;
      % set running state
      if ~strcmp(obj.state,'running')
        obj.state='running';
      end
      % Update diagnostics display if showing
      if obj.guiExists('fbDiagnostics') && ~get(obj.gui.main_diagnostics,'Value')
        obj.fbDiagnostics_update(bpmValsX,bpmErrX,bpmValsY,bpmErrY,cval,climLow,climHigh);
      end
    end
    function toggle(obj,~,~)
      % Toggle between running or stopped state
      if ismember(obj.state,{'stopped' 'error'})
        obj.start;
      elseif strcmp(obj.timer.running,'off')
        obj.state='stopping';
        obj.state='stopped';
      else
        obj.stop;
      end
    end
    function start(obj)
      % Start timer object which runs feedback
      if ~strcmp(obj.state,'running') && ~strcmp(obj.state,'starting')
        obj.state='starting';
      end
    end
    function stop(obj)
      % stop the timer which runs the feedback
      if ~strcmp(obj.state,'stopped') && ~strcmp(obj.state,'stopping')
        obj.state='stopping';
      end
    end
    function stateChangeEvnt(obj)
      % control what happens when the feedback state changes
      switch obj.state
        case 'starting'
          start(obj.timer);
        case {'stopping' 'error'}
          stop(obj.timer);
          obj.state='stopped';
      end
    end
    function takeNewFbSetpoint(obj)
      % Set new setpoint for FB BPMs
      % Update BPMs
      obj.acquire(obj.FL);
      % Get BPM readings
      obj.setRef;
    end
    function resetFbSetpoint(obj)
      % reset FB setpoint to zero for BPMs
      obj.setRef('zero');
    end
    function fbTimer(obj,~,~,cmd)
      % timer function to run the feedback
      try
        if exist('cmd','var')
          if strcmp(cmd,'start')
            obj.state='running';
            obj.lastStartDate=now;
            obj.fbResetVals=obj.Ampl;
          elseif strcmp(cmd,'stop')
            obj.state='stopped';
          end
        else
          obj.run;
        end
      catch ME
        obj.lasterr=ME;
        obj.state='error';
        disp(ME.message)
      end
    end
    function resetFB(obj,~,~)
      % reset FB actuators to their pre-stored reset values
      obj.SetPt=obj.fbResetVals;
      obj.FL.hwSet(obj);
    end
    function newFB(obj,~,~)
      % Make new feedback object
      obj.fbList{end+1}=obj.copy;
      obj.fbList{end}.Name=sprintf('newFB%d',length(obj.fbList));
      obj.fbList{end}.fbID=length(obj.fbList); % give the obj its ID in the list
      % Copy new list of FB objects to all FB objects
      for ifb=1:length(obj.fbList)
        obj.fbList{ifb}.fbList=obj.fbList;
      end
      obj.fbList{end}.state='stopped';
      % Relaunch main gui if it is there
      if obj.guiExists('guiMain')
        obj.guiMain;
      end
    end
    function rmFB(obj,~,~)
      % delete feedback object
      
      % Cannot delete last feedback object
      if length(obj.fbList)<2
        return
      end
      % Delete this FB object in list
      obj.fbList(obj.fbID)=[];
      fblist=obj.fbList{1};
      % Update lists in all FB's
      for ifb=1:length(obj.fbList)
        obj.fbList{ifb}.fbList=obj.fbList;
      end
      
      % Now delete the object
      isgui=obj.guiExists('guiMain');
      delete(obj);
      
      % relaunch the gui if there is one
      if isgui
        fblist.guiMain;
      end
      
    end
    function han=guiMain(obj,~,~)
      % Create main GUI figure window
      if obj.guiExists('guiMain'); delete(obj.gui.guiMain); end;
      obj.guiCreateFigure('guiMain','Orbit Feedback(s)',[800 75*length(obj.fbList)]);
      border=0.045;
      psize=[1-2*border (1-border*(length(obj.fbList)+1))/length(obj.fbList)];
      csize=[(1-border*5)/4 1-2*border];
      han=obj.gui.guiMain;
      if isempty(obj.fbList); return; end; % if no FB's created, just return leaving GUI with menus to create new ones
      for ifb=1:length(obj.fbList)
        obj.fbList{ifb}.gui.guiMain=obj.gui.guiMain;
      end
      % Create menu items
      em=uimenu('Parent',obj.gui.guiMain,'Label','Edit');
      uimenu('Parent',em,'Label','Create new feedback object','Callback',...
        @(src,event)obj.newFB(src,event));
      if length(obj.fbList)>1
        dm=uimenu('Parent',em,'Label','Delete feedback object');
        for ifb=1:length(obj.fbList)
          uimenu('Parent',dm,'Label',obj.fbList{ifb}.Name,'Callback', @(src,event)obj.fbList{ifb}.rmFB(src,event));
        end
      end
      % Make feedback panels
      for ifb=1:length(obj.fbList)
        % Feedback panel
        obj.fbList{ifb}.guiCreatePanel('main_panel',sprintf('%s (cal = %s)',obj.fbList{ifb}.Name,obj.fbList{ifb}.manualCalName),...
          obj.fbList{ifb}.gui.guiMain,[border border+border*(ifb-1)+psize(2)*(ifb-1) psize]);
        % Run/stop button
        obj.fbList{ifb}.guiCreatePushbutton('main_runstop',obj.fbList{ifb}.state,'main_panel',[border border csize]);
        set(obj.fbList{ifb}.gui.main_runstop,'BackgroundColor',obj.fbList{ifb}.stateCol)
        set(obj.fbList{ifb}.gui.main_runstop,'Callback',@(src,event)toggle(obj.fbList{ifb},src,event));
        % Settings button
        obj.fbList{ifb}.guiCreatePushbutton('main_settings','Settings','main_panel',[border*2+csize(1) border csize]);
        set(obj.fbList{ifb}.gui.main_settings,'Callback',@(src,event)fbSettings(obj.fbList{ifb},src,event))
        set(obj.fbList{ifb}.gui.main_settings,'Interruptible','off')
        % Diagnostics button
        obj.fbList{ifb}.guiCreatePushbutton('main_diagnostics','Diagnostics','main_panel',[border*3+csize(1)*2 border csize]);
        set(obj.fbList{ifb}.gui.main_diagnostics,'Callback',@(src,event)fbDiagnostics(obj.fbList{ifb},src,event))
        set(obj.fbList{ifb}.gui.main_diagnostics,'Interruptible','off')
        % change set point button
        obj.fbList{ifb}.guiCreatePushbutton('main_setpt','Set Pt','main_panel',[border*4+csize(1)*3 border csize]);
        set(obj.fbList{ifb}.gui.main_setpt,'Callback',@(src,event)changeSetPt(obj.fbList{ifb},src,event))
        set(obj.fbList{ifb}.gui.main_setpt,'Interruptible','off')
      end
    end
  end
  
  %% GUI callbacks and sub-GUI elements
  methods(Hidden)
    function fbDiagnostics_prCallback(obj,src,~)
      % callback function for diagnostics
      if src==obj.gui.fbDiagnostics_plotlinear
        if ~get(src,'Value')
          set(src,'Value',1)
        else
          set(obj.gui.fbDiagnostics_plotlog,'Value',0)
        end
      elseif src==obj.gui.fbDiagnostics_plotlog
        if ~get(src,'Value')
          set(src,'Value',1)
        else
          set(obj.gui.fbDiagnostics_plotlinear,'Value',0)
        end
      elseif src==obj.gui.fbDiagnostics_plotval
        if ~get(src,'Value')
          set(src,'Value',1)
        else
          set(obj.gui.fbDiagnostics_plotnorm,'Value',0)
        end
      elseif src==obj.gui.fbDiagnostics_plotnorm
        if ~get(src,'Value')
          set(src,'Value',1)
        else
          set(obj.gui.fbDiagnostics_plotval,'Value',0)
        end
      end
    end
    function takeNewManualCal_tableEdit(obj,src,event)
      tableIndex=get(src,'UserData');
      inds=event.Indices;
      indx=tableIndex{inds(1)}(1);
      ichan=tableIndex{inds(1)}(2);
      if get(obj.gui.takeNewManualCal_dunits_setpt,'Value')
        conv=1;
      else
        conv=obj.Ampl2ACT{indx}(ichan);
      end
      switch inds(2)
        case 2
          setpt=obj.SetPt;
          setpt{indx}(ichan)=event.NewData/conv;
          obj.SetPt=setpt;
          obj.FL.hwSet(obj,indx);
        case 3
          obj.manualCalStepsLow{indx}(ichan)=event.NewData/conv;
        case 4
          obj.manualCalStepsHigh{indx}(ichan)=event.NewData/conv;
        case 5
          obj.manualCalNsteps{indx}(ichan)=event.NewData;
        case 6
          obj.manualCalUseCntrlChan{indx}(ichan)=event.NewData;
      end
    end
    function takeNewManualCal_name(obj,src,~)
      obj.manualCalName=get(src,'String');
    end
    function takeNewManualCal_nbpm(obj,src,~)
      obj.ndata=str2double(get(src,'String'));
    end
    function takeNewManualCal_dunits(obj,src,~)
      if strcmp(get(src,'String'),'SetPt')
        if get(obj.gui.takeNewManualCal_dunits_setpt,'Value')
          set(obj.gui.takeNewManualCal_dunits_act,'Value',false)
          obj.takeNewManualCal_tableFill;
        else
          set(obj.gui.takeNewManualCal_dunits_setpt,'Value',true)
        end
      else % ACT
        if get(obj.gui.takeNewManualCal_dunits_act,'Value')
          set(obj.gui.takeNewManualCal_dunits_setpt,'Value',false)
          obj.takeNewManualCal_tableFill;
        else
          set(obj.gui.takeNewManualCal_dunits_act,'Value',true)
        end
      end
    end
    function guiStopEvent(obj,~,~)
      obj.stopReq=true;
%       takeNewManualCal(obj,[],[]);
    end
    function guiCheckSlopes(obj,~,~)
      obj.guiCreateFigure('guiCheckSlopes','Check Cal Slopes',[700 500]);
      border=0.02;
      obj.guiCreateAxes('checkSlopes_axis1',obj.gui.guiCheckSlopes,[0.1 0.65 0.8 0.28]);
      obj.guiCreateAxes('checkSlopes_axis2',obj.gui.guiCheckSlopes,[0.1 0.3 0.8 0.28]);
      bsize=(0.4-3*border)/4;
      obj.guiCreatePushbutton('checkSlopes_prevact','<','guiCheckSlopes',[0.1 border 0.1 bsize])
      set(obj.gui.checkSlopes_prevact,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      obj.guiCreatePushbutton('checkSlopes_nextact','>','guiCheckSlopes',[0.8 border 0.1 bsize])
      set(obj.gui.checkSlopes_nextact,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      obj.guiCreatePushbutton('checkSlopes_prevbpm','<','guiCheckSlopes',[0.1 border*2+bsize 0.1 bsize])
      set(obj.gui.checkSlopes_prevbpm,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      obj.guiCreatePushbutton('checkSlopes_nextbpm','>','guiCheckSlopes',[0.8 border*2+bsize 0.1 bsize])
      set(obj.gui.checkSlopes_nextbpm,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      obj.guiCreatePopupmenu('checkSlopes_actmenu',obj.INDXused,'guiCheckSlopes',[0.2+border border 0.6-2*border bsize*0.8])
      set(obj.gui.checkSlopes_actmenu,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      obj.guiCreatePopupmenu('checkSlopes_bpmmenu',[{'All BPMs'} obj.instrName(obj.useInstr)],'guiCheckSlopes',...
        [0.2+border border*2+bsize 0.6-2*border bsize*0.8])
      set(obj.gui.checkSlopes_bpmmenu,'Callback',@(src,event)guiCheckSlopesCallback(obj,src,event))
      guiCheckSlopesCallback(obj,obj.gui.checkSlopes_bpmmenu);
    end
    function guiCheckSlopesCallback(obj,src,~)
      if src==obj.gui.checkSlopes_prevact
        valnow=get(obj.gui.checkSlopes_actmenu,'Value');
        if valnow>1; valnow=valnow-1; end;
        set(obj.gui.checkSlopes_actmenu,'Value',valnow)
      elseif src==obj.gui.checkSlopes_nextact
        valnow=get(obj.gui.checkSlopes_actmenu,'Value');
        if valnow<length(get(obj.gui.checkSlopes_actmenu,'String')); valnow=valnow+1; end;
        set(obj.gui.checkSlopes_actmenu,'Value',valnow)
      elseif src==obj.gui.checkSlopes_prevbpm
        valnow=get(obj.gui.checkSlopes_bpmmenu,'Value');
        if valnow>1; valnow=valnow-1; end;
        set(obj.gui.checkSlopes_bpmmenu,'Value',valnow)
      elseif src==obj.gui.checkSlopes_nextbpm
        valnow=get(obj.gui.checkSlopes_bpmmenu,'Value');
        if valnow<length(get(obj.gui.checkSlopes_bpmmenu,'String')); valnow=valnow+1; end;
        set(obj.gui.checkSlopes_bpmmenu,'Value',valnow)
      end
      initDataDate=obj.lastdata;
      chans=obj.INDXchannels;
      nc=0;
      bpmStr=get(obj.gui.checkSlopes_bpmmenu,'String');
      ibpm=get(obj.gui.checkSlopes_bpmmenu,'Value');
      for indx=find(obj.useCntrl)
        for ichan=find(obj.useCntrlChan{indx})
          if ~ismember(ichan,chans{2,indx}); continue; end;
          nc=nc+1;
          if nc~=get(obj.gui.checkSlopes_actmenu,'Value'); continue; end;
          steps=linspace(obj.manualCalStepsLow{indx}(ichan),obj.manualCalStepsHigh{indx}(ichan),obj.manualCalNsteps{indx}(ichan));
          bpmValsX=NaN(sum(obj.useInstr),length(steps));
          bpmValsY=bpmValsX;
          bpmErrX=bpmValsX;
          bpmErrY=bpmValsX;
          for istep=obj.manualCalStepsUsed{indx,ichan}
            obj.lastdata=obj.manualCalDataDate(indx,ichan,istep);
            [data dataerr]=obj.meanstd;
            bpmValsX(:,istep)=data(:,1); bpmErrX(:,istep)=dataerr(:,1);
            bpmValsY(:,istep)=data(:,2); bpmErrY(:,istep)=dataerr(:,2);
          end
        end
      end
      bx=bpmValsX; bxe=bpmErrX;
      by=bpmValsY; bye=bpmErrY;
      szx=size(bx); szy=size(by);
      if strcmp(bpmStr{ibpm},'All BPMs')
        plot(obj.gui.checkSlopes_axis1,repmat(steps,szx(1),1)',bx'.*1e6)
        plot(obj.gui.checkSlopes_axis2,repmat(steps,szy(1),1)',by'.*1e6)
        lines1=findall(obj.gui.checkSlopes_axis1,'Type','line');
        lines2=findall(obj.gui.checkSlopes_axis2,'Type','line');
        names=obj.instrName(obj.useInstr);
        for iline=1:length(lines1)
          hcmenu=uicontextmenu;
          uimenu(hcmenu,'Label',names{end-iline+1});
          set(lines1(iline),'uicontextmenu',hcmenu);
          set(lines2(iline),'uicontextmenu',hcmenu);
        end
      else
        errorbar(obj.gui.checkSlopes_axis1,steps,bx(ibpm-1,:).*1e6,bxe(ibpm-1,:).*1e6)
        errorbar(obj.gui.checkSlopes_axis2,steps,by(ibpm-1,:).*1e6,bye(ibpm-1,:).*1e6)
      end
      grid(obj.gui.checkSlopes_axis1,'on')
      grid(obj.gui.checkSlopes_axis2,'on')
      axis(obj.gui.checkSlopes_axis1,'tight')
      axis(obj.gui.checkSlopes_axis2,'tight')
      ylabel(obj.gui.checkSlopes_axis1,'x / um')
      ylabel(obj.gui.checkSlopes_axis2,'y / um')
      obj.lastdata=initDataDate;
    end
    function changeSetPt(obj,~,~)
      % Create main figure window
      obj.guiCreateFigure('changeSetPt','Change FB Setpoint',[250 200]);
      border=0.02;
      optsize=[1-2*border (1-3*border)/2];
      % Create option buttons and callbacks
      obj.guiCreatePushbutton('changeSetPt_new','Take New','changeSetPt',[border 1-border-optsize(2) optsize]);
      set(obj.gui.changeSetPt_new,'Callback',@(src,event)guiChangeSetPt_Callback(obj,src,event,1));
      obj.guiCreatePushbutton('changeSetPt_setzero','Set to Zero','changeSetPt',[border border optsize]);
      set(obj.gui.changeSetPt_setzero,'Callback',@(src,event)guiChangeSetPt_Callback(obj,src,event,0));
    end
    function guiChangeSetPt_Callback(obj,~,~,opt)
      if opt
        obj.takeNewFbSetpoint;
      else
        obj.resetFbSetpoint;
      end
      delete(obj.gui.changeSetPt)
    end
    function fbSettings_calCB(obj,~,~)
      cn=get(obj.gui.settings_manualCalInfo,'String');
      if ~iscell(cn); cn={cn}; end;
      obj.manualCalDataLoad(cn{get(obj.gui.settings_manualCalInfo,'Value')});
      % Update actuator names
      chnames={{'main'} {'x' 'dx' 'y' 'dy' 'z' 'dz'} {'ampl' 'pha'}};
      cstr={};
      names=obj.INDXnames;
      pslist=obj.PS_list; girlist=obj.GIRDER_list;
      for ic=find(obj.useCntrl)
        chanlist=find(obj.useCntrlChan{ic});
        if ~isempty(chanlist)
          for ichan=chanlist
            if ismember(ic,pslist)
              cstr{end+1}=sprintf('%s::%s',names{ic},chnames{1}{ichan});
            elseif ismember(ic,girlist)
              cstr{end+1}=sprintf('%s::%s',names{ic},chnames{2}{ichan});
            else
              cstr{end+1}=sprintf('%s::%s',names{ic},chnames{3}{ichan});
            end
          end
        end
      end
      set(obj.gui.settings_corrList,'Value',1)
      set(obj.gui.settings_corrList,'String',cstr)
      % Update INSTR names
      set(obj.gui.settings_bpmList,'Value',1)
      set(obj.gui.settings_bpmList,'String',obj.instrName(obj.useInstr))
    end
    function fbSettings(obj,~,~)
      % Create main figure window
      obj.guiCreateFigure('fbSettings',sprintf('''%s'' Settings',obj.Name),[600 450]);
      % Define border size
      border=0.02;
      % Define panel element sizes
      size_ver=[(1-border*5)/4 (1-border*3)/2];
      size_hor=(1-border*3)/2;
      % define panels
      obj.guiCreatePanel('settingsNamePanel','FB Name','fbSettings',[border 1-border-size_ver(1) size_hor size_ver(1)]);
      obj.guiCreatePanel('settingsManCalPanel','Calibration','fbSettings',[border 1-border*2-size_ver(1)*2 size_hor size_ver(1)]);
      obj.guiCreatePanel('settingsFbPanel','Feedback','fbSettings',[border 1-border*3-size_ver(1)*3 size_hor size_ver(1)]);
      obj.guiCreatePanel('settingsCommitPanel','Commit Changes','fbSettings',[border border size_hor size_ver(1)]);
      obj.guiCreatePanel('settingsCorrListPanel','Actuators','fbSettings',[border*2+size_hor 1-border-size_ver(2) size_hor size_ver(2)]);
      obj.guiCreatePanel('settingsBpmListPanel','BPMs','fbSettings',[border*2+size_hor border size_hor size_ver(2)]);
      % FB name
      obj.guiCreateEdit('settings_fbName',obj.Name,'settingsNamePanel',[border 0.33 1-2*border 0.33]);
      % manual calibration elements
      obj.guiCreatePopupmenu('settings_manualCalInfo','---','settingsManCalPanel',[border 0.66 1-2*border 0.2]);
      set(obj.gui.settings_manualCalInfo,'Callback',@(src,event)fbSettings_calCB(obj,src,event));
      dataList=obj.manualCalDataList;
      if ~isempty(dataList)
        set(obj.gui.settings_manualCalInfo,'Value',1) % select most recent
        set(obj.gui.settings_manualCalInfo,'String',dataList)
      end
      obj.guiCreatePushbutton('settings_takeNewManualCal','Take New','settingsManCalPanel',[border 0.33 (1-3*border)/2 0.2]);
      set(obj.gui.settings_takeNewManualCal,'Callback',@(src,event)guiTakeNewManualCal(obj,src,event));
      % feedback elements
      enVal={'off' 'on'};
      gainVal=obj.Gain;
      pidVal=obj.PID;
      if strcmp(obj.feedbackAlgorithm,'Gain')
        fbVal=[1 0];
      else
        fbVal=[0 1];
      end
      obj.guiCreateRadiobutton('settings_useGainCal','gain',fbVal(1),'settingsFbPanel',[border 0.66 0.3 0.2]);
      set(obj.gui.settings_useGainCal,'Callback',@(src,event)guiChooseGainMethod(obj,src,event));
      obj.guiCreateRadiobutton('settings_usePidCal','PID',fbVal(2),'settingsFbPanel',[border 0.23 0.2 0.2]);
      set(obj.gui.settings_usePidCal,'Callback',@(src,event)guiChoosePidMethod(obj,src,event));
      obj.guiCreateEdit('settings_gainVal',num2str(gainVal),'settingsFbPanel',[0.4 0.66 0.3 0.2]);
      obj.guiCreateEdit('settings_pVal',num2str(pidVal(1)),'settingsFbPanel',[0.25 0.23 0.2 0.2]);
      set(obj.gui.settings_pVal,'Enable',enVal{1+fbVal(2)});
      obj.guiCreateEdit('settings_iVal',num2str(pidVal(2)),'settingsFbPanel',[0.5 0.23 0.2 0.2]);
      set(obj.gui.settings_iVal,'Enable',enVal{1+fbVal(2)});
      obj.guiCreateEdit('settings_dVal',num2str(pidVal(3)),'settingsFbPanel',[0.75 0.23 0.2 0.2]);
      set(obj.gui.settings_dVal,'Enable',enVal{1+fbVal(2)});
      % commit elements
      obj.guiCreatePushbutton('settings_commit','OK','settingsCommitPanel',[0.15 0.35 0.3 0.3]);
      set(obj.gui.settings_commit,'Callback',@(src,event)guiAcceptSettings(obj,src,event)) ;
      obj.guiCreatePushbutton('settings_reject','Cancel','settingsCommitPanel',[0.55 0.35 0.3 0.3]);
      set(obj.gui.settings_reject,'Callback',@(src,event)guiAcceptSettings(obj,src,event)) ;
      % correction units elements
      names=obj.INDXnames(obj.useCntrl);
      obj.guiCreateListbox('settings_corrList',names,'settingsCorrListPanel',[border border*2+0.1 1-2*border 1-3*border-0.1]);
      obj.guiCreatePushbutton('settings_changeCorrList','Change','settingsCorrListPanel',[border border 0.2 0.1]);
      set(obj.gui.settings_changeCorrList,'Callback',@(src,event)guiChangeUnits(obj,src,event)) ;
      if ~isempty(obj.lastStartDate)
        obj.guiCreatePushbutton('settings_resetFB',sprintf('Reset to: %s',datestr(obj.lastStartDate)),...
          'settingsCorrListPanel',[border*2+0.2 border 0.8-border*4 0.1]);
        set(obj.gui.settings_resetFB,'Callback',@(src,event)resetFB(obj,src,event));
      end
      % BPM elements
      obj.guiCreateListbox('settings_bpmList',obj.instrName(obj.useInstr),'settingsBpmListPanel',[border border*2+0.1 1-2*border 1-3*border-0.1]);
      obj.guiCreatePushbutton('settings_changeBpmList','Change','settingsBpmListPanel',[border border 0.2 0.1]);
      set(obj.gui.settings_changeBpmList,'Callback',@(src,event)guiChangeUnits(obj,src,event)) ;
      obj.guiCreatePushbutton('settings_changeBpmCuts','Change Cuts','settingsBpmListPanel',[border*2+0.2 border 0.4 0.1]);
      set(obj.gui.settings_changeBpmCuts,'Callback',@(src,event)FlInstr_selectCuts(obj,src,event)) ;
      % Update actuators / BPMs list
      try
        fbSettings_calCB(obj);
      catch
      end
    end
    function guiChangeUnits(obj,src,~)
      if src==obj.gui.settings_changeCorrList
        uiwait(obj.guiIndexChoice);
        if isempty(obj.indexChoiceFromGui); return; end;
        useCntrl_temp={obj.indexChoiceFromGui obj.indexChanChoiceFromGui};
        chnames={{'main'} {'x' 'dx' 'y' 'dy' 'z' 'dz'} {'ampl' 'pha'}};
        cstr={};
        names=obj.INDXnames;
        pslist=obj.PS_list; girlist=obj.GIRDER_list;
        for ic=find(useCntrl_temp{1})
          chanlist=find(useCntrl_temp{2}{ic});
          if ~isempty(chanlist)
            for ichan=chanlist
              if ismember(ic,pslist)
                cstr{end+1}=sprintf('%s::%s',names{ic},chnames{1}{ichan});
              elseif ismember(ic,girlist)
                cstr{end+1}=sprintf('%s::%s',names{ic},chnames{2}{ichan});
              else
                cstr{end+1}=sprintf('%s::%s',names{ic},chnames{3}{ichan});
              end
            end
          end
        end
        set(obj.gui.settings_corrList,'Value',1)
        set(obj.gui.settings_corrList,'String',cstr)
        set(obj.gui.settings_corrList,'UserData',useCntrl_temp)
      else
        uiwait(obj.guiInstrChoice);
        if isempty(obj.instrChoiceFromGui); return; end;
        useInstr_temp=obj.instrChoiceFromGui;
        set(obj.gui.settings_bpmList,'Value',1)
        set(obj.gui.settings_bpmList,'String',obj.instrName(useInstr_temp))
        set(obj.gui.settings_bpmList,'UserData',useInstr_temp)
      end
    end
    function guiTakeNewManualCal(obj,src,event)
      obj.calret=false;
      instrVals=get(obj.gui.settings_bpmList,'UserData');
      if ~isempty(instrVals)
        useInstrOrig=obj.useInstr;
        obj.useInstr=instrVals;
      end
      useCntrlVals=get(obj.gui.settings_corrList,'UserData');
      if ~isempty(useCntrlVals)
        cntrlOrig={obj.useCntrl obj.useCntrlChan};
        obj.useCntrl=useCntrlVals{1};
        obj.useCntrlChan=useCntrlVals{2};
      end
      uiwait(takeNewManualCal(obj,src,event));
      if ~isempty(instrVals)
        obj.useInstr=useInstrOrig;
      end
      if ~isempty(useCntrlVals)
        obj.useCntrl=cntrlOrig{1};
        obj.useCntrlChan=cntrlOrig{2};
      end
      dataList=obj.manualCalDataList;
      if ~isempty(dataList) && obj.calret
        set(obj.gui.settings_manualCalInfo,'Value',1) % select most recent
        set(obj.gui.settings_manualCalInfo,'String',dataList)
      end
    end
    function guiChooseGainMethod(obj,~,~)
      if get(obj.gui.settings_useGainCal,'Value')
        set(obj.gui.settings_pVal,'Enable','off')
        set(obj.gui.settings_iVal,'Enable','off')
        set(obj.gui.settings_dVal,'Enable','off')
        set(obj.gui.settings_gainVal,'Enable','on')
        set(obj.gui.settings_usePidCal,'Value',false)
      else
        set(obj.gui.settings_useGainCal,'Value',true)
      end
    end
    function guiChoosePidMethod(obj,~,~)
      if get(obj.gui.settings_usePidCal,'Value')
        set(obj.gui.settings_pVal,'Enable','on')
        set(obj.gui.settings_iVal,'Enable','on')
        set(obj.gui.settings_dVal,'Enable','on')
        set(obj.gui.settings_gainVal,'Enable','off')
        set(obj.gui.settings_useGainCal,'Value',false)
      else
        set(obj.gui.settings_usePidCal,'Value',true)
      end
    end
    function guiAcceptSettings(obj,src,~)
      try
        if src==obj.gui.settings_commit
          cn=get(obj.gui.settings_manualCalInfo,'String');
          if ~iscell(cn); cn={cn}; end;
          try
            obj.manualCalDataLoad(cn{get(obj.gui.settings_manualCalInfo,'Value')});
          catch ME
            errordlg(sprintf('Error loading specified calibration data or none yet taken:\n%s',ME.message),'Cal load error')
            return
          end
          if get(obj.gui.settings_useGainCal,'Value')
            obj.feedbackAlgorithm='Gain';
            obj.Gain=str2double(get(obj.gui.settings_gainVal,'String'));
          else
            obj.feedbackAlgorithm='PID';
            obj.PID=[str2double(get(obj.gui.settings_pVal,'String')) str2double(get(obj.gui.settings_iVal,'String')) ...
              str2double(get(obj.gui.settings_dVal,'String'))];
          end
          useCntrlVals=get(obj.gui.settings_corrList,'UserData');
          if ~isempty(useCntrlVals) % cntrls changed
            if length(useCntrlVals{1})==length(obj.useCntrl)
              obj.useCntrl=useCntrlVals{1};
            end
            if length(useCntrlVals{2})==length(obj.useCntrlChan)
              obj.useCntrlChan=useCntrlVals{2};
              obj.manualCalUseCntrlChan=useCntrlVals{2};
            end
          end
          if length(get(obj.gui.settings_bpmList,'UserData'))==length(obj.useInstr)
            obj.useInstr=get(obj.gui.settings_bpmList,'UserData');
          end
          obj.Name=get(obj.gui.settings_fbName,'String');
          set(obj.gui.main_panel,'Title',sprintf('%s (cal = %s)',obj.Name,obj.manualCalName))
        end
        delete(obj.src.settings)
      catch ME
        if strcmp(ME.message,'Cal load error')
          rethrow(lasterror)
        end
        try
          delete(obj.src.settings)
        catch
          delete(gcf)
        end
      end
    end
    function han=takeNewManualCal(obj,src,~)
      % Create main figure window
      obj.guiCreateFigure('takeNewManualCal','Manual FB Calibration',[625 400]);
      border=0.01;
      tsize=0.65;
      psize=[(1-5*border)/4 (1-tsize-3*border)/2];
      han=obj.gui.takeNewManualCal;
      % Make table for calibration settings
      colnames={'name' 'value' 'low' 'high' 'nsteps' 'do cal?'};
      colfmt={'char' 'numeric' 'numeric' 'numeric' 'numeric' 'logical'};
      colwid={200,'auto','auto','auto','auto','auto'};
      coledit=[false true true true true true];
      obj.guiCreateTable('takeNewManualCal_table',colnames,colfmt,colwid,coledit,...
        'takeNewManualCal',[border 1-tsize+3*border 1-border*2 tsize-4*border]);
      set(obj.gui.takeNewManualCal_table,'CellEditCallback',@(src,event)takeNewManualCal_tableEdit(obj,src,event));
      % other controls
      obj.guiCreatePanel('unitpanel','display units','takeNewManualCal',[border border*2+psize(2) psize]);
      rsize=(1-border*3)/2;
      obj.guiCreateRadiobutton('takeNewManualCal_dunits_setpt','SetPt',1,'unitpanel',[border 1-border-rsize 1-2*border rsize]);
      set(obj.gui.takeNewManualCal_dunits_setpt,'Callback',@(src,event)takeNewManualCal_dunits(obj,src,event));
      obj.guiCreateRadiobutton('takeNewManualCal_dunits_act','ACT',0,'unitpanel',[border border 1-2*border rsize])
      set(obj.gui.takeNewManualCal_dunits_act,'Callback',@(src,event)takeNewManualCal_dunits(obj,src,event))
      obj.guiCreatePanel('nbpmpanel','Nbpm ave.','takeNewManualCal',[border*2+psize(1) border*2+psize(2) psize]);
      obj.guiCreateEdit('takeNewManualCal_nbpm',num2str(obj.ndata),'nbpmpanel',[border border 1-2*border 1-2*border]);
      set(obj.gui.takeNewManualCal_nbpm,'Callback',@(src,event)takeNewManualCal_nbpm(obj,src,event)) ;
      obj.guiCreatePushbutton('takeNewManualCal_changeCuts','INSTR Cuts','takeNewManualCal',[border*3+psize(1)*2 border*2+psize(2) psize]);
      set(obj.gui.takeNewManualCal_changeCuts,'Callback',@(src,event)FlInstr_selectCuts(obj,src,event));
      obj.guiCreatePushbutton('takeNewManualCal_go','GO','takeNewManualCal',[border*4+psize(1)*3 border*2+psize(2) psize]);
      set(obj.gui.takeNewManualCal_go,'Callback',@(src,event)guiDoManualCal(obj,src,event));
      obj.guiCreatePanel('takeNewManualCal_namepanel','cal name','takeNewManualCal',[border border 1-2*border psize(2)]);
      if exist('src','var') && isfield(obj.gui,'settings_takeNewManualCal') && isequal(src,obj.gui.settings_takeNewManualCal)
        mname=datestr(now,30);
      else
        mname=obj.manualCalName;
      end
      obj.guiCreateEdit('takeNewManualCal_name',mname,'takeNewManualCal_namepanel',[border border 1-2*border 1-2*border]);
      set(obj.gui.takeNewManualCal_name,'Callback',@(src,event)takeNewManualCal_name(obj,src,event));
      set(obj.gui.takeNewManualCal_name,'Max',10);
      % Fill table data
      obj.takeNewManualCal_tableFill;
      % Init do cal checkboxes
      obj.manualCalUseCntrlChan=obj.useCntrlChan;
    end
    function guiDoManualCal(obj,~,~)
      obj.ndata=str2double(get(obj.gui.takeNewManualCal_nbpm,'String'));
      obj.manualCalName=get(obj.gui.takeNewManualCal_name,'String');
      obj.manualCal;
    end
    function fbDiagnostics(obj,~,~)
      % Create feedback diagnostics figure window
      obj.guiCreateFigure('fbDiagnostics',sprintf('%s Diagnostics',obj.Name),[1000 700]);
      border=0.02;
      usize=(1-3*border);
      axsize=usize*0.75;
      bsize=[(1-4*border)/3 0.2*usize];
      % axes
      obj.guiCreateAxes('fbDiagnostics_axes1','fbDiagnostics',[border*3 1-border-axsize 1-5*border axsize*0.95]);
      % plot select
      obj.guiCreateButtonGroup('fbDiagnostics_plotSel','Select Plot','fbDiagnostics',[border border bsize]);
      rsize=(1-4*border)/3;
      obj.guiCreateRadiobutton('fbDiagnostics_plotBpmsX','BPMs (x)',1,'fbDiagnostics_plotSel',[border border*3+rsize*2 1-2*border rsize]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotBpmsY','BPMs (y)',0,'fbDiagnostics_plotSel',[border border*2+rsize 1-2*border rsize]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotActs','Actuators',0,'fbDiagnostics_plotSel',[border border 1-2*border rsize]);
      % plot ranges
      obj.guiCreatePanel('fbDiagnostics_plotrangePanel','Plot ranges','fbDiagnostics',[border*2+bsize(1) border bsize]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotlinear','linear',1,'fbDiagnostics_plotrangePanel',...
        [border border (1-3*border)/2 (1-3*border)/2]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotlog','log',0,'fbDiagnostics_plotrangePanel',...
        [border border+(1-3*border)/2 (1-3*border)/2 (1-3*border)/2]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotval','value',1,'fbDiagnostics_plotrangePanel',...
        [1-border-(1-3*border)/2 border (1-3*border)/2 (1-3*border)/2]);
      obj.guiCreateRadiobutton('fbDiagnostics_plotnorm','normalised',0,'fbDiagnostics_plotrangePanel',...
        [1-border-(1-3*border)/2 border+(1-3*border)/2 (1-3*border)/2 (1-3*border)/2]);
      set(obj.gui.fbDiagnostics_plotlinear,'Callback',@(src,event)fbDiagnostics_prCallback(obj,src,event));
      set(obj.gui.fbDiagnostics_plotlog,'Callback',@(src,event)fbDiagnostics_prCallback(obj,src,event));
      set(obj.gui.fbDiagnostics_plotval,'Callback',@(src,event)fbDiagnostics_prCallback(obj,src,event));
      set(obj.gui.fbDiagnostics_plotnorm,'Callback',@(src,event)fbDiagnostics_prCallback(obj,src,event));
      % Menu
      calmenu=uimenu('Parent',obj.gui.fbDiagnostics,'Label','Cal');
      obj.gui.fbDiagnostics_menuShowCal=uimenu('Parent',calmenu,'Label','Show Cal Slopes','Callback',...
        @(src,event)guiCheckSlopes(obj,src,event));
      % Update
      rsize=(1-3*border)/2;
      obj.guiCreatePanel('fbDiagnostics_updatePanel','Time window / s','fbDiagnostics',[border*3+bsize(1)*2 border bsize]);
      obj.guiCreateTogglebutton('fbDiagnostics_update','Update','fbDiagnostics_updatePanel',[border border 1-2*border rsize]);
      obj.guiCreateEdit('fbDiagnostics_dt','120','fbDiagnostics_updatePanel',[border 1-border-rsize 1-2*border rsize]);
    end
  end
  
  %% Private methods
  methods(Access=private)
    function fbDiagnostics_update(obj,xbpms,xerr,ybpms,yerr,cvals,limLow,limHigh)
      % Fill data array
      diagdata=get(obj.gui.fbDiagnostics,'UserData');
      if isempty(diagdata) || ~iscell(diagdata)
        diagdata=[];
      else
        sz1=size(diagdata{1}); sz2=size(diagdata{3}); sz3=size(diagdata{5}); sz4=size(diagdata{6});
        if sz1(1)~=length(xbpms) || sz2(1)~=length(cvals) || sz3(1)~=length(xerr) || sz4(1)~=length(yerr); diagdata=[]; end;
      end
      if isempty(diagdata)
        diagdata{1}=xbpms;
        diagdata{2}=ybpms;
        diagdata{3}=cvals';
        diagdata{4}=now;
        diagdata{5}=xerr;
        diagdata{6}=yerr;
      else
        diagdata{1}=[diagdata{1} xbpms];
        diagdata{2}=[diagdata{2} ybpms];
        diagdata{3}=[diagdata{3} cvals'];
        diagdata{4}=[diagdata{4} now];
        diagdata{5}=[diagdata{5} xerr];
        diagdata{6}=[diagdata{6} yerr];
      end
      % Throw away data beyond time window
      dataAge=(now-diagdata{4}).*3600.*24;
      oldData=dataAge>str2double(get(obj.gui.fbDiagnostics_dt,'String'));
      if any(~oldData)
        for idata=[1:3 5 6]
          diagdata{idata}=diagdata{idata}(:,~oldData);
        end
        diagdata{4}=diagdata{4}(~oldData);
      else
        diagdata=[];
      end
      set(obj.gui.fbDiagnostics,'UserData',diagdata);
      % Asking for update?
      if ~get(obj.gui.fbDiagnostics_update,'Value')
        return
      end
      % make requested plot
      logplot=get(obj.gui.fbDiagnostics_plotlog,'Value');
      normplot=get(obj.gui.fbDiagnostics_plotnorm,'Value');
      timevals=(diagdata{4}-now).*3600.*24;
      if get(obj.gui.fbDiagnostics_plotBpmsX,'Value')
        if normplot
          data=diagdata{1}./max(abs(diagdata{1}(:)));
          dataerr=diagdata{5}./max(abs(diagdata{1}(:)));
        else
          data=diagdata{1};
          dataerr=diagdata{5};
        end
        if logplot
          set(obj.gui.fbDiagnostics_axes1,'YScale','log')
          if obj.ndata>1
            errorbar(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',abs(data)',dataerr');
          else
            plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',abs(data)')
          end
        else
          set(obj.gui.fbDiagnostics_axes1,'YScale','linear')
          if obj.ndata>1
            errorbar(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',data',dataerr');
          else
            plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',data')
          end
        end
      elseif get(obj.gui.fbDiagnostics_plotBpmsY,'Value')
        if normplot
          data=diagdata{2}./max(abs(diagdata{2}(:)));
          dataerr=diagdata{6}./max(abs(diagdata{2}(:)));
        else
          data=diagdata{2};
          dataerr=diagdata{6};
        end
        if logplot
          set(obj.gui.fbDiagnostics_axes1,'YScale','log')
          if obj.ndata>1
            errorbar(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',abs(data)',dataerr');
          else
            plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',abs(data)')
          end
        else
          set(obj.gui.fbDiagnostics_axes1,'YScale','linear')
          if obj.ndata>1
            errorbar(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',data',dataerr');
          else
            plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz1(1),1)',data')
          end
        end
      elseif get(obj.gui.fbDiagnostics_plotActs,'Value')
        if normplot
          for isz=1:sz2(1)
            data(isz,:)=diagdata{3}(isz,:)./abs(limHigh(isz)-limLow(isz));
          end
        else
          data=diagdata{3};
        end
        if logplot
          set(obj.gui.fbDiagnostics_axes1,'YScale','log')
          plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz2(1),1)',abs(data)');
        else
          set(obj.gui.fbDiagnostics_axes1,'YScale','linear')
          plot(obj.gui.fbDiagnostics_axes1,repmat(timevals,sz2(1),1)',data');
        end
      end
      grid(obj.gui.fbDiagnostics_axes1,'on')
      drawnow('expose')
    end
    function takeNewManualCal_tableFill(obj)
      % Get data for table
      irow=0;
      if ~get(obj.gui.takeNewManualCal_dunits_setpt,'Value')
        sp=obj.Ampl2ACT;
        act=obj.ACT;
      end
      for indx=1:length(obj.INDXnames)
        if obj.useCntrl(indx)
          for ichan=1:length(obj.INDXchannels(indx))
            if obj.useCntrlChan{indx}(ichan)
              irow=irow+1;
              if obj.MasterRef(indx,1)
                channames={'main'};
              elseif obj.MasterRef(indx,2)
                channames={'x' 'dx' 'y' 'dy' 'z' 'dz'};
              elseif obj.MasterRef(indx,3)
                channames={'ampl' 'phase'};
              end
              tableData{irow,1}=sprintf('%s::%s',obj.INDXnames{indx},channames{ichan});
              if get(obj.gui.takeNewManualCal_dunits_setpt,'Value')
                tableData{irow,2}=obj.Ampl{indx}(ichan);
                tableData{irow,3}=obj.manualCalStepsLow{indx}(ichan);
                tableData{irow,4}=obj.manualCalStepsHigh{indx}(ichan);
              else
                if iscell(act{indx}(ichan))
                  act{indx}(ichan)=cell2mat(act{indx}(ichan));
                end
                tableData{irow,2}=act{indx}(ichan);
                if iscell(obj.manualCalStepsLow{indx}(ichan))
                  ml=cell2mat(obj.manualCalStepsLow{indx}(ichan)).*sp{indx}(ichan);
                  mh=cell2mat(obj.manualCalStepsHigh{indx}(ichan)).*sp{indx}(ichan);
                else
                  ml=obj.manualCalStepsLow{indx}(ichan).*sp{indx}(ichan);
                  mh=obj.manualCalStepsHigh{indx}(ichan).*sp{indx}(ichan);
                end
                tableData{irow,3}=ml;
                tableData{irow,4}=mh;
              end
              tableData{irow,5}=obj.manualCalNsteps{indx}(ichan);
              tableData{irow,6}=true;
              tableIndex{irow}=[indx ichan];
            end
          end
        end
      end
      set(obj.gui.takeNewManualCal_table,'Data',tableData)
      set(obj.gui.takeNewManualCal_table,'UserData',tableIndex)
    end
  end
end