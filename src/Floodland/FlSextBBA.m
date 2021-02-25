classdef FlSextBBA < handle & FlApp & FlGui & matlab.mixin.Copyable
  %FLSEXTBBA Sextupole BBA class + GUI
  %   Perform BBA of any sextupole with a mover using downstream BPMs
  
  properties
    bpms % List of BPMs
    sextSelect % index into obj.sextChoice
    sextDimSelect='x';
    nMoverSteps=10;
    moverRange=[-2e-3 2e-3];
    bpmNAve % Number of BPM readings to average with analysis
    bpmRelTmitCut=2; % sigma
  end
  properties(Hidden)
    moverSpeed={}; % speed of mover (m/s) length of sextChoice * 2 (x & y)]
    movingPV={}; % PVs to inform of if sextupole is moving
    sextAllPS
    sextAllInitBDES
    sextChoice % Allowable sextupoles (FlIndex GIRDER list)
    sextName={};
    sextPS % FlIndex PS index
    sextBpm % Closest BPM to each sextupole (INSTR index)
    bpmChoice % Allowable BPMs (INSTR object list)
    bpmName={};
    bpmIgnore={}; % list of BPM names to ignore
    FL % pointer to Floodland object
    INSTR % pointer to INSTR object
    FLI % pointer to FlIndex object
    data % internal data structure
    status % current status
    runTimer % main BBA running timer
    simModel
  end
  properties(Dependent)
    sextBDES
    sextID % display name for this sextupole:dimension
  end
  properties(Hidden) % GUI properties
    guiDetailPanel='None';
  end
  properties(Constant)
    appName='Sextupole BBA'; % Application name to appear in menus
    guiDetailPanelChoice={'None' 'Analysis' 'Mover Control' 'BPM' 'SEXT PS' 'Messages'};
    statusChoice={'OK' 'Error'};
    defaultMoverRange={[-2e-3 2e-3] [-1.4e-3 1.4e-3] [-5e-3 5e-3]};
  end
  
  methods % constructor and get/set methods
    function obj=FlSextBBA(FL,instObj,indxObj,simModel,glist)
      % FlSextBBA(FL,instObj,indxObj,simModel [,glist])
      % FL/instObj/indxObj : required Floodland, FlInstr and FlIndex
      %                      objects
      % simModel : Model parameters from lattice file
      % glist : optional list of GIRDER pointer to sextupoles to restrict
      %         list of possible sextupole to use with this object
      global BEAMLINE GIRDER
      % Need to pass FlInstr object
      if ~exist('instObj','var') || ~strcmp(class(instObj),'FlInstr')
        error('Must pass FlInstr object as second argument');
      end
      % Need to pass FlIndex object
      if ~exist('indxObj','var') || ~strcmp(class(indxObj),'FlIndex')
        error('Must pass FlIndex object as third argument');
      end
      % Need to pass Floodland object
      if ~exist('FL','var') || ~strcmp(class(FL),'Floodland')
        error('Must pass Floodland object as first argument');
      end
      % Need to pass Model structure
      if ~exist('simModel','var')
        error('Must pass Model structure');
      end
      obj.FL=FL;
      obj.INSTR=instObj;
      obj.FLI=indxObj;
      obj.simModel=simModel;
      % Find list of possible sextupoles and BPMs
      girs=obj.FLI.GIRDER;
      if exist('glist','var') && ~isempty(glist)
        girs=girs(ismember(girs,glist));
      end
      if isempty(girs)
        error('No GIRDER present in FlIndex object (that is present in glist if provided)')
      end
      for igir=1:length(girs)
        issext=false;
        speed=GIRDER{girs(igir)}.velo;
        for iele=GIRDER{girs(igir)}.Element(1):GIRDER{girs(igir)}.Element(end)
          if strcmp(BEAMLINE{iele}.Class,'SEXT')
            issext=true;
            sname=BEAMLINE{iele}.Name;
            sps=BEAMLINE{iele}.PS;
            % Find closest BPM to associate with this sextupole
            [~, sbpm]=min(abs(BEAMLINE{iele}.S-arrayfun(@(x) BEAMLINE{x}.S,obj.INSTR.Index)));
            break
          end
        end
        if issext
          if ~isempty(GIRDER{girs(igir)}.pvname)
            obj.sextChoice(end+1)=igir;
            obj.sextPS(end+1)=find(ismember(obj.FLI.PS,sps),1);
            obj.sextName{end+1}=sname;
            obj.sextBpm(end+1)=sbpm;
            if obj.FL.issim
              obj.moverSpeed{end+1}=[1 1];
            else
              obj.moverSpeed{end+1}=speed([1 3]);
            end
          end
        end
      end
      if isempty(obj.sextChoice)
        error('No Sextupoles in FlIndex object that has a hardware link this application can use')
      end
      % All sextupoles
      obj.sextAllPS=find(ismember(obj.FLI.PS,unique(arrayfun(@(x) BEAMLINE{x}.PS,findcells(BEAMLINE,'Class','SEXT')))));
      obj.sextAllInitBDES=obj.FLI.PS_B(obj.sextAllPS);
      % default Nbpm
      obj.bpmNAve=10;
      % setup first sextupole as default (also triggers BPM selection)
      obj.sextSelect=1;
      % Initialise message box data buffer
      obj.guiCreateMessagebox;
      % make timer object
      obj.runTimer=timer('Tag','sextBBA','TimerFcn',@(src,event)runTimerFn(obj,src,event,'run'),'BusyMode','drop',...
        'ExecutionMode','fixedSpacing','Period',1,'StartFcn',@(src,event)runTimerFn(obj,src,event,'start'),...
        'StopFcn',@(src,event)runTimerFn(obj,src,event,'stop'));
    end
    function set.sextSelect(obj,val)
      global BEAMLINE
      if val<1 || val>length(obj.sextChoice)
        error('Selection out of range')
      end
      obj.sextSelect=val;
      try
        obj.bpmChoice=find(ismember(obj.INSTR.Class,'MONI')&(obj.INSTR.Index>max(BEAMLINE{obj.FLI.GIRDER_indx(obj.sextChoice(val))}.Block))&...
          cellfun(@(x) ~isempty(x),obj.INSTR.pvname(:))');
        if isempty(obj.bpmChoice)
          error('No downstream BPMs present from any possible sextupoles')
        end
        obj.bpmName=arrayfun(@(x) BEAMLINE{x}.Name,obj.INSTR.Index(obj.bpmChoice),'UniformOutput',false);
      catch ME
        obj.bpmChoice=[];
        obj.bpmName=[];
        error('error selecting BPMs for this Sextupole: %s',ME.message)
      end
      if isfield(obj.gui,'bbaAxes')
        cla(obj.gui.bbaAxes)
        obj.guiPlotSelCallback;
      end
      if strcmp(obj.guiDetailPanel,'Analysis')
        obj.doAnal;
      end
      if isfield(obj.gui,'plotSel')
        set(obj.gui.plotSel,'String',{'All' obj.bpmName{~ismember(obj.bpmName,obj.bpmIgnore)}})
      end
    end
    function set.guiDetailPanel(obj,val)
      if ismember(val,obj.guiDetailPanelChoice)
        % redisplay GUI if choice changes
        if ~strcmp(val,obj.guiDetailPanel)
          obj.guiDetailPanel=val;
          obj.guiMain; % redisplay GUI
        end
      else
        error('Available Detail Panel choices are restricted to: %s',obj.guiDetailPanelChoice)
      end
    end
    function set.sextDimSelect(obj,val)
      if ~ismember(val,{'x' 'y'})
        error('Must select ''x'' or ''y''')
      end
      obj.sextDimSelect=val;
      dims='xy';
      obj.moverRange=obj.defaultMoverRange{dims==obj.sextDimSelect};
      if strcmp(obj.guiDetailPanel,'Mover Control')
        set(obj.gui.moverRangeLow,'String',obj.moverRange(1)*1e3)
        set(obj.gui.moverRangeHigh,'String',obj.moverRange(2)*1e3)
      end
    end
    function set.moverRange(obj,val)
      try
        if length(val)~=2
          error('moverRange must be supplied as 2 element vector')
        end
        dims='xy';
        if val(1)<obj.defaultMoverRange{dims==obj.sextDimSelect}(1) || ...
            val(2)>obj.defaultMoverRange{dims==obj.sextDimSelect}(2)
          error('Range outside max permissible by mover system')
        end
        % Check dynamic range
        gpos=obj.FLI.GIRDER_POS;
        gpos=gpos(obj.sextSelect,:);
        for ival=val
          if obj.sextDimSelect=='x'
            gpos(1)=ival;
          else
            gpos(3)=ival;
          end
          camangles=obj.camCheck(gpos(1),gpos(3),gpos(6));
          if any(isnan(camangles))
            error('Range outside max permissible by mover system')
          end
        end
        obj.moverRange=val;
      catch ME
        obj.guiUpdateStatusDisplay('statusDisplay',-1);
        obj.guiAddToMessagebox('Error setting mover range:');
        obj.guiAddToMessagebox(ME.message);
      end
    end
    function set.bpmNAve(obj,val)
      if val>0
        obj.bpmNAve=val;
        obj.INSTR.ndata=val;
      end
    end
    function set.nMoverSteps(obj,val)
      if val>=3
        obj.nMoverSteps=val;
      end
    end
    function sextID=get.sextID(obj)
      sStr=obj.sextName{obj.sextSelect};
      dimStr=obj.sextDimSelect;
      sextID=[sStr dimStr];
    end
  end
  methods(Access=private) % run timer and BBA process functions
    function ret=runTimerFn(obj,~,~,cmd)
      persistent state moveSteps moveTime t0 pos1 posInit
      ret=[];
      switch cmd
        case 'getdata'
          ret={state moveSteps moveTime t0 pos1 posInit};
        case 'start'
          state=1;
          obj.FL.hwGet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
          obj.guiUpdateStatusDisplay('statusDisplay',2);
          obj.guiAddToMessagebox(sprintf('Starting BBA procedure (%s)',obj.sextName{obj.sextSelect}));
          moveSteps=linspace(obj.moverRange(1),obj.moverRange(2),obj.nMoverSteps); moveSteps=moveSteps(randperm(length(moveSteps)));
          t0=clock;
          moveTime=-1;
          pos1=obj.FLI.GIRDER_POS; posInit=pos1;
          % reset data structures
          obj.data.(obj.sextID)=[];
          cla(obj.gui.bbaAxes);
          title(obj.gui.bbaAxes,'');
          xlabel(obj.gui.bbaAxes,'');
          ylabel(obj.gui.bbaAxes,'');
          % Check range is good
          try
            obj.moverRange=obj.moverRange;
          catch
            state=-1;
            obj.guiAddToMessagebox('Mover range will put mover outside limits during scan');
            stop(obj.runTimer)
          end
        case 'stop'
          obj.FL.hwGet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
          % Put movers back
          obj.FLI.GIRDER_POS=posInit;
          obj.guiAddToMessagebox(sprintf('Moving Sextpole back to init position: %.3f / %.3f (mm)',posInit(obj.sextChoice(obj.sextSelect),[1 3])));
          obj.FL.hwSet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
          % Update status display and messages
          if state>length(moveSteps)
            obj.guiUpdateStatusDisplay('statusDisplay',1);
            obj.guiAddToMessagebox('Finished BBA');
          elseif state<0
            obj.guiUpdateStatusDisplay('statusDisplay',-1);
          else
            obj.guiUpdateStatusDisplay('statusDisplay',3);
            obj.guiAddToMessagebox('User requested STOP');
          end
        case 'run'
          try
            if state>length(moveSteps)
              stop(obj.runTimer);
              return
            end
            obj.FL.hwGet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
            if state<=length(moveSteps) && moveTime<0
              set(obj.gui.statusDisplay,'String',sprintf('RUN (Step %d of %d)',state,length(moveSteps)))
              % move to next position and set move timer
              gpos=obj.FLI.GIRDER_POS;
              if obj.sextDimSelect=='x'
                gpos(obj.sextChoice(obj.sextSelect),1)=moveSteps(state);
                moveTime=abs(pos1(obj.sextChoice(obj.sextSelect),1)-moveSteps(state))/obj.moverSpeed{obj.sextSelect}(1);
              else
                gpos(obj.sextChoice(obj.sextSelect),3)=moveSteps(state);
                moveTime=abs(pos1(obj.sextChoice(obj.sextSelect),3)-moveSteps(state))/obj.moverSpeed{obj.sextSelect}(2);
              end
              obj.guiAddToMessagebox(sprintf('Moving Sextpole to: %.3f mm [wait %.1f s]',moveSteps(state)*1e3,moveTime));
              pos1=gpos;
              t0=clock;
              obj.FLI.GIRDER_POS=gpos;
              obj.FL.hwSet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
            elseif etime(clock,t0)>moveTime && ~obj.isMoving
              obj.takeBpmData(state);
              % Take data and update analysis once have more than 3 rows of data
              if state>=3
                obj.guiPlotSelCallback;
                if strcmp(obj.guiDetailPanel,'Analysis')
                  obj.guiFillAnalData;
                end
              end
              state=state+1;
              moveTime=-1;
            end
          catch ME
            obj.guiAddToMessagebox(sprintf('Error: (Step %d) Code error: %s',state,ME.message));
            obj.guiUpdateStatusDisplay('statusDisplay',-1);
            state=-1;
            stop(obj.runTimer)
          end
        case 'getstate'
          ret=state;
      end
    end
    function takeBpmData(obj,ival)
      % Take the data and fill local structures
      % - try 3 times
      sname=obj.sextID;
      nanCheck=false; ntry=0;
      ret=obj.runTimerFn([],[],'getdata');
      obj.data.(sname).scanvals.(obj.sextDimSelect)=ret{2};
      obj.data.(sname).name=obj.bpmName;
      while ~nanCheck && ntry<3
        ntry=ntry+1;
        obj.INSTR.acquire(obj.FL,obj.bpmNAve);
        [md, sd]=obj.INSTR.meanstd;
        obj.data.(sname).date=now;
        obj.data.(sname).x(:,ival)=md(obj.bpmChoice,1);
        nanCheck=true;
        if all(isnan(obj.data.(sname).x(:,ival))); nanCheck=false;end;
        obj.data.(sname).x_err(:,ival)=sd(obj.bpmChoice,1);
        if all(isnan(obj.data.(sname).x_err(:,ival))); nanCheck=false;end;
        obj.data.(sname).y(:,ival)=md(obj.bpmChoice,2);
        if all(isnan(obj.data.(sname).y(:,ival))); nanCheck=false;end;
        obj.data.(sname).y_err(:,ival)=sd(obj.bpmChoice,2);
        if all(isnan(obj.data.(sname).y_err(:,ival))); nanCheck=false;end;
        obj.data.(sname).t(:,ival)=md(obj.bpmChoice,3);
        obj.data.(sname).t_err(:,ival)=sd(obj.bpmChoice,3);
      end
      if ~nanCheck
        error('No BPM data passed cuts, check BPM selection')
      end
    end
    function doAnal(obj)
      try
        sname=obj.sextID;
        sz=size(obj.data.(sname).x);
        [scanvals, bI]=sort(obj.data.(sname).scanvals.(obj.sextDimSelect)(1:sz(2)));
        % Which bpms to analyse for?
        str=get(obj.gui.plotSel,'String'); thisBpmName=str{get(obj.gui.plotSel,'Value')};
        for ibpm=1:length(obj.data.(sname).name)
          [~, ~, chi2]=noplot_polyfit(scanvals,obj.data.(sname).x(ibpm,bI),obj.data.(sname).x_err(ibpm,bI),2);
          [q, dq]=noplot_parab(scanvals,obj.data.(sname).x(ibpm,bI),obj.data.(sname).x_err(ibpm,bI)*sqrt(chi2));
          obj.data.(sname).anal.fit_A(ibpm)=q(1);
          obj.data.(sname).anal.fit_B(ibpm)=q(2);
          obj.data.(sname).anal.fit_C(ibpm)=q(3);
          obj.data.(sname).anal.fit_A_err(ibpm)=dq(1);
          obj.data.(sname).anal.fit_B_err(ibpm)=dq(2);
          obj.data.(sname).anal.fit_C_err(ibpm)=dq(3);
        end
        if strcmp(thisBpmName,'All')
          usebpm=~ismember(obj.data.(sname).name,obj.bpmIgnore);
        else
          usebpm=ismember(obj.data.(sname).name,thisBpmName);
        end
        if sum(usebpm)>1
          [q, dq]=noplot_polyfit(1:sum(usebpm),obj.data.(sname).anal.fit_B(usebpm),obj.data.(sname).anal.fit_B_err(usebpm),0);
          obj.data.(sname).anal.BBA.(obj.sextDimSelect)=q;
          obj.data.(sname).anal.BBA.(sprintf('%c_err',obj.sextDimSelect))=dq;
        else
          obj.data.(sname).anal.BBA.(obj.sextDimSelect)=obj.data.(sname).anal.fit_B(usebpm);
          obj.data.(sname).anal.BBA.(sprintf('%c_err',obj.sextDimSelect))=obj.data.(sname).anal.fit_B_err(usebpm);
        end
      catch
      end
      if strcmp(obj.guiDetailPanel,'Analysis')
        obj.guiFillAnalData;
      end
    end
  end
  methods % GUI
    function han=guiMain(obj)
      % Main graphical user interface for this object
      % Border size, dimension variables etc
      border=0.02;
      figW=1-2*border;
      if strcmp(obj.guiDetailPanel,'None')
        psize=500;
        sH=(1-border*3)/2;
        S2=[border border];
        figH=(1-border*3)/2;
        figHOff=0;
      else
        figH=(1-border*4)/3;
        figHOff=0;
        psize=900;
        sH=(1-border*4)/3;
        S2=[border 1-border*2-figH-sH];
      end
      S1=[border 1-border-figH];
      S3=[border border];
      
      % Create main figure window
      redisp=obj.guiCreateFigure('guiMain','Sextupole BBA',[750 psize]);
      if redisp; return; end;
      han=obj.gui.guiMain;
      
      % ====================
      % ---  fig window  ---
      % ====================
      obj.guiCreateAxes('bbaAxes','guiMain',[S1+[0.05 figHOff-0.03] figW-0.05 figH])
      
      % =====================
      % --- main controls ---
      % =====================
      mcH=(sH-2*border)/3; mcH=mcH./1.6;
      mcW=(1-4*border)/3;
      % Plot Select
      obj.guiCreatePanel('plotSelPanel','Plot/analysis Select','guiMain',[S2+[0 2*border+2*mcH] mcW mcH]);
      obj.guiCreatePopupmenu('plotSel',{'All' obj.bpmName{~ismember(obj.bpmName,obj.bpmIgnore)}},'plotSelPanel',[border border 1-border*2 1-border*2]);
      set(obj.gui.plotSel,'Callback',@(src,event)guiPlotSelCallback(obj,src,event));
      % Sextupole Select
      obj.guiCreatePanel('sextSelPanel','Sextupole Select','guiMain',[S2+[0 border+mcH] mcW mcH]);
      obj.guiCreatePopupmenu('sextSel',obj.sextName,'sextSelPanel',[border border 0.8-border*3 1-border*2]);
      set(obj.gui.sextSel,'Callback',@(src,event)guiSextSelCallback(obj,src,event));
      obj.guiCreatePopupmenu('sextDimSel',{'x' 'y'},'sextSelPanel',[1-border-0.2 border 0.2-border*3 1-border*2]);
      set(obj.gui.sextDimSel,'Callback',@(src,event)guiSextSelCallback(obj,src,event));
      set(obj.gui.sextDimSel,'Value',find(ismember({'x' 'y'},obj.sextDimSelect)))
      set(obj.gui.sextSel,'Value',obj.sextSelect)
      obj.guiSextSelCallback(obj.gui.sextSel,[]);
      
      % Plot all button
      obj.guiCreatePushbutton('displayAllPlots','Plot All','guiMain',[S2 mcW mcH]);
      set(obj.gui.displayAllPlots,'Callback',@(src,event)guiDisplayAllPlotsCallback(obj,src,event));
      % Status display
      obj.guiCreatePanel('statusDisplayPanel','Status','guiMain',[S2+[border+mcW border*2+mcH*2] mcW mcH]);
      obj.guiCreateStatusDisplay('statusDisplay','statusDisplayPanel',[border*2 border*4 1-border*4 1-border*4]);
      set(obj.gui.statusDisplay,'ButtonDownFcn',@(src,event)guiStatusDisplayButDownFn(obj,src,event))
      % Start button
      obj.guiCreatePushbutton('startBBA','Start BBA','guiMain',[S2+[border+mcW border+mcH] mcW mcH]);
      set(obj.gui.startBBA,'Callback',@(src,event)guiStartBBACallback(obj,src,event));
      % Stop button
      obj.guiCreatePushbutton('stopBBA','STOP','guiMain',[S2+[border+mcW 0] mcW mcH]);
      set(obj.gui.stopBBA,'Callback',@(src,event)guiStopBBACallback(obj,src,event));
      % Sext BPM display
      obj.guiCreatePanel('sextBPMDisplayPanel','Sext. BPM / mm','guiMain',[S2+[border*2+mcW*2 border*2+mcH*2] mcW mcH]);
      obj.guiCreateReadbackText('sextBPMDisplay','sextBPMDisplayPanel','guiUpdateFn','bpm',20,[border border*3 1-border*2 1-border*4]);
      % Move to Offset button
      obj.guiCreatePushbutton('moveSextToBBA','Move to Offset','guiMain',[S2+[border*2+mcW*2 border+mcH] mcW mcH]);
      set(obj.gui.moveSextToBBA,'Callback',@(src,event)guiMoveSextToBBACallback(obj,src,event));
      % Commit BBA button
      obj.guiCreatePushbutton('commitBBA','CommitBBA','guiMain',[S2+[border*2+mcW*2 0] mcW mcH]);
      set(obj.gui.commitBBA,'Callback',@(src,event)guiCommitBBACallback(obj,src,event));
      
      % ========================
      % --- Analysis display ---
      % ========================
      if strcmp(obj.guiDetailPanel,'Analysis')
        tw=0.8;
        mcH=(sH-2*border)/3;
        % Analysis table
        colnames={'BPM' 'Center (B)' 'Center Err (dB)' 'A' 'C' ''};
        colfmt={'char' 'numeric' 'numeric' 'numeric' 'numeric' 'logical'};
        colwid={'auto','auto','auto','auto','auto','auto'};
        coledit=[false false false false false true];
        obj.guiCreateTable('analTable',colnames,colfmt,colwid,coledit,...
          'guiMain',[S3 tw-border*2 sH]);
        % Current BBA data
        obj.guiCreatePanel('analCurrentBBADataPanel','BBA Data / mm','guiMain',[S3+[tw-border*1.5 border*2+mcH*2] (1-tw) (mcH-border)/1.5]);
        obj.guiCreateReadbackText('analCurrentBBAData','analCurrentBBADataPanel','guiUpdateFn','bba',10,[border border*4 1-border*3 1-border*4]);
        % New calc data
        obj.guiCreatePanel('analNewBBADataPanel','BBA Calc / mm','guiMain',[S3+[tw-border*1.5 border+mcH*1.5] (1-tw) (mcH-border)/1.5]);
        obj.guiCreateReadbackText('analNewBBAData','analNewBBADataPanel','guiUpdateFn','bbaNew',10,[border border*4 1-border*3 1-border*4]);
        % Remove BPM button
        obj.guiCreatePushbutton('analRemoveBPM','Remove BPM','guiMain',[S3+[tw-border*1.5 0] (1-tw) (mcH-border)/1.5]);
        set(obj.gui.analRemoveBPM,'Callback',@(src,event)guiRemoveBPMCallback(obj,src,event));
        % Select BPM button
        obj.guiCreatePushbutton('analSelBPM','Select BPM','guiMain',[S3+[tw-border*1.5 mcH/1.5-border/2] (1-tw) (mcH-border)/1.5]);
        set(obj.gui.analSelBPM,'Callback',@(src,event)guiSelBPMCallback(obj,src,event));
        % Fill data fields if the data exists
        if isfield(obj.data,obj.sextID) && isfield(obj.data.(obj.sextID),'anal')
          obj.guiFillAnalData;
        end
      end
      
      % =============================
      % --- Mover Control display ---
      % =============================
      if strcmp(obj.guiDetailPanel,'Mover Control')
        tH=(sH-2*border)/3;
        tW=0.75-border*3;
        % Mover steps
        obj.guiCreatePanel('moverNumStepsPanel','N Steps','guiMain',[S3+[0 border*2+tH*2] 1-tW-border tH*0.6]);
        obj.guiCreateEdit('moverNumSteps',num2str(obj.nMoverSteps),'moverNumStepsPanel',[border border*3 1-border*3 1-border*4]);
        set(obj.gui.moverNumSteps,'Callback',@(src,event)guiMoverNumStepsCallback(obj,src,event));
        % Mover Range
        obj.guiCreatePanel('moverRangePanel','Mover Range / mm','guiMain',[S3+[0 border+tH] 1-tW-border tH*0.6]);
        obj.guiCreateEdit('moverRangeLow',num2str(obj.moverRange(1)*1e3,4),'moverRangePanel',[border border*3 0.45-border*3 1-border*4]);
        set(obj.gui.moverRangeLow,'Callback',@(src,event)guiMoverRangeCallback(obj,src,event));
        obj.guiCreateEdit('moverRangeHigh',num2str(obj.moverRange(2)*1e3,4),'moverRangePanel',[1-2*border-0.45 border*3 0.45-border*3 1-border*4]);
        set(obj.gui.moverRangeHigh,'Callback',@(src,event)guiMoverRangeCallback(obj,src,event));
        % Mover Set
        obj.guiCreatePanel('moverSetPanel','Set [x / y / tilt] (mm/mrad)','guiMain',[S3+[1-tW border*2+tH*2] tW-border*2 tH*0.6]);
        xW=0.25-border*2;
        gpos=obj.FLI.GIRDER_POS;
        gpos=gpos(obj.sextChoice(obj.sextSelect),:);
        obj.guiCreateEdit('moverSetX',num2str(gpos(1)*1e3,4),'moverSetPanel',[border border*3 xW 1-border*4]);
        obj.guiCreateEdit('moverSetY',num2str(gpos(3)*1e3,4),'moverSetPanel',[border*2+xW border*3 xW 1-border*4]);
        obj.guiCreateEdit('moverSetT',num2str(gpos(6)*1e3,4),'moverSetPanel',[border*3+xW*2 border*3 xW 1-border*4]);
        obj.guiCreatePushbutton('moverMoveCmd','Move','moverSetPanel',[1-xW-border border*3 xW 1-border*4]);
        set(obj.gui.moverMoveCmd,'Callback',@(src,event)guiMoverMoveCmdCallback(obj,src,event));
        % Mover Readback
        obj.guiCreatePanel('moverReadbackPanel','Readback [x / y / tilt] (mm/mrad)','guiMain',[S3+[1-tW border+tH] tW-border*2 tH*0.6]);
        xW=(1-border*4)/3;
        obj.guiCreateReadbackText('moverReadbackX','moverReadbackPanel','guiUpdateFn','moverX',10,[border border*3 xW 1-border*6]);
        obj.guiCreateReadbackText('moverReadbackY','moverReadbackPanel','guiUpdateFn','moverY',10,[border*2+xW border*3 xW 1-border*6]);
        obj.guiCreateReadbackText('moverReadbackT','moverReadbackPanel','guiUpdateFn','moverT',10,[border*3+xW*2 border*3 xW 1-border*6]);
      end
      
      % ===================
      % --- BPM display ---
      % ===================
      if strcmp(obj.guiDetailPanel,'BPM')
        tw=0.8;
        mcH=(sH-2*border)/3;
        % BPM data table
        colnames={'BPM' 'X' 'Xerr' 'Y' 'Yerr' 'TMIT' 'TMITerr' ''};
        colfmt={'char' 'char' 'char' 'char' 'char' 'char' 'char' 'logical'};
        colwid={'auto','auto','auto','auto','auto','auto','auto','auto'};
        coledit=[false false false false false false false true];
        obj.guiCreateReadbackTable('bpmTable',colnames,colfmt,colwid,coledit,...
          'guiMain','guiUpdateFn','bpmtable',5,[S3 tw-border*2 sH]);
        % N ave
        obj.guiCreatePanel('bpmNAvePanel','N ave.','guiMain',[S3+[tw border*2+mcH*2] (1-tw)-border*3 mcH/2]);
        obj.guiCreateEdit('bpmNAve',num2str(obj.bpmNAve),'bpmNAvePanel',[border border*3 1-border*2 1-border*5]);
        set(obj.gui.bpmNAve,'Callback',@(src,event)guiBpmNAveCallback(obj,src,event));
        % Select BPM button
        obj.guiCreatePushbutton('bpmSelBPM','Select BPM','guiMain',[S3+[tw-border*1.5 mcH/1.5-border/2] (1-tw) (mcH-border)/1.5]);
        set(obj.gui.bpmSelBPM,'Callback',@(src,event)guiSelBPMCallback(obj,src,event));
        % Remove BPM button
        obj.guiCreatePushbutton('bpmRemoveBPM','Remove Selected','guiMain',[S3+[tw-border*1.5 0] (1-tw) (mcH-border)/1.5]);
        set(obj.gui.bpmRemoveBPM,'Callback',@(src,event)guiRemoveBPMCallback(obj,src,event));
      end
      
      % =======================
      % --- SEXT PS display ---
      % =======================
      if strcmp(obj.guiDetailPanel,'SEXT PS')
        mcH=(sH-2*border)/3;
        tw1=0.6666-3*border;
        tw2=0.3333-3*border;
        % Sextupole selction
        obj.guiCreatePanel('psSextSelPanel','Sextupole [BDES]','guiMain',[S3 tw1 sH]);
        obj.guiCreateListbox('psSextSel',obj.FLI.PS_names(obj.sextAllPS),'psSextSelPanel',[border*2 border*2 1-border*4 1-border*4]);
        set(obj.gui.psSextSel,'Value',find(ismember(obj.FLI.PS_names(obj.sextAllPS),obj.sextName{obj.sextSelect}))) % make current sextupole selected
        % Sextupole BDES
        obj.guiCreatePanel('psSextBDESPanel','BDES','guiMain',[S3+[border+tw1 border*2+mcH*2] tw2 mcH/1.5]);
        obj.guiCreateReadbackText('psSextBDES','psSextBDESPanel','guiUpdateFn','sbdes',4,[border border*3 1-border*2 1-border*5]);
        % Set BDES
        obj.guiCreatePanel('psSextSetBDESPanel','Set BDES','guiMain',[S3+[border+tw1 border+mcH] tw2 mcH/1.5]);
        obj.guiCreateEdit('psSextSetBDESVal','0','psSextSetBDESPanel',[border border*3 0.75-border*3 1-border*6]);
        obj.guiCreatePushbutton('psSextSetBDES','Set','psSextSetBDESPanel',[border+0.75 border*3 0.25-border*3 1-border*6]);
        set(obj.gui.psSextSetBDES,'Callback',@(src,event)guiPsSextSetBDESCallback(obj,src,event));
        % Sextupole Power
        obj.guiCreatePanel('psSextPowerPanel','Power Supply','guiMain',[S3+[border+tw1 0] tw2 mcH]);
        obj.guiCreatePushbutton('psSextPowerOn','ON','psSextPowerPanel',[border border*3 0.5-2*border 1-2*border*6]);
        set(obj.gui.psSextPowerOn,'Callback',@(src,event)guiPsSextPowerCallback(obj,src,event));
        obj.guiCreatePushbutton('psSextPowerOff','OFF','psSextPowerPanel',[border+(0.5-border) border*3 0.5-2*border 1-2*border*6]);
        set(obj.gui.psSextPowerOff,'Callback',@(src,event)guiPsSextPowerCallback(obj,src,event));
      end
      
      % =======================
      % --- Message display ---
      % =======================
      if strcmp(obj.guiDetailPanel,'Messages')
        obj.guiCreatePanel('messagePanel','Mesasges','guiMain',[S3 1-2*border sH]);
        obj.guiCreateMessagebox('messagePanel',[border border 1-2*border 1-2*border]);
      end
      
      % =============
      % --- Menus ---
      % =============
      % File
      obj.gui.menuFile=uimenu('Parent',obj.gui.guiMain,'Label','File');
      obj.gui.menuSaveData=uimenu('Parent',obj.gui.menuFile,'Label','Save Data','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
      obj.gui.menuLoadData=uimenu('Parent',obj.gui.menuFile,'Label','Load Data','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
%       obj.gui.menuSaveSettings=uimenu('Parent',obj.gui.menuFile,'Label','Save Settings','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
      obj.gui.menuSaveToLog=uimenu('Parent',obj.gui.menuFile,'Label','Save To Logbook','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
      % Edit
      obj.gui.menuEdit=uimenu('Parent',obj.gui.guiMain,'Label','Edit');
      obj.gui.menuUpdateRate=uimenu('Parent',obj.gui.menuEdit,'Label','GUI Update Rate (Hz)');
      for ir=1:length(obj.guiAvailableUpdateRates)
        obj.gui.(sprintf('menuUpdateRate%d',ir))=...
          uimenu('Parent',obj.gui.menuUpdateRate,'Label',sprintf('%.1f',obj.guiAvailableUpdateRates(ir)),'Callback',@(src,event)guiMenuEditCallback(obj,src,event));
        if obj.guiAvailableUpdateRates(ir)==obj.guiUpdateRate
          set(obj.gui.(sprintf('menuUpdateRate%d',ir)),'Checked','on')
        else
          set(obj.gui.(sprintf('menuUpdateRate%d',ir)),'Checked','off')
        end
      end
      % Detail Panel
      obj.gui.menuDetail=uimenu('Parent',obj.gui.guiMain,'Label','Detail');
      for idet=1:length(obj.guiDetailPanelChoice)
        mname=sprintf('menuDetail%s',regexprep(obj.guiDetailPanelChoice{idet},' ',''));
        obj.gui.(mname)=...
          uimenu('Parent',obj.gui.menuDetail,'Label',obj.guiDetailPanelChoice{idet},'Callback',@(src,event)guiMenuDetailCallback(obj,src,event,idet));
        if strcmp(obj.guiDetailPanel,obj.guiDetailPanelChoice{idet})
          set(obj.gui.(mname),'Checked','on');
        else
          set(obj.gui.(mname),'Checked','off');
        end
      end
      % Help
      obj.gui.menuHelp=uimenu('Parent',obj.gui.guiMain,'Label','Help');
      obj.gui.menuHelpProcedures=uimenu('Parent',obj.gui.menuHelp,'Label','Show Procedure List','Callback',@(src,event)guiMenuHelpCallback(obj,src,event));
      
      % trigger activities related to sext list callback
      obj.guiSextSelCallback(obj.gui.sextSel); obj.guiSextSelCallback(obj.gui.sextDimSel);
      
    end
    function guiProcedures(obj) % List procedures (eventually each button push should have action
      ptxt={'>=5 Hz to FACET dump' 'Sext Magnet Mover GUI running' 'Check Ready status' 'Switch off downstream sextupoles' 'Check BPM list and mover range' 'Center Sextupole Mover Readings' ...
        'Run BBA data taking procedure' 'Check data- remove any bad BPMs if required' 'Commit BBA data to database if good' 'Switch on and normalise all Sextupoles' ...
        'Save to Logbook'};
      border=0.005; nbut=length(ptxt);
      bW=1-2*border; bH=(1-border*(nbut*1.1))/nbut;
      obj.guiCreateFigure('procedures','SEXT BBA Procedure List',[400 600]);
      for ibut=nbut:-1:1
        obj.guiCreateTogglebutton(sprintf('proc%d',nbut-ibut+1),ptxt{nbut-ibut+1},'procedures',[border border*ibut+bH*(ibut-1) bW bH]);
        set(obj.gui.(sprintf('proc%d',nbut-ibut+1)),'Callback',@(src,event)guiProcButtonCallback(obj,src,event,ibut,nbut));
      end
    end
  end
  % Internal GUI callbacks etc
  methods(Hidden)
    function guiSelBPMCallback(obj,~,~)
      fhan=obj.INSTR.guiInstrChoice;
      desel=find(~ismember(obj.INSTR.instrName,obj.bpmName) | ismember(obj.INSTR.instrName,obj.bpmIgnore));
      set(obj.INSTR.gui.instrCbox1,'Value',desel);
      obj.INSTR.guiInstrCallback(obj.INSTR.gui.selbutton1,[]);
      waitfor(fhan);
      if ~isempty(obj.INSTR.instrChoiceFromGui)
        obj.bpmIgnore=obj.bpmName(~ismember(obj.bpmChoice,find(obj.INSTR.instrChoiceFromGui)));
        obj.guiSextSelCallback(obj.gui.sextSel); obj.guiSextSelCallback(obj.gui.sextDimSel);
      end
    end
    function guiProcButtonCallback(obj,~,~,ibut,nbut)
      % If last button then put all back to unpushed
      if ibut==nbut && get(obj.gui.(sprintf('proc%d',nbut)),'Value')
        for ibut=1:nbut
          set(obj.gui.(sprintf('proc%d',ibut)),'Value',0)
        end
      end
    end
    function guiMenuFileCallback(obj,src,~)
      switch src
        case obj.gui.menuSaveData
          [filename, pathname]=uiputfile(sprintf('FlSextBBA_data_%s.mat',datestr(now,30)));
          if filename
            data=obj.data; %#ok<NASGU>
            guiMessageData=obj.guiMessageData; %#ok<NASGU>
            guiMessageDataOrder=obj.guiMessageDataOrder; %#ok<NASGU>
            save(fullfile(pathname,filename),'data','guiMessageData','guiMessageDataOrder');
          end
        case obj.gui.menuLoadData
          [filename, pathname]=uigetfile;
          if ~filename
            return
          end
          ld=load(fullfile(pathname,filename));
          if isfield(ld,'data')
            obj.data=ld.data;
          end 
          obj.guiAddToMessagebox(sprintf('Data loaded from file: %s',fullfile(pathname,filename)));
          obj.guiSextSelCallback(obj.gui.sextSel); obj.guiSextSelCallback(obj.gui.sextDimSel);
        case obj.gui.menuSaveSettings
          errordlg('Not yet implemented','No Implementation')
        case obj.gui.menuSaveToLog
          errordlg('Not yet implemented','No Implementation')
      end
    end
    function guiMenuDetailCallback(obj,src,~,idet)
      for ir=1:length(obj.guiAvailableUpdateRates)
        mname=sprintf('menuDetail%s',regexprep(obj.guiDetailPanelChoice{idet},' ',''));
        if src==obj.gui.(mname);
          obj.guiDetailPanel=obj.guiDetailPanelChoice{idet};
          return
        end
      end
    end
    function guiMenuEditCallback(obj,src,~)
      for ir=1:length(obj.guiAvailableUpdateRates)
        if src==obj.gui.(sprintf('menuUpdateRate%d',ir))
          obj.guiUpdateRate=obj.guiAvailableUpdateRates(ir);
          set(obj.gui.(sprintf('menuUpdateRate%d',ir)),'Checked','on')
        else
          set(obj.gui.(sprintf('menuUpdateRate%d',ir)),'Checked','off')
        end
      end
    end
    function guiMenuHelpCallback(obj,~,~)
      obj.guiProcedures;
    end
    function guiPsSextSetBDESCallback(obj,~,~)
      val=str2double(get(obj.gui.psSextSetBDESVal,'String'));
      if isnan(val)
        errordlg('Invalid BDES entry','Invalid entry')
        return
      end
      bdesVal=val;
      try
        bvals=obj.FLI.PS_B;
        bvals(obj.sextAllPS(get(obj.gui.psSextSel,'Value')))=bdesVal/10;
        obj.FLI.PS_B=bvals;
        obj.FL.hwSet(obj.FLI,obj.FLI.PS_list(obj.sextAllPS(get(obj.gui.psSextSel,'Value'))));
      catch ME
        errordlg(sprintf('Error setting Power Supply\n%s',ME.message),'PS set error')
        return
      end
    end
    function guiPsSextPowerCallback(obj,src,~)
      switch src
        case obj.gui.psSextPowerOn
          bvals=obj.FLI.PS_B;
          bvals(obj.sextAllPS(get(obj.gui.psSextSel,'Value')))=obj.sextAllInitBDES(get(obj.gui.psSextSel,'Value'));
          obj.FLI.PS_B=bvals;
          obj.FL.hwSet(obj.FLI,obj.FLI.PS_list(obj.sextAllPS(get(obj.gui.psSextSel,'Value'))));
        case obj.gui.psSextPowerOff
          set(obj.gui.psSextSetBDES,'String','0')
          obj.guiPsSextSetBDESCallback(obj.gui.psSextSetBDES);
      end
    end
    function guiBpmNAveCallback(obj,~,~)
      val=str2double(get(obj.gui.bpmNAve,'String'));
      if ~isnan(val) && val>0
        obj.bpmNAve=val;
      else
        set(obj.gui.bpmNAve,'String',num2str(obj.bpmNAve))
      end
    end
    function guiMoverMoveCmdCallback(obj,~,~)
      vals=[str2double(get(obj.gui.moverSetX,'String')) str2double(get(obj.gui.moverSetY,'String')) str2double(get(obj.gui.moverSetT,'String'))].*1e-3;
      try
        if any(isnan(vals)) || vals(1)<obj.defaultMoverRange{1}(1) || vals(1)>obj.defaultMoverRange{1}(2) || ...
            vals(2)<obj.defaultMoverRange{2}(1) || vals(2)>obj.defaultMoverRange{2}(2) ||...
          vals(3)<obj.defaultMoverRange{3}(1) || vals(3)>obj.defaultMoverRange{3}(2)
          error('Invalid or out of range mover values')
        end
        gpos=obj.FLI.GIRDER_POS;
        gpos(obj.sextChoice(obj.sextSelect),[1 3 6])=vals;
        % Check dynamic range
        camangles=obj.camCheck(vals(1),vals(2),vals(3));
        if any(isnan(camangles))
          error('Range outside max permissible by mover system')
        end
        obj.FLI.GIRDER_POS=gpos;
        obj.FL.hwSet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
        obj.guiAddToMessagebox(sprintf('Magnet %s moved to: %.4f %.4f %.4f (mm/mrad)',obj.sextName{obj.sextSelect},vals.*1e3))
      catch ME
        obj.guiUpdateStatusDisplay('statusDisplay',-1);
        obj.guiAddToMessagebox('Error setting move coords:');
        obj.guiAddToMessagebox(ME.message);
      end
    end
    function guiMoverRangeCallback(obj,src,~)
      if obj.sextDimSelect=='x'
        idim=1;
      else
        idim=2;
      end
      if src==obj.gui.moverRangeLow
        iran=1;
        fname='moverRangeLow';
        val=str2double(get(obj.gui.moverRangeLow,'String'))*1e-3;
      else
        iran=2;
        fname='moverRangeHigh';
        val=str2double(get(obj.gui.moverRangeHigh,'String'))*1e-3;
      end
      if isnan(val) || (iran==1 && val<obj.defaultMoverRange{idim}(1)) || (iran==2 && val>obj.defaultMoverRange{idim}(2))
        set(obj.gui.(fname),'String',num2str(obj.defaultMoverRange{idim}(iran)*1e3))
        return
      end
      try
        obj.moverRange(iran)=val;
      catch ME
        obj.guiUpdateStatusDisplay('statusDisplay',-1);
        obj.guiAddToMessagebox('Error Setting Mover Range:');
        obj.guiAddToMessagebox(ME.message);
      end
    end 
    function guiMoverNumStepsCallback(obj,~,~)
      val=str2double(get(obj.gui.moverNumSteps,'String'));
      if isnan(val) || val<0 || val>100
        set(obj.gui.moverNumSteps,'String',num2str(obj.nMoverSteps))
        return
      end
      obj.nMoverSteps=val;
    end
    function guiRemoveBPMCallback(obj,src,~)
      if isfield(obj.gui,'analRemoveBPM') && src==obj.gui.analRemoveBPM
          tdata=get(obj.gui.analTable,'Data');
      elseif isfield(obj.gui,'bpmRemoveBPM') && src==obj.gui.bpmRemoveBPM
          tdata=get(obj.gui.bpmTable,'Data');
      end
      delsel=cell2mat(tdata(:,end));
      if any(delsel)
        for isel=find(delsel)
          if ~ismember(tdata{isel,1},obj.bpmIgnore)
            obj.bpmIgnore{end+1}=tdata{isel,1};
          end
        end
      end
    end
    function guiPlotSelCallback(obj,~,~)
      selStr=get(obj.gui.plotSel,'String');
      obj.doAnal;
      obj.doPlot(obj.gui.bbaAxes,selStr{get(obj.gui.plotSel,'Value')});
    end
    function guiSextSelCallback(obj,src,~)
      switch src
        case obj.gui.sextSel
          obj.sextSelect=get(obj.gui.sextSel,'Value');
        case obj.gui.sextDimSel
          dimStr=get(obj.gui.sextDimSel,'String'); obj.sextDimSelect=dimStr{get(obj.gui.sextDimSel,'Value')};
          obj.sextSelect=get(obj.gui.sextSel,'Value');
      end
    end
    function guiDisplayAllPlotsCallback(obj,~,~)
      for ibpm=1:length(obj.bpmChoice)
        if ~ismember(obj.bpmName{ibpm},obj.bpmIgnore)
          figure('Name',obj.bpmName{ibpm},'NumberTitle','off'); ah=axes; %#ok<LAXES>
          obj.doPlot(ah,obj.bpmName{ibpm});
        end
      end
    end
    function guiStartBBACallback(obj,~,~)
      if ~strcmp(obj.runTimer.Running,'on')
        start(obj.runTimer);
      end
    end
    function guiStopBBACallback(obj,~,~)
      stop(obj.runTimer);
    end
    function guiMoveSextToBBACallback(obj,~,~)
      sname=obj.sextID;
      if ~isfield(obj.data,sname) || ~isfield(obj.data.(sname),'anal') || ~isfield(obj.data.(sname).anal,'BBA')
        errordlg('No data taken yet for this Sextupole','No data')
        return
      end
      if get(obj.gui.sextDimSel,'Value')==1 % x
        dim='x'; idim=1;
      else
        dim='y'; idim=3;
      end
      if ~isfield(obj.data.(sname).anal.BBA,dim)
        errordlg('No data taken yet for this Sextupole','No data')
        return
      end
      data=obj.data.(sname).anal;
      newBBA=data.BBA.(dim);
      gpos=obj.FLI.GIRDER_POS;
      if ~strcmp(questdlg(sprintf('Mover sextupole to %c measured BBA offset?\nOffset %.4f -> %.4f (mm)',dim,...
          gpos(obj.sextChoice(obj.sextSelect),idim)*1e3,(gpos(obj.sextChoice(obj.sextSelect),idim)+newBBA)*1e3)),'Yes')
        return
      end
      gpos(obj.sextChoice(obj.sextSelect),idim)=gpos(obj.sextChoice(obj.sextSelect),idim)+newBBA;
      obj.FLI.GIRDER_POS=gpos;
      obj.FL.hwSet(obj.FLI,obj.FLI.GIRDER_list(obj.sextChoice(obj.sextSelect)));
    end
    function guiCommitBBACallback(obj,~,~)
      sname=obj.sextID;
      data=obj.data.(sname);
      if get(obj.gui.sextDimSel,'Value')==1 % x
        dim='x';
      else
        dim='y';
      end
      if ~isfield(obj.data.(sname).anal.BBA,dim)
        errordlg('No data taken yet for this Sextupole','No data')
        return
      end
      newBBA=data.anal.BBA.(dim); newBBA_err=data.anal.BBA.(sprintf('%c_err',dim));
      if ~isfield(obj.data.(sname),'BBA') || ~isfield(obj.data.(sname).BBA,dim) || isempty(obj.data.(sname).BBA.(dim))
        currentBBA=0;
      else
        currentBBA=obj.data.(sname).BBA.(dim);
      end
      if ~strcmp(questdlg(sprintf('Commit BBA offset?\nOffset = %.6f + (%.6f +/- %.6f) = %.6f (mm)',currentBBA*1e3,newBBA*1e3,newBBA_err*1e3,(newBBA+currentBBA)*1e3)),'Yes')
        return
      end
      errordlg('Not yet implemented','No Implementation')
    end
    function guiFillAnalData(obj)
      try
        sname=obj.sextID;
        if ~isfield(obj.data,sname) || isempty(obj.data.(sname).name) || ~isfield(obj.gui,'analTable') || ~ishandle(obj.gui.analTable)
          set(obj.gui.analTable,'Data',[]);
          return
        end
        data=obj.data.(sname);
        tdata={}; cbpm=0;
        for ibpm=1:length(obj.bpmChoice)
          if ismember(obj.bpmName{ibpm},obj.bpmIgnore)
            continue
          end
          nbpm=find(ismember(obj.data.(sname).name,obj.bpmName{ibpm}));
          cbpm=cbpm+1;
          tdata{cbpm,1}=obj.bpmName{ibpm};
          if isempty(nbpm)
            tdata{cbpm,2}='---'; tdata{cbpm,3}='---'; tdata{cbpm,4}='---'; tdata{cbpm,5}='---';
          else
            tdata{cbpm,2}=data.anal.fit_B(nbpm);
            tdata{cbpm,3}=data.anal.fit_B_err(nbpm);
            tdata{cbpm,4}=data.anal.fit_A(nbpm);
            tdata{cbpm,5}=data.anal.fit_C(nbpm);
          end
          tdata{cbpm,6}=false;
        end
        set(obj.gui.analTable,'Data',tdata);
      catch
      end
    end
    function doPlot(obj,phan,bpmName)
      try
        cla(phan);
        cols='rgbkmcy';
        sname=obj.sextID;
        if ~isfield(obj.data,sname) || isempty(obj.data.(sname)); return; end;
        sz=size(obj.data.(sname).x);
        [sv, bI]=sort(obj.data.(sname).scanvals.(obj.sextDimSelect)(1:sz(2)));
        data=obj.data.(sname);
        if strcmp(bpmName,'All')
          for ibpm=1:length(data.name)
            if ismember(data.name{ibpm},obj.bpmIgnore)
              continue
            end
            errorbar(phan,sv.*1e3,(abs(data.x(ibpm,bI))-min(abs(data.x(ibpm,bI)))).*1e3,data.x_err(ibpm,bI).*1e3,...
              sprintf('%c.',cols(1+mod(ibpm,7))));
            if ibpm==1; hold(phan,'on'); end;
            if isfield(data,'anal') && isfield(data.anal,'fit_A')
              xfit=linspace(sv(1),sv(end),1000);
              ydat=data.anal.fit_A(ibpm).*(xfit-data.anal.fit_B(ibpm)).^2 + data.anal.fit_C(ibpm) ;
              plot(phan,xfit.*1e3,(abs(ydat)-min(abs(ydat))).*1e3,sprintf('%c',cols(1+mod(ibpm,7))))
            end
          end
          hold(phan,'off')
        else
          ibpm=find(ismember(data.name,bpmName),1);
          errorbar(phan,sv.*1e3,(abs(data.x(ibpm,bI))-min(abs(data.x(ibpm,bI)))).*1e3,...
            data.x_err(ibpm,bI).*1e3,sprintf('%c.',cols(1+mod(ibpm,7))))
          if isfield(data,'anal') && isfield(data.anal,'fit_A')
            xfit=linspace(sv(1),sv(end),1000);
            ydat=data.anal.fit_A(ibpm).*(xfit-data.anal.fit_B(ibpm)).^2 + data.anal.fit_C(ibpm) ;
            hold(phan,'on')
            plot(phan,xfit.*1e3,(abs(ydat)-min(abs(ydat))).*1e3,sprintf('%c',cols(1+mod(ibpm,7))))
            hold(phan,'off')
          end
        end
        % Add in BBA fits
        if isfield(data,'anal') && isfield(data.anal,'BBA')
          ax=axis(phan);
          bba=data.anal.BBA.(obj.sextDimSelect);
          bba_err=data.anal.BBA.(sprintf('%c_err',obj.sextDimSelect));
          hold(phan,'on')
          line([bba bba].*1e3,ax(3:4),'Parent',phan,'Color','blue');
          line([bba bba].*1e3+bba_err.*1e3,ax(3:4),'Parent',phan,'Color','red');
          line([bba bba].*1e3-bba_err.*1e3,ax(3:4),'Parent',phan,'Color','red');
          hold(phan,'off')
        end
        xlabel(phan,sprintf('%s Position / mm',obj.sextName{obj.sextSelect}));
        ylabel(phan,'BPM position / mm');
        if strcmp(bpmName,'All'); col='b'; else col=cols(1+mod(ibpm,7)); end;
        title(phan,sprintf('[BPM: %s] BBA = %.4f +/- %.4f (mm)',bpmName,obj.data.(sname).anal.BBA.(obj.sextDimSelect)*1e3,...
          obj.data.(sname).anal.BBA.(sprintf('%c_err',obj.sextDimSelect))*1e3),'Color',col)
        grid(phan,'on')
      catch
      end
    end
    function guiStatusDisplayButDownFn(obj,~,~)
      if ~strcmp(obj.guiDetailPanel,'Messages')
        obj.guiDetailPanel='Messages';
      end
      obj.guiUpdateStatusDisplay('statusDisplay',1);
    end
  end
  methods(Hidden)
    function updateReturn=guiUpdateFn(obj,~,inp)
      % Update required GUI fields
      % Update all values internally on bpm call as this happens each timer
      % cycle, then just distribute to the other fields as needed
      persistent ldata snames
      if isempty(snames)
        snames=obj.FLI.PS_names(obj.sextAllPS);
      end
      switch inp
        case 'bpm'
          % Only try and update from HW if program is not running
          if ~strcmp(obj.runTimer.Running,'on')
            obj.INSTR.acquire(obj.FL,1);
            obj.FL.hwGet(obj.FLI);
          end
          [meanData, stdData] = obj.INSTR.meanstd;
          if isempty(meanData)
            meanData=squeeze(obj.INSTR.Data(:,1,:));
            stdData=zeros(size(meanData));
          end
          if obj.sextDimSelect=='x'
            idim=1;
          else
            idim=2;
          end
          updateReturn=sprintf('%.3f +/- %.3f',meanData(obj.sextBpm(obj.sextSelect),idim)*1e3, ...
            stdData(obj.sextBpm(obj.sextSelect),idim)*1e3);
          if strcmp(obj.guiDetailPanel,'Analysis')
            ldata{1}=obj.FLI.GIRDER_POS(obj.sextChoice,:);
          elseif strcmp(obj.guiDetailPanel,'BPM')
            ldata{1}=meanData; ldata{2}=stdData;
          elseif strcmp(obj.guiDetailPanel,'SEXT PS')
            ldata{1}=obj.FLI.PS_B(obj.sextAllPS)*10; % in kG
          elseif strcmp(obj.guiDetailPanel,'Mover Control')
            ldata{1}=obj.FLI.GIRDER_POS;
          end
          % >>> Update BBA field from controls here <<<
        case 'bbaNew'
          sname=obj.sextID;
          if isfield(obj.data,sname) && isfield(obj.data.(sname),'anal')
            updateReturn=num2str(obj.data.(sname).anal.BBA.(obj.sextDimSelect)*1e3,5);
          else
            updateReturn=0;
          end
        case 'bba'
          sname=obj.sextID;
          if isfield(obj.data,sname) && isfield(obj.data.(sname),'BBA') && ~isempty(obj.data.(sname).BBA.(obj.sextDimSelect))
            updateReturn=num2str(obj.data.(sname).BBA.(obj.sextDimSelect)*3,5);
          else
            updateReturn=0;
          end
        case 'moverX'
          updateReturn=ldata{1}(obj.sextSelect,1)*1e3;
        case 'moverY'
          updateReturn=ldata{1}(obj.sextSelect,3)*1e3;
        case 'moverT'
          updateReturn=ldata{1}(obj.sextSelect,6)*1e3;
        case 'bpmtable' % colnames={'BPM' 'X' 'Xerr' 'Y' 'Yerr' 'TMIT' 'TMITerr' ''};
          tdata={};
          nbpm=0;
          for ibpm=1:length(obj.bpmChoice)
            if ismember(obj.bpmName{ibpm},obj.bpmIgnore)
              continue
            end
            nbpm=nbpm+1;
            tdata{nbpm,1}=obj.bpmName{ibpm};
            tdata{nbpm,2}=num2str(ldata{1}(obj.bpmChoice(ibpm),1)*1e3,3);
            tdata{nbpm,3}=num2str(ldata{2}(obj.bpmChoice(ibpm),1)*1e3,3);
            tdata{nbpm,4}=num2str(ldata{1}(obj.bpmChoice(ibpm),2)*1e3,3);
            tdata{nbpm,5}=num2str(ldata{2}(obj.bpmChoice(ibpm),2)*1e3,3);
            tdata{nbpm,6}=num2str(ldata{1}(obj.bpmChoice(ibpm),3)*1e3,3);
            tdata{nbpm,7}=num2str(ldata{2}(obj.bpmChoice(ibpm),3)*1e3,3);
            tdata{nbpm,8}=false;
          end
          updateReturn=tdata;
        case 'sbdes'
          val=get(obj.gui.psSextSel,'Value');
          updateReturn=ldata{1}(val(1));
          for isext=1:length(obj.sextAllPS)
            sstr{isext}=sprintf('%s [%g]',snames{isext},ldata{1}(isext));
          end
          set(obj.gui.psSextSel,'String',sstr);
        case 'sbdes-all' % special case- this one never called by main updater, just used by PS panel stuff locally
          obj.FL.hwGet(obj.FLI);
          updateReturn=obj.FLI.PS_B(obj.sextPS);
      end
    end
    function camangles = camCheck(obj,x,y,rot)
      
      Model=obj.simModel;
      x_1 = x + (Model.camsettings.sext.a*cos(rot)) + (Model.camsettings.sext.b*sin(rot)) - Model.camsettings.sext.a;
      y_1 = y - (Model.camsettings.sext.b*cos(rot)) + (Model.camsettings.sext.a*sin(rot)) + Model.camsettings.sext.c;
      betaminus = pi/4 - rot;
      betaplus  = pi/4 + rot;
      
      camangles(1) = rot - asin((1/Model.camsettings.sext.L) * ...
        ((x_1+Model.camsettings.sext.S2)*sin(rot) - y_1*cos(rot) + (Model.camsettings.sext.c - Model.camsettings.sext.b)));
      
      camangles(2) = rot - asin((1/Model.camsettings.sext.L) * ...
        ((x_1+Model.camsettings.sext.S1)*sin(betaminus) + y_1*cos(betaminus) - Model.camsettings.sext.R));
      
      camangles(3) = rot - asin((1/Model.camsettings.sext.L) * ...
        ((x_1-Model.camsettings.sext.S1)*sin(betaplus) - y_1*cos(betaplus) + Model.camsettings.sext.R));
      
      if ~isreal(camangles(1)) || ~isreal(camangles(2)) || ~isreal(camangles(3))
        camangles = [NaN NaN NaN];
      end
    end
    function ismov=isMoving(obj)
      ismov=false;
      thisSext=obj.sextSelect;
      if ~isempty(obj.movingPV) && length(obj.movingPV)>=thisSext
        try
          resp=lcaGet(obj.movingPV{thisSext});
          ismov=any(resp);
        catch
        end
      end
    end
  end
end