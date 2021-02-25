classdef FlSaveRestore < handle & FlIndex & FlApp & FlGui & matlab.mixin.Copyable
  %FLSAVERESTORE Save/Restore program for Floodland controls
  %   Define 1 or more lists of controls to store save points and ability
  %   to restore lists to those previously saved values
  %
  % This class is designed to be mainly used from its graphical user
  % interface (GUI) which can be launched using the guiMain method. It can
  % however be driven from the class methods alone if needed.
  %
  % Constructor:
  %  SR=FlSaveRestore(FL,indxObj)
  %    Provide Floodland object and FlIndex object which forms a master
  %    list of available hardware objects with controls links that can be
  %    saved/restored. Sub-selections of hardware elements happen through
  %    the FlIndex class methods.
  %    You can generate several "files", each one containing a different
  %    list of save/restore links.
  %
  % Main public methods:
  %  addFile - add a new save/restore list
  %  rmFile - remove a list (cannot remove last one)
  %  setRestoreVals - get current hardware values for selected hardware
  %                   channels and move to restore vals
  %  addChannel - add a new hardware channel from list provided at
  %               construction time (extends FlIndex method)
  %  rmChannel - remove a selected channel (cannot reomve last one)
  %  guiMain - launch main application GUI
  %
  % See also:
  %  Floodland FlInstr FlIndex FlGui FlApp
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc FlSaveRestore">doc FlSaveRestore</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  
  properties
    fileName % file name associated with this instance
    restoreVals % user defined restore values for selected actuators
  end
  properties(Access=protected)
    fileList={}; % list of all created other instances (pointers)
    FL % pointer to Floodland object
    guiDisplayUnits='setpt'; % SetPt or ACT display units on GUI
    selectChan={}; % channel selection
  end
  properties(Constant)
    appName='Save/Restore'; % Application name to appear in menus
  end
  
  
  %% Main public methods
  methods
    function obj=FlSaveRestore(FL,indxObj)
      % obj=FlSaveRestore(FL,indxObj)
      % Constructor, provide Floodland object and master FlIndex object
      
      % Need to pass FlIndex object
      if exist('indxObj','var') && ~strcmp(class(indxObj),'FlIndex')
        error('Must pass FlInstr object');
      end
      % Need to pass Floodland object
      if exist('FL','var') && ~strcmp(class(FL),'Floodland')
        error('Must pass Floodland object as first argument');
      end
      obj.FL=FL;
      obj.refIndx=indxObj; % reference FlIndex object with all known actuators
      % Initialise file list
      obj.fileList{1}=obj;
      obj.fileName=datestr(now,30);
    end
    function addFile(obj)
      % Add a new file (save/restore list of actuators)
      
      obj.fileList{end+1}=obj.copy;
      obj.fileList{end}.fileList=obj.fileList;
      obj.fileList{end}.useCntrl=true(size(obj.useCntrl));
      obj.fileList{end}.fileName=datestr(now,30);
      for ifile=1:length(obj.fileList)
        obj.fileList{ifile}.fileList=obj.fileList;
      end
    end
    function rmFile(obj,ind)
      % Remove file (reference=ind)
      
      dfl=obj.fileList{ind};
      obj.fileList(ind)=[];
      for ifile=1:length(obj.fileList)
        obj.fileList{ifile}.fileList=obj.fileList;
      end
      delete(dfl);
    end
    function setRestoreVals(obj)
      % Set restore values in restoreVals property to selected controls
      % in useCntrl and useCntrlChan properties, OR if 'sel' supplied as
      % {useCntrl}=useCntrlChan list, then apply restoreVals to these
      % instead
      
      % Check there is anything to set
      if isempty(obj.useCntrl) || ~any(obj.useCntrl)
        return
      end
      
      % Update hw vals
      obj.FL.hwGet(obj);
      newVals=obj.Ampl;
      
      % Set new vals
      for indx=find(obj.useCntrl)
        chans=find(obj.selectChan{indx});
        if isempty(chans); continue; end;
        for ichan=chans
          newVals{indx}(ichan)=obj.restoreVals{indx}(ichan);
        end
      end
      obj.SetPt=newVals;
      
      % Write new controller value to control system
      obj.FL.hwSet(obj);
    end
    function addChannel(obj,allRefInd)
      % Overload FlIndex addChannel method- add initialisation of
      % restoreVals property
      % allRefInd = vector of FlInstr superset indices to add to this
      % object
      
      % First call the FlIndex class addChannel method
      for refInd=allRefInd
        addChannel@FlIndex(obj,refInd);
      end
      
      % Now initialise restore vals for this channel with current values
      % from control system
      obj.FL.hwGet(obj);
      newVals=obj.SetPt;
      for ind=1:length(allRefInd)
        obj.restoreVals{end+1}=newVals{end-length(allRefInd)+ind};
        obj.selectChan{end+1}=obj.useCntrlChan{end-length(allRefInd)+ind};
      end
    end
    function rmChannel(obj,ind)
      % Overload FlIndex rmChannel method
      % ind = vector of control indices to remove from this object
      
       % First call the FlIndex class rmChannel method
      rmChannel@FlIndex(obj,ind);

      % Now sych restoreVals
      obj.restoreVals(ind)=[];
      obj.selectChan(ind)=[];
    end
    function han=guiMain(obj)
      % Main graphical user interface for this object
      
      % Create main figure window
      obj.guiCreateFigure('guiMain','Save - Restore',[625 400]);
      han=obj.gui.guiMain;
      % Border size
      border=0.02;
      % File name edit window
      fsize=0.4;
      obj.guiCreatePanel('fileNamePanel','File Name','guiMain',[border border fsize 0.1]);
      obj.guiCreateEdit('fileName',obj.fileName,'fileNamePanel',[border border 1-2*border 1-2*border]);
      set(obj.gui.fileName,'Callback',@(src,event)guiEditFilename(obj,src,event));
      % Save/restore and update buttons
      bsize=(1-border*5-fsize)/3;
      obj.guiCreatePushbutton('doSave','Save Vals','guiMain',[border*2+fsize border bsize 0.1]);
      set(obj.gui.doSave,'Callback',@(src,event)guiDoSave(obj,src,event))
      obj.guiCreatePushbutton('doRestore','Restore','guiMain',[border*3+fsize+bsize border bsize 0.1]);
      set(obj.gui.doRestore,'Callback',@(src,event)guiDoRestore(obj,src,event))
      obj.guiCreatePushbutton('doUpdate','Update','guiMain',[border*4+fsize+bsize*2 border bsize 0.1]);
      set(obj.gui.doUpdate,'Callback',@(src,event)guiDoUpdate(obj,src,event))
      % Main table
      colnames={'out of tol' 'control' 'current value' 'restore value' 'Sel'};
      colfmt={'logical' 'char' 'numeric' 'numeric' 'logical'};
      colwid={'auto',258,'auto','auto','auto'};
      coledit=[false false false true true];
      obj.guiCreateTable('table',colnames,colfmt,colwid,coledit,...
        'guiMain',[border border*2+0.1 1-border*2 1-border*3-0.1]);
      set(obj.gui.table,'CellEditCallback',@(src,event)guiTableEdit(obj,src,event));
      % Menus
      fm=uimenu('Parent',obj.gui.guiMain,'Label','File');
      obj.gui.menuNew=uimenu('Parent',fm,'Label','New','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
      om=uimenu('Parent',fm,'Label','Open');
      dm=uimenu('Parent',fm,'Label','Delete');
      for ifile=1:length(obj.fileList)
        if ~strcmp(obj.fileList{ifile}.fileName,obj.fileName)
          uimenu('Parent',dm,'Label',obj.fileList{ifile}.fileName,'Callback',...
            @(src,event)guiMenuFileCallback(obj,src,event,-ifile));
          uimenu('Parent',om,'Label',obj.fileList{ifile}.fileName,'Callback',...
            @(src,event)guiMenuFileCallback(obj,src,event,ifile));
        end
      end
      obj.gui.menuEditList=uimenu('Parent',fm,'Label','Edit Controls List','Callback',@(src,event)guiMenuFileCallback(obj,src,event));
      % Table context menu
      hcmenu=uicontextmenu;
      sm=uimenu('Parent',hcmenu,'Label','Select');
      obj.gui.menuSelectAll=uimenu('Parent',sm,'Label','all','Callback',@(src,event)guiMenuSelectCallback(obj,src,event));
      obj.gui.menuSelectNone=uimenu('Parent',sm,'Label','none','Callback',@(src,event)guiMenuSelectCallback(obj,src,event));
      obj.gui.menuSelectTrim=uimenu('Parent',sm,'Label','out of tol','Callback',@(src,event)guiMenuSelectCallback(obj,src,event));
      dm=uimenu('Parent',hcmenu,'Label','Display Units');
      obj.gui.menuDisplayUnitsSetpt=uimenu('Parent',dm,'Label','SetPt','Callback',@(src,event)guiMenuSelectDisplayUnits(obj,src,event));
      obj.gui.menuDisplayUnitsAct=uimenu('Parent',dm,'Label','ACT','Callback',@(src,event)guiMenuSelectDisplayUnits(obj,src,event));
      if strcmp(obj.guiDisplayUnits,'setpt')
        set(obj.gui.menuDisplayUnitsSetpt,'Checked','on')
        set(obj.gui.menuDisplayUnitsAct,'Checked','off')
      else
        set(obj.gui.menuDisplayUnitsSetpt,'Checked','off')
        set(obj.gui.menuDisplayUnitsAct,'Checked','on')
      end
      set(obj.gui.table,'UIContextMenu',hcmenu)
      guiDoUpdate(obj);
    end
  end
  %% GUI callbacks
  methods(Hidden)
    function guiMenuSelectCallback(obj,src,~)
      data=get(obj.gui.table,'Data');
      sz=size(data);
      switch src
        case obj.gui.menuSelectAll
          for idata=1:sz(1)
            data{idata,5}=true;
          end
        case obj.gui.menuSelectNone
          for idata=1:sz(1)
            data{idata,5}=false;
          end
        case obj.gui.menuSelectTrim
          for idata=1:sz(1)
            if data{idata,1}
              data{idata,5}=true;
            else
              data{idata,5}=false;
            end
          end
      end
      set(obj.gui.table,'Data',data);
    end
    function guiMenuSelectDisplayUnits(obj,src,~)
      switch src
        case obj.gui.menuDisplayUnitsSetpt
          obj.guiDisplayUnits='setpt';
          set(obj.gui.menuDisplayUnitsSetpt,'Checked','on')
          set(obj.gui.menuDisplayUnitsAct,'Checked','off')
        case obj.gui.menuDisplayUnitsAct
          obj.guiDisplayUnits='act';
          set(obj.gui.menuDisplayUnitsSetpt,'Checked','off')
          set(obj.gui.menuDisplayUnitsAct,'Checked','on')
      end
      guiDoUpdate(obj);
    end
    function guiEditFilename(obj,src,~)
      for ifile=1:length(obj.fileList)
        if strcmp(obj.fileList{ifile}.fileName,get(src,'String'))
          errordlg('Must supply unique name, this one already exists for another file','File Name Error')
          return
        end
      end
      obj.fileName=get(src,'String');
    end
    function guiDoSave(obj,~,~)
      data=get(obj.gui.table,'Data');
      nc=0;
      if strcmp(obj.guiDisplayUnits,'setpt')
        cvals=obj.Ampl2ACT;
      end
      for indx=1:length(obj.useCntrl)
        chans=find(obj.selectChan{indx});
        if isempty(chans); continue; end;
        for ichan=chans
          nc=nc+1;
          if data{nc,5}
            if strcmp(obj.guiDisplayUnits,'setpt')
              obj.restoreVals{indx}(ichan)=data{nc,3};
            else
              obj.restoreVals{indx}(ichan)=data{nc,3}/cvals{indx}(ichan);
            end
          end
        end
      end
      guiDoUpdate(obj);
    end
    function guiDoRestore(obj,~,~)
      setRestoreVals(obj);
      guiDoUpdate(obj);
    end
    function guiDoUpdate(obj,~,~)
      if isempty(obj.MasterInd); return; end; % Return if no actuators added
      obj.FL.hwGet(obj);
      if strcmp(obj.guiDisplayUnits,'setpt')
        newVals=obj.Ampl;
        restoreVals=obj.restoreVals;
      else
        newVals=obj.ACT;
        cvals=obj.Ampl2ACT;
        restoreVals=obj.restoreVals;
        restoreVals=arrayfun(@(x) restoreVals{x}.*cvals{x},1:length(restoreVals),'UniformOutput',false);
      end
      nc=0;
      names=obj.INDXused;
      tableData={};
      for indx=1:length(restoreVals)
        chans=find(obj.useCntrlChan{indx});
        if isempty(chans); continue; end;
        for ichan=chans
          nc=nc+1;
          if abs((newVals{indx}(ichan)-restoreVals{indx}(ichan))/restoreVals{indx}(ichan))>0.02
            data{nc,1}=true;
          else
            data{nc,1}=false;
          end
          data{nc,2}=names{nc};
          data{nc,3}=newVals{indx}(ichan);
          data{nc,4}=restoreVals{indx}(ichan);
          data{nc,5}=obj.selectChan{indx}(ichan);
          tableData{nc}=[indx ichan];
        end
      end
      set(obj.gui.table,'Data',data)
      set(obj.gui.table,'UserData',tableData)
    end
    function guiTableEdit(obj,src,event)
      inds=event.Indices;
      tableData=get(src,'UserData');
      switch inds(2)
        case 4
          obj.restoreVals{tableData{inds(1)}(1)}(tableData{inds(1)}(2))=event.NewData;
        case 5
          obj.useCntrlChan{tableData{inds(1)}(1)}(tableData{inds(1)}(2))=event.NewData;
      end
    end
    function guiMenuFileCallback(obj,src,~,ifile)
      if exist('ifile','var')
        if ifile>0
          oldPos=get(obj.gui.guiMain,'Position');
          delete(obj.gui.guiMain);
          obj.gui=[];
          guiMain(obj.fileList{ifile});
          obj.gui=obj.fileList{ifile}.gui;
          set(obj.fileList{ifile}.gui.guiMain,'Position',oldPos)
        else
          if ~strcmp(questdlg(sprintf('Delete file %s ?',obj.fileList{abs(ifile)}.fileName)),'Yes')
            return
          end
          obj.rmFile(abs(ifile));
          obj.guiMain;
        end
      else
        if strcmp(get(src,'Label'),'New')
          obj.addFile;
          oldPos=get(obj.gui.guiMain,'Position');
          delete(obj.gui.guiMain);
          obj.gui=[];
          guiMain(obj.fileList{end});
          obj.gui=obj.fileList{end}.gui;
          set(obj.fileList{end}.gui.guiMain,'Position',oldPos)
        else
          for ic=1:length(obj.refIndx.useCntrl)
            if ismember(ic,obj.refInd)
              obj.refIndx.useCntrl(ic)=true;
              obj.refIndx.useCntrlChan{ic}=obj.useCntrlChan{ismember(obj.refInd,ic)};
            else
              obj.refIndx.useCntrl(ic)=false;
            end
          end
          uiwait(obj.refIndx.guiIndexChoice);
          if isempty(obj.refIndx.indexChoiceFromGui) || ~any(obj.refIndx.indexChoiceFromGui); return; end;
          if ~isempty(obj.refInd)
            chanDel=[];
            for ind=1:length(obj.refInd)
              if obj.refIndx.indexChoiceFromGui(obj.refInd(ind))
                obj.useCntrlChan{ind}=obj.refIndx.indexChanChoiceFromGui{obj.refInd(ind)};
              else
                chanDel(end+1)=ind;
              end
            end
            if ~isempty(chanDel)
              obj.rmChannel(chanDel);
            end
          end
          chanAdd=[];
          for ind=find(obj.refIndx.indexChoiceFromGui)
            if ~ismember(ind,obj.refInd)
              chanAdd(end+1)=ind;
            end
          end
          if ~isempty(chanAdd)
            obj.addChannel(chanAdd);
          end
          guiDoUpdate(obj);
        end
      end
    end
  end
end

