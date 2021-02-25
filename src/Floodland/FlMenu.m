classdef FlMenu < FlGui
  %FLMENU Application menu for Floodland applications and graphical
  %interface to general Floodland functionality
  %   
  % Create with Floodland object
  %  FM=FlMenu(FL)
  %
  % Main public methods are:
  %  addApp - add an application by passing object to this methods
  %  rmApp - remove previously added application
  %  guiMain - display application menu (intended main user interface for
  %           launching and high level control of Floodland applications)
  %  simModeChange - change sim mode
  %
  % See also:
  %  Floodland FlGui FlApp
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc FlMenu">doc FlMenu</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  properties(Access=private)
    FL % pointer to Floodland object
    appList={}; % list of applications (links to Floodland app objects)
  end
  properties(Dependent)
    appListNames % Names of Floodland apps attached to this menu
  end
  
  %% Get/Set methods
  methods
    function names=get.appListNames(obj)
      if ~isempty(obj.appList)
        names=cell(1,length(obj.appList));
        for iList=1:length(obj.appList)
          names{iList}=obj.appList{iList}.appName;
        end
      end
    end
  end
  
  %% Main public methods
  methods
    function obj=FlMenu(FL)
      % obj=FlMenu(FL)
      % Need to pass Floodland object (FL)
      if exist('FL','var') && ~strcmp(class(FL),'Floodland')
        error('Must pass Floodland object as first argument');
      end
      obj.FL=FL;
      obj.guiTitle='Lucretia:Floodland'; % Title to display on menu GUI 
    end
    function addApp(obj,appObj)
      % addApp(obj,appObj)
      % Add application object (FlApp class) to menu
      
      % Check requested object to add is a subclass of FlApp
      mc=metaclass(appObj);
      isFlApp=false;
      if ~isempty(mc.SuperclassList)
        for imc=1:length(mc.SuperclassList)
          isFlApp=strcmp(mc.SuperclassList(imc).Name,'FlApp');
          if isFlApp; break; end;
        end
      end
      if ~isFlApp; error('Can only add Floodland application objects to menu, i.e. objects that inherit from FlApp class'); end;
      obj.appList{end+1}=appObj;
    end
    function rmApp(obj,appID)
      % rmApp(obj,appID)
      % Delete application from menu
      if exist('appId','var') && isnumeric(appID) && appID>0 && appID<=length(obj.appList)
        obj.appList(appID)=[];
      else
        error('Must supply appID = appList entry (see display(obj))')
      end
    end
    function handle=guiMain(obj,~,~)
      % Main gui
      nApp=length(obj.appList);
      border=0.05;
      appButtonSize=50;
      % Generate GUI
      obj.guiCreateFigure('guiMain',obj.guiTitle,[350 max([appButtonSize*nApp,50])]);
      handle=obj.gui.guiMain;
      % Menu & sim mode settings
      mm=uimenu('Parent',obj.gui.guiMain,'Label','Mode');
      obj.gui.menu_simMode=uimenu('Parent',mm,'Label','Sim','Callback',@(src,event)simModeChange(obj,src,event));
      obj.gui.menu_liveMode=uimenu('Parent',mm,'Label','Live','Callback',@(src,event)simModeChange(obj,src,event));
      mc=uimenu('Parent',obj.gui.guiMain,'Label','Controls');
      msaf=uimenu('Parent',mc,'Label','Safeties');
      obj.gui.menu_safeOn=uimenu('Parent',msaf,'Label','on','Callback',@(src,event)safetiesOn(obj,src,event));
      obj.gui.menu_safeOff=uimenu('Parent',msaf,'Label','off','Callback',@(src,event)safetiesOff(obj,src,event));
      if strcmp(obj.FL.writeSafety,'on')
        set(obj.gui.menu_safeOn,'Checked','on')
        set(obj.gui.menu_safeOff,'Checked','off')
      else
        set(obj.gui.menu_safeOn,'Checked','off')
        set(obj.gui.menu_safeOff,'Checked','on')
      end
      if obj.FL.issim
        set(obj.gui.menu_simMode,'Checked','on')
        set(obj.gui.menu_liveMode,'Checked','off')
        obj.simModeChange(obj,obj.gui.menu_simMode);
      else
        set(obj.gui.menu_simMode,'Checked','off')
        set(obj.gui.menu_liveMode,'Checked','on')
        obj.simModeChange(obj,obj.gui.menu_liveMode);
      end
      
      
      % Application buttons
      if ~isempty(obj.appList)
        bh=(1-border*(length(obj.appList)+1))/length(obj.appList);
        for iapp=1:length(obj.appList)
          obj.guiCreatePushbutton(class(obj.appList{iapp}),obj.appList{iapp}.appName,obj.gui.guiMain,...
            [border border+border*(iapp-1)+bh*(iapp-1) 1-2*border bh]);
          set(obj.gui.(class(obj.appList{iapp})),'Callback',@(src,event)guiMain(obj.appList{iapp}))
        end
      end
      drawnow('expose')
    end
    function display(obj)
      % display functionality for workspace printing
      names=obj.appListNames;
      fprintf('Index   Application\n')
      fprintf('-----   -----------\n')
      for iapp=1:length(names)
        fprintf('%5d   %s\n',iapp,names{iapp})
      end
    end
  end
  
  %% GUI callbacks
  methods(Hidden)
    function safetiesOn(obj,~,~)
      obj.FL.writeSafety='on';
      set(obj.gui.menu_safeOn,'Checked','on')
      set(obj.gui.menu_safeOff,'Checked','off')
      obj.setBkgCol;
    end
    function safetiesOff(obj,~,~)
      obj.FL.writeSafety='off';
      set(obj.gui.menu_safeOn,'Checked','off')
      set(obj.gui.menu_safeOff,'Checked','on')
      obj.setBkgCol;
    end
    function simModeChange(obj,src,~)
      % simModeChange(obj,src)
      % callback for change mode menu items
      if src==obj.gui.menu_simMode
        obj.FL.issim=true;
        set(obj.gui.menu_simMode,'Checked','on')
        set(obj.gui.menu_liveMode,'Checked','off')
      else
        obj.FL.issim=false;
        set(obj.gui.menu_simMode,'Checked','off')
        set(obj.gui.menu_liveMode,'Checked','on')
      end
      obj.setBkgCol;
    end
    function setBkgCol(obj)
      if obj.FL.issim
        obj.FL.issim=true;
        set(obj.gui.guiMain,'Color',[170,255,128]./255)
        set(obj.gui.guiMain,'Name',sprintf('%s (%s)',obj.guiTitle,'Sim'))
      elseif strcmp(obj.FL.writeSafety,'on')
        set(obj.gui.guiMain,'Color',[255,190,0]./255)
        set(obj.gui.guiMain,'Name',sprintf('%s (%s)',obj.guiTitle,'Live'))
      else
        set(obj.gui.guiMain,'Color','red')
        set(obj.gui.guiMain,'Name',sprintf('%s (%s)',obj.guiTitle,'Live, safeties off'))
      end
      drawnow('expose')
    end
  end
  
  %% Status save/load methods
  methods(Static)
    function statusSave(tag,varargin)
      % statusSave(tag,[datalist...])
      % Status saves in dated entry in directory name 'tag'
      if ~exist(tag,'dir')
        mkdir(tag);
      end
      % Remove load block if present
      if exist(fullfile(tag,'NOLOAD'),'file')
        delete(fullfile(tag,'NOLOAD'))
      end
      ds=sprintf('%s.mat',datestr(now,30));
      save(fullfile(tag,ds),'varargin')
      
      % Keep max 100 files
      tf=dir(sprintf('%s/*.mat',tag));
      if length(tf)>100
        [a b]=sort({tf.date});
        for ifile=1:length(tf)-100
          delete(fullfile(tag,tf(b(ifile).name)))
        end
      end
    end
    function savedata=statusLoad(tag,varargin)
      % savedata=statusLoad(tag,[date])
      % fetch data saved in tag directory, optionally with given date/time
      % ID, else just get latest
      if ~exist(tag,'dir')
        error('No saved status with this tag')
      end
      % abort if noload tag put on this directory
      if exist(fullfile(tag,'NOLOAD'),'file')
        error('Block put on loading data from this directory, removed with next statusSave')
      end
      tf=dir(sprintf('%s/*.mat',tag));
      [a b]=sort({tf.date});
      sd=load(fullfile(tag,tf(b(end)).name));
      if ~isfield(sd,'varargin') || length(sd.varargin)~=nargin-1
        error('Different number of requested variables than available in saved file')
      end
      for idata=1:length(sd.varargin)
        savedata.(varargin{idata})=sd.varargin{idata};
      end
    end
    function statusLoadBlock(tag)
      % Block loading from a given tag directory
      evalc(sprintf('!touch %s',fullfile(tag,'NOLOAD')));
    end
  end
end

