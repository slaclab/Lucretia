classdef FlGui < handle
  %FLGUI Template for Floodland GUIs
  
  properties
    guiFont
    guiNumScreens
    guiScreenSize
    guiUpdateRate=1;
    guiTitle
  end
  properties
    gui
  end
  properties(Access=protected)
    guiTimer
    guiTimerRate=1; % update rate of timer / Hz
    guiUpdateParams={}; % user-defined parameters to pass to update method
    guiUpdateMethod={}; % methods to call for updates
    guiMessageData={};
    guiMessageDataOrder
  end
  properties(Constant)
    guiAvailableUpdateRates=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10];
    guiMsgBoxSize=100;
  end
  methods % get/set
    function set.guiUpdateRate(obj,val)
      if ~ismember(val,obj.guiAvailableUpdateRates)
        error('Must choose from choices given in ''guiAvailableUpdateRates'' property')
      end
      obj.guiUpdateRate=val;
      try
        if ~isempty(obj.guiTimer)
          stop(obj.guiTimer)
          set(obj.guiTimer,'Period',1/val)
          start(obj.guiTimer)
        end
      catch
        warning('Lucretia:FlGui','Timer defined but cannot change Period')
      end
    end
  end
  methods
    %% constructor
    function obj=FlGui
      mp=get(0,'MonitorPositions');
      sz=size(mp);
      obj.guiNumScreens=sz(1);
      obj.guiScreenSize=mp;
      obj.guiFont=get(0,'FixedWidthFontName');
    end
  end
  methods(Access=protected)
    %% test for gui existence
    function resp=guiExists(obj,name)
      resp = isprop(obj,'gui') && isfield(obj.gui,name) && ~isempty(obj.gui.(name)) && ishandle(obj.gui.(name)) && ...
        strcmp(get(obj.gui.(name),'Name'),obj.guiTitle);
    end
    %% figure
    function redisp=guiCreateFigure(obj,name,title,size)
      persistent forceNewPos
      redisp=false;
      if ~exist('size','var') || (length(size)~=2 && length(size)~=4)
        error('Size variable should be supplied as vector length 2 or 4');
      end
      if ~exist('name','var') || ~ischar(name)
        error('Must supply name for figure handle')
      end
      if ~exist('title','var') || ~ischar(name)
        error('Must supply title for figure')
      end
      % If GUI already exists, re-display it
      if obj.guiExists(name)
        oldpos=get(obj.gui.(name),'Position');
        if ~isempty(obj.guiTimer) && strcmp(obj.guiTimer.Running,'on')
          stop(obj.guiTimer)
        end
        close(obj.gui.(name))
        obj.gui=[];
        obj.guiUpdateParams={};
        obj.guiUpdateMethod={};
        forceNewPos=oldpos; forceNewPos(2)=forceNewPos(2)+forceNewPos(4);
        obj.gui.(name)=obj.(name);
        redisp=true;
        return
      end
      % if size is 1*2, default to making figure window in centre of
      % current screen
      % - find which screen we are in
      thisScreen=0;
      curpos=get(0,'PointerLocation');
      for iscreen=1:obj.guiNumScreens
        if inpolygon(curpos(1),curpos(2),[obj.guiScreenSize(iscreen,1) obj.guiScreenSize(iscreen,1)+obj.guiScreenSize(iscreen,3)-1],...
            [obj.guiScreenSize(iscreen,2) obj.guiScreenSize(iscreen,2)+obj.guiScreenSize(iscreen,4-1)])
          thisScreen=iscreen;
          break
        end
      end
      if thisScreen>0
        if ~isempty(forceNewPos)
          if length(size)==4
            size(1:2)=forceNewPos(1:2)-[0 size(4)];
          else
            size=[forceNewPos(1:2)-[0 size(2)] size(:)'];
          end
          forceNewPos=[];
        end
        figSize=[min([size(1) obj.guiScreenSize(thisScreen,3)]) min([(obj.guiScreenSize(thisScreen,4)) size(2)])];
        if length(size)==2 % put it in the middle of current screen
          xpos=obj.guiScreenSize(iscreen,1)+obj.guiScreenSize(thisScreen,3)/2;
          ypos=obj.guiScreenSize(iscreen,2)+obj.guiScreenSize(thisScreen,4)/2;
          pos=[xpos-figSize(1)/2 ypos+figSize(2)/2 figSize];
        else % otherwise put it exactly where requested
          pos=size;
        end
        obj.gui.(name)=figure('position',pos,'Name',title,'NumberTitle','off','MenuBar','none');
      else % If finding which screen we are in failed, just put it where Matlab wants it to go
        obj.gui.(name)=figure('Name',title,'NumberTitle','off','MenuBar','none');
      end
      if isempty(obj.guiTimer)
        obj.guiTitle=title;
        % make timer object
        obj.guiTimer=timer('Tag',sprintf('%s Timer',title),'TimerFcn',@(src,event)guiTimerFn(obj,src,event,'run'),'BusyMode','drop',...
          'ExecutionMode','fixedSpacing','Period',1/obj.guiTimerRate,'StartFcn',@(src,event)guiTimerFn(obj,src,event,'start'),...
          'StopFcn',@(src,event)guiTimerFn(obj,src,event,'stop'),'StartDelay',2);
      end
    end
    %% panel
    function guiCreatePanel(obj,name,title,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name) = uipanel('Parent',parent,'FontName',obj.guiFont,'Title',title,'Units','normalized','Position',pos);
    end
    %% button group
    function guiCreateButtonGroup(obj,name,title,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name) = uibuttongroup('Parent',parent,'FontName',obj.guiFont,'Title',title,'Units','normalized','Position',pos);
    end
    %% axes
    function guiCreateAxes(obj,name,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=axes('Parent',parent,'Units','normalized','FontName',obj.guiFont,'Position',pos);
    end
    %% pushbutton
    function guiCreatePushbutton(obj,name,title,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','pushbutton','String',title,...
          'FontWeight','bold','FontName',obj.guiFont,'Units','normalized','Position',pos);
    end
    %% edit
    function guiCreateEdit(obj,name,txt,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','edit','FontName',obj.guiFont,'Units','normalized','String',txt,...
        'Position',pos,'BackgroundColor','white') ;
    end
    %% text
    function guiCreateText(obj,name,txt,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
       obj.gui.(name)=uicontrol(parent,'Style','text','FontName',obj.guiFont,'Units','normalized','String',txt,...
        'Position',pos,'FontWeight','bold') ;
    end
    %% status Display
    function guiCreateStatusDisplay(obj,name,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','text','FontName',obj.guiFont,'Units','normalized','Position',pos,'FontWeight','bold','Enable','inactive') ;
      guiUpdateStatusDisplay(obj,name,'last')
    end
    function guiUpdateStatusDisplay(obj,name,status)
      persistent laststatus
      if isequal(status,'last')
        if isempty(laststatus)
          status=1;
        else
          status=laststatus;
        end
      end
      if status==1
        txt='Ready';
        bgc='green';
      elseif status==0
        txt='WARNING';
        bgc=[0.8 0 0];
      elseif status==2
        txt='RUNNING';
        bgc='green';
      elseif status==3
        txt='USR STOP';
        bgc='green';
      else
        txt='ERROR';
        bgc='red';
      end
      set(obj.gui.(name),'BackgroundColor',bgc);set(obj.gui.(name),'String',txt);
      laststatus=status;
      set(obj.gui.(name),'Enable','inactive')
    end
    %% readback text
    function guiCreateReadbackText(obj,name,parent,rbkMethod,rbkParams,prec,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','text','FontName',obj.guiFont,'Units','normalized','String','---',...
        'Position',pos,'FontWeight','bold','BackgroundColor','black','ForegroundColor','green') ;
      obj.guiUpdateParams{end+1}{1}=name;
      obj.guiUpdateParams{end}{2}='text';
      obj.guiUpdateParams{end}{3}=rbkParams;
      obj.guiUpdateParams{end}{4}=prec;
      obj.guiUpdateMethod{end+1}=rbkMethod;
      if ~strcmp(obj.guiTimer.Running,'on')
        start(obj.guiTimer)
      end
    end
    %% popupmenu
    function guiCreatePopupmenu(obj,name,menutxt,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','popupmenu','String',menutxt,'FontName',obj.guiFont,'Units','normalized',...
        'Position',pos,'BackgroundColor','white','FontWeight','bold');
    end
    %% radiobutton
    function guiCreateRadiobutton(obj,name,txt,val,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','radiobutton','String',txt,'FontName',obj.guiFont,...
        'Units','normalized','Position',pos,'FontWeight','bold','Value',val);
    end
    %% listbox
    function guiCreateListbox(obj,name,txt,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','listbox','String',txt,'FontName',obj.guiFont,'Units','normalized',...
        'Position',pos,'Min',1,'Max',10000,'FontWeight','bold') ;
    end
    %% message box
    function guiCreateMessagebox(obj,parent,pos)
      if exist('parent','var') && ~isempty(parent)
        obj.guiCreateListbox('msgBox','',parent,pos);
        set(obj.gui.msgBox,'Max',obj.guiMsgBoxSize)
        if ~isempty(obj.guiMessageData)
          set(obj.gui.msgBox,'String',obj.guiMessageData)
        end
      end
      if isempty(obj.guiMessageData)
        obj.guiMessageData=cell(obj.guiMsgBoxSize,1);
        obj.guiMessageDataOrder=zeros(obj.guiMsgBoxSize,1);
      end
    end
    function guiLoadMessageboxData(obj,data,dataOrder)
      [~, nI]=sort(dataOrder);
      obj.guiAddToMessagebox('<<>>clear<<>>');
      if ~isempty(data)
        for idata=1:length(nI)
          if nI(idata)>0
            obj.guiAddToMessagebox(data{nI(idata)},'nodate');
          end
        end
        obj.guiAddToMessagebox(obj,'Old data loaded');
      end
    end
    function guiAddToMessagebox(obj,msg,cmd)
      persistent lastind
      if isequal(msg,'<<>>clear<<>>')
        lastind=[];
        obj.guiMessageData=cell(obj.guiMsgBoxSize,1);
        obj.guiMessageDataOrder=zeros(obj.guiMsgBoxSize,1);
        return
      end
      if isempty(lastind)
        lastind=0;
      end
      lastind=lastind+1;
      if lastind>length(obj.guiMessageData)
        lastind=1;
      end
      if exist('cmd','var') && strcmp(cmd,'nodate')
        obj.guiMessageData{lastind}=msg;
      else
        obj.guiMessageData{lastind}=sprintf('%s: %s',datestr(now,31),msg);
      end
      obj.guiMessageDataOrder(lastind)=max(obj.guiMessageDataOrder)+1;
      obj.guiDisplayMessageboxData;
    end
    function guiDisplayMessageboxData(obj)
      if ~isfield(obj.gui,'msgBox') || ~ishandle(obj.gui.msgBox); return; end;
      [val ind]=sort(obj.guiMessageDataOrder,'descend');
      set(obj.gui.msgBox,'String',obj.guiMessageData(ind(val~=0)));
    end
    %% table
    function guiCreateTable(obj,name,colnames,colfmt,colwid,coledit,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uitable(parent,'FontName',obj.guiFont,'ColumnName',colnames,'ColumnFormat', colfmt,...
        'ColumnWidth', colwid, 'ColumnEditable',coledit,'Units','normalized','Position',pos);
    end
    %% readback table
    function guiCreateReadbackTable(obj,name,colnames,colfmt,colwid,coledit,parent,rbkMethod,rbkParams,prec,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uitable(parent,'FontName',obj.guiFont,'ColumnName',colnames,'ColumnFormat', colfmt,...
        'ColumnWidth', colwid, 'ColumnEditable',coledit,'Units','normalized','Position',pos,'CellEditCallback',@(src,event)guiTimerFn(obj,src,event,'tableEdit'));
      obj.guiUpdateParams{end+1}{1}=name;
      obj.guiUpdateParams{end}{2}='table';
      obj.guiUpdateParams{end}{3}=rbkParams;
      obj.guiUpdateParams{end}{4}=prec;
      obj.guiUpdateMethod{end+1}=rbkMethod;
      if ~strcmp(obj.guiTimer.Running,'on')
        start(obj.guiTimer)
      end
    end
    %% togglebutton
    function guiCreateTogglebutton(obj,name,txt,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','togglebutton','String',txt,'FontName',obj.guiFont,'Units','normalized',...
        'Position',pos,'FontWeight','bold');
    end
    %% checkbox
    function guiCreateCheckbox(obj,name,txt,val,parent,pos)
      if ischar(parent); parent=obj.gui.(parent); end;
      obj.gui.(name)=uicontrol(parent,'Style','checkbox','String',txt,'FontName',obj.guiFont,...
          'Units','normalized','Position',pos,'FontWeight','bold','Value',val);
    end
    %% timer callback function
    function guiTimerFn(obj,~,~,cmd)
      switch cmd
        case 'start'
        case 'stop'
        otherwise
          if isempty(obj.guiUpdateMethod)
            stop(obj.guiTimer)
            return
          end
          for iUpdate=1:length(obj.guiUpdateMethod)
            if isempty(obj.guiUpdateParams{iUpdate}{1}) || ~isfield(obj.gui,obj.guiUpdateParams{iUpdate}{1}) || ~ishandle(obj.gui.(obj.guiUpdateParams{iUpdate}{1}))
              continue
            end
            try
              switch obj.guiUpdateParams{iUpdate}{2}
                case 'text'
                  updateVal=obj.(obj.guiUpdateMethod{iUpdate})(obj.guiUpdateParams{iUpdate}{1},obj.guiUpdateParams{iUpdate}{3});
                  if ~isempty(updateVal)
                    if ischar(updateVal)
                      set(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'String',updateVal(1:min(length(updateVal),obj.guiUpdateParams{iUpdate}{4})))
                    else
                      set(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'String',num2str(updateVal,obj.guiUpdateParams{iUpdate}{4}))
                    end
                  end
                case 'table'
                  tdata=get(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'Data');
                  updateVal=obj.(obj.guiUpdateMethod{iUpdate})(obj.guiUpdateParams{iUpdate}{1},obj.guiUpdateParams{iUpdate}{3});
                  % Only update non-editable cells
                  ce=get(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'ColumnEditable');
                  if isequal(size(tdata),size(updateVal))
                    updateVal(:,ce)=tdata(:,ce);
                  end
                  set(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'Data',updateVal);
                case 'tableEdit'
              end
            catch
              try
                set(obj.gui.(obj.guiUpdateParams{iUpdate}{1}),'String','???')
              catch
              end
            end
          end
      end
    end
  end
  
end

