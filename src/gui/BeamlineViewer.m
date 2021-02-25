function varargout = BeamlineViewer(varargin)
%
% BEAMLINEVIEWER Create a graphical user interface (GUI) viewer for
%   Lucretia beamline data structures.
%
% BeamlineViewer( ) creates a GUI viewer for viewing Lucretia BEAMLINE, 
%    GIRDER, KLYSTRON, and PS data.  The viewer allows users to scroll
%    through any of the aforementioned data structures and view the full
%    data fields of any one beamline element, girder, klystron or power
%    supply.  Searches of the BEAMLINE cell array are supported.  Clicking
%    on Element, Girder, Klystron, PS, Block, or Slices fields in the
%    detailed display brings up or highlights the relevant data objects.
%    Visually scroll through the beamline elements using the viewer,
%    clicking on magnet elements highlights them in the listboxes below
%    To be able to use the Twiss/bpm plotting functions, pass a Lucretia
%    Initial and/or Beam structure : BeamlineViewer(Initial,Beam)
% -------------------------------------------------------
%
% Version date:  26-Sept-2008.

%==========================================================================
%
% Here we have all the code which Matlab uses to manage the GUI itself at
% the lowest level:  respond to button pushes, initialize, etc.
%
%==========================================================================

% Last Modified by GUIDE v2.5 26-Sep-2008 16:30:40

% Begin initialization code - DO NOT EDIT

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BeamlineViewer_OpeningFcn, ...
                   'gui_OutputFcn',  @BeamlineViewer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% End initialization code - DO NOT EDIT

%==========================================================================

function BeamlineViewer_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
%
% This function has no output args, see OutputFcn.  It executes immediately
% prior to the Beamline Viewer being displayed to the user.
% 
% Input Arguments:
%   hObject    handle to figure
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)
%   varargin   command line arguments to BeamlineViewer (see VARARGIN)

% data field to remember which bit of data is displayed in the detailed
% view window
  global INSTR FL
  
  handles.DisplayedElement = 0 ;
  handles.DisplayType = 0 ;

% Update handles structure with the DisplayedElement field

  guidata(hObject, handles);

% populate listbox and set status based on success or failure

  status = GetBeamline(handles) ;
  handles = guidata(hObject) ;
  if isempty(FL)
    handles.output = status ;
  else
    handles.output = handles ;
  end % if using Floodland environment
  
  guidata(hObject, handles);
  
% on failure (due to problems with the BEAMLINE array) do the following:  

  if (status == 0)
      beep
      msgbox('Global cell array BEAMLINE is empty', ...
          'Beamline Viewer Error','error','modal') ;
      return ;
  elseif (status == -1)
      beep
      msgbox('Global cell array BEAMLINE is malformed', ...
          'Beamline Viewer Error','error','modal') ;
      return ;
  end

% clear the element display and the detailed display

  DisplayElement(handles,0,0) ;
  AuxDisplay(handles,0) ;
  
  % Floodland-specific functions
  if isempty(INSTR)
    set(handles.pushbutton7,'Visible','off');
  else
    set(handles.pushbutton7,'Visible','on');
  end % if INSTR global filled
  if isempty(FL)
    set(handles.pushbutton8,'Visible','off');
    set(handles.pushbutton9,'Visible','off');
    set(handles.pushbutton10,'Visible','off');
    set(handles.popupmenu3,'String',{'All'});
  else
    set(handles.pushbutton8,'Visible','on');
    set(handles.pushbutton9,'Visible','on');
    set(handles.pushbutton10,'Visible','on');
    if isfield(FL,'Section')
      sections=fieldnames(FL.Section);
      if ~isempty(sections)
        secs=get(handles.popupmenu3,'String');
        for isec=1:length(sections)
          secs{end+1}=sections{isec};
        end % for isec
        set(handles.popupmenu3,'String',secs);
      end % if ~empty secs
    end % if FL.Section
  end % if FL global filled
  
  if length(varargin)==1
    plotFunc(handles,'init',varargin{1});
  elseif length(varargin)==2
    plotFunc(handles,'init',varargin{1},varargin{2});
  else
    plotFunc(handles,'init')
  end % if passed Initial and/or Beam info

%==========================================================================
  
function varargout = BeamlineViewer_OutputFcn(hObject, eventdata, handles)
%
% Function which executes on exit from the BeamlineViewer GUI
%
% Output arguments:
%   varargout  cell array for returning output args (see VARARGOUT);
%
% Input arguments:
%   hObject    handle to figure
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  varargout{1} = handles.output;
  if ( (isnumeric(handles.output)) && (handles.output ~= 1) )
      delete(handles.figure1) ;
  end
  varargout{2}=[];

%==========================================================================

function listbox1_Callback(hObject, eventdata, handles)
%
% Function which executes when the selection in the Beamline listbox has
% changed
%
% Input arguments:
%   hObject    handle to listbox1 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

% On a double-click, get the element number and display that BEAMLINE
% member in the detailed display box

  if (strcmp(get(handles.figure1,'SelectionType'),'open')) || (~isempty(eventdata) && eventdata==2)
    eptr = get(handles.listbox1,'Value') ;
    DisplayElement(handles,eptr,0) ;
    plotFunc(handles,'init');
    set(handles.slider1,'Value',eptr);
    slider1_Callback(handles.slider1, eventdata, handles);
  end

%==========================================================================  
  
function listbox1_CreateFcn(hObject, eventdata, handles) %#ok<*INUSD,*DEFNU>
%
% Function which executes on creation of the beamline listbox
%
% Input arguments:
%   hObject    handle to listbox1 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    empty - handles not created until after all CreateFcns called

  if ispc && isequal(get(hObject,'BackgroundColor'), ...
          get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
  end

%==========================================================================  

function listbox2_Callback(hObject, eventdata, handles)
%
% Function manages the activity when the user selects a line in the
% detailed display.
%
% Input arguments:
%   hObject    handle to listbox2 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  global BEAMLINE GIRDER KLYSTRON PS

  if (strcmp(get(handles.figure1,'SelectionType'),'open')) || ~isempty(eventdata);
      
    SelectLine = get(handles.listbox2,'Value') ;
    ielem = handles.DisplayedElement ;
    
% we do different things depending on whether it's an element display or a
% display of one of the other types

    switch handles.DisplayType
     
      case 0 % element is displayed
        switch SelectLine
            case handles.Gptr
              if (BEAMLINE{ielem}.Girder ~= 0)
                AuxDisplay(handles,1) ;
                handles.AuxDisp = 1 ;
                set(handles.listbox3,'Value',BEAMLINE{ielem}.Girder) ;
                DisplayElement(handles,BEAMLINE{ielem}.Girder,1) ;
              end
            case handles.Kptr
              if (BEAMLINE{ielem}.Klystron ~= 0)
                AuxDisplay(handles,2) ;
                handles.AuxDisp = 2 ;
                set(handles.listbox3,'Value',BEAMLINE{ielem}.Klystron) ;
                DisplayElement(handles,BEAMLINE{ielem}.Klystron,2) ;
              end
            case handles.Pptr
              if (BEAMLINE{ielem}.PS ~= 0)
                AuxDisplay(handles,3) ;
                handles.AuxDisp = 3 ;
                set(handles.listbox3,'Value',BEAMLINE{ielem}.PS(1)) ;
                DisplayElement(handles,BEAMLINE{ielem}.PS(1),3) ;
              end
            case handles.Bptr
                set(handles.listbox1,'Value',...
                    BEAMLINE{ielem}.Block(1):BEAMLINE{ielem}.Block(2)) ;
            case handles.Sptr
                set(handles.listbox1,'Value',BEAMLINE{ielem}.Slices) ;
        end
        
      case {1,2,3} % girder, klystron or power supply is displayed
        switch handles.DisplayType
            case 1
                displayed = GIRDER{ielem} ;
            case 2
                displayed = KLYSTRON(ielem) ;
            case 3
                displayed = PS(ielem) ;
        end
        if (SelectLine == handles.Eptr)
            set(handles.listbox1,'Value',displayed.Element) ;
        end
        
    end
  end

%==========================================================================  

function listbox2_CreateFcn(hObject, eventdata, handles)
%
% Function which executes on creation of the detailed display listbox
%
% Input arguments:
%   hObject    handle to listbox1 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    empty - handles not created until after all CreateFcns called

if ispc && isequal(get(hObject,'BackgroundColor'), ...
        get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%==========================================================================

function listbox3_Callback(hObject, eventdata, handles)
%
% Executes on change in the selction of the auxiliary data structure
% (GIRDER, KLYSTRON, PS) display.
%
% Input arguments:
%   hObject    handle to listbox3 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  if (strcmp(get(handles.figure1,'SelectionType'),'open')) ;
    eptr = get(handles.listbox3,'Value') ;
    dt = get(handles.text1,'String') ;
    switch dt
        case 'GIRDER'
            dtype = 1 ;
        case 'KLYSTRON'
            dtype = 2 ;
        case 'PS'
            dtype = 3 ;
        case 'INSTR'
            dtype = 4 ;
    end
    DisplayElement(handles,eptr,dtype) ;
  end

%==========================================================================

function listbox3_CreateFcn(hObject, eventdata, handles)
%
% Function which executes on creation of the auxiliary listbox
%
% Input arguments:
%   hObject    handle to listbox1 (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    empty - handles not created until after all CreateFcns called

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%==========================================================================

function NewSearchButton_Callback(hObject, eventdata, handles)
%
% Handles the response to pressing the "New Search" button.
%
% Input arguments:
%   hObject    handle to NewSearchButton (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

% the response is completely handled in exterior code, but we tell it
% whether to do a new or repeat search with the 2nd argument...

  DoBeamlineSearch(handles,1) ;

%==========================================================================

function RepeatSearchButton_Callback(hObject, eventdata, handles)
%
% Handles the response to pressing the "Repeat Search" button.
%
% Input arguments:
%   hObject    handle to NewSearchButton (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

% the response is completely handled in exterior code, but we tell it
% whether to do a new or repeat search with the 2nd argument...

  DoBeamlineSearch(handles,0) ;

%==========================================================================

function RefreshBeamline_Callback(hObject, eventdata, handles)
%
% Executes in response to the "Refresh" button-push.  
%
% Input arguments:
%   hObject    handle to RefreshBeamline (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  status = GetBeamline(handles) ;
  handles = guidata(hObject) ;
  handles.output = status ;
      DisplayElement(handles,handles.DisplayedElement,...
                             handles.DisplayType) ;

%==========================================================================

function Girder_Callback(hObject, eventdata, handles)
%
% Executes in response to the "Girder" button push.
%
% Input arguments:
%   hObject    handle to Girder (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  AuxDisplay(handles,1) ;

%==========================================================================

function Klystron_Callback(hObject, eventdata, handles)
%
% Executes in response to the "Klystron" button push.
%
% Input arguments:
%   hObject    handle to Girder (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  AuxDisplay(handles,2) ;

%==========================================================================

function PS_Callback(hObject, eventdata, handles)
%
% Executes in response to the "Power Supply" button push.
%
% Input arguments:
%   hObject    handle to Girder (see GCBO)
%   eventdata  reserved - to be defined in a future version of MATLAB
%   handles    structure with handles and user data (see GUIDATA)

  AuxDisplay(handles,3) ;

%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================
%==========================================================================

% The functions below provide more of the logical "guts" of the Beamline
% Viewer.

%==========================================================================

function status = GetBeamline(handles)
%
% fills up the data structures needed to display the information in the
% Beamline view list box (on the left of the display)

  global BEAMLINE KLYSTRON PS GIRDER
  
% construct the string version of each name in BEAMLINE, along with its
% element class and its index
  
  if (isempty(BEAMLINE))
      status = 0 ;
      return ;
  elseif (~iscell(BEAMLINE))
      status = -1 ;
      return ;
  else
      status = 1 ;
  end
  eindex = 1:length(BEAMLINE) ;
  elem_shorthand = cell(length(BEAMLINE),1) ;
  svec = zeros(length(BEAMLINE),1) ;
  classvec = elem_shorthand ;
  namevec = elem_shorthand ;
  smax = 0 ;
  for count = eindex
      if (  (~isfield(BEAMLINE{count},'S'))     || ...
            (~isfield(BEAMLINE{count},'Class')) || ...
            (~isfield(BEAMLINE{count},'Name'))          )
         status = -1 ;
         return ;
      end
      svec(count) = BEAMLINE{count}.S ;
      if (abs(BEAMLINE{count}.S) > smax)
          smax = abs(BEAMLINE{count}.S) ;
      end
  end
  NIndex = floor(log(max(eindex))/log(10))+1 ;
  NLeftDigit = floor(log(smax)/log(10)) ;
  NDigit = NLeftDigit + 6 ;
  Format = ['%',num2str(NDigit,'%d'),'.4f'] ;
  for count = eindex
      indx = num2str(count,'%d') ;
      lindx = length(indx) ;
      if (lindx<NIndex)
          indx(lindx+1:NIndex) = ' ' ;
      end
      ist = num2str(BEAMLINE{count}.S,Format) ;
      lst = length(ist) ;
      spos = [] ;
      if (lst < NDigit)
          spos(1:(NDigit-lst)) = ' ' ;
      end
      lcl = length(BEAMLINE{count}.Class) ;
      classvec{count} = BEAMLINE{count}.Class ;
      cstr = [] ;
      if (lcl < 8)
          cstr(1:8-lcl) = ' ' ;
      end
      cstr = [cstr,BEAMLINE{count}.Class] ;
      spos = [spos,ist] ;
      namevec{count} = BEAMLINE{count}.Name ;
      eshort = [indx,'  ',spos,'  ', ...
          cstr,'  ',BEAMLINE{count}.Name] ;
      elem_shorthand{count} = eshort ;
  end
  handles.elements = elem_shorthand ;
  handles.class = classvec ;
  handles.name = namevec ;
  handles.spos = svec ;
  handles.index = eindex ;
  
% get the number of girders, klystrons, power supplies and store them

  handles.ngirder = length(GIRDER) ;
  handles.nklys = length(KLYSTRON) ;
  handles.nps = length(PS) ;
  
  guidata(handles.figure1,handles) ;
  set(handles.listbox1,'String',handles.elements,...
	'Value',1)

% and that's it

%==========================================================================

function DoBeamlineSearch(handles,NewSearch)

% function which does all the work of performing searches, managing the
% search dialog box and its associated error dialog boxes, and storing data
% from one call to the next when needed.

  persistent SearchClass SearchName SearchMinS SearchMaxS SearchDirection
  persistent SearchCount
  
% is this the first call?  if so, initialize

  if (isempty(SearchDirection))
      SearchClass = [] ;
      SearchName = [] ;
      SearchMinS = -Inf ;
      SearchMaxS =  Inf ;
      SearchDirection = 1 ;
      SearchCount = 1 ;
  end
  
  % first step:  if this is a new search, bring up the dialog box which
  % gets the user's desired search parameters
  
  if (NewSearch)
      
    if (SearchMinS == -Inf)
      MinS = [] ;
    else
      MinS = SearchMinS ;
    end
    if (SearchMaxS == Inf)
      MaxS = [] ;
    else
      MaxS = SearchMaxS ;
    end
    
    GoodReturn = 0 ;
    while (GoodReturn ~= 1)
      [a,b,c,d,e,f,g] = BeamlineSearch(SearchClass,SearchName,...
                                   num2str(MinS),num2str(MaxS),...
                                   SearchDirection,SearchCount) ;
      if (a==0)
        return ;
      end

% since it turns out that there are a couple of ways that an empty string
% can be empty, make sure that it's the right kind of empty now

      if (isempty(d))
          d = [] ;
      end
      if (isempty(e))
          e = [] ;
      end
      
      SearchClass = d ; SearchName = e ; SearchDirection = f ;
      SearchCount = g ;
      if (isempty(b))
          b = -Inf ;
      else
          b = str2double(b) ;
      end
      if (isempty(c))
          c = Inf ;
      else
          c = str2double(c) ;
      end
      if ( (isnan(b)) || (isnan(c)) )
          beep
          uiwait(msgbox('Invalid S limits specified',...
              'Beamline Search Error','error','modal')) ;
      else
          GoodReturn = 1 ;
          SearchMinS = b ;
          SearchMaxS = c ;
      end
    end
    
  end
  
% now we have all of the parameters, it is time to do the search!  Start by
% getting the current location of the highlighted element

  eptr = get(handles.listbox1,'Value') ;
  if (SearchDirection == 1)
      eptr = min(eptr) ;
  else
      eptr = max(eptr) ;
  end
  found = 0 ; found_index = [] ;
  while ( (found < SearchCount) && ...
          (eptr+SearchDirection > 1) && ...
          (eptr+SearchDirection < length(handles.index)) )
      
      eptr = eptr + SearchDirection ;
      
% apply the screens which are selected by the user; since the S position
% screen uses +/- infinity, it's always active

      if ( (handles.spos(eptr) < SearchMinS) || ...
           (handles.spos(eptr) > SearchMaxS)        )
         continue ;
      end
      
      if ( (~isempty(SearchClass)) && ...
           (~strcmpi(SearchClass,handles.class{eptr}))  )
         continue ;
      end

      if ( (~isempty(SearchName)) && ...
           (~strcmpi(SearchName,handles.name{eptr}))  )
         continue ;
      end

% if we got this far, then we found the next match

      found = found+1 ;
      found_index = [found_index eptr] ;
      
  end
  
% did we find something?  if so, move the focus to the found item

  if (found>0)
      set(handles.listbox1,'Value',found_index) ;
  else
      uiwait(msgbox('No matches found for the selected search parameters',...
          'Beamline Search: No Matches','warn','modal')) ;
  end

%==========================================================================

function DisplayElement(handles,eptr,idisp)

% gets the data for the detailed view list box and puts it into the
% relevant data structure for displaying.  The idisp argument tells it
% whether to display a BEAMLINE, GIRDER, KLYSTRON or PS structure.

  global BEAMLINE GIRDER KLYSTRON PS INSTR FL;
  
% clear the display on initialization  
  
  if (eptr == 0)
      set(handles.listbox2,'String',' ') ;
      set(handles.text3,'String',' ') ;
      handles.DisplayedElement = 0 ;
      guidata(handles.figure1,handles) ;
      return ;
  end
  
% if this is a for-real call, we display the element's properties:
 
  disptext = [] ;
  switch idisp
      case 0
          if (eptr <= length(BEAMLINE))
            dispdat = BEAMLINE{eptr} ;
            disptext = ['BEAMLINE{',num2str(eptr,'%d'),'}'] ;
          else
            eptr = 0 ;
          end
      case 1
          if (eptr <= length(GIRDER))
            dispdat = GIRDER{eptr} ;
            disptext = ['GIRDER{',num2str(eptr,'%d'),'}'] ;
          else
            eptr = 0 ;
          end
      case 2
          if (eptr <= length(KLYSTRON))
            dispdat = KLYSTRON(eptr) ;
            disptext = ['KLYSTRON(',num2str(eptr,'%d'),')'] ;
          else
            eptr = 0 ;
          end
      case 3 
          if (eptr <= length(PS))
            dispdat = PS(eptr) ;
            disptext = ['PS(',num2str(eptr,'%d'),')'] ;
          else
            eptr = 0 ;
          end
      case 4
          if (eptr <= length(INSTR))
            dispdat = INSTR{eptr} ;
            disptext = ['INSTR{',num2str(eptr,'%d'),'}'] ;
          else
            eptr = 0 ;
          end
  end
  if (eptr ~= 0)
    [bdata,e,g,k,p,b,s,in] = GetStringsFromStructureFields(dispdat,0) ;
  else
    bdata = {' '} ;
    e = 0 ; g = 0 ; k = 0 ; p = 0 ; b = 0 ; s = 0 ;
  end
  
  % Floodland data
  if ~isempty(FL)
    bdata{end+1}=' ';
    bdata{end+1}='Access permission: 0';
    if isfield(handles,'UserData') && ~isempty(handles.UserData) && iscell(handles.UserData) && ~isempty(handles.UserData{1})
      for ireq=1:length(handles.UserData{1})
        if idisp==handles.UserData{1}(ireq).reqType && ismember(eptr,handles.UserData{1}(ireq).req)
          bdata{end}=['Access permission: ',handles.UserData{1}(ireq).reqResp];
        end % if access granted
      end % for ireq
    end % if access request made
  end % ~empty FL
  
  set(handles.listbox2,'String',bdata,...
	'Value',1)
  set(handles.text3,'String',disptext) ;
  handles.DisplayedElement = eptr ;
  handles.DisplayType = idisp ;
  handles.Eptr = e ;
  handles.Gptr = g ;
  handles.Kptr = k ;
  handles.Pptr = p ;
  handles.Bptr = b ;
  handles.Sptr = s ;
  handles.Iptr = in ;
  guidata(handles.figure1,handles) ;
  
%==========================================================================

function [bdata,varargout] = GetStringsFromStructureFields(structure,indent)

% go to a data structure, get the field names and field values, and return
% them as a cell array.  If a field has subfields, process it later than
% the fields which have simple numeric values.  While we're at it, if the
% data structure has Element, Girder, Klystron, PS, Block, or Slices fields
% return their positions in the display for later use by the listbox
% managers.

  indstep = 4 ;
  e = 0 ; g = 0 ; k = 0 ; p = 0 ; b = 0 ; s = 0 ; in = 0 ;
  
% get the field names from the structure

  fn = fieldnames(structure) ;
  fnmax = 0 ;
  for count = 1:length(fn)
%      end
      fnmax = max([fnmax length(fn{count})]) ;
  end
  bdata = cell(1) ;
  
% get the data values and build an overall string...  
  
  linecount = 0 ;
  for count = 1:length(fn)
      f = getfield(structure,fn{count}) ; %#ok<GFLD>
      if (isstruct(f))
          continue ;
      end
      linecount = linecount + 1 ;
      switch fn{count}
          case 'Element' 
              e = linecount ;
          case 'Girder'
              g = linecount ;
          case 'Klystron'
              k = linecount ;
          case 'PS'
              p = linecount ;
          case 'Block'
              b = linecount ;
          case 'Slices'
              s = linecount ;
          case 'Instrument'
              in = linecount ;
      end
      fstring = [] ;
      fstring(1:fnmax-length(fn{count})+indent) = ' ' ;
      fstring = [fstring,fn{count},': '] ;
      if (isnumeric(f))
          fstring = [fstring,num2str(f)] ;
      elseif (ischar(f))
          fstring = [fstring,f] ;
      elseif (iscell(f))
          fstring = [fstring,'{cell array}'] ;
      end
      bdata{linecount} = fstring ;
  end
  
% now do the nested data structures via a recursive call to this very data
% structure

  for count = 1:length(fn)
      f = getfield(structure,fn{count}) ; %#ok<GFLD>
      if (~isstruct(f))
          continue ;
      end
      linecount = linecount+2 ;
      fstring = [] ;
      fstring(1:fnmax-length(fn{count})+indent) = ' ' ;
      fstring = [fstring,fn{count},': '] ;
      bdata{linecount} = fstring ;
      breturn = GetStringsFromStructureFields(f,indent+indstep) ;
      for count2 = 1:length(breturn)
          linecount = linecount + 1 ;
          bdata{linecount} = breturn{count2} ;
      end
  end
  if (nargout > 1)
      varargout{1} = e ;
      varargout{2} = g ;
      varargout{3} = k ;
      varargout{4} = p ;
      varargout{5} = b ;
      varargout{6} = s ;
      varargout{7} = in ;
  end

%==========================================================================

function AuxDisplay(handles, idisp)

% fill the center list box, which shows GIRDER, KLYSTRON, or PS index
% numbers.

global GIRDER KLYSTRON PS INSTR

  handles.AuxDisp = idisp ;
  
  switch idisp
      case 1 % GIRDER
          dtype = 'GIRDER' ;
          ndisp = length(GIRDER) ;
      case 2 % KLYSTRON
          dtype = 'KLYSTRON' ;
          ndisp = length(KLYSTRON) ;
      case 3 % PS
          dtype = 'PS' ;
          ndisp = length(PS) ;
      case 4 % INSTR
          dtype = 'INSTR' ;
          ndisp = length(INSTR) ;
      case 0 % no selection
          dtype = [] ;
          ndisp = 0 ;
  end
  disp = cell(ndisp,1) ;
  if (ndisp > 0)
      for count = 1:ndisp
          disp{count} = num2str(count,'%d') ;
      end
  end
  
  guidata(handles.figure1,handles) ;
  set(handles.listbox3,'String',disp,...
	'Value',min([ndisp 1])) ;
  set(handles.text1,'String',dtype) ;


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% INSTRUMENT data request
AuxDisplay(handles,4) ;



% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Make access request
reqInd=get(handles.listbox3,'Value');
if ~isempty(reqInd)
  switch handles.AuxDisp
    case 1
      [stat reqID] = AccessRequest({reqInd [] []});
    case 3
      [stat reqID] = AccessRequest({[] reqInd []});
    otherwise
      msgbox('Nothing to request here');
      return
  end % switch handles.AuxDisp
end % if any requests
if stat{1}~=1
  errordlg(stat{2},'Access Request error'); return;
end
if ~reqID
  errordlg('Access Refused','Access Request');
  granted='0';
else
  granted='1';
end % if refused
if ~isfield(handles,'UserData') || isempty(handles.UserData) || ~iscell(handles.UserData)
  handles.UserData{1}(1).reqResp = granted ;
  handles.UserData{1}(1).reqID = reqID;
  handles.UserData{1}(1).req = reqInd;
  handles.UserData{1}(1).reqType = handles.AuxDisp ;
else
  handles.UserData{1}(end+1).reqResp = granted ;
  handles.UserData{1}(end).reqID = reqID;
  handles.UserData{1}(end).req = reqInd;
  handles.UserData{1}(end).reqType = handles.AuxDisp ;
end % if request data empty
guidata(handles.figure1,handles) ;

listbox2_Callback(handles.listbox2, 1, handles)

% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Update status of access request(s) if made
if isfield(handles,'UserData') && ~isempty(handles.UserData) && iscell(handles.UserData) && ~isempty(handles.UserData{1})
  for ireq=1:length(handles.UserData{1})
    [stat resp] = AccessRequest('status',handles.UserData{1}(ireq).reqID);
    handles.UserData{1}(ireq).reqResp = resp ;
  end % for ireq
end % if request made
guidata(handles.figure1,handles) ;

listbox2_Callback(handles.listbox2, 1, handles)



% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
guiCloseFn('BeamlineViewer',handles);



% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
try
  guiCloseFn('BeamlineViewer',handles);
catch %#ok<CTCH>
  delete(hObject);
end % try/catch




% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over listbox2.
function listbox2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Set zoom slider parameters
set(handles.slider2,'Min',1);
set(handles.slider2,'Max',max(min(get(handles.slider1,'Max')-round(get(handles.slider1,'Value')),round(get(handles.slider1,'Value'))-get(handles.slider1,'Min')),1));
if round(get(handles.slider2,'Value')) < get(handles.slider2,'Min')
  set(handles.slider2,'Value',get(handles.slider2,'Min'));
elseif round(get(handles.slider2,'Value')) > get(handles.slider2,'Max')
  set(handles.slider2,'Value',get(handles.slider2,'Max'));
end % if past min/max
if get(handles.slider1,'Value') == get(handles.slider1,'Min')
  set(handles.slider2,'Max',get(handles.slider2,'Max')+0.1);
elseif get(handles.slider1,'Value') == get(handles.slider1,'Max')
  set(handles.slider2,'Min',get(handles.slider2,'Min')-0.1);
end % if at min or max

% redo plot
plotFunc(handles);



% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

% redo plot
plotFunc(handles);

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3


% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function plotFunc(handles,cmd,Initial_in,Beam_in)
% Function to handle plotting of stuff in axis
global BEAMLINE FL INSTR
persistent Initial Beam

if exist('Initial_in','var')
  Initial=Initial_in;
end
if exist('Beam_in','var')
  Beam=Beam_in;
end

if exist('cmd','var')
  if isequal(cmd,'init')
    secstr_in=get(handles.popupmenu3,'String');
    if ~iscell(secstr_in)
      secstr{1}=secstr_in;
    else
      secstr=secstr_in;
    end % if secstr cell
    switch lower(secstr{get(handles.popupmenu3,'Value')})
      case 'all'
        firstele=1; lastele=length(BEAMLINE);
      otherwise
        firstele=FL.Section.(secstr{get(handles.popupmenu3,'Value')}).inds(1);
        lastele=FL.Section.(secstr{get(handles.popupmenu3,'Value')}).inds(2);
    end % switch section
    set(handles.slider1,'Min',1);
    set(handles.slider1,'Max',lastele);
    set(handles.slider1,'Value',floor((lastele-firstele)/2));
    set(handles.slider2,'Min',1);
    set(handles.slider2,'Max',max(min(get(handles.slider1,'Max')-round(get(handles.slider1,'Value')),round(get(handles.slider1,'Value'))-get(handles.slider1,'Min')),1));
    set(handles.slider2,'Value',get(handles.slider2,'Max'));
  else
    error('unknown input to BeamlineViewer:plotFunc')
  end % if init
end % if cmd given

firstele=max(round(get(handles.slider1,'Value'))-round(get(handles.slider2,'Value')),get(handles.slider1,'Min'));
lastele=min(round(get(handles.slider1,'Value'))+round(get(handles.slider2,'Value')),get(handles.slider1,'Max'));

selval=round(get(handles.listbox4,'Value'));
% ahan=get(handles.uipanel1,'Children');
ahan=subplot(1,1,1,'Parent',handles.uipanel1);
if length(selval)==1 && selval==1
  if BEAMLINE{firstele}.S==BEAMLINE{lastele}.S
    return % nothing to plot
  end % if first and last S the same
  axis(ahan(1),[BEAMLINE{firstele}.S BEAMLINE{lastele}.S 1 10]);
  AddMagnetPlot(firstele,lastele,handles.uipanel1,'replace');
elseif lastele>firstele
  pnames_in=get(handles.listbox4,'String');
  if ~iscell(pnames_in)
    pnames{1}=pnames_in;
  else
    pnames=pnames_in;
  end % if pnames cell
  if sum(selval~=1)>1
    selval=[selval(end-1) selval(end)];
  else
    selval=selval(end);
  end % if multiple selects
  for isel=1:length(selval)
    if selval(isel)>=2 && selval(isel)<=9
      if (isempty(FL) || ~isfield(FL,'SimModel') || ~isfield(FL.SimModel,'Initial')) && isempty(Initial)
        errordlg('Must provide Lucretial Initial structure in BeamlinViewer call (Beamlineviewer(Initial)) or in FL.SimModel.Initial for twiss plots',...
          'Twiss Plot Error')
        return
      end % if have the right stuff for Twiss plot
      if ~isempty(FL) && isfield(FL,'SimModel') && isfield(FL.SimModel,'Initial')
        Initial_this=FL.SimModel.Initial;
      else
        Initial_this=Initial;
      end % get Initial
      [stat, twiss] = GetTwiss(firstele,lastele,Initial_this.x.Twiss,Initial_this.y.Twiss) ;
      if stat{1}~=1; return; end
      xplot(isel,:)=twiss.S(2:end);
      yplot(isel,:)=twiss.(pnames{selval(isel)})(2:end);
    elseif selval(isel)>9
      if (isempty(INSTR) || ~isfield(INSTR{end},'Index') || INSTR{end}.Index<lastele) && isempty(Beam)
        errordlg('Must provide Lucretia Beam structure in BeamlineViewer call or Floodland global INSTR structure for bpm plot',...
          'BPM plot error')
        return
      end % if right stuff for BPM plot
      if ~isempty(INSTR)
        xp=cellfun(@(x) BEAMLINE{x.Index}.S,INSTR);
        if selval(isel)==10
          yp=cellfun(@(x) x.Data(1),INSTR);
        elseif selval(isel)==11
          yp=cellfun(@(x) x.Data(2),INSTR);
        end % if x or y
        yplot(isel,:)=yp(cellfun(@(x) x.Index>=firstele&x.Index<=lastele,INSTR));
        xplot(isel,:)=xp(cellfun(@(x) x.Index>=firstele&x.Index<=lastele,INSTR));
      else
        [stat,beamout,instdata] = TrackThru( firstele, lastele, Beam, 1, 1, 0 );if stat{1}~=1; errordlg(stat{2:end},'Tracking error'); return; end;
        xplot(isel,:)=[instdata{1}.S];
        if selval(isel)==10
          yplot(isel,:)=[instdata{1}([instdata{1}.Index]>=firstele&[instdata{1}.Index]<=lastele).x];
        elseif selval(isel)==1
          yplot(isel,:)=[instdata{1}([instdata{1}.Index]>=firstele&[instdata{1}.Index]<=lastele).y];
        end % if x or y
        [xplot(isel,:) I]=sort(xplot(isel,:));
        yplot(isel,:)=yplot(isel,I);
      end % if FL or not
    end % if a twiss plot demand
  end % for isel
  if length(selval)==1
    plot(ahan(1),xplot,yplot)
  else
    plotyy(ahan(1),xplot(1,:),yplot(1,:),xplot(2,:),yplot(2,:))
  end % if 1 or 2 plots
  % Addmagnet plot doesn't work properly when adding on top of another plot
  % in uipanel because axes doesn't select new axes properly- need to fix
  % somehow -- use subplot and addmagnet replace on top subplot?---
%   ahan=subplot(2,1,1,'Parent',handles.uipanel1);
%   AddMagnetPlot(firstele,lastele,handles.uipanel1);
end % if plotmenu


% --- Executes on selection change in listbox4.
function listbox4_Callback(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns listbox4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox4


% --- Executes during object creation, after setting all properties.
function listbox4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over listbox4.
function listbox4_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to listbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




% --------------------------------------------------------------------
function uipanel1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to uipanel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ele=get(hObject,'UserData');
if ~isempty(ele)
  set(handles.slider1,'Value',ele);
  slider1_Callback(handles.slider1,[],handles);
  set(handles.listbox1,'Value',ele)
  listbox1_Callback(handles.listbox1,2,handles);
end % if not empty ele




% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
plotFunc(handles);

