classdef FlInstr < handle & FlGui & FlUtils
  %FLINSTR Class to assign and manipulate hardware associated with Instruments & beam diagnostics systems
  %   Use to store and manipulate lists of Instruments (e.g. BPMs) and read
  %   data from controls system and perform data reduction tasks (e.g. BPM
  %   data averaging and quality cuts). See the property descriptions for
  %   the available beam quality cuts which can be applied.
  %   Use objects of this class to both get hardware and store tracked data
  %   from simulation (hardware access provided by Floodland object in
  %   conjunction with the acquire method when not in sim mode [defined in
  %   Floodland object]).
  %   All instrument objects are loaded from the BEAMLINE global array at
  %   object creation time, attach hardware access details with the
  %   defineInstrHW method.
  %
  %   Get at data, applying quality cuts with the data property.
  %
  % Main public methods:
  %  + operator to merge another FlInstr object
  %  clearData - clears all internal data buffers
  %  defineInstrHW - define a variety of properties for hardware access to
  %                  a given instrument in this object
  %  meanstd - return mean and standard deviation data from data buffer
  %            according to selected cuts
  %  acquire - acquire a new batch of data (simulated or real depending on
  %            Floodland simmode property)
  %  setRef - set reference values for instruments based on current
  %           in-buffer values
  %  plot - plot functionality for current in-buffer data
  %  setResolution - Set resolution properties
  %  guiInstrChoice - main graphical user interface for this object, allows
  %                   selection from available instruments.
  %
  % See also:
  %  Floodland FlIndex FlGui FlApp Track
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc FlInstr">doc FlInstr</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  
  properties(Dependent)
    data % processed data
    dataDate % date stamps of processed data
  end
  properties
    lastdata = 'end' ; % index in data array of last required BPM reading, array index or 'end' for the last available
    ndata = 1 ; % depth of data buffer to reduce data from (averaging size)
    hwMultGet = false ; % if true then force getting of hardware values to be one PV at a time, else group all hardware requests together
    minq = [] ; % Minimum absolute bunch charge (Coulombs), cut any pulses or individual BPMs that reported lower than this
    maxRMS = [] ; % Maximum deviation from an RMS value over the last ndata points to allow
    maxVal = [] ; % Maximum total absolute value to allow
    qData = [] ; % Charge data
    databuffer = [] ; % Internal buffer of instrument data
    buffersize = 1000 ; % Maximum buffer depth
    maxStorage = 100 ; % MB - Max buffer size (buffersize adjusted lower if this cannot be fulfilled with buffersize setting)
    Beam = []; % Beam used in simulation, if blank, use Floodland one
    preCommand = {} ; % Control commands (PVs) to issue before getting hardware readout vals
    postCommand = {} ; % Control commands (PVs) to issue after getting hardware readout vals
    timeStamp = [] ; % Buffer of date/time stamps for data (timestamps from local hardware IOC's where available)
    monitorPV = []; % PV (string) or list of PVs (cell of strings) to monitor (INSTR hardware acquition happens upon this/these PV(s) posting new data)
    Class = {} ; % Instrument class ('MONI', 'PROF', 'WIRE',...)
    simBeam='BeamMacro'; % Preferred Floodland Beam type to use for simulation
    maxDataGap=10; % max gap between subsequent data requested in data field in secs
    useInstr % logical array to indicate instrument choice (only get hardware updates from this list)
    instrName % Names of INSTR BEAMLINE elements
    stopReq=false; % flag for requesting abort of specific in-class operations
    Type; % Instrument type ('stripline', 'ccav', etc...)
  end
  properties(Hidden)
    AIDA_BPMD = {} ; % AIDA BPMD allocation
    AIDA_NAMES = {} ; % AIDA Names
    instrChoiceFromGui % INSTR choice returned from user in GUI
  end
  properties(SetAccess=protected)
    Resolution = [] ; % instrument resolution array
    INSTRchannels = {} ; % readout channels active (see obj.chnames)
    pvname = {} ; % cell array of PV names to connect to hardware channels
    hwconv = {} ; % Conversion factors between Lucretia and hardware units
    protocol = {} ; % Hardware protocol: currently 'AIDA' or 'EPICS' supported
    bufpos = 1 ; % current position in the cyclic data buffer
    Data % [ bpmind , bufind , chanind ] - raw data
    DataDate % date/time stamp for raw data
    ref % reference Instrument readings (e.g. BPM set points/electrical offsets)
    referr % statistical error on above readings
    dispref % reference dispersion values at Instrument locations
    dispreferr % statistical error on above measurements
    pulseID % unique pulse ID number allocated to each new buffer entry
  end
  properties(Constant)
    chnames={'x' 'y' 'z' 'sig11' 'sig33' 'sig13' 'sig55' 'Q'}; % Allowed channel names
  end
  
  %% Set/Get methods
  methods
    function dataout = get.data(obj)
      % dataout = get.data(obj)
      % Unpack raw data from internal buffer and apply any requested cuts
      
      % raw data format
      % - get date sorted order
      [Y s1]=sort(obj.DataDate,'descend');
      dord=s1(~isnan(Y));
      dates=Y(~isnan(Y));
      if isequal(obj.lastdata,'end')
        d1=1;
      else
        d1=find(dates<obj.lastdata,1,'first');
      end
      if length(dord)<obj.ndata
        dataout=[];
        return
      end
      % Get max time between subsequent stored data requested
      maxDate=max(diff(dates(dord(d1:d1-1+obj.ndata))).*24.*3600);
      if obj.ndata>1 && ~isempty(obj.maxDataGap) && maxDate>obj.maxDataGap
        warning('Lucretia:Floodland:FlInstr:dateGapExceeded','Too large time gap between requested pulses: (%d s)',maxDate)
        dataout=[];
        return
      end
      dataout=obj.Data(obj.useInstr,dord(d1:d1-1+obj.ndata),:);
      % +++cuts+++
      % Q cut
      if ~isempty(obj.minq) && ~isempty(obj.qData)
        for ict=1:length(obj.qData)
          dataout(:,(dataout(ict,:,length(obj.chnames))<obj.minq),:)=NaN;
        end
      end
      % RMS cut
      if obj.ndata>4 && ~isempty(obj.maxRMS)
        for idim=1:2
          [I J]=find(abs(dataout(:,:,idim))>(repmat(mean(dataout(:,:,idim),2),1,obj.ndata)+repmat(std(dataout(:,:,idim),[],2),1,obj.ndata)));
          dataout(I,J,idim)=NaN;
        end
      end
      % Max value cut
      if ~isempty(obj.maxVal)
        for idim=1:2
          [I J]=find(abs(dataout(:,:,idim))>obj.maxVal);
          dataout(I,J,idim)=NaN;
        end
      end
      % subtract ref signal
      if any(obj.ref(:))
        try
          sz=size(dataout);
          for idata=1:sz(2)
            dataout(:,idata,:)=dataout(:,idata,:)-reshape(obj.ref(obj.useInstr,:),sz(1),1,sz(3));
          end
        catch ME
          if strcmp(ME.identifier,'MATLAB:getReshapeDims:notSameNumel')
            warning('Lucretia:refOrbitChange','INSTR selection choice changed since last reference orbit, reseting ref orbit to zero')
            obj.setRef('zero');
          end
        end
      end
    end
    function set.simBeam(obj,val)
      if ischar(val) && ismember(val,{'BeamMacro' 'BeamSingle' 'BeamSparse'})
        obj.simBeam=val;
      else
        error('Invalid beam name, allowed beam names are from Floodland class: (''BeamMacro'' ''BeamSingle'' ''BeamSparse'')')
      end
    end
  end
  
  %% Main public methods
  methods
    function obj = FlInstr
      global BEAMLINE
      tf=findcells(BEAMLINE,'TrackFlag');
      iind=0;
      for itf=1:length(tf)
        if isfield(BEAMLINE{tf(itf)}.TrackFlag,'GetBPMData')
          BEAMLINE{tf(itf)}.TrackFlag.GetBPMData=1;
        elseif isfield(BEAMLINE{tf(itf)}.TrackFlag,'GetBPMBeamPars')
          BEAMLINE{tf(itf)}.TrackFlag.GetBPMBeamPars=1;
        end
        if isfield(BEAMLINE{tf(itf)}.TrackFlag,'GetBPMData') || isfield(BEAMLINE{tf(itf)}.TrackFlag,'GetBPMBeamPars')
          iind=iind+1;
          obj.Class{iind} = BEAMLINE{tf(itf)}.Class;
          obj.instrName{iind} = BEAMLINE{tf(itf)}.Name;
          obj.Index(iind) = tf(itf) ;
          obj.Type{iind} = 'unknown' ;
          obj.Resolution(iind,:) = zeros(1,length(obj.chnames)) ;
          if isfield(BEAMLINE{tf(itf)},'Resolution')
            obj.Resolution(iind,1:2) = [BEAMLINE{tf(itf)}.Resolution BEAMLINE{tf(itf)}.Resolution] ;
          end
          obj.INSTRchannels{iind} = {};
          obj.hwconv{iind}=[];
          obj.pvname{iind}={};
          obj.protocol{iind}={};
          obj.preCommand{iind}={};
          obj.postCommand{iind}={};
          obj.timeStamp(iind)=0;
          obj.AIDA_BPMD{iind}=[];
          obj.AIDA_NAMES{iind}={};
        end
      end
      obj.useInstr = true(1,length(obj.Index)) ;
      % Initialise Data Array (x y z sig11 sig33 sig13 sig55 Q)
      % - check requested buffer size < requested max storage size else set
      % max possible based on requested max storage size
      maxdbsize=floor(obj.maxStorage/(8*length(obj.Index)*length(obj.chnames)/(1024*1024)));
      if obj.buffersize>maxdbsize
        obj.buffersize=maxdbsize;
      end
      % - initialise complete data array with NaN's
      obj.Data=NaN(length(obj.Index),obj.buffersize,length(obj.chnames));
      obj.DataDate=NaN(1,obj.buffersize);
      obj.pulseID = NaN(1,obj.buffersize);
      % Initialise reference values
      obj.setRef('zero');
      obj.dispref = zeros(size(obj.ref)) ;
      obj.dispreferr = zeros(size(obj.ref)) ;
    end
    function obj=plus(obj,A)
      % plus(obj,A)
      % Merge an object of FlInstr class to this one
      
      % If adding an FlIndex object- use that plus method instead
      if strcmp(class(A),'FlIndex')
        plus@FlIndex(obj,A);
        return
      end
      mc=metaclass(A);
      isFlInstr=strcmp(class(A),'FlInstr');
      if ~isempty(mc.SuperclassList) && ~isFlInstr
        for imc=1:length(mc.SuperclassList)
          isFlInstr=strcmp(mc.SuperclassList(imc).Name,'FlInstr');
          if isFlInstr; break; end;
        end
      end
      if ~isFlInstr; error('Addition only supported for 2 classes that are or inherit from FlInstr'); end;
      % get original data arrays
      origData=obj.Data;
      origDataDate=obj.DataDate;
      origPulseID=obj.pulseID;
      fn=properties(A);
      for ifn=1:length(fn)
        mp=findprop(A,fn{ifn});
        if ~strcmp(mp.DefiningClass.Name,'FlInstr')
          continue
        end
        if strcmpi(fn{ifn},'Data')
          continue
        elseif strcmpi(fn{ifn},'DataDate')
          continue
        elseif strcmp(fn{ifn},'buffersize')
          if A.buffersize>obj.buffersize
            obj.buffersize=A.buffersize;
          end
        elseif strcmp(fn{ifn},'maxStorage')
          if A.maxStorage>obj.maxStorage
            obj.maxStorage=A.maxStorage;
          end
        elseif strcmp(fn{ifn},'chnames')
          continue
        else
          obj.(fn{ifn})=A.(fn{ifn});
        end
      end
      asize=size(A.Data);
      osize=size(obj.Data);
      if asize(1)~=osize(1)
        error('Two data sets must contain the same number of BPMs')
      end
      if isempty(A.Data)
        newData=origData;
        dates=origDataDate;
        pulseid=origPulseID;
      elseif ~isempty(origData) % merge data
        dates=cat(2,obj.DataDate,A.DataDate);
        [dates Idates]=sort(dates);
        newData=cat(2,obj.Data,A.Data);
        newData=newData(:,Idates,:);
        pulseid=cat(2,obj.pulseID,A.pulseID);
        pulseid=pulseid(Idates);
      else
        newData=A.Data;
        dates=A.DataDate;
        pulseid=A.pulseID;
      end
      for ibuf=1:min(obj.buffersize,length(dates))
        obj.Data(:,ibuf,:)=newData(:,ibuf,:);
        obj.DataDate(ibuf)=dates(ibuf);
        obj.pulseID(ibuf)=pulseid(ibuf);
      end
      % Make buffer pointer largest non-NaN entry + 1
      nanData=find(squeeze(~isnan(obj.Data(1,:,1))));
      obj.bufpos=max(nanData)+1;
      if isempty(obj.bufpos) || obj.bufpos>length(nanData)
        obj.bufpos=1;
      end
      obj.AIDA_BPMD=A.AIDA_BPMD;
      obj.AIDA_NAMES=A.AIDA_NAMES;
    end
    function clearData(obj)
      % Clear internal data buffers
      obj.Data=NaN(length(obj.Index),obj.buffersize,length(obj.chnames));
      obj.DataDate=NaN(1,obj.buffersize);
      obj.pulseID = NaN(1,obj.buffersize);
      obj.bufpos = 1 ;
    end
    function defineInstrHW(obj,ind,chan,pvname,proto,conv,varargin)
      % defineInstrHW(obj,ind,chan,pvname,proto,conv,[AIDA_BPMD,AIDA_NAMES])
      % Setup INSTR HW channel
      % Provide BEAMLINE index (ind), channel name (chan, from obj.chnames
      % list), pvname string required to link to hardware, protocol to be
      % used (proto='AIDA' or 'EPICS'), conversion factor (conv)
      % For AIDA BPMs, provide AIDA BPMD parameter and AIDA_NAMES if
      % getting BPM data from a larger local array on the hardware
      global BEAMLINE
      if ~exist('ind','var') || ind<1 || ind>length(BEAMLINE)
        error('Must supply BEAMLINE index')
      end
      chanNames={'x' 'y' 'z' 'sig11' 'sig33' 'sig13' 'sig55' 'Q'};
      if exist('chan','var') && ~all(ismember(chan,chanNames))
        error('Must supply channel name chan = one of: ''x'' ''y'' ''z'' ''sig11'' ''sig33'' ''sig13'' ''sig55'' ''Q''')
      end
      if ~iscell(chan); chan={chan}; end;
      % Deal with deletion
      if nargin<=4 && ischar(pvname) && strcmp(pvname,'delete')
        if exist('chan','var')
          wch=ismember(obj.INSTRchannels{ind},chan);
        else
          wch=true(size(obj.INSTRchannels{ind}));
        end
        if any(wch)
          obj.INSTRchannels{ind}=obj.INSTRchannels{ind}(~wch);
          obj.pvname{ind}=obj.pvname{ind}(~wch);
          obj.protocol{ind}=obj.protocol{ind}(~wch);
          obj.hwconv{ind}=obj.hwconv{ind}(~wch);
          if ~isempty(obj.AIDA_BPMD{ind})
            if sum(wch)>1 && length(obj.AIDA_BPMD)>1
              obj.AIDA_BPMD{ind}=obj.AIDA_BPMD{ind}(~wch);
            else
              obj.AIDA_BPMD{ind}=[];
            end
          end
        end
        return
      end
      if ~exist('chan','var')
        error('Must supply channel name chan = one of: ''x'' ''y'' ''z'' ''sig11'' ''sig33'' ''sig13'' ''sig55'' ''Q''')
      end
      if ~iscell(pvname); pvname={pvname}; end;
      if ~exist('pvname','var') || length(pvname)~=length(chan)
        error('Must supply pvname = char or cell of chars same length as chan')
      end
      if ~exist('proto','var') || (iscell(proto)&&length(proto)~=length(chan))
        error('Must supply ''proto'', either string or cell of strings = length of ''pvnames'' given')
      end
      if ~iscell(proto)
        proto_in=proto; proto=cell(1,length(chan));
        for ichan=1:length(chan); proto{ichan}=proto_in; end;
      end
      if ~all(ismember(proto,{'EPICS' 'AIDA'}))
        error('Only supported protocols currently are ''EPICS'' or ''AIDA''')
      end
      if length(conv)>1 && length(conv)~=length(chan)
        error('Must supply ''conv'', either scalar or vector of same length as ''chan''')
      end
      if length(conv)==1
        conv_in=conv; conv=zeros(1,length(chan));
        for ichan=1:length(chan); conv(ichan)=conv_in; end;
      end
      % Process optional additional args
      aida_bpmd=[]; aida_names={};
      if ~isempty(varargin)
        for iarg=1:length(varargin)
          if strcmpi(varargin{iarg},'AIDA_BPMD')
            if length(varargin)<iarg+1
              error('Must supply must supply vector of same length as chan or scalar as next argument after ''AIDA_BPMD''')
            end
            if length(varargin{iarg+1})~=length(chan) && length(varargin{iarg+1})==1
              for ichan=1:length(chan)
                aida_bpmd(ichan)=varargin{iarg+1};
              end
            elseif length(aida_bpmd)~=length(chan)
              error('If specifying AIDA_BPMD, then must supply vector of same length as chan or scalar')
            else
              aida_bpmd=varargin{iarg+1};
            end
          elseif strcmpi(varargin{iarg},'AIDA_NAMES')
            if length(varargin)<iarg+1
              error('Must supply must supply vector of same length as chan or scalar as next argument after ''AIDA_NAMES''')
            end
            if length(varargin{iarg+1})~=length(chan) && (ischar(varargin{iarg+1}) || length(varargin{iarg+1})==1)
              for ichan=1:length(chan)
                aida_names{ichan}=varargin{iarg+1};
              end
            elseif length(aida_bpmd)~=length(chan)
              error('If specifying AIDA_NAMES, then must supply cell vector of same length as chan or char or cell scalar')
            else
              aida_names=varargin{iarg+1};
            end
          end
        end
      end
      for ichan=1:length(chan)
        if isempty(obj.INSTRchannels{ind}) || ~ismember(chan{ichan},obj.INSTRchannels{ind})
          obj.INSTRchannels{ind}{end+1}=chan{ichan};
          obj.pvname{ind}{end+1}=pvname{ichan};
          obj.protocol{ind}{end+1}=proto{ichan};
          obj.hwconv{ind}(end+1)=conv(ichan);
          if ~isempty(aida_bpmd)
            obj.AIDA_BPMD{ind}(end+1)=aida_bpmd(ichan);
          end
          if ~isempty(aida_names)
            obj.AIDA_NAMES{ind}{end+1}=aida_names{ichan};
          end
        elseif ismember(chan{ichan},obj.INSTRchannels{ind})
          thisInd=ismember(obj.INSTRchannels{ind},chan{ichan});
          obj.INSTRchannels{ind}{thisInd}=chan{ichan};
          obj.pvname{ind}{thisInd}=pvname{ichan};
          obj.protocol{ind}{thisInd}=proto{ichan};
          obj.hwconv{ind}(thisInd)=conv(ichan);
          if ~isempty(aida_bpmd)
            obj.AIDA_BPMD{ind}(thisInd)=aida_bpmd(ichan);
          end
          if ~isempty(aida_names)
            obj.AIDA_NAMES{ind}(thisInd)=aida_names(ichan);
          end
        end
      end
    end
    function [meanData stdData] = meanstd(obj)
      % [meanData stdData] = meanstd(obj)
      % Return mean and standard deviation of data in buffer to depth
      % obj.ndata
      thisData=obj.data;
      if isempty(thisData)
        meanData=[]; stdData=[]; return;
      end
      sz=size(thisData);
      meanData=zeros(sz(1),sz(3));
      stdData=meanData;
      for inst=1:sz(1)
        for ich=1:sz(3)
          d=squeeze(thisData(inst,:,ich)); d=d(~isnan(d));
          if ~isempty(d)
            meanData(inst,ich)=mean(d);
            stdData(inst,ich)=std(d);
          else
            meanData(inst,ich)=NaN;
            stdData(inst,ich)=NaN;
          end
        end
      end
    end
    function acquire(obj,FL,npulse)
      % acquire(obj,FL,npulse)
      % acquire obj.ndata pulse(s) of new data from hardware (if monitor PVs defined,
      % then wait for these to post new data first)
      % Supply Floodland object FL
      % If required, overide value in obj.ndata with 'npulse' variable
      global BEAMLINE
      if ~exist('FL','var') || ~strcmp(class(FL),'Floodland')
        error('Must pass an ''FL'' handle of class ''Floodland''')
      end
      if ~exist('npulse','var')
        npulse=obj.ndata;
      end
      % simulated data or real data
      if FL.issim
        % set BEAMLINE resolutions to match this object
        initRes=zeros(1,length(obj.Index));
        for indx=1:length(obj.Index)
          if isfield(BEAMLINE{obj.Index(indx)},'Resolultion')
            initRes(indx)=BEAMLINE{obj.Index(indx)}.Resolution;
            BEAMLINE{obj.Index(indx)}.Resolution=min(obj.Resolution(indx,1:2));
          end
        end
        if isempty(obj.Beam)
          Beam=FL.(obj.simBeam);
        else
          Beam=obj.Beam;
        end
        for ipulse=1:npulse
          if obj.stopReq
            return
          end
          obj.accum_sim(Beam);
        end
        % Return BEAMLINE resolution
        for indx=1:length(obj.Index)
          if isfield(BEAMLINE{obj.Index(indx)},'Resolultion')
            BEAMLINE{obj.Index(indx)}.Resolution=initRes(indx);
          end
        end
      else
        obj.accum_real(FL,npulse);
      end
    end
    function setRef(obj,cmd)
      % setRef(obj,cmd)
      % set reference orbit
      if exist('cmd','var') && strcmp(cmd,'zero')
        sz=size(obj.Data);
        obj.ref=zeros(sz(1),sz(3));
      else
        [data dataerr]=obj.meanstd;
        obj.ref(obj.useInstr,:)=data;
        obj.referr(obj.useInstr,:)=dataerr;
      end
    end
    function plot(obj,varargin)
      % plot(obj,dim,[type, magbar])
      % Plot data in buffer according to buffer size and cuts requested
      % dim=one of channels in obj.chnames
      % type is 'orbit' or 'data' (default orbit) data shows points with
      % error bars, orbit shows connected lines without errors
      % magbar=true|false (default true), show graphical depiction of
      % magnets above plot or not
      global BEAMLINE
      % Extract arguments
      if nargin<2
        error('Must at least supply ''dim'' argument')
      end
      types={'orbit' 'data'};
      newarg={}; ax=[]; type='orbit';
      if isnumeric(varargin{1})
        ax=varargin{1};
        if nargin<3; error('Must supply dimension to plot choose from:\n%s',evalc('disp(obj.chnames)')); end;
        for iarg=1:nargin-2
          newarg{end+1}=varargin{iarg+1};
        end
      else
        newarg=varargin;
      end
      if ~all(ismember(newarg{1},obj.chnames))
        error('Must supply ''dim'' to plot: choose from,\n%s',evalc('disp(obj.chnames)'))
      end
      dim=newarg{1}; if ~iscell(dim); dim={dim}; end;
      if length(newarg)>1
        type=newarg{2};
      end
      popts=[]; domagplot=true;
      if ~ischar(type) || ~ismember(lower(type),types)
        if islogical(type)
          domagplot=type;
        else
          type='orbit';
          popts=type;
        end
      end
      if length(newarg)>2 && isempty(popts)
        popts=newarg{3};
      elseif length(newarg)>2 && islogical(popts)
        domagplot=newarg{3};
      end
      if length(newarg)>3
        domagplot=newarg{4};
      end
      % Force orbit type if ndata=1
      if obj.ndata==1; type='orbit'; end;
      
      % Perform plot
      s=arrayfun(@(x) BEAMLINE{x}.S,obj.Index(obj.useInstr));
      
      % get mean and rms data
      [data dataerr]=obj.meanstd;
      
      % make new axis if none provided
      if isempty(ax)
        ax=axes;
      end
      phan=get(ax,'Parent');
      switch lower(type)
        case 'orbit'
          for idim=1:length(dim)
            if length(dim)==1
              if isempty(popts)
                plot(ax,s,data(:,ismember(obj.chnames,dim{idim})))
              else
                plot(ax,s,data(:,ismember(obj.chnames,dim{idim})),popts)
              end
            else
              ax(idim)=subplot(length(dim),1,idim,'Parent',phan);
              if isempty(popts)
                plot(s,data(:,ismember(obj.chnames,dim{idim})))
              else
                plot(s,data(:,ismember(obj.chnames,dim{idim})),popts)
              end
            end
          end
        case 'data'
          for idim=1:length(dim)
            if length(dim)==1
              if isempty(popts)
                errorbar(ax,s,data(:,ismember(obj.chnames,dim{idim})),dataerr(:,ismember(obj.chnames,dim{idim})))
              else
                errorbar(ax,s,data(:,ismember(obj.chnames,dim{idim})),dataerr(:,ismember(obj.chnames,dim{idim})),popts)
              end
            else
              ax(idim)=subplot(length(dim),1,idim,'Parent',phan);
              if isempty(popts)
                errorbar(s,data(:,ismember(obj.chnames,dim{idim})),dataerr(:,ismember(obj.chnames,dim{idim})))
              else
                errorbar(s,data(:,ismember(obj.chnames,dim{idim})),dataerr(:,ismember(obj.chnames,dim{idim})),popts)
              end
            end
          end
      end
      axes(ax(1)); %#ok<MAXES>
      if domagplot
        AddMagnetPlot(obj.Index(1),obj.Index(end));
      end
    end
    function setResolution(obj,ind,chan,val)
      % setResolution(obj)
      %  Set internal resolution values for instrument devices to BEAMLINE
      %  values
      % setResolution(obj,ind,chan,val)
      %  Set resolution of instr entry 'ind', channel 'chan' to 'val'
      
      global BEAMLINE
      % Set according to BEAMLINE
      if ~exist('ind','var')
        for indx=1:length(obj.Index)
          obj.Resolution(indx,[1 2])=repmat(BEAMLINE{obj.Index(indx)}.Resolution,1,2);
        end
        return
      end
      obj.Resolution(ind,ismember(obj.chnames,chan))=val;
    end
    function han=guiInstrChoice(obj)
      % Setup GUI and display
      
      % Create main figure window
      obj.guiCreateFigure('guiInstrChoice','Choose INSTR',[800 600]);
      set(obj.gui.guiInstrChoice,'CloseRequestFcn',@(src,event)guiInstrCallback(obj,src,event));
      han=obj.gui.guiInstrChoice;
      % Define borders
      border_top=0.02; 
      border_bottom=0.02;
      border_left=0.02;
      border_right=0.02;
      % Define element areas
      area_ver=[80 5 15].*0.01; %
      area_hor={[45 10 45].*0.01 [22 4 22 4 22 4 22].*0.01};
      % User Area
      userSize=[(1-(border_left+border_right)) (1-(border_top+border_bottom))];
      % Define choice box panels
      cpanpos{1}=[border_left border_bottom+(area_ver(3)+area_ver(2))*userSize(2)];
      cpanpos{2}=[border_left+(area_hor{1}(1)+area_hor{1}(2))*userSize(1) ...
        border_bottom+(area_ver(3)+area_ver(2))*userSize(2)];
      cpansize(1)=userSize(1)*area_hor{1}(1) ;
      cpansize(2)=userSize(2)*area_ver(1) ;
      cpanborder=[0.01 0.01];
      obj.guiCreatePanel('cbox1_panel','Instr Selection','guiInstrChoice',[cpanpos{1}(1) cpanpos{1}(2) cpansize]);
      obj.guiCreatePanel('cbox2_panel','Instr Available','guiInstrChoice',[cpanpos{2}(1) cpanpos{2}(2) cpansize]);
      % Define choice listboxes
      obj.guiCreateListbox('instrCbox1','','cbox1_panel',[cpanborder 1-2*cpanborder]);
      obj.guiCreateListbox('instrCbox2','','cbox2_panel',[cpanborder 1-2*cpanborder]);
      % Define selection buttons
      selborder=0.01;
      selsize=[area_hor{1}(2)*userSize(1)*0.9 cpansize(2)*0.1];
      selpos=[border_left+userSize(1)*(area_hor{1}(1)+area_hor{1}(2)*selborder) ...
        border_bottom+userSize(2)*(area_ver(3)+area_ver(2))]+[0 cpansize(2)*0.6];
      obj.guiCreatePushbutton('selbutton1','->','guiInstrChoice',[selpos selsize]);
      set(obj.gui.selbutton1,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      selpos=selpos-[0 cpansize(2)*0.2];
      obj.guiCreatePushbutton('selbutton2','<-','guiInstrChoice',[selpos selsize]);
      set(obj.gui.selbutton2,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      % Define option panels
      oppansize{1}=[area_hor{2}(1).*userSize(1) area_ver(3)*userSize(2)];
      oppansize{2}=[area_hor{2}(3).*userSize(1) area_ver(3)*userSize(2)];
      oppansize{3}=[area_hor{2}(5).*userSize(1) area_ver(3)*userSize(2)];
      oppansize{4}=[area_hor{2}(7).*userSize(1) area_ver(3)*userSize(2)];
      oppanpos={[border_left border_bottom] ...
        [border_left+userSize(1)*sum(area_hor{2}(1:2)) border_bottom] ...
        [border_left+userSize(1)*sum(area_hor{2}(1:4)) border_bottom] ...
        [border_left+userSize(1)*sum(area_hor{2}(1:6)) border_bottom]};
      obj.guiCreatePanel('oppan1','Sort by...','guiInstrChoice',[oppanpos{1} oppansize{1}]);
      obj.guiCreatePanel('oppan4','Make Selection','guiInstrChoice',[oppanpos{4} oppansize{4}]);
      % List options
      % - display
      border=0.02;
      osize=(1-3*border)/2;
      obj.guiCreateRadiobutton('display_alpha','Name',0,'oppan1',[border border 1-2*border osize]);
      set(obj.gui.display_alpha,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      obj.guiCreateRadiobutton('display_s','S',1,'oppan1',[border border*2+osize 1-2*border osize]);
      set(obj.gui.display_s,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      % - make selection
      obj.guiCreatePushbutton('select_accept','Accept','oppan4',[border border 1-2*border osize]);
      set(obj.gui.select_accept,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      set(obj.gui.select_accept,'BackgroundColor','green');
      obj.guiCreatePushbutton('select_cancel','Cancel','oppan4',[border border*2+osize 1-2*border osize]);
      set(obj.gui.select_cancel,'Callback',@(src,event)guiInstrCallback(obj,src,event));
      set(obj.gui.select_cancel,'BackgroundColor','red');
      % Update fields
      obj.instrChoiceFromGui=obj.useInstr;
      obj.updateInstrGui;
    end
  end
  
  %% GUI callbacks
  methods(Hidden)
    function guiInstrCallback(obj,src,~)
      % Process GUI callbacks
      if src==obj.gui.display_s
        if get(obj.gui.display_s,'Value')
          set(obj.gui.display_alpha,'Value',false)
          obj.updateInstrGui;
        else
          set(obj.gui.display_s,'Value',true)
        end
      elseif src==obj.gui.display_alpha
        if get(obj.gui.display_alpha,'Value')
          set(obj.gui.display_s,'Value',false)
          obj.updateInstrGui;
        else
          set(obj.gui.display_alpha,'Value',true)
        end
      elseif src==obj.gui.select_accept
        delete(obj.gui.guiInstrChoice);
        obj.gui.guiInstrChoice=[];
      elseif src==obj.gui.select_cancel || src==obj.gui.guiInstrChoice
        obj.instrChoiceFromGui=[];
        delete(obj.gui.guiInstrChoice);
        obj.gui.guiInstrChoice=[];
      elseif src==obj.gui.selbutton1
        sel=get(obj.gui.instrCbox1,'Value');
        mI=get(obj.gui.instrCbox1,'UserData');
        obj.instrChoiceFromGui(mI(sel))=false;
        set(obj.gui.instrCbox1,'Value',1)
        obj.updateInstrGui;
      elseif src==obj.gui.selbutton2
        sel=get(obj.gui.instrCbox2,'Value');
        mI=get(obj.gui.instrCbox2,'UserData');
        obj.instrChoiceFromGui(mI(sel))=true;
        set(obj.gui.instrCbox2,'Value',1)
        obj.updateInstrGui;
      else
        obj.updateInstrGui;
      end
    end
    function FlInstr_selectCuts(obj,~,~)
      % GUI selection cuts
      
      % Create main figure window
      obj.guiCreateFigure('FlInstr_selectCuts','Select BPM Cuts',[700 150]);
      border=0.02;
      psize=[(1-6*border)/5 (1-3*border)/2];
      csize=[1-border*2 (1-border*3)/2];
      % min Q panel
      obj.guiCreatePanel('minqpanel','min Q','FlInstr_selectCuts',[border border+psize(2) psize]) ;
      obj.guiCreateCheckbox('FlInstr_selectCuts_minq_enable','Enable',~isempty(obj.minq),'minqpanel',[border 1-border-csize(2) csize]);
      obj.guiCreateEdit('FlInstr_selectCuts_minq_val',num2str(obj.minq),'minqpanel',[border border csize]);
      % max RMS panel
      obj.guiCreatePanel('maxrmspanel','max RMS','FlInstr_selectCuts',[border*2+psize(1) border+psize(2) psize]) ;
      obj.guiCreateCheckbox('FlInstr_selectCuts_maxrms_enable','Enable',~isempty(obj.maxRMS),'maxrmspanel',[border 1-border-csize(2) csize]);
      obj.guiCreateEdit('FlInstr_selectCuts_maxrms_val',num2str(obj.maxRMS),'maxrmspanel',[border border csize]);
      % max Val panel
      obj.guiCreatePanel('maxvalpanel','max Val','FlInstr_selectCuts',[border*3+psize(1)*2 border+psize(2) psize]) ;
      obj.guiCreateCheckbox('FlInstr_selectCuts_maxval_enable','Enable',~isempty(obj.maxVal),'maxvalpanel',[border 1-border-csize(2) csize]);
      obj.guiCreateEdit('FlInstr_selectCuts_maxval_val',num2str(obj.maxVal),'maxvalpanel',[border border csize]);
      % max data gap
      obj.guiCreatePanel('maxdgappanel','max data gap / s','FlInstr_selectCuts',[border*4+psize(1)*3 border+psize(2) psize]) ;
      obj.guiCreateCheckbox('FlInstr_selectCuts_maxdgap_enable','Enable',~isempty(obj.maxDataGap),'maxdgappanel',[border 1-border-csize(2) csize]);
      obj.guiCreateEdit('FlInstr_selectCuts_maxdgap_val',num2str(obj.maxDataGap),'maxdgappanel',[border border csize]);
      % # BPM ave
      obj.guiCreatePanel('nbpmpanel','#Pulses','FlInstr_selectCuts',[border*5+psize(1)*4 border+psize(2) psize]) ;
      obj.guiCreateEdit('FlInstr_selectCuts_nbpm',num2str(obj.ndata),'nbpmpanel',[border border csize]);
      % commit
      obj.guiCreatePushbutton('FlInstr_selectCuts_commit','Commit','FlInstr_selectCuts',[border border psize(1) psize(2)*0.5])
      set(obj.gui.FlInstr_selectCuts_commit,'Callback',@(src,event)guiSelectCuts_Callback(obj,src,event));
      % cancel
      obj.guiCreatePushbutton('FlInstr_selectCuts_cancel','Cancel','FlInstr_selectCuts',[border*2+psize(1) border psize(1) psize(2)*0.5]);
      set(obj.gui.FlInstr_selectCuts_cancel,'Callback',@(src,event)guiSelectCuts_Callback(obj,src,event));
    end
    function guiSelectCuts_Callback(obj,src,~)
      if src==obj.gui.FlInstr_selectCuts_commit
        if get(obj.gui.FlInstr_selectCuts_minq_enable,'Value')
          obj.minq=str2double(get(obj.gui.FlInstr_selectCuts_minq_val,'String'));
        else
          obj.minq=[];
        end
        if get(obj.gui.FlInstr_selectCuts_maxrms_enable,'Value')
          obj.maxRMS=str2double(get(obj.gui.FlInstr_selectCuts_maxrms_val,'String'));
        else
          obj.maxRMS=[];
        end
        if get(obj.gui.FlInstr_selectCuts_maxval_enable,'Value')
          obj.maxVal=str2double(get(obj.gui.FlInstr_selectCuts_maxval_val,'String'));
        else
          obj.maxVal=[];
        end
        if get(obj.gui.FlInstr_selectCuts_maxdgap_enable,'Value')
          obj.maxDataGap=str2double(get(obj.gui.FlInstr_selectCuts_maxdgap_val,'String'));
        else
          obj.maxDataGap=[];
        end
        obj.ndata=str2double(get(obj.gui.FlInstr_selectCuts_nbpm,'String'));
        delete(obj.gui.FlInstr_selectCuts);
        obj.gui=rmfield(obj.gui,'FlInstr_selectCuts');
      elseif obj.gui.FlInstr_selectCuts_cancel
        delete(obj.gui.FlInstr_selectCuts);
        obj.gui=rmfield(obj.gui,'FlInstr_selectCuts');
      end
    end
  end
  
  %% Internal private methods
  methods(Access=protected)
    function accum_real(obj,FL,npulse)
      % accum_real(obj,FL,npulse)
      % Accumulate data from control system
      % Supply Floodland object FL and supply npulses to acquire
      persistent moni_set lastpulseID
      
      % form PV lists by protocol type
      pvlist{1}={}; pvlist{2}={};
      for ind=1:length(obj.Index)
        if ~obj.useInstr(ind)
          continue
        end
        if ~isempty(obj.pvname{ind})
          for ipv=1:length(obj.pvname{ind})
            proto=obj.protocol{ind}{ipv};
            switch proto
              case 'EPICS'
                pc=1;
              case 'AIDA'
                pc=2;
              otherwise
                warning('Lucretia:Floodland:FlInstrData:missingprotocol','protocol not supported (%s, Index: %d)',proto,ind)
                continue
            end
            pvlist{pc}{end+1,1}=obj.pvname{ind}{ipv};
            pvlist{pc}{end,2}={ind find(ismember(obj.chnames,obj.INSTRchannels{ind}{ipv})) obj.INSTRchannels{ind}{ipv}};
            pvlist{pc}{end,3}=obj.hwconv{ind}(ipv);
            if ~isempty(obj.AIDA_BPMD{ind})
              pvlist{pc}{end,4}=obj.AIDA_BPMD{ind}(ipv);
            end
            if ~isempty(obj.AIDA_NAMES{ind})
              pvlist{pc}{end,5}=obj.AIDA_NAMES{ind}{ipv};
            end
          end
        end
      end
      
      % Fetch INSTR data from controls
      npc=[];
      if obj.hwMultGet
        ploop=1;
      else
        ploop=npulse;
      end
      for idata=1:ploop
        if obj.stopReq
          return
        end
        % Wait for new data?
        try
          if ~isempty(obj.monitorPV)
            if iscell(obj.monitorPV)
              for ipv=1:length(obj.monitorPV)
                if isempty(moni_set)
                  lcaSetMonitor(obj.monitorPV{ipv});
                end
                lcaNewMonitorWait(obj.monitorPV{ipv});
              end
            else
              if isempty(moni_set)
                lcaSetMonitor(obj.monitorPV);
              end
              lcaNewMonitorWait(obj.monitorPV);
            end
            if isempty(moni_set); moni_set=true; end;
          else % or wait for 1 pulse duration if repRate defined and not sim
            if ~isempty(FL.repRate) && ~FL.issim
              pause(1/FL.repRate);
            end
          end
        catch ME
          warning('Lucretia:Floodland:FlInstrData:newMoniFail','Failed to set or use EPICS monitor PV to wait for new event:\n%s\n',ME.message)
        end
        % --- EPICS data
        try
          if ~isempty(pvlist{1})
            [cv tv]=FlCA('lcaGet',pvlist{1}(:,1));
            if iscell(tv); tv=cell2mat(tv); end;
            if iscell(cv); cv=cell2mat(cv); end;
            if obj.hwMultGet
              for iidata=1:npulse
                tvals(iidata,1)=tv(:,iidata);
                cvals(iidata,1)=cv(:,iidata);
              end
            else
              tvals{idata,1}=tv;
              cvals{idata,1}=cv;
            end
            if ~ismember(1,npc); npc(end+1)=1; end;
          end
          % --- AIDA data
          if ~isempty(pvlist{2})
            [cv tv]=FlCA('aidaGet',pvlist{2}(:,1),pvlist{2}(:,5),cellfun(@(x) x{3},pvlist{2}(:,2),'UniformOutput',false),pvlist{2}(:,4));
            if iscell(tv); tv=cell2mat(tv); end;
            if iscell(cv); cv=cell2mat(cv); end;
            if obj.hwMultGet
              for iidata=1:npulse
                tvals(iidata,2)=tv(:,iidata);
                cvals(iidata,2)=cv(:,iidata);
              end
            else
              tvals{idata,2}=tv;
              cvals{idata,2}=cv;
            end
            if ~ismember(2,npc); npc(end+1)=2; end;
          end
        catch ME
          warning('Lucretia:Floodland:FlInstrData:newDataFail','Failed to get new INSTR data:\n%s\n',ME.message)
        end
      end
      
      % put data into local array (applying hw->lucretia conversion factors)
      for idata=1:npulse
        for ipc=npc
          for ind=1:length(pvlist{ipc}(:,2))
            obj.Data(pvlist{ipc}{ind,2}{1},obj.bufpos,pvlist{ipc}{ind,2}{2})=cvals{idata,ipc}(ind)*pvlist{ipc}{ind,3};
          end
        end
        % increment buffer position if any good data here
        if any(any(~isnan(obj.Data(:,obj.bufpos,:))))
          if isempty(lastpulseID)
            obj.pulseID(obj.bufpos)=1;
          else
            obj.pulseID(obj.bufpos)=lastpulseID+1;
          end
          lastpulseID=obj.pulseID(obj.bufpos);
          obj.DataDate(obj.bufpos)=now;
          obj.bufpos=obj.bufpos+1; if obj.bufpos>obj.buffersize; obj.bufpos=1; end;
        else
          obj.DataDate(obj.bufpos)=nan;
        end
      end
      
    end
    function accum_sim(obj,Beam)
      % accum_sim(obj,Beam)
      % Simulation - track bunch and fill instrument data array
      % Provide Lucretia Beam to track
      persistent lastpulseID
      lastele=obj.Index(find(obj.useInstr,1,'last'));
      [~, bo instdata]=TrackThru(1,lastele,Beam,1,1,0);
      fnames={'x' 'y' 'z' 'sig11' 'sig33' 'sig13' 'sig55'};
      ind2=[0 0 0];
      for ind=1:length(obj.Index)
        if strcmp(obj.Class{ind},'MONI')
          ind1=1;
        elseif ~isempty(regexp(obj.Class{ind},'CAV$','once'))
          ind1=3;
        else
          ind1=2;
        end
        ind2(ind1)=ind2(ind1)+1;
        if ind1<=length(instdata) && ind2(ind1)<=length(instdata{ind1})
          for ifn=1:length(fnames)
            if isfield(instdata{ind1}(ind2(ind1)),fnames{ifn}) && ~isempty(instdata{ind1}(ind2(ind1)).(fnames{ifn}))
              obj.Data(ind,obj.bufpos,ifn)=instdata{ind1}(ind2(ind1)).(fnames{ifn}) ;
            else
              obj.Data(ind,obj.bufpos,ifn)=NaN;
            end
          end
          obj.Data(ind,obj.bufpos,8)=...
            sum(bo.Bunch.Q(bo.Bunch.stop>instdata{ind1}(ind2(ind1)).Index|bo.Bunch.stop==0)) ;
        else
          obj.Data(ind,obj.bufpos,:)=NaN;
        end
      end
      % Increment buffer position pointer
      if any(any(~isnan(obj.Data(:,obj.bufpos,:))))
        obj.DataDate(obj.bufpos)=now;
        if isempty(lastpulseID)
          obj.pulseID(obj.bufpos)=1;
        else
          obj.pulseID(obj.bufpos)=lastpulseID+1;
        end
        lastpulseID=obj.pulseID(obj.bufpos);
        obj.bufpos=obj.bufpos+1;
        if obj.bufpos>obj.buffersize
          obj.bufpos=1;
        end
      else
        obj.DataDate(obj.bufpos)=NaN;
        obj.pulseID(obj.bufpos)=NaN;
      end
    end
    function instdata_out = get_instdata(obj)
      % instdata_out = get_instdata(obj)
      % Convert FlInstrData provided data (last provided pulse) into TrackThru's instdata format
      global BEAMLINE
      data=squeeze(obj.Data(:,obj.bufpos,:));
      for ind=1:length(obj.Index)
        if strcmp(obj.Class{ind},'MONI')
          ind1=1;
        elseif ~isempty(regexp(obj.Class{ind},'CAV$','once'))
          ind1=3;
        else
          ind1=2;
        end
        instdata_out{ind1}(end+1).x=data(ind,1);
        instdata_out{ind1}(end).y=data(ind,2);
        instdata_out{ind1}(end).z=data(ind,3);
        instdata_out{ind1}(end).Index=obj.Index(ind);
        instdata_out{ind1}(end).S=BEAMLINE{obj.Index(ind)}.S;
        instdata_out{ind1}(end).Pmod=BEAMLINE{obj.Index(ind)}.P;
        if ind1==2
          instdata_out{2}(end).sig11=data(ind,4);
          instdata_out{2}(end).sig33=data(ind,5);
          instdata_out{2}(end).sig13=data(ind,6);
          instdata_out{2}(end).sig55=data(ind,7);
        end
      end
    end
    function updateInstrGui(obj)
      % Update GUI fields
      global BEAMLINE
      % Form right display of all available controls that meet with options
      % and not selected in left display
      displayStr2={}; s2=[]; ind=0; instind=[];
      name2=obj.instrName;
      for ibl=obj.Index
        ind=ind+1;
        instind(end+1)=ind;
        displayStr2{end+1}=sprintf('%d: %s [%d]',ibl,BEAMLINE{ibl}.Name,ind) ;
        s2(end+1)=BEAMLINE{ibl}.S;
      end
      sel=get(obj.gui.instrCbox2,'Value');
      % - put in required sort order
      if max(sel)>length(displayStr2)
        set(obj.gui.instrCbox2,'Value',1)
      end
      name1=name2(obj.instrChoiceFromGui);
      s1=s2(obj.instrChoiceFromGui);
      name2=name2(~obj.instrChoiceFromGui);
      s2=s2(~obj.instrChoiceFromGui);
      displayStr1=displayStr2(obj.instrChoiceFromGui);
      displayStr2=displayStr2(~obj.instrChoiceFromGui);
      ic1=find(obj.instrChoiceFromGui);
      ic2=find(~obj.instrChoiceFromGui);
      if get(obj.gui.display_alpha,'Value')
        [~, I]=sort(name2);
        [~, I1]=sort(name1);
      else
        [~, I]=sort(s2);
        [~, I1]=sort(s1);
      end
      set(obj.gui.instrCbox2,'String',displayStr2(I))
      set(obj.gui.instrCbox2,'UserData',ic2(I))
      set(obj.gui.instrCbox1,'String',displayStr1(I1))
      set(obj.gui.instrCbox1,'UserData',ic1(I1))
      drawnow('expose');
    end
  end
end