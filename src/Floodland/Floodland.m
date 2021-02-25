classdef Floodland < handle & matlab.mixin.Copyable
  %FLOODLAND Floodland main Class (Hardware interaction module for Lucretia)
  %
  % Specify Model lattice data and tracking data for simulation and methods
  % for getting at the control system (for power supplies, movers and
  % klystrons)
  %
  % Main public methods:
  %
  %   hwGet / hwSet : Get and set hardware values (requiring FlIndex object
  %   to define what hardware values required)
  %
  % See also:
  %   FlIndex
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc Floodland">doc Floodland</a>
  %
  % Full lucretia documentation available online:
  %   <a href="http://www.slac.stanford.edu/accel/ilc/codes/Lucretia">Lucretia</a>
  
  properties
    expt % Experiment name to associate with this object
    Initial % Lucretia Initial beam structure
    BeamSingle % Single ray
    BeamMacro % Macro-particle beam
    BeamSparse % Sparse beam
    Twiss % Lattice twiss parameters (generated from Initial beam structure)
    Nmacro=1000; % Number of macro-particles in BeamMacro type
    beamSigmaCut=3; % Number of sigmas to cut when generating Macro-particle beam
    sparse_Nslice=31; % Number of longitudinal slices per beam for sparse tracking
    sparse_Nener=11; % Number of energies per slice for sparse beam tracking
    issim=true; % Simulation switch, if true tracking and hardware get/put simulated else the control system is accessed
    repRate=10; % repetition rate of machine
    latticeName % Name of lattice associated with this Floodland instance
    latticeDate % Date of lattice, datenum format
    magTrimStyle='TRIM'; % PTRB or TRIM (implemented by low level controls)
    timezone % hours east GMT (set by constructor)
    serverMode=false; % Operating in text only (possibly bg) mode (supress GUI operations)
  end
  properties(Access=private)
    seed % random number seed
    slices % BEAMLINE element slices
    blocks % BEAMLINE element blocks
  end
  properties
    writeSafety='on'; % 'on' or 'off' : if on then don't at any point try to write to a control system, just print out data instead
  end
  
  %% Get/Set methods
  methods
    function set.issim(obj,val)
      obj.issim=logical(val);
      if ~obj.issim
        ws=warning('QUERY','Lucretia:Floodland:issim_set');
        if strcmp(ws.state,'on'); beep; end;
        warning('Lucretia:Floodland:issim_set','Sim mode disabled, connections to real hardware now permitted')
      end
    end
    function set.Nmacro(obj,val)
      if ~isnumeric(val) || val<2 || val>1e6
        error('Supply # macro particles 2-1e6')
      end
      obj.Nmacro=val;
      try
        obj.beamGen;
      catch
      end
    end
    function set.sparse_Nslice(obj,val)
      if ~isnumeric(val) || val<1 || val>1e6
        error('Supply # sparse slices 1-1e6')
      end
      obj.sparse_Nslice=val;
      try
        obj.beamGen;
      catch
      end
    end
    function set.sparse_Nener(obj,val)
      if ~isnumeric(val) || val<1 || val>1e6
        error('Supply # energies per slice 1-1e6')
      end
      obj.sparse_Nener=val;
      try
        obj.beamGen;
      catch
      end
    end
    function set.Initial(obj,newInitial)
      obj.Initial=newInitial;
      try
        obj.genTwiss;
        obj.beamGen;
      catch
      end
    end
  end
  
  %% Main public methods
  methods
    function obj=Floodland(exptName,latticeName,latticeDate,Initial)
      % Floodland(exptName,latticeName,latticeDate,Initial)
      %   exptName: string describing experiment associated with this
      %             object
      %   latticeName: name describing lattice associated with this object
      %   latticeDate: lattice date tag (any supported Matlab date format)
      %   Initial: Lucretia Initial structure
      %
      % See also:
      %    InitCondStruc
      global BEAMLINE
      if ~exist('exptName','var')
        error('Must supply an experiment (accelerator) name for this Floodland instance');
      end
      if ~exist('Initial','var')
        error('Must supply Lucretia ''Initial'' structure')
      end
      if ~exist('latticeName','var') || ~ischar(latticeName)
        error('Must supply ''latticeName'' (string)')
      end
      if ~exist('latticeDate','var')
        error('Must supply ''latticeDate'' (any supported Matlab date format)')
      end
      % assign element slices and blocks
      [~,obj.slices] = SetElementSlices( 1, length(BEAMLINE) );
      [~,obj.blocks] = SetElementBlocks( 1, length(BEAMLINE) );
      obj.latticeName=latticeName;
      obj.latticeDate=datenum(latticeDate);
      obj.expt=exptName;
      obj.Initial=Initial; % calls beamGen & genTwiss
      obj.seed=rng;
      % Set current timezone from OS
      try
        if ispc
          t=regexp(evalc('systeminfo | findstr  /C:"Time Zone"'),'\(GMT(.+)\)','tokens','once');
          obj.timezone=rem(datenum(t{1}),1)*24;
          if t{1}(1)=='-'; obj.timezone=-obj.timezone; end;
        else
          str2double(evalc('!date +%:::z'));
        end
      catch
        obj.timezone=0;
      end
    end
    function hwGet(obj,indxObj,getList)
      % hwGet(obj,indxObj,getList)
      % Read hardware channels into Lucretia
      %   indxObj: FlIndex object or an object that inherits from this
      %   class
      %   getList: (optional) [double vector] Only get these elements of
      %            the provided indxOnj
      %
      % See also:
      %   FlIndex
      
      % Need to pass FlIndex object or object that inherits from FlIndex Class
      mc=metaclass(indxObj);
      isFlIndex=strcmp(class(indxObj),'FlIndex');
      if ~isempty(mc.SuperclassList) && ~isFlIndex
        for imc=1:length(mc.SuperclassList)
          isFlIndex=strcmp(mc.SuperclassList(imc).Name,'FlIndex');
          if isFlIndex; break; end;
        end
      end
      if ~isFlIndex; error('Must pass FlIndex object'); end;
      props={'PS' 'KLYSTRON' 'GIRDER'};
      prop={};
      for iprop=1:length(props)
        if ~isempty(indxObj.(props{iprop}))
          prop{end+1}=props{iprop};
        end
      end
      % If getList provided, only get these values
      if exist('getList','var')
        doget=false(size(indxObj.useCntrl));
        doget(getList)=true;
      else
        doget=true(size(indxObj.useCntrl));
      end
      % perform hw fetch action
      hwChans=indxObj.INDXchannels;
      % PS
      proplist=find(indxObj.useCntrl(indxObj.PS_list)&doget(indxObj.PS_list));
      if ~isempty(proplist)
        psGet(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.PS_list(proplist)),hwChans(1,indxObj.PS_list(proplist)));
      end
      % KLYSTRON
      proplist=find(indxObj.useCntrl(indxObj.KLYSTRON_list)&doget(indxObj.KLYSTRON_list));
      if ~isempty(proplist)
        klyGet(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.KLYSTRON_list(proplist)),hwChans(1,indxObj.KLYSTRON_list(proplist)));
      end
      % GIRDER
      proplist=find(indxObj.useCntrl(indxObj.GIRDER_list)&doget(indxObj.GIRDER_list));
      if ~isempty(proplist)
        girGet(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.GIRDER_list(proplist)),hwChans(1,indxObj.GIRDER_list(proplist)));
      end
    end
    function hwSet(obj,indxObj,putList)
      % hwSet(obj,indxObj,putList)
      % Trim hardware channels to values in SetPt
      %   indxObj: FlIndex object or an object that inherits from this
      %   class
      %   putList: (optional) [double vector] Only set these elements of
      %            the provided indxObj
      %
      % See also:
      %   FlIndex
      
      % Need to pass FlIndex object or object that inherits from FlIndex Class
      mc=metaclass(indxObj);
      isFlIndex=strcmp(class(indxObj),'FlIndex');
      if ~isempty(mc.SuperclassList) && ~isFlIndex
        for imc=1:length(mc.SuperclassList)
          isFlIndex=strcmp(mc.SuperclassList(imc).Name,'FlIndex');
          if isFlIndex; break; end;
        end
      end
      if ~isFlIndex; error('Must pass FlIndex object'); end;
      props={'PS' 'KLYSTRON' 'GIRDER'};
      prop={};
      for iprop=1:length(props)
        if ~isempty(indxObj.(props{iprop}))
          prop{end+1}=props{iprop};
        end
      end
      % If putList provided, only put these values
      if exist('putList','var')
        doput=false(size(indxObj.useCntrl));
        doput(putList)=true;
      else
        doput=true(size(indxObj.useCntrl));
      end
      % Only set values where ampl and setpt are different
      ampl=indxObj.Ampl; setpt=indxObj.SetPt;
      doput(arrayfun(@(x) isequal(ampl{x},setpt{x}),1:length(ampl)))=false;
      % perform trim action
      hwChans=indxObj.INDXchannels;
      % PS
      proplist=find(indxObj.useCntrl(indxObj.PS_list)&doput(indxObj.PS_list));
      if ~isempty(proplist)
        psTrim(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.PS_list(proplist)),hwChans(2,indxObj.PS_list(proplist)));
      end
      % KLYSTRON
      proplist=find(indxObj.useCntrl(indxObj.KLYSTRON_list)&doput(indxObj.KLYSTRON_list));
      if ~isempty(proplist)
        klyTrim(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.KLYSTRON_list(proplist)),hwChans(2,indxObj.KLYSTRON_list(proplist)));
      end
      % GIRDER
      proplist=find(indxObj.useCntrl(indxObj.GIRDER_list)&doput(indxObj.GIRDER_list));
      if ~isempty(proplist)
        girTrim(obj,indxObj,proplist,indxObj.useCntrlChan(indxObj.GIRDER_list(proplist)),hwChans(2,indxObj.GIRDER_list(proplist)));
      end
    end
  end
  
  %% Internal methods
  methods(Access=protected)
    function b=loadobj(a)
      b=a.copy;
    end
    function genTwiss(obj)
      global BEAMLINE
      Tx.alpha=obj.Initial.x.Twiss.alpha;
      Tx.beta=obj.Initial.x.Twiss.beta;
      Tx.eta=obj.Initial.x.Twiss.eta;
      Tx.etap=obj.Initial.x.Twiss.etap;
      Tx.nu=obj.Initial.x.Twiss.nu;
      Ty.alpha=obj.Initial.y.Twiss.alpha;
      Ty.beta=obj.Initial.y.Twiss.beta;
      Ty.eta=obj.Initial.y.Twiss.eta;
      Ty.etap=obj.Initial.y.Twiss.etap;
      Ty.nu=obj.Initial.y.Twiss.nu;
      [stat, T] = GetTwiss(1,length(BEAMLINE),Tx,Ty);
      if stat{1}~=1; error('Twiss generation error:\n%s\n',stat{2}); end;
      obj.Twiss=T;
    end
    function beamGen(obj,newseed)
      % store current random generator info
      if ~exist('newseed','var') || ~newseed
        try
          s=rng;
          if isempty(obj.seed); obj.seed=s.Seed; end;
          olds=s; olds.Seed=obj.seed;
          rng(olds);
        catch
          obj.seed=s.Seed;
        end
      end
      % Generate beam
      obj.BeamMacro = MakeBeam6DGauss( obj.Initial, obj.Nmacro, obj.beamSigmaCut, 1 );
      obj.BeamSparse = MakeBeam6DSparse( obj.Initial, obj.beamSigmaCut,obj.sparse_Nslice,obj.sparse_Nener);
      obj.BeamSingle = MakeBeam6DGauss( obj.Initial, 1, 1, 1 );
      % Make centroid of all beams initially the same
      for idim=1:6
        obj.BeamSparse.Bunch.x(idim,:)=obj.BeamSparse.Bunch.x(idim,:)+(obj.BeamSingle.Bunch.x(idim)-mean(obj.BeamSparse.Bunch.x(idim,:)));
        obj.BeamMacro.Bunch.x(idim,:)=obj.BeamMacro.Bunch.x(idim,:)+(obj.BeamSingle.Bunch.x(idim)-mean(obj.BeamMacro.Bunch.x(idim,:)));
      end
      % Restore current generator seed
      if ~exist('newseed','var') || ~newseed
        rng(s);
      end
    end
    function klyGet(obj,indxObj,propindx,chanindx,hwindx)
      global KLYSTRON
      if obj.issim; return; end;
      klist=indxObj.KLYSTRON(propindx);
      V=indxObj.KLYSTRON_VMDL(propindx); PHA=indxObj.KLYSTRON_PHASEMDL(propindx);
      comstack={}; comstackVals=[]; comstackProto={};
      comstackP={}; comstackValsP=[]; comstackProtoP={};
      comstack_get={}; comstackProto_get={}; comstack_conv={}; comstack_PHA=[]; comstack_V=[]; comstack_kly=[];
      % Form get/set & monitor lists
      for n=1:length(klist)
        ik=klist(n);
        pvname=KLYSTRON(ik).pvname;
        if (isempty(pvname{1,2}) && isempty(pvname{1,1})) || isequal(KLYSTRON(ik).conv,0) || isempty(KLYSTRON(ik).conv)
          continue
        end
        % Add any pre-commands to stack
        if ~isempty(KLYSTRON(ik).preCommand{1})
          comstack{end+1}=KLYSTRON(ik).preCommand{1}{1};
          comstackVals(end+1)=KLYSTRON(ik).preCommand{1}{2};
          comstackProto{end+1}=KLYSTRON(ik).protocol;
        end
        % Main PV to get value from
        if ~isempty(pvname{1,1}) && chanindx{n}(1) && ismember(1,hwindx{n})
          comstack_get{end+1,1}=pvname{1,1};
          comstackProto_get{end+1,1}=KLYSTRON(ik).protocol;
          comstack_conv{end+1,1}=KLYSTRON(ik).conv{1,1};
          comstack_V(end+1)=V(n);
          comstack_PHA(end+1)=NaN;
          comstack_kly(end+1)=ik;
        end
        if ~isempty(pvname{1,2}) && chanindx{n}(2) && ismember(2,hwindx{n})
          comstack_get{end+1,1}=pvname{1,1};
          comstackProto_get{end+1,1}=KLYSTRON(ik).protocol;
          comstack_conv{end+1,1}=KLYSTRON(ik).conv{1,2};
          comstack_V(end+1)=NaN;
          comstack_PHA(end+1)=PHA(n);
          comstack_kly(end+1)=ik;
        end
        % Any post processing commands?
        if ~isempty(KLYSTRON(ik).postCommand{1})
          comstackP{end+1}=KLYSTRON(ik).postCommand{1}{1};
          comstackValsP(end+1,1)=KLYSTRON(ik).postCommand{1}{2};
          comstackProtoP{end+1}=KLYSTRON(ik).protocol;
        end
      end
      % Issue control system command(s)
      if ~isempty(comstack)
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      end
      cvals=obj.cntrlGet(comstack_get,comstackProto_get,comstack_conv);
      if ~isempty(comstackP)
        obj.cntrlSet(comstackP,comstackProtoP,comstackValsP);
      end
      % Deal out values to KLYSTRON amp and phase
      for ival=1:length(cvals)
        if ~isnan(cvals{ival})
          if ~isnan(comstack_V(ival))
            KLYSTRON(comstack_kly(ival)).Ampl=cvals{ival}/comstack_V(ival);
          elseif ~isnan(comstack_PHA(ival))
            KLYSTRON(comstack_kly(ival)).Phase=cvals{ival}-comstack_PHA(ival);
          end
        end
      end
    end
    function klyTrim(obj,indxObj,propindx,chanindx,hwindx)
      global KLYSTRON
      klist=indxObj.KLYSTRON(propindx);
      if obj.issim
        stat=KlystronTrim(klist);
        if stat{1}~=1; error('Klystron simulation trim error:\n%s',stat{2}); end;
        return
      end
      V=indxObj.KLYSTRON_VMDL; PHA=indxObj.KLYSTRON_PHASEMDL;
      comstack={}; comstackVals=[]; comstackProto={};
      % Form get/set & monitor lists
      for n=1:length(klist)
        ik=klist(n);
        pvname=KLYSTRON(ik).pvname;
        if (isempty(pvname{2,1}) && isempty(pvname{2,2})) || isequal(KLYSTRON(ik).conv,0) || isempty(KLYSTRON(ik).conv)
          continue
        end
        % Add any pre-commands to stack
        if ~isempty(KLYSTRON(ik).preCommand{2})
          comstack{end+1}=KLYSTRON(ik).preCommand{2}{1};
          comstackVals(end+1)=KLYSTRON(ik).preCommand{2}{2};
          comstackProto{end+1}=KLYSTRON(ik).protocol;
        end
        % Main PV
        conv=KLYSTRON(ik).conv;
        if ~isempty(pvname{2,1}) && chanindx{n}(1) && ismember(1,hwindx{n})
          comstackProto{end+1,1}=KLYSTRON(ik).protocol;
          comstack{end+1,1}=pvname{2,1};
          if length(conv{1})==1
            comstackVals(end+1,1)=(KLYSTRON(ik).AmplSetPt*V(n))/conv{2,1};
          else
            comstackVals(end+1,1)=interp1(conv{1}(2,:),conv{1}(1,:),KLYSTRON(ik).V*V(n),'linear');
          end
        end
        if ~isempty(pvname{2,2}) && chanindx{n}(2) && ismember(2,hwindx{n})
          comstackProto{end+1,1}=KLYSTRON(ik).protocol;
          comstack{end+1,1}=pvname{2,2};
          if length(conv{2})==1
            comstackVals(end+1,1)=(PHA(n)+KLYSTRON(ik).PhaseSetPt)/conv{2,2};
          else
            comstackVals(end+1,1)=interp1(conv{2}(2,:),conv{2}(1,:),PHA(n)+KLYSTRON(ik).Phase,'linear');
          end
        end
        % Any post processing commands?
        if ~isempty(KLYSTRON(ik).postCommand{2})
          comstack{end+1}=KLYSTRON(ik).postCommand{2}{1};
          comstackVals(end+1,1)=KLYSTRON(ik).postCommand{2}{2};
          comstackProto{end+1}=KLYSTRON(ik).protocol;
        end
      end
      % Issue control system command(s)
      if strcmp(obj.writeSafety,'off')
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      else
        disp('Control safety switch on in Floodland object, NOT written to control system:')
        for ic=1:length(comstack)
          fprintf('PROTO: %s PV: %s VAL: %g\n',comstackProto{ic},comstack{ic},comstackVals(ic))
        end
      end
    end
    function psGet(obj,indxObj,propindx,chanindx,hwindx)
      global PS
      if obj.issim; return; end;
      pslist=indxObj.PS(propindx);
      B=indxObj.PS_BMDL(propindx);
      
      comstack={}; comstackVals=[]; comstackProto={};
      comstackP={}; comstackValsP=[]; comstackProtoP={};
      comstack_get={}; comstackProto_get={}; comstack_conv={}; comstack_ps=[]; comstack_B=[];
      comstack_pconv={};
      % Form get/set & monitor lists
      for n=1:length(pslist)
        ips=pslist(n);
        pvname=PS(ips).pvname;
        if isempty(pvname{2}) || isequal(PS(ips).conv{1},0) || isempty(PS(ips).conv{1})
          continue
        end
        if ~iscell(PS(ips).protocol)
          proto=PS(ips).protocol;
        else
          proto=PS(ips).protocol{1};
        end
        % Add any pre-commands to stack
        if ~isempty(PS(ips).preCommand{1})
          comstack{end+1}=PS(ips).preCommand{1}{1};
          comstackVals(end+1)=PS(ips).preCommand{1}{2};
          comstackProto{end+1}=proto;
        end
        if  chanindx{n}(1) && ~isempty(hwindx{n})
          % Main PV to get value from
          trimVal=0;
          if ~isempty(PS(ips).trimpv{1})
            try
              trimpv=PS(ips).trimpv;
              trimconv=PS(ips).trimconv;
              comstack_get{end+1,1}=trimpv{1};
              comstackProto_get{end+1,1}=proto;
              comstack_conv{end+1,1}=trimconv{1};
              comstack_B(end+1)=B(n);
              comstack_ps(end+1)=ips;
              comstack_pconv{end+1,1}=PS(ips).conv2{1};
            catch ME
              disp(PS(ips).trimpv{1})
              disp(ips)
              error('Trim Controls not available:\n%s',ME.message)
            end
            if isnan(trimVal) || ~isnumeric(trimVal)
              error('Lucretia:Floodland:psTrim:noTrimControls','Bad response from Trim controls for PS %d',ips)
            end % if bad val returned
          end % if 2 pv's (main and trim)
          comstack_get{end+1,1}=PS(ips).pvname{1};
          comstackProto_get{end+1,1}=proto;
          comstack_conv{end+1,1}=PS(ips).conv{1};
          comstack_B(end+1)=B(n);
          comstack_ps(end+1)=ips;
          comstack_pconv{end+1,1}=PS(ips).conv2{1};
        end
        % Any post processing commands?
        if ~isempty(PS(ips).postCommand{1})
          comstackP{end+1}=PS(ips).postCommand{1}{1};
          comstackValsP(end+1,1)=PS(ips).postCommand{1}{2};
          comstackProtoP{end+1}=proto;
        end
      end
      % Issue control system command(s)
      if ~isempty(comstack)
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      end
      cvals=obj.cntrlGet(comstack_get,comstackProto_get,comstack_conv);
      if ~isempty(comstackP)
        obj.cntrlSet(comstackP,comstackProtoP,comstackValsP);
      end
      % Dealt out values to PS ampl
      % - sum returned (converted) values from control system, then apply a
      % final conversion
      for ival=1:length(cvals)
        if ~any(isnan(cvals{ival}))
          cntrlVal=0;
          for ival2=find(ismember(comstack_ps,comstack_ps(ival)))
            if ~isnan(cvals{ival2})
              cntrlVal=cntrlVal+cvals{ival2};
            end
          end
          conv=comstack_pconv{ival};
          if length(conv)==1
            cntrlVal=cntrlVal.*conv;
          else
            cntrlVal=interp1(conv(1,:),conv(2,:),cntrlVal,'linear');
          end
          PS(comstack_ps(ival)).Ampl=(length(PS(comstack_ps(ival)).Element)/2)*(cntrlVal/comstack_B(ival));
        end
      end
    end
    function psTrim(obj,indxObj,propindx,chanindx,hwindx)
      global PS
      pslist=indxObj.PS(propindx);
      B=indxObj.PS_BMDL(propindx);
      % If sim, use PSTrim function and return
      if obj.issim
        stat=PSTrim(pslist);
        if stat{1}~=1; error(stat{2}); end;
        return
      end
      comstack={}; comstackVals=[]; comstackProto={};
      for n=1:length(pslist)
        ips=pslist(n);
        pvname=PS(ips).pvname;
        if isempty(pvname{2}) || isequal(PS(ips).conv{2},0) || isempty(PS(ips).conv{2})
          continue
        end
         if ~iscell(PS(ips).protocol)
          proto=PS(ips).protocol;
        else
          proto=PS(ips).protocol{2};
        end
        % Add any pre-commands to stack
        if ~isempty(PS(ips).preCommand{2})
          comstack{end+1,1}=PS(ips).preCommand{2}{1};
          comstackVals(end+1,1)=PS(ips).preCommand{2}{2};
          comstackProto{end+1,1}=proto;
        end
        conv2=PS(ips).conv2{2};
        if chanindx{n}(1) && ~isempty(hwindx{n})
          % Get trim or main pv names to use and vals to set
          trimVal=0;
          if ~isempty(PS(ips).trimpv{2})
            try
              trimVal=cell2mat(obj.cntrlGet(PS(ips).trimpv{1},proto,PS(ips).trimconv{1},'force'));
            catch ME
              error('Trim Controls not available:\n%s',ME.message)
            end
            if isnan(trimVal) || ~isnumeric(trimVal)
              error('Lucretia:Floodland:psTrim:noTrimControls','Bad response from Trim controls for PS %d',ips)
            end % if bad val returned
            if PS(ips).trimUnipolar; trimVal=abs(trimVal); end;
            if length(conv2)>1
              trimVal=interp1(conv2(1,:),conv2(2,:),trimVal,'linear')/B(n);
            else
              trimVal=(trimVal*conv2)/B(n);
            end
          end % if 2 pv's (main and trim)
          % Use trim PV if available and in range, else use main
          if ~isempty(PS(ips).trimpv{2}) && (isempty(PS(ips).trimHigh(1)) || (PS(ips).SetPt-mainVal)<PS(ips).trimHigh(1)) && ...
              (isempty(PS(ips).trimLow(1)) || (PS(ips).SetPt-mainVal)>PS(ips).trimLow(1))
            mainVal=cell2mat(obj.cntrlGet(PS(ips).pvname{1},proto,PS(ips).conv{2}),'force');
            if PS(ips).unipolar; mainVal=abs(mainVal); end;
            if length(conv2)>1
              mainVal=interp1(conv2(1,:),conv2(2,:),mainVal,'linear')/B(n);
            else
              mainVal=(mainVal*conv2)/B(n);
            end
            comstack{end+1}=PS(ips).trimpv{2};
            comstackProto{end+1}=proto;
            val=PS(ips).SetPt-mainVal;
            % unipolar device?
            unipolar=PS(ips).trimUnipolar;
            if unipolar && val<0
              error('Trying to set negative value on unipolar Power Supply (%d)',ips)
            end
            % get correct conversion factor / lookup
            conv=PS(ips).trimconv{2};
            if (~PS(ips).unipolar && length(conv)>1 && val<0 && ~any(conv(2,end)<0)) || ...
                (~PS(ips).unipolar && length(conv)>1 && val>0 && ~any(conv(2,end)>0))
              conv=-conv;
            end
          % else use the main curret control device
          elseif (isempty(PS(ips).high(2)) || (PS(ips).SetPt-trimVal)<PS(ips).high(2)) && ...
              (isempty(PS(ips).low(2)) || (PS(ips).SetPt-trimVal)>PS(ips).low(2))
            comstack{end+1}=PS(ips).pvname{2};
            comstackProto{end+1}=proto;
            val=PS(ips).SetPt-trimVal;
            unipolar=PS(ips).unipolar;
            % unipolar device?
            if unipolar && val<0
              error('Trying to set negative value on unipolar Power Supply (%d)',ips)
            end
            % get correct conversion factor / lookup
            conv=PS(ips).conv{2};
            if (~PS(ips).unipolar && length(conv)>1 && val<0 && ~any(conv(2,end)<0)) || ...
                (~PS(ips).unipolar && length(conv)>1 && val>0 && ~any(conv(2,end)>0))
              conv=-conv;
            end
          else
            error('Attempting to set PS beyond software limits (PS %d)',ips)
          end
          % Form trim vals to send to control system
          if length(conv2)==1
            if unipolar
              comstackVals(end+1,1)=(abs(val)/conv2)*abs(B(n));
            else
              comstackVals(end+1,1)=(val/conv2)*B(n);
            end
          else
            if unipolar
              comstackVals(end+1,1)=interp1(conv2(2,:),conv2(1,:),abs(val*B(n)),'linear');
            else
              comstackVals(end+1,1)=interp1(conv2(2,:),conv2(1,:),val*B(n),'linear');
            end
          end
          if length(conv)==1
            if unipolar
              comstackVals(end,1)=comstackVals(end,1)/conv;
            else
              comstackVals(end,1)=comstackVals(end,1)/conv;
            end
          else
            if unipolar
              comstackVals(end,1)=interp1(conv(2,:),conv(1,:),comstackVals(end,1),'linear');
            else
              comstackVals(end,1)=interp1(conv(2,:),conv(1,:),comstackVals(end,1),'linear');
            end
          end
          % If multiple magnets sharing same boost, set average B field to
          % each magnet
          comstackVals(end)=comstackVals(end)/length(PS(ips).auxbst);
        end
        % Any post processing commands?
        if ~isempty(PS(ips).postCommand{2})
          comstack{end+1}=PS(ips).postCommand{2}{1};
          comstackVals(end+1,1)=PS(ips).postCommand{2}{2};
          comstackProto{end+1}=proto;
        end
      end
      % Issue control system command(s)
      if strcmp(obj.writeSafety,'off')
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      else
        disp('Control safety switch on in Floodland object, NOT written to control system:')
        for ic=1:length(comstack)
          fprintf('PROTO: %s PV: %s VAL: %g\n',comstackProto{ic},comstack{ic},comstackVals(ic))
        end
      end
    end    
    function girGet(obj,indxObj,propindx,chanindx,hwindx)
      global GIRDER
      if obj.issim; return; end;
      glist=indxObj.GIRDER(propindx);
      
      comstack={}; comstackVals=[]; comstackProto={};
      comstackP={}; comstackValsP=[]; comstackProtoP={};
      comstack_get={}; comstackProto_get={}; comstack_conv={};
      gReadInd={};
      % Form get/set & monitor lists
      for n=1:length(glist)
        ig=glist(n);
        pvname=GIRDER{ig}.pvname;
        if isempty(pvname{2}) || isequal(GIRDER{ig}.conv{1},0) || isempty(GIRDER{ig}.conv{1})
          continue
        end
        % Add any pre-commands to stack
        if ~isempty(GIRDER{ig}.preCommand{1})
          comstack{end+1}=GIRDER{ig}.preCommand{1}{1};
          comstackVals(end+1)=GIRDER{ig}.preCommand{1}{2};
          comstackProto{end+1}=GIRDER{ig}.protocol;
        end
        % Main PV to get value from
        for idim=1:6
          if ~isempty(pvname{1,idim}) && chanindx{n}(idim) && ismember(idim,hwindx{n})
            comstack_get{end+1,1}=pvname{1,idim};
            comstackProto_get{end+1,1}=GIRDER{ig}.protocol;
            comstack_conv{end+1,1}=GIRDER{ig}.conv{1,idim};
            gReadInd{end+1}=[ig idim];
          end
        end
        % Any post processing commands?
        if ~isempty(GIRDER{ig}.postCommand{1})
          comstackP{end+1}=GIRDER{ig}.postCommand{1}{1};
          comstackValsP(end+1,1)=GIRDER{ig}.postCommand{1}{2};
          comstackProtoP{end+1}=GIRDER{ig}.protocol;
        end
      end
      % Issue control system command(s)
      if ~isempty(comstack)
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      end
      cvals=obj.cntrlGet(comstack_get,comstackProto_get,comstack_conv);
      if ~isempty(comstackP)
        obj.cntrlSet(comstackP,comstackProtoP,comstackValsP);
      end
      % Deal out mover positions
      for ival=1:length(cvals)
        if ~isnan(cvals{ival})
          GIRDER{gReadInd{ival}(1)}.MoverPos(gReadInd{ival}(2))=cvals{ival};
        end
      end
    end
    function girTrim(obj,indxObj,propindx,chanindx,hwindx)
      global GIRDER
      glist=indxObj.GIRDER(propindx);
      if obj.issim
        stat=MoverTrim(glist);
        if stat{1}~=1; error('GIRDER simulation trim error:\n%s',stat{2}); end;
        return
      end
      comstack={}; comstackVals=[]; comstackProto={};
      comstackP={}; comstackValsP=[]; comstackProtoP={};
      % Form get/set & monitor lists
      for n=1:length(glist)
        ig=glist(n);
        pvname=GIRDER{ig}.pvname;
        if (isempty(pvname{2,1}) && isempty(pvname{2,2})) || isequal(GIRDER{ig}.conv,0) || isempty(GIRDER{ig}.conv)
          continue
        end
        % Add any pre-commands to stack
        if ~isempty(GIRDER{ig}.preCommand{2})
          comstack{end+1}=GIRDER{ig}.preCommand{2}{1};
          comstackVals(end+1)=GIRDER{ig}.preCommand{2}{2};
          comstackProto{end+1}=GIRDER{ig}.protocol;
        end
        % Main PV
        conv=GIRDER{ig}.conv;
        for idim=1:6
          if ~isempty(pvname{2,idim}) && chanindx{n}(1) && ismember(idim,hwindx{n})
            comstackProto{end+1,1}=GIRDER{ig}.protocol;
            comstack{end+1,1}=pvname{2,idim};
            if length(conv{1})==1
              comstackVals(end+1,1)=(GIRDER{ig}.MoverSetPt(idim))/conv{2,idim};
            else
              comstackVals(end+1,1)=interp1(conv{1}(2,:),conv{1}(1,:),GIRDER{ig}.MoverSetPt(idim),'linear');
            end
          end
        end
        % Any post processing commands?
        if ~isempty(GIRDER{ig}.postCommand{2})
          comstackP{end+1}=GIRDER{ig}.postCommand{2}{1};
          comstackValsP(end+1,1)=GIRDER{ig}.postCommand{2}{2};
          comstackProtoP{end+1}=GIRDER{ig}.protocol;
        end
      end
      % Issue control system command(s)
      if strcmp(obj.writeSafety,'off')
        obj.cntrlSet(comstack,comstackProto,comstackVals);
      else
        disp('Control safety switch on in Floodland object, NOT written to control system:')
        for ic=1:length(comstack)
          fprintf('PROTO: %s PV: %s VAL: %g\n',comstackProto{ic},comstack{ic},comstackVals(ic))
        end
      end
    end
    function cntrlSet(obj,pv,proto,vals)
      % Set vals to controls (raw)
      if ~iscell(pv); pv={pv,1}; end;
      if ~iscell(proto); proto={proto}; end;
      % seperate into protocols and get values
      sz=size(proto); if sz(2)>sz(1); proto=proto'; end;
      sz=size(pv); if sz(2)>sz(1); pv=pv'; end;
      isEpics=ismember(proto,'EPICS');
      isAida=ismember(proto,'AIDA');
      if any(isEpics)
        FlCA('lcaPut',pv(isEpics),vals(isEpics));
      end
      if any(isAida)
        FlCA('aidaPut',pv(isAida),vals(isAida),obj.magTrimStyle);
      end
    end
  end
  
  %% Utility methods
  methods(Static)
    function cvals=cntrlGet(pv,proto,conv,cmd)
      % cvals=Floodland.cntrlGet(pv,proto,conv)
      %
      % Get vals from controls (and convert)
      persistent monitorList
      if ~iscell(pv); pv={pv}; end;
      if ~iscell(proto); proto={proto}; end;
      if ~iscell(conv); conv={conv}; end;
      sz=size(proto); if sz(2)>sz(1); proto=proto'; end;
      sz=size(pv); if sz(2)>sz(1); pv=pv'; end;
      cvals=cell(length(pv),1); tvals=NaN(size(cvals));
      % seperate into protocols and get values
      isEpics=ismember(proto,'EPICS');
      isAida=ismember(proto,'AIDA');
      if any(isEpics)
        epv=pv(isEpics);
        if isempty(monitorList)
          lcaSetMonitor(epv);
          monitorList=epv;
          moniUpdate=true(size(epv));
        elseif any(~ismember(epv,monitorList))
          newpv=~ismember(epv,monitorList);
          lcaSetMonitor(epv(newpv));
          monitorList(end+1:end+sum(newpv))=epv(newpv);
          moniUpdate=true(size(epv));
        else
          moniUpdate=FlCA('lcaNewMonitorValue',epv);
        end
        if exist('cmd','var') && isequal(cmd,'force')
          [epicsVals epicsts]=FlCA('lcaGet',epv);
        else
          [epicsVals epicsts]=FlCA('lcaGet',epv(logical(moniUpdate)));
        end
        epind=find(isEpics);
        ind=0;
        for ival=1:length(moniUpdate)
          if moniUpdate(ival)
            ind=ind+1;
            tvals(epind(ival))=epicsts(ind);
            tvals(epind(ival))=addtodate(datenum(1990,1,1),floor(real(tvals(epind(ival)))*1e3+imag(tvals(epind(ival)))*1e-6),'millisecond');
            if iscell(epicsVals)
              cvals{epind(ival)}=epicsVals{ind};
            else
              cvals{epind(ival)}=epicsVals(ind);
            end
          else
            cvals{epind(ival)}=NaN;
          end
        end
      end
      if any(isAida)
        [aidaVals aidats]=FlCA('aidaGet',pv(isAida));
        ind=find(isAida);
        for ival=1:length(aidaVals)
          tvals(ind(ival))=aidats(ival);
          if ~iscell(aidaVals)
            cvals{ind(ival)}=aidaVals(ival);
          else
            cvals{ind(ival)}=aidaVals{ival};
          end
        end
      end
      % Do conversion
      for ival=1:length(cvals)
        if ~any(isnan(cvals{ival}))
          if length(conv{ival})==1
            cvals{ival}=cvals{ival}.*conv{ival};
          else
            try
              cvals{ival}=interp1(conv{ival}(1,:),conv{ival}(2,:),cvals{ival},'linear');
            catch ME
              ME.getReport
              error(ME.message)
            end
          end
        end
      end
    end
    function micr=bitid2micr(c2bitid)
      % micr=bitid2micr(c2bitid)
      %
      %
      % microname/bitid data
      %
      % from REF_DBSFILE:MICRONAME.DAT (07-AUG-2009)
      
      micrs=[ ...
        'LI00'; ...
        'LI01';'LI02';'LI03';'LI04';'LI05';'LI06';'LI07';'LI08';'LI09';'LI10'; ...
        'LI11';'LI12';'LI13';'LI14';'LI15';'LI16';'LI17';'LI18';'LI19';'LI20'; ...
        'LI21';'LI22';'LI23';'LI24';'LI25';'LI26';'LI27';'LI28';'LI29';'LI30'; ...
        'LI31';'DR01';'DR02';'DR03';'FB31';'FB30';'MP00';'CL01';'MP01';'DR11'; ...
        'DR12';'DR13';'CA01';'CA02';'CA03';'CA11';'CA12';'CA13';'FB73';'FF01'; ...
        'FF11';'FB69';'MC00';'LI32';'NP25';'FG00';'EP01';'EP02';'EP05';'FB88'; ...
        'PT01';'TL01';'TL00';'IA20';'IB20';'LA21';'LB21';'LC21';'IM20';'LD21'; ...
        'ID20';'LM21';'IR20';                     'PR02';'PR04';'PR06';'XD01'; ...
        'TA01';'TA02';'PI00';'PI01';'PI11';'PR08';'PR10';'AM00';'PR00';'LI33'; ...
        'LI34';'CB00';'CB01';'CB02';'AB01';'LC00';'IG00';'PR03';'PR12';'LI35'; ...
        'BD01';'RF00';'TA03'; ...
        'XL03';'XL04';'XL05';'LI36';'LM22';'LM23';'LM24';'LM25'; ...
        'LM26';'LM27';'LM28';'LM29';'LM30';'LA25';'LD24';              'LA24'; ...
        'LA22';'LB22';'LA23';'LA27';'LB27';'LA28';'LB28';'BA00';'BM00';'BM01'; ...
        'PC00';                                                 'MV01'; ...
        'PX00';'PX01';                                          'SD01'; ...
        'CS00';'CS01';'CS02';'CS03';'CS04';'CS05'; ...
        'CS06';'CS08';'CS09';'CS10';'CS11';'CS12';'CS07';'CS13';'CS14'; ...
        'FC00'; ...
        'FC01';'FC02'; ...
        'CW01';'CW02';'CW03';'CW04';'CW05'; ...
        'CW06';'CW07';'CW08';'CW09';'FBUS';'TEST'; ...
        ];
      bitids=[ ...
        0; ...
        1;  2;  3;  4;  5;  6;  7;  8;  9; 10; ...
        11; 12; 13; 14; 15; 16; 17; 18; 19; 20; ...
        21; 22; 23; 24; 25; 26; 27; 28; 29; 30; ...
        31; 32; 33; 34; 35; 36; 37; 38; 39; 40; ...
        41; 42; 43; 44; 45; 46; 47; 48; 49; 50; ...
        51; 52; 53; 54; 55; 56; 57; 58; 59; 60; ...
        61; 62; 63; 64; 65; 66; 67; 68; 69; 70; ...
        71; 72; 73;             77; 78; 79; 80; ...
        81; 82; 83; 84; 85; 86; 87; 88; 89; 90; ...
        91; 92; 93; 94; 95; 96; 97; 98; 99;100; ...
        103;104;105; ...
        123;124;125;126;127;128;129;130; ...
        131;132;133;134;135;136;137;        140; ...
        141;142;143;144;145;146;147;148;149;150; ...
        151;                            159; ...
        160;161;                        169; ...
        185;186;187;188;189;190; ...
        191;192;193;194;195;196;197;198;199; ...
        220; ...
        221;222; ...
        256;257;258;259;260; ...
        261;262;263;264;265;266; ...
        ];
      
      % convert "C2" bitid to integer bitid
      
      % converts character 2 & 3 of model device name to integer micro bitid
      
      bitid = -1;
      
      if strcmp(c2bitid(1:1),'0')
        bitid = 0;
      elseif strcmp(c2bitid(1:1),'1')
        bitid = 10;
      elseif strcmp(c2bitid(1:1),'2')
        bitid = 20;
      elseif strcmp(c2bitid(1:1),'3')
        bitid = 30;
      elseif strcmp(c2bitid(1:1),'4')
        bitid = 40;
      elseif strcmp(c2bitid(1:1),'5')
        bitid = 50;
      elseif strcmp(c2bitid(1:1),'6')
        bitid = 60;
      elseif strcmp(c2bitid(1:1),'7')
        bitid = 70;
      elseif strcmp(c2bitid(1:1),'8')
        bitid = 80;
      elseif strcmp(c2bitid(1:1),'9')
        bitid = 90;
      elseif strcmp(c2bitid(1:1),'A')
        bitid = 100;
      elseif strcmp(c2bitid(1:1),'B')
        bitid = 110;
      elseif strcmp(c2bitid(1:1),'C')
        bitid = 120;
      elseif strcmp(c2bitid(1:1),'D')
        bitid = 130;
      elseif strcmp(c2bitid(1:1),'E')
        bitid = 140;
      elseif strcmp(c2bitid(1:1),'F')
        bitid = 150;
      end
      
      if bitid == -1
        error(['Bit ID ' c2bitid ' not found'])
      else
        bitid = bitid + str2double(c2bitid(2:2));
      end
      
      % find and return micro name
      id=find(bitids==bitid);
      if (isempty(id))
        error('c2bitid %s not found',c2bitid)
      else
        micr=micrs(id,:);
      end
      
    end
  end
end
