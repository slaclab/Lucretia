classdef DeckTool < handle
  % DECKTOOL Class for providing synchronization functionality with
  % external accelerator deck descriptions
  % Supports following deck standards:
  %  * XSIF: Compatible with LIBXSIF version 2.1. Also copes with
  %  most MAD8 formating.
  %  * BMAD (write only)
  %  * Elegant
  properties % GetPropInfo relies on order of public properties, only add to end
    MARK
    DRIF
    QUAD
    SEXT
    OCTU
    MULT
    SBEN
    SOLENOID
    LCAV
    TCAV
    XCOR
    YCOR
    XYCOR
    MONI
    HMON
    VMON
    INST
    PROF
    WIRE
    BLMO
    SLMO
    IMON
    COLL
    COORD
    TMAP
  end % CLASS names
  properties(Access=private)
    nsigfig=9; % max # of significant figures to use in numeric output
    splitMag % If empty do nothing, if false then no split magnets, if true then all split magnets
    BEAMLINE % Store initial full loaded BEAMLINE
    PS % Store initial full PS structure
    GIRDER % Store initial full GIRDER structure
    KLYSTRON % Store initial full KLYSTRON structure
    deckType='XSIF'; % XSIF, BMAD or Elegant
    maxcol % max # columns (set by deck type selection)
    blrange % required beamline range to operate on
    wakefiles % filenames to associate with WF structure entries
    doSplitDrifts=false; % split drift elements with Split command or not (change with SetDoSplitDrifts(val))
    basedir % base directory to reference files from (the main deck file calling directory)
    parsedWFfiles % list of already parsed wakefield files
    verinfo % version information about running Matlab instance
  end
  methods
    function obj=DeckTool(deckType,splitMag)
      % DT=DeckTool(rtdir,deckType,splitMag)
      global BEAMLINE PS GIRDER %#ok<GVMIS>
      % Parse desired deck type (default to XSIF if not provided)
      if ~exist('deckType','var')
        deckType='XSIF';
      end
      obj.SetDeckType(deckType);
      % Store copies of original Lucretia data structures
      obj.BEAMLINE=BEAMLINE;
      obj.PS=PS;
      obj.GIRDER=GIRDER;
      % Flag split magnet treatment or not
      if exist('splitMag','var')
        obj.SetSplitMag(splitMag);
      end
      if ~isempty(BEAMLINE)
        obj.SetProps(); % Initialize object properties from Lucretia globals
        % Need slice and block information
        SetElementSlices(1,length(BEAMLINE));
        SetElementBlocks(1,length(BEAMLINE));
        % Need floor coordinates defined
        if ~isfield(BEAMLINE{end},'Coordi')
          SetFloorCoordinates(1,length(BEAMLINE),zeros(1,6));
        end
        if ~isfield(BEAMLINE{end},'S') || isempty(BEAMLINE{end}.S) || BEAMLINE{end}.S==0
          SetSPositions(1,length(BEAMLINE),0);
        end
      end
      obj.verinfo=ver('Matlab'); % store info about current Matlab instance
    end
    function SetWakeFile(obj,WFno,type,filename)
      % Assign a wakefield file to a WF entry
      % WFno= WF structure index
      % type= 'ZSR' or 'TSR'
      % filename= wakefield data file name
      switch type
        case 'ZSR'
          obj.wakefiles{WFno,1}=filename;
        case 'TSR'
          obj.wakefiles{WFno,2}=filename;
      end
    end
    function SelectSubBeamline(obj,name1,name2)
      % Set beamline range for methods to operate on
      global BEAMLINE
      i1=findcells(BEAMLINE,'Name',name1);
      if isempty(i1)
        error('Unknown BEAMLINE element name: %s',name1)
      else
        i1=i1(1);
      end
      i2=findcells(BEAMLINE,'Name',name2);
      if isempty(i2)
        error('Unknown BEAMLINE element name: %s',name2)
      else
        i2=i2(end);
      end
      obj.blrange=[i1 i2];
    end
    function SetSplitMag(obj,dosplit)
      obj.splitMag=logical(dosplit);
    end
    function SetDoSplitDrifts(obj,val)
      if exist('val','var') && val
        obj.doSplitDrifts=true;
      else
        obj.doSplitDrifts=false;
      end
    end
    function Initial = ReadDeck(obj,filename,linename,betaname,beamname)
      % [Initial,Beam] = ReadDeck(obj,filename [,linename,betaname,beamname])
      % Read deck in chosen file format (obj.deckType)
      % filename: filename to read (char array for xsif, bmad, 2 element cell array for Elegant- with {lat file, ele file})
      % linename (optional): Name of line to use in deck
      % betaname (optional): Initial BETA0 parameters to use
      % beamname (optional): Use data from beamname
      global BEAMLINE
      if ~exist('linename','var')
        linename='';
      end
      if ~exist('betaname','var')
        betaname='';
      end
      if ~exist('beamname','var')
        beamname='';
      end
      if ~exist('filename','var')
        error('Must provide file name argument');
      end
      % Initialize some internal variables
      obj.parsedWFfiles={};
      if iscell(filename)
        D=dir(filename{1});
      else
        D=dir(filename);
      end
      obj.basedir=D.folder;% Assume all called files are relative to main deck file
      if iscell(filename)
        D2=dir(filename{2});
        filename={D.name D2.name};
      else
        filename=D.name;
      end
      switch obj.deckType
        case 'XSIF'
          % read in raw deck file
          txt=obj.XSIFRead(filename);
          % parse deck to Lucretia format
          Initial = obj.XSIFParse(txt,linename,betaname,beamname) ;
        case 'Elegant'
          % read in raw deck file
          txt=obj.ElegantRead(filename);
          % parse deck to Lucretia format
          Initial = obj.ElegantParse(txt,linename,betaname,beamname) ;
      end
      % Convert any bends with K2 elements temp stored in B(3)- use Multipole components
      isb=findcells(BEAMLINE,'Class','SBEN');
      isb=isb(arrayfun(@(x) length(BEAMLINE{x}.B)>2,isb));
      if ~isempty(isb)
        if isb(1)>1
          BLnew=BEAMLINE(1:isb(1)-1);
        else
          BLnew={};
        end
        nsb=1;
        while nsb<=length(isb)
          bele=BEAMLINE{isb(nsb)};
          k2ele = MultStruc(0,bele.B(3)/2,0,2,[0 0],1,sprintf('%s_K2',bele.Name)) ;
          k2ele.P=bele.P;
          if bele.B(2)==0
            bele.B=bele.B(1);
          else
            bele.B=bele.B(1:2);
          end
          BLnew=[BLnew; k2ele; bele; k2ele];
          nsb=nsb+1;
          if nsb>length(isb) && length(BLnew)<length(BEAMLINE)
            BLnew=[BLnew; BEAMLINE(isb(end)+1:end)];
          elseif nsb<=length(isb) && (isb(nsb)-isb(nsb-1))>1
            BLnew=[BLnew; BEAMLINE(isb(nsb-1)+1:isb(nsb)-1)];
          end
        end
        if isb(end)<length(BEAMLINE)
          BEAMLINE=[BLnew; BEAMLINE(isb(end)+1:end)];
        else
          BEAMLINE=BLnew;
        end
      end
      SetSPositions(1,length(BEAMLINE),0);
      SetFloorCoordinates(1,length(BEAMLINE),zeros(1,6));
    end
    function WriteDeck(obj,Initial,filename,linename,useline)
      % Write deck out to chosen file format
      % Initial = Lucretia Initial beam structire
      % filename: filename to write to
      % linename: Name of BEAMLINE line to use in deck
      % useline: set true to embed use command in file for this linename
      % uses whole BEAMLINE unless sub-BEAMLINE selected with
      % SelectSubBeamline method
      if ~isempty(obj.blrange)
        obj.SetSubBeamline();
      end
      % try
        if ~isempty(obj.splitMag) && obj.splitMag
          obj.splitMags();
        elseif ~isempty(obj.splitMag)
          obj.unsplitMags();
        end
        if ~exist('useline','var')
          useline=false;
        end
        obj.deckWrite(Initial,filename,linename,useline);
      % catch ME
      %   if ~isempty(obj.splitMag) || ~isempty(obj.blrange)
      %     obj.restoreLucretiaData();
      %   end
      %   rethrow(ME)
      % end
      if ~isempty(obj.splitMag) || ~isempty(obj.blrange)
        obj.restoreLucretiaData();
      end
    end
    function SetDeckType(obj,deckType)
      switch deckType
        case 'XSIF'
          obj.deckType='XSIF';
          obj.maxcol=79;
        case 'BMAD'
          obj.deckType='BMAD';
          obj.maxcol=79;
        case 'Elegant'
          obj.deckType='Elegant';
          obj.maxcol=79;
        otherwise
          error('Unsupported Deck Type')
      end
    end
    function WriteSurvey(obj,filename,Initial,i1,i2,SC,espr)
      %  WriteSurvey(filename,Initial,i1,i2 [,SC])
      % Write "survey" file- csv file listing model elements and
      % coordinates
      % filename= name of survey file to write
      % Initial= Lucretia Initial structure @ i1
      % File written for elements i1:i2
      % SC (optional) - write beam stay clear defined as N x beta beam size + M x dispersive beam size
      % (added in quadrature) where SC=[N M]
      % espr (optional) - use given energy spread (dE/E) instead of that
      % provided by Initial - provide vector input of length [i1:i2] to
      % provide at each element or scalar to apply for all
      global BEAMLINE
      % Stay clear defined as 10X beta beam size + dispersive beam size
      % (added in quadrature)
      useclass={'SBEN' 'SEXT' 'QUAD' 'LCAV' 'TCAV' 'XCOR' 'YCOR' 'OCTU' 'SOLENOID' 'MONI' 'PROF' 'INST' 'WIRE' 'BLMO' 'SLMO' 'IMON' 'COLL' 'MARK' 'DRIF'};
      ind=0;
      Type={};
      if exist('SC','var')
        [~,T]=GetTwiss(i1,i2,Initial.x.Twiss,Initial.y.Twiss);
      end
      if ~exist('espr','var')
        espr=ones(1,i2).*(Initial.SigPUncorrel/Initial.Momentum);
      elseif length(espr)==1
        espr=ones(1,1+i2-i1).*espr;
      end
      nespr=0;
      for iele=i1:i2
        nespr=nespr+1;
        if ~ismember(BEAMLINE{iele}.Class,useclass)
          continue
        end
        ind=ind+1;
        if isfield(BEAMLINE{iele},'Type')
          Type{ind}=BEAMLINE{iele}.Type;
        else
          Type{ind}=BEAMLINE{iele}.Name; %#ok<*AGROW>
        end
        ModelName{ind}=BEAMLINE{iele}.Name;
        X(ind)=BEAMLINE{iele}.Coordi(1);
        Y(ind)=BEAMLINE{iele}.Coordi(2);
        Z(ind)=BEAMLINE{iele}.Coordi(3);
        PHI(ind)=BEAMLINE{iele}.Anglei(2); % Pitch
        THETA(ind)=BEAMLINE{iele}.Anglei(1); % Yaw
        PSI(ind)=BEAMLINE{iele}.Anglei(3); % Roll
        if isfield(BEAMLINE{iele},'B')
          BL(ind)=BEAMLINE{iele}.B(1);
        else
          BL(ind)=0;
        end
        if ~exist('SC','var')
          SC_X(ind)=0; SC_Y(ind)=0;
        else
          gamma=BEAMLINE{iele}.P/0.511e-3;
          sx=sqrt(T.betax(1+iele-i1)*(Initial.x.NEmit/gamma));
          sy=sqrt(T.betay(1+iele-i1)*(Initial.y.NEmit/gamma));
          sx_D=T.etax(1+iele-i1) * espr(nespr) ;
          sy_D=T.etay(1+iele-i1) * espr(nespr) ;
          SC_X(ind)=sqrt((sx*SC(1))^2+(sx_D*SC(2))^2); SC_Y(ind)=sqrt((sy*SC(1))^2+(sy_D*SC(2))^2);
        end
        [L,K0L,K1L,K2L,K3L]=obj.getProps(iele);
        props.L(ind)=L; props.K0L(ind)=K0L; props.K1L(ind)=K1L; props.K2L(ind)=K2L;
        props.K3L(ind)=K3L;
      end
      X(abs(X)<1e-6)=0; Y(abs(Y)<1e-6)=0; Z(abs(Z)<1e-6)=0; PHI(abs(PHI)<1e-6)=0; THETA(abs(THETA)<1e-6)=0; PSI(abs(PSI)<1e-6)=0;
      T=table(ModelName',Type',X',Y',Z',THETA',PHI',PSI',SC_X',SC_Y',props.L',BL',props.K0L',props.K1L',props.K2L',props.K3L','VariableNames',...
        {'ModelName';'Type';'X';'Y';'Z';'THETA';'PHI';'PSI';'SC_X';'SC_Y';'L';'BL';'K0L';'K1L';'K2L';'K3L'}); % ,'RowNames',Label'
      writetable(T,filename);
    end
    function Diff(obj,filename,linename,betaname,beamname)
      % Diff(filename [,linename,betaname,beamname])
      % Difference of in-memory lattice to given file
      global BEAMLINE PS
      if ~exist('linename','var')
        linename='';
      end
      if ~exist('betaname','var')
        betaname='';
      end
      if ~exist('beamname','var')
        beamname='';
      end
      BL0=BEAMLINE; % Original lattice
      SetSPositions(1,length(BEAMLINE),0);
      SetFloorCoordinates(1,length(BEAMLINE),zeros(1,6));
      BL1=BEAMLINE;
      obj.ReadDeck(filename,linename,betaname,beamname); % lattice from file to compare loaded into BEAMLINE global
      SetSPositions(1,length(BEAMLINE),0);
      SetFloorCoordinates(1,length(BEAMLINE),zeros(1,6));
      cf={'B' 'S' 'Coordi' 'Coordf' 'L' 'Angle' 'Tilt' 'HGAP' 'FINT' 'EdgeAngle' 'P'}; % BEAMLINE fields to compare
      % - Write out where parsed BEAMLINE file differs from in-memory one
      for iele=1:length(BL1)
        if length(BEAMLINE)<iele
          fprintf('%s(%d): > EOF\n',BL1{iele}.Name,iele)
          continue
        end
        if ~strcmp(BEAMLINE{iele}.Name,BL1{iele}.Name)
          fprintf('%s(%d): Name Mismatch: < %s > %s\n',BL1{iele}.Name,iele,BL1{iele}.Name,BEAMLINE{iele}.Name)
          continue
        end
        if ~strcmp(BEAMLINE{iele}.Class,BL1{iele}.Class)
          fprintf('%s(%d): Class Mismatch: < %s > %s\n',BL1{iele}.Name,iele,BL1{iele}.Class,BEAMLINE{iele}.Class)
          continue
        end
        for icf=1:length(cf)
          if isfield(BL1{iele},cf{icf})
            if strcmp(cf{icf},'B') && isfield(BL1{iele},'PS') && BL1{iele}.PS>0
              lv=BL1{iele}.B.*PS(BL1{iele}.PS).Ampl;
            else
              lv=BL1{iele}.(cf{icf});
            end
            rv=BEAMLINE{iele}.(cf{icf});
            if length(lv)>length(rv) && isequal(lv(end),0)
              lv(end)=[];
            end
            if length(lv)<length(rv) && isequal(rv(end),0)
              rv(end)=[];
            end
            if ~isequal(num2str(lv,obj.nsigfig),num2str(rv,obj.nsigfig))
              % if numeric comparison and closer than 1ppm then ignore
              % difference
              if isnumeric(lv) && isnumeric(rv) && length(lv)==length(rv) && abs(1-max(abs(lv./rv)))<1e-6
                continue
              end
              fprintf('%s(%d) [%s]: < %s > %s\n',BL1{iele}.Name,iele,cf{icf},num2str(lv,obj.nsigfig),num2str(rv,obj.nsigfig));
            end
          end
        end
      end
      BEAMLINE=BL0; % Put back original beamline
    end
  end
  methods(Access=private)
    function SetSubBeamline(obj)
      global BEAMLINE PS GIRDER KLYSTRON
      BEAMLINE=BEAMLINE(obj.blrange(1):obj.blrange(2));
      for iPS=1:length(PS)
        PS(iPS).Element=PS(iPS).Element-(obj.blrange(1)-1);
      end
      for iGir=1:length(GIRDER)
        GIRDER{iGir}.Element=GIRDER{iGir}.Element-(obj.blrange(1)-1);
      end
      for iKly=1:length(KLYSTRON)
        KLYSTRON(iKly).Element=KLYSTRON(iKly).Element-(obj.blrange(1)-1);
      end
      SetElementSlices(1,length(BEAMLINE));
      SetElementBlocks(1,length(BEAMLINE));
      obj.SetProps();
    end
    function pinfo=GetPropInfo(obj,iele,prop,val)
      global BEAMLINE
      persistent dz
      if isempty(dz)
        dz=1;
      end
      pinfo.name={};
      pinfo.val=[];
      Cb=1e9/2.99792458e8;
      switch obj.deckType
        case 'XSIF'
          % Mapping of classnames (properties of this object) to XSIF elements types
          classnames={'MARKER' 'DRIFT' 'QUADRUPOLE' 'SEXTUPOLE' 'OCTUPOLE' 'MULTIPOLE' 'SBEND' 'SOLENOID' 'LCAVITY' 'DRIFT' 'HKICKER' ...
            'VKICKER' 'KICKER' 'MONITOR' 'HMONITOR' 'VMONITOR' 'INSTRUMENT' 'PROFILE' 'WIRE' 'BLMONITOR' 'SLMONITOR' 'IMONITOR' ...
            'COLL' 'GKICK' 'MATRIX'};
        case 'BMAD'
          % Mapping of classnames (properties of this object) to BMAD elements types
          classnames={'marker' 'drift' 'quadrupole' 'sextupole' 'octupole' 'multipole' 'sbend' 'solenoid' ...
            'lcavity, cavity_type=traveling_wave' 'crab_cavity' 'hkicker' ...
            'vkicker' 'kicker' 'instrument' 'instrument' 'instrument' 'instrument' 'instrument' 'instrument' 'instrument' 'instrument' 'instrument' ...
            'COLL' 'patch' 'taylor'};
        case 'Elegant'
          % Mapping of classnames (properties of this object) to Elegant elements types
          classnames={'MARK' 'DRIF' 'QUAD' 'SEXT' 'OCTU' 'MULT' 'CSBEND' 'SOLE' 'RFCW' 'RFDF' 'HKICK' ...
            'VKICK' 'KICKER' 'MONI' 'MONI' 'MONI' 'MONI' 'MONI' 'MONI' 'MONI' 'MONI' 'MONI' ...
            'COLL' 'MARK' 'EMATRIX'};
        otherwise
          error('Unknown conversion type')
      end
      pinfo.classname=classnames{ismember(properties(obj),BEAMLINE{iele}.Class)};
      if nargin==2
        return
      end
      % Alternate collimator classes depending on deck type
      if strcmp(pinfo.classname,'COLL')
        if strcmp(BEAMLINE{iele}.Geometry,'Ellipse')
          switch obj.deckType
            case 'Elegant'
              pinfo.classname='ECOL';
            otherwise
              pinfo.classname='ECOLLIMATOR';
          end          
        elseif strcmp(BEAMLINE{iele}.Geometry,'Rectangle')
          switch obj.deckType
            case 'Elegant'
              pinfo.classname='RCOL';
            otherwise
              pinfo.classname='RCOLLIMATOR';
          end  
        end
      end
      % If CSR or LSC enabled, use alternate element types for Elegant
      if strcmp(obj.deckType,'Elegant')
        if strcmp(BEAMLINE{iele}.Class,'SBEN') && isfield(BEAMLINE{iele}.TrackFlag,'CSR') && BEAMLINE{iele}.TrackFlag.CSR~=0
          pinfo.classname='CSRCSBEND';
        end
        % Use CSRDRIFT element for handling CSR and/or LSC simulations
        if strcmp(BEAMLINE{iele}.Class,'DRIF') && isfield(BEAMLINE{iele}.TrackFlag,'CSR') && BEAMLINE{iele}.TrackFlag.CSR~=0
          pinfo.classname='CSRDRIFT';
        end
        if strcmp(BEAMLINE{iele}.Class,'DRIF') && isfield(BEAMLINE{iele}.TrackFlag,'LSC') && BEAMLINE{iele}.TrackFlag.LSC~=0
          pinfo.classname='CSRDRIFT';
        end
      end
      switch prop
        case 'ISR'
          if strcmp(obj.deckType,'Elegant')
            if BEAMLINE{iele}.TrackFlag.SynRad>0
              pinfo.name{end+1}='ISR';
              pinfo.val(end+1)=1;
            end
          end
        case 'LSC'
          if isfield(BEAMLINE{iele},'TrackFlag') && isfield(BEAMLINE{iele}.TrackFlag,'LSC') && BEAMLINE{iele}.TrackFlag.LSC>0
            pinfo.name{end+1}='LSC_BINS';
            if isfield(BEAMLINE{iele}.TrackFlag,'LSC_npow2Bins') && BEAMLINE{iele}.TrackFlag.LSC_npow2Bins>0
              pinfo.val(end+1)=2^BEAMLINE{iele}.TrackFlag.LSC_npow2Bins;
            else
              pinfo.val(end+1)=4096;
            end
            if ~strcmp(BEAMLINE{iele}.Class,'DRIF')
              pinfo.name{end+1}='LSC';
              pinfo.val(end+1)=1;
              if isfield(BEAMLINE{iele}.TrackFlag,'LSC_smoothFactor') && BEAMLINE{iele}.TrackFlag.LSC_smoothFactor>0
                pinfo.name{end+1}='SMOOTHING';
                pinfo.val(end+1)=1;
              end
            end
            if isfield(BEAMLINE{iele}.TrackFlag,'LSC_cutoffFreq')
              pinfo.name{end+1}='LSC_HIGH_FREQUENCY_CUTOFF1';
              pinfo.val(end+1)=BEAMLINE{iele}.TrackFlag.LSC_cutoffFreq;
            end
            pinfo.name{end+1}='N_KICKS';
            pinfo.val(end+1)=ceil(BEAMLINE{iele}.L/0.01);
            if ismember(BEAMLINE{iele}.Class,{'SBEN','DRIF'})
              if ~isfield(BEAMLINE{iele}.TrackFlag,'CSR') || BEAMLINE{iele}.TrackFlag.CSR==0
                pinfo.name{end+1}='CSR';
                pinfo.val(end+1)=0;
                pinfo.name{end+1}='USE_STUPAKOV';
                pinfo.val(end+1)=1;
              end
            end
          end
        case 'CSR'
          if isfield(BEAMLINE{iele}.TrackFlag,'CSR') && ismember(BEAMLINE{iele}.Class,{'SBEN','DRIF'})
            if BEAMLINE{iele}.TrackFlag.CSR~=0
              pinfo.name{end+1}='CSR';
              pinfo.val(end+1)=1;
              if strcmp(BEAMLINE{iele}.Class,'SBEN')
                pinfo.name{end+1}='BINS';
                pinfo.val(end+1)=BEAMLINE{iele}.TrackFlag.CSR;
                pinfo.name{end+1}='SG_HALFWIDTH';
                pinfo.val(end+1)=1;
                pinfo.name{end+1}='N_SLICES';
                pinfo.val(end+1)=BEAMLINE{iele}.TrackFlag.Split;
              elseif strcmp(BEAMLINE{iele}.Class,'DRIF')
                pinfo.name{end+1}='USE_STUPAKOV';
                pinfo.val(end+1)=1;
                pinfo.name{end+1}='DZ';
                pinfo.val(end+1)=BEAMLINE{iele}.TrackFlag.CSR;
              end
            end
          end
        case {'L' 'Angle'}
          if val==0; return; end
          pinfo.name{end+1}=upper(prop);
          pinfo.val(end+1)=val;
        case 'Freq'
          switch obj.deckType
            case 'XSIF'
              pinfo.name{end+1}='FREQ';
              pinfo.val(end+1)=val;
            case 'BMAD'
              pinfo.name{end+1}='rf_frequency';
              pinfo.val(end+1)=val*1e6;
            case 'Elegant'
              if strcmp(BEAMLINE{iele}.Class,'TCAV')
                pinfo.name{end+1}='FREQUENCY';
              else
                pinfo.name{end+1}='FREQ';
              end
              pinfo.val(end+1)=val*1e6;
          end
        case 'Lrad'
          if val==0; return; end
          switch obj.deckType
            case {'XSIF','Elegant'}
              pinfo.name{end+1}='LRAD';
              pinfo.val(end+1)=val;
          end
        case 'Tilt'
          if ~any(val); return; end
          switch obj.deckType
            case 'Elegant'
              pinfo.name{end+1}='TILT';
              pinfo.val(end+1)=val(1);
            otherwise
              switch BEAMLINE{iele}.Class
                case 'MULT'
                  for ival=1:length(BEAMLINE{iele}.B)
                    pinfo.name{end+1}=sprintf('T%d',BEAMLINE{iele}.PoleIndex(ival));
                    pinfo.val(end+1)=BEAMLINE{iele}.Tilt(ival);
                  end
                case 'SBEN'
                  switch obj.deckType
                    case 'XSIF'
                      pinfo.name{end+1}='TILT';
                      pinfo.val(end+1)=val;
                    case 'BMAD'
                      pinfo.name{end+1}='ref_tilt';
                      pinfo.val(end+1)=val;
                  end
                otherwise
                  pinfo.name{end+1}='TILT';
                  pinfo.val(end+1)=val;
              end
          end
        case 'aper'
          switch BEAMLINE{iele}.Class
            case 'COLL'
              switch obj.deckType
                case 'XSIF'
                  pinfo.name{end+1}='XSIZE';
                  pinfo.name{end+1}='YSIZE';
                case 'BMAD'
                  pinfo.name{end+1}='x_limit';
                  pinfo.name{end+1}='y_limit';
                case 'Elegant'
                  pinfo.name{end+1}='X_MAX';
                  pinfo.name{end+1}='Y_MAX';
              end
              pinfo.val(end+1)=val(1);
              if length(val)==2
                pinfo.val(end+1)=val(2);
              else
                pinfo.val(end+1)=val(1);
              end
            otherwise
              switch obj.deckType
                case 'Elegant'
                  if strcmp(pinfo.classname,'MULT')
                    pinfo.name{end+1}='BORE';
                    pinfo.val(end+1)=norm(val);
                  end
                otherwise
                  pinfo.name{end+1}='APERTURE';
                  pinfo.val(end+1)=norm(val);
              end
          end
        case 'B'
          switch BEAMLINE{iele}.Class
            case 'QUAD'
              pinfo.name{end+1}='K1';
              pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
            case 'SEXT'
              pinfo.name{end+1}='K2';
              pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
            case 'OCTU'
              pinfo.name{end+1}='K3';
              pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
            case 'MULT'
              for ival=1:length(BEAMLINE{iele}.B)
                switch obj.deckType
                  case 'Elegant'
                    pinfo.name{end+1}='KNL';
                    KNL=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
                    pinfo.val(end+1)=KNL(1);
                    pinfo.name{end+1}='ORDER';
                    pinfo.val(end+1)=BEAMLINE{iele}.PoleIndex(1);
                  otherwise
                    pinfo.name{end+1}=sprintf('K%dL',BEAMLINE{iele}.PoleIndex(ival));
                    pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
                end
              end
            case 'SBEN'
              B=GetTrueStrength(iele);
              if length(B)>1 && abs(B(2))>0
                pinfo.name{end+1}='K1';
                pinfo.val(end+1)=B(2)/(Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
              end
            case 'SOLENOID'
              pinfo.name{end+1}='KS';
              pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P*BEAMLINE{iele}.L);
            case 'XCOR'
              switch obj.deckType
                case 'Elegant'
                  pinfo.name{end+1}='KICK';
                  pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
                otherwise
                  pinfo.name{end+1}='HKICK';
                  pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
              end
            case 'YCOR'
              switch obj.deckType
                case 'Elegant'
                  pinfo.name{end+1}='KICK';
                  pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
                otherwise
                  pinfo.name{end+1}='VKICK';
                  pinfo.val(end+1)=GetTrueStrength(iele)/(Cb*BEAMLINE{iele}.P);
              end
            case 'XYCOR'
              pinfo.name{end+1}='HKICK';
              pinfo.name{end+1}='VKICK';
              B=GetTrueStrength(iele)./(Cb*BEAMLINE{iele}.P);
              pinfo.val(end+1)=B(1);
              pinfo.val(end+1)=B(2);
          end
        case 'P'
          switch obj.deckType
            case 'XSIF'
              if strcmp(pinfo.classname,'LCAVITY')
                pinfo.name{end+1}='E0';
                pinfo.val(end+1)=val;
              end
          end
        case 'EdgeAngle'
          if ~any(val); return; end
          pinfo.name{end+1}='E1';
          pinfo.val(end+1)=val(1);
          pinfo.name{end+1}='E2';
          if length(val)==2
            pinfo.val(end+1)=val(2);
          else
            pinfo.val(end+1)=val(1);
          end
        case 'EdgeCurvature'
          if ~any(val); return; end
          pinfo.name{end+1}='H1';
          pinfo.val(end+1)=val(1);
          pinfo.name{end+1}='H2';
          if length(val)==2
            pinfo.val(end+1)=val(2);
          else
            pinfo.val(end+1)=val(1);
          end
        case 'Egain'
          if strcmp(obj.deckType,'Elegant')
            if val~=0
              pinfo.name{end+1}='CHANGE_P0';
              pinfo.val(end+1)=1;
            end
          end
        case 'HGAP'
          if ~any(val); return; end
          switch obj.deckType
            case 'XSIF'
              pinfo.name{end+1}='HGAP';
              pinfo.val(end+1)=val(1);
              if length(val)==2 && val(2)~=val(1)
                pinfo.name{end+1}='HGAPX';
                pinfo.val(end+1)=val(2);
              end
            case 'BMAD'
              pinfo.name{end+1}='h1';
              pinfo.val(end+1)=val(1);
              pinfo.name{end+1}='h2';
              if length(val)==2
                pinfo.val(end+1)=val(2);
              else
                pinfo.val(end+1)=val(1);
              end
          end
        case 'FINT'
          if ~any(val); return; end
          switch obj.deckType
            case 'Elegant'
              pinfo.name{end+1}='FINT';
              pinfo.val(end+1)=max(val);
              if length(val)==2 && val(2)~=val(1)
                pinfo.name{end+1}='EDGE1_EFFECTS';
                pinfo.val(end+1)=sign(abs(val(1)));
                pinfo.name{end+1}='EDGE2_EFFECTS';
                pinfo.val(end+1)=sign(abs(val(2)));
              end
            otherwise
              pinfo.name{end+1}='FINT';
              pinfo.val(end+1)=val(1);
              if length(val)==2 && val(2)~=val(1)
                pinfo.name{end+1}='FINTX';
                pinfo.val(end+1)=val(2);
              end
          end
        case 'Volt'
          switch obj.deckType
            case 'XSIF'
              pinfo.name{end+1}='DELTAE';
              pinfo.val(end+1)=GetTrueVoltage(iele);
            case 'BMAD'
              pinfo.name{end+1}='gradient';
              pinfo.val(end+1)=1e6*BEAMLINE{iele}.Volt/BEAMLINE{iele}.L;
            case 'Elegant'
              if strcmp(BEAMLINE{iele}.Class,'TCAV')
                pinfo.name{end+1}='VOLTAGE';
              else
                pinfo.name{end+1}='VOLT';
              end
              pinfo.val(end+1)=GetTrueVoltage(iele)*1e6;
              if GetTrueVoltage(iele)>0
                pinfo.name{end+1}='CHANGE_P0';
                pinfo.val(end+1)=1;
              end
          end
        case 'Phase'
          switch obj.deckType
            case 'Elegant'
              pinfo.name{end+1}='PHASE';
              pinfo.val(end+1)=90+GetTruePhase(iele);
            otherwise
              pinfo.name{end+1}='PHI0';
              pinfo.val(end+1)=GetTruePhase(iele)/360;
          end
        case 'Kloss'
          switch obj.deckType
            case 'XSIF'
              pinfo.name{end+1}='ELOSS';
            case 'BMAD'
              pinfo.name{end+1}='e_loss';
          end
          pinfo.val(end+1)=val*BEAMLINE{iele}.L;
        case 'Change'
          switch obj.deckType
            case 'XSIF'
              pinfo.name{end+1}='DX';
              pinfo.name{end+1}='DXP';
              pinfo.name{end+1}='DY';
              pinfo.name{end+1}='DYP';
              pinfo.name{end+1}='DL';
              pinfo.name{end+1}='ANGLE';
            case 'BMAD'
              pinfo.name{end+1}='x_offset';
              pinfo.name{end+1}='x_pitch';
              pinfo.name{end+1}='y_offset';
              pinfo.name{end+1}='y_pitch';
              pinfo.name{end+1}='z_offset';
              pinfo.name{end+1}='tilt';
          end
          for ival=1:6
            pinfo.val(end+1)=val(ival);
          end
        case 'R'
          for iarg1=1:6
            for iarg2=1:6
              rv=BEAMLINE{iele}.R(iarg1,iarg2);
              if abs(rv)>0
                switch obj.deckType
                  case 'XSIF'
                    pinfo.name{end+1}=sprintf('RM(%d,%d)',iarg1,iarg2);
                  case 'BMAD'
                    pinfo.name{end+1}=sprintf('tt%d%d',iarg1,iarg2);
                  case 'Elegant'
                    for i1=1:6
                      for i2=1:6
                        if abs(BEAMLINE{iele}.R(i1,i2))>0
                          pinfo.name{end+1}=sprintf('R%d%d',i1,i2);
                          pinfo.val(end+1)=BEAMLINE{iele}.R(i1,i2);
                        end
                      end
                    end
                end
                pinfo.val(end+1)=rv;
              end
            end
          end
        case 'T'
          for iT=1:length(val)
            switch obj.deckType
              case 'XSIF'
                ns=num2str(BEAMLINE{iele}.Tinds(iT));
                pinfo.name{end+1}=sprintf('TM(%d,%d,%d)',arrayfun(@(x) str2double(ns(x)),1:3));
              case 'BMAD'
                pinfo.name{end+1}=sprintf('tt%d',BEAMLINE{iele}.Tinds(iT));
              case 'Elegant'
                if ~isempty(BEAMLINE{iele}.Tinds)
                  for tind=1:length(BEAMLINE{iele}.Tinds)
                    pinfo.name{end+1}=sprintf('T%d',num2str(BEAMLINE{iele}.Tinds(tind)));
                    pinfo.val(end+1)=BEAMLINE{iele}.T(tind);
                  end
                end
            end
            pinfo.val(end+1)=val(iT);
          end
        case 'U'
          for iU=1:length(val)
            switch obj.deckType
              case 'BMAD'
                pinfo.name{end+1}=sprintf('tt%d',BEAMLINE{iele}.Uinds(iU));
            end
            pinfo.val(end+1)=val(iU);
          end
        case 'V'
          for iV=1:length(val)
            switch obj.deckType
              case 'BMAD'
                pinfo.name{end+1}=sprintf('tt%d',BEAMLINE{iele}.Vinds(iV));
            end
            pinfo.val(end+1)=val(iV);
          end
        case 'W'
          for iW=1:length(val)
            switch obj.deckType
              case 'BMAD'
                pinfo.name{end+1}=sprintf('tt%d',BEAMLINE{iele}.Winds(iW));
            end
            pinfo.val(end+1)=val(iW);
          end
      end
    end
    function deckWrite(obj,Initial,filename,linename,useline)
      % Write out external deck format file
      global BEAMLINE WF SDDS_SETUP
      if isempty(SDDS_SETUP)
        SDDS_SETUP=false;
      end
      % Constants and derived parameters
      qe=1.60217662e-19; % electron charge / C
      me=5.109989465237626e-04; % electron rest mass / GeV
      gamma=Initial.Momentum/me;
      % - Open file for writing
      if strcmp(obj.deckType,'Elegant')
        filename=[regexprep(filename,'\.lte','') '.lte'];
      end
      fid=fopen(filename,'w');
      if ~fid; error('Error opening %s',filename); end
      % - Write header
      fprintf(fid,'! Beam lines auto-generated from Lucretia: (%s)\n\n',datetime);
      fprintf(fid,'! ===============================================\n');
      
      % - Loop through CLASSes and write out beamline elements and properties
      BL0=BEAMLINE;
      try
        if strcmp(obj.deckType,'Elegant')
          obj.AssignCSR; % Assign CSR tags to DRIFTs downstream of CSR BENDS
        end
        classList=properties(obj);
        fprintf(fid,'\n');
        for iclass=1:length(classList)
          if isempty(obj.(classList{iclass}))
            continue
          end
          fprintf(fid,'\n');
          fprintf(fid,sprintf('! %s Definitions:\n\n',classList{iclass}));
          plist=fieldnames(obj.(classList{iclass})); plist=plist(~ismember(plist,'index'));
          % ignore repeated names
          classind=obj.(classList{iclass}).index;
          [~,IA]=unique(arrayfun(@(x) BEAMLINE{x}.Name,classind,'UniformOutput',false));
          IA=sort(IA);
          BL1=BEAMLINE;
          itype=0;
          while itype<length(IA)
            itype=itype+1;
            eleind=classind(IA(itype));
            if isempty(plist)
              pinfo=obj.GetPropInfo(eleind);
              fprintf(fid,sprintf('%s: %s\n',BEAMLINE{eleind}.Name,pinfo.classname));
              continue
            end
            addprop=false;
            for iprop=1:length(plist)
              pinfo=obj.GetPropInfo(eleind,plist{iprop},obj.(classList{iclass}).(plist{iprop}){IA(itype)});
              % Additional properties for LCAV/TCAV?
              if ~addprop && ismember(BEAMLINE{eleind}.Class,{'LCAV','TCAV'}) && strcmp(obj.deckType,'Elegant') % switch on edge-focusing treatment in RF structures
                pinfo.name{end+1}='END1_FOCUS';
                pinfo.val(end+1)=1;
                pinfo.name{end+1}='END2_FOCUS';
                pinfo.val(end+1)=1;
                addprop=true;
              end
              if iprop==1
                bname=BEAMLINE{eleind}.Name;
                newstr=sprintf('%s: %s',bname,pinfo.classname);
                fprintf(fid,newstr);
                nline=length(newstr);
              end
              if isempty(pinfo.name)
                continue
              end
              for iname=1:length(pinfo.name)
                switch obj.deckType
                  case {'XSIF','Elegant'}
                    newstr=sprintf('%s=%s',upper(pinfo.name{iname}),num2str(pinfo.val(iname),obj.nsigfig));
                  case 'BMAD'
                    newstr=sprintf('%s=%s',lower(pinfo.name{iname}),num2str(pinfo.val(iname),obj.nsigfig));
                end
                nline=nline+length(newstr);
                if (nline+2)>obj.maxcol
                  fprintf(fid,sprintf(',&\n%s',newstr));
                  nline=length(newstr);
                else
                  fprintf(fid,sprintf(',%s',newstr));
                  nline=nline+1;
                end
              end
            end
            newstr={};
            if isfield(BEAMLINE{eleind},'Type') && ~strcmp(BEAMLINE{eleind}.Type,'UNKNOWN')
              switch obj.deckType
                case 'XSIF'
                  newstr=sprintf('TYPE="%s"',regexprep(BEAMLINE{eleind}.Type,'\s+','_'));
                case 'BMAD'
                  newstr=sprintf('type="%s"',regexprep(BEAMLINE{eleind}.Type,'\s+','_'));
              end
              if ~isempty(newstr)
                nline=nline+length(newstr);
                if (nline+2)>obj.maxcol
                  fprintf(fid,sprintf(',&\n%s',newstr));
                  nline=length(newstr);
                else
                  fprintf(fid,sprintf(',%s',newstr));
                  nline=nline+1;
                end
              end
            end
            % Add wakefield file descriptors and write out wake files
            newstr={}; newstr{1}=[]; newstr{2}=[];
            if strcmp(obj.deckType,'XSIF')
              if isfield(BEAMLINE{eleind},'LWFfile') && ~isempty(BEAMLINE{eleind}.LWFfile)
                newstr{1}=sprintf('LFILE="%s"',BEAMLINE{eleind}.LWFfile);
              end
              if isfield(BEAMLINE{eleind},'TWFfile') && ~isempty(BEAMLINE{eleind}.TWFfile)
                newstr{2}=sprintf('TFILE="%s"',BEAMLINE{eleind}.TWFfile);
              end
            elseif strcmp(obj.deckType,'Elegant') && strcmp(pinfo.classname,'RFCW')
              newstr{1}=sprintf('ZWAKE=%d',BEAMLINE{eleind}.TrackFlag.SRWF_Z);
              newstr{2}=sprintf('TRWAKE=%d',BEAMLINE{eleind}.TrackFlag.SRWF_T);
              d=dir(filename);
              zwname=sprintf('zwake_%d.sdds',BEAMLINE{eleind}.Wakes(1));
              twname=sprintf('trwake_%d.sdds',BEAMLINE{eleind}.Wakes(2));
              dowake=false;
              if BEAMLINE{eleind}.Wakes(1)>0 && BEAMLINE{eleind}.TrackFlag.SRWF_Z
                newstr{end+1}=sprintf('ZWAKEFILE="%s"',zwname);
                dowake=true;
              end
              if length(BEAMLINE{eleind}.Wakes)>1 && BEAMLINE{eleind}.Wakes(2)>0 && BEAMLINE{eleind}.TrackFlag.SRWF_T
                newstr{end+1}=sprintf('TRWAKEFILE="%s"',twname);
                dowake=true;
              end
              if dowake
                newstr{end+1}='TCOLUMN=T';
                newstr{end+1}='WXCOLUMN=X';
                newstr{end+1}='WYCOLUMN=X';
                newstr{end+1}='WZCOLUMN=Z';
                newstr{end+1}='CELL_LENGTH=1';
                if ~SDDS_SETUP
                  sddspath=regexprep(which('DeckTool'),'LatticeGeneration/DeckTool.m','LatticeGeneration/SDDS');
                  addpath(sddspath);
                  javaaddpath(fullfile(sddspath,'SDDS.jar'), '-end');
                  SDDS_SETUP=true;
                end
                import SDDS.java.SDDS.* %#ok<SIMPT>
                if ~exist(zwname,'file')
                  iwf=BEAMLINE{eleind}.Wakes(1);
                  tvals = WF.ZSR(iwf).z./physConsts.clight ;
                  wvals = WF.ZSR(iwf).K ;
                  wake.column_names='TZ'; wake.column_names=wake.column_names';
                  wake.column.T.type='double';
                  wake.column.T.units='s';
                  wake.column.T.page1=tvals(:);
                  wake.column.Z.type='double';
                  wake.column.Z.units='V/C'; % really V/C/m but always set CELL_LENGTH=1
                  wake.column.Z.page1=wvals(:);
                  sddssave(wake,zwname);
                end
                if ~exist(twname,'file')
                  iwf=BEAMLINE{eleind}.Wakes(1);
                  tvals = WF.TSR(iwf).z./physConsts.clight ;
                  wvals = WF.TSR(iwf).K ;
                  wake.column_names='TX'; wake.column_names=wake.column_names';
                  wake.column.T.type='double';
                  wake.column.T.units='s';
                  wake.column.T.page1=tvals(:);
                  wake.column.X.type='double';
                  wake.column.X.units='V/C/m'; % really V/C/m^2 but always set CELL_LENGTH=1
                  wake.column.X.page1=wvals(:);
                  sddssave(wake,twname);
                end
              end
            end
            for istr=1:length(newstr)
              if ~isempty(newstr{istr})
                nline=nline+length(newstr{istr});
                if (nline+2)>obj.maxcol
                  fprintf(fid,sprintf(',&\n%s',newstr{istr}));
                  nline=length(newstr{istr});
                else
                  fprintf(fid,sprintf(',%s',newstr{istr}));
                  nline=nline+1;
                end
              end
            end
            fprintf(fid,'\n');
            % If multipole element and writing Elegant lattice, repeat multipole elements per PoleInex
            if strcmp(obj.deckType,'Elegant') && strcmp(BEAMLINE{eleind}.Class,'MULT')
              if length(BEAMLINE{eleind}.PoleIndex)>1
                BEAMLINE{eleind}.PoleIndex=BEAMLINE{eleind}.PoleIndex(2:end);
                BEAMLINE{eleind}.B=BEAMLINE{eleind}.B(2:end);
                BEAMLINE{eleind}.Tilt=BEAMLINE{eleind}.Tilt(2:end);
                itype=itype-1;
              end
            end
          end
          BEAMLINE=BL1;
        end
      catch ME
        BEAMLINE=BL0;
        throw(ME);
      end
      BEAMLINE=BL0;
      switch obj.deckType
        case 'XSIF'
          fprintf(fid,'TWSS0: BETA0,');
          fprintf(fid,'BETX=%s,',num2str(Initial.x.Twiss.beta,obj.nsigfig));
          fprintf(fid,'ALFX=%s,&\n',num2str(Initial.x.Twiss.alpha,obj.nsigfig));
          fprintf(fid,'MUX=%s,',num2str(Initial.x.Twiss.nu,obj.nsigfig));
          fprintf(fid,'DX=%s,',num2str(Initial.x.Twiss.eta,obj.nsigfig));
          fprintf(fid,'DPX=%s,&\n',num2str(Initial.x.Twiss.eta,obj.nsigfig));
          fprintf(fid,'BETY=%s,',num2str(Initial.y.Twiss.beta,obj.nsigfig));
          fprintf(fid,'ALFY=%s,&\n',num2str(Initial.y.Twiss.alpha,obj.nsigfig));
          fprintf(fid,'MUY=%s,',num2str(Initial.y.Twiss.nu,obj.nsigfig));
          fprintf(fid,'DY=%s,',num2str(Initial.y.Twiss.eta,obj.nsigfig));
          fprintf(fid,'DPY=%s,&\n',num2str(Initial.y.Twiss.eta,obj.nsigfig));
          fprintf(fid,'ENERGY=%s',num2str(Initial.Momentum,obj.nsigfig));
          fprintf(fid,'\n\n');
          fprintf(fid,'BEAM0: BEAM, ENERGY=%s,&\n',num2str(Initial.Momentum,obj.nsigfig));
          fprintf(fid,'NPART=%s,',num2str(Initial.Q/1.60217653e-19,obj.nsigfig));
          fprintf(fid,'EXN=%s,EYN=%s,&\n',num2str(Initial.x.NEmit,obj.nsigfig),num2str(Initial.y.NEmit,obj.nsigfig));
          fprintf(fid,'SIGT=%s,SIGE=%s',num2str(Initial.sigz,obj.nsigfig),num2str(Initial.SigPUncorrel/Initial.Momentum));
          emitz=sqrt(Initial.sigz^2*(Initial.SigPUncorrel/Initial.Momentum)^2 - Initial.PZCorrel^2);
          betaz=Initial.sigz^2/emitz;
          alphaz=-betaz*((Initial.PZCorrel./Initial.Momentum)/Initial.sigz^2);
          fprintf(fid,'\n!EMITZ=%s,BETAZ=%s,ALPHAZ=%s\n\n',num2str(emitz,obj.nsigfig),...
            num2str(betaz,obj.nsigfig),num2str(alphaz,obj.nsigfig));
          fprintf(fid,'\n\n');
          fprintf(fid,sprintf('%s: LINE=(%s',linename,BEAMLINE{1}.Name));
        case 'BMAD'
          fprintf(fid,'parameter[e_tot] = %.15g * 1e9\n',Initial.Momentum);
          fprintf(fid,'parameter[geometry] = Open\n');
          fprintf(fid,'parameter[lattice] = %s\n',linename);
          fprintf(fid,'parameter[n_part] = %d\n',ceil(Initial.Q/qe));
          fprintf(fid,'parameter[particle] = electron\n');
          fprintf(fid,'beam_start[x] = %g\n',Initial.x.pos);
          fprintf(fid,'beam_start[px] = %g\n',Initial.x.ang);
          fprintf(fid,'beam_start[y] = %g\n',Initial.y.pos);
          fprintf(fid,'beam_start[py] = %g\n',Initial.y.ang);
          fprintf(fid,'beam_start[z] = %g\n',Initial.zpos);
          fprintf(fid,'beam_start[emittance_a] = %g\n',Initial.x.NEmit/gamma);
          fprintf(fid,'beam_start[emittance_b] = %g\n',Initial.y.NEmit/gamma);
          fprintf(fid,'beam_start[sig_z] = %g\n',Initial.sigz);
          fprintf(fid,'beam_start[sig_e] = %g\n',Initial.SigPUncorrel/Initial.Momentum);
          fprintf(fid,'beginning[beta_a] = %.15g\n',Initial.x.Twiss.beta);
          fprintf(fid,'beginning[beta_b] = %.15g\n',Initial.y.Twiss.beta);
          fprintf(fid,'beginning[alpha_a] = %.15g\n',Initial.x.Twiss.alpha);
          fprintf(fid,'beginning[alpha_b] = %.15g\n',Initial.y.Twiss.alpha);
          fprintf(fid,'beginning[eta_x] = %.15g\n',Initial.x.Twiss.eta);
          fprintf(fid,'beginning[etap_x] = %.15g\n',Initial.x.Twiss.etap);
          fprintf(fid,'beginning[eta_y] = %.15g\n',Initial.y.Twiss.eta);
          fprintf(fid,'beginning[etap_y] = %.15g\n',Initial.y.Twiss.etap);
          fprintf(fid,'beginning[phi_a] = %.15g\n',Initial.x.Twiss.nu);
          fprintf(fid,'beginning[phi_b] = %.15g\n',Initial.y.Twiss.nu);
          fprintf(fid,'beginning[p0c] = %g * 1e9\n',Initial.Momentum);
          fprintf(fid,'beginning[s] = %.15g\n',BEAMLINE{1}.S);
          if isfield(BEAMLINE{1},'Coordi') && ~isempty(BEAMLINE{1}.Coordi)
            if abs(BEAMLINE{1}.Anglei(1))>0
              fprintf(fid,'beginning[theta_position] = %.15g\n',BEAMLINE{1}.Anglei(1));
            end
            if abs(BEAMLINE{1}.Anglei(2))>0
              fprintf(fid,'beginning[phi_position] = %.15g\n',BEAMLINE{1}.Anglei(2));
            end
            if abs(BEAMLINE{1}.Anglei(3))>0
              fprintf(fid,'beginning[psi_position] = %.15g\n',BEAMLINE{1}.Anglei(3));
            end
            if abs(BEAMLINE{1}.Coordi(1))>0
              fprintf(fid,'beginning[x_position] = %.15g\n',BEAMLINE{1}.Coordi(1));
            end
            if abs(BEAMLINE{1}.Coordi(2))>0
              fprintf(fid,'beginning[y_position] = %.15g\n',BEAMLINE{1}.Coordi(2));
            end
            if abs(BEAMLINE{1}.Coordi(3))>0
              fprintf(fid,'beginning[z_position] = %.15g\n',BEAMLINE{1}.Coordi(3));
            end
          end
          fprintf(fid,'\n\n');
          fprintf(fid,sprintf('%s: line=(%s',linename,BEAMLINE{1}.Name));
        case 'Elegant'
          cfid = fopen(regexprep(filename,'\.lte','.ele'),'w');
          fprintf(cfid,'&run_setup\n');
          fprintf(cfid,'    lattice = %s,\n',filename);
          fprintf(cfid,'    use_beamline = %s,\n',linename);
          fprintf(cfid,'    echo_lattice = 1,\n');
          fprintf(cfid,'    output = %%s.out,\n');
          fprintf(cfid,'    centroid = %%s.cen,\n');
          fprintf(cfid,'    sigma = %%s.sig,\n');
          fprintf(cfid,'    final = %%s.fin,\n');
          fprintf(cfid,'    magnets = %%s.mag,\n');
          fprintf(cfid,'    parameters = %%s.par,\n');
          fprintf(cfid,'    p_central_mev = %g,\n',Initial.Momentum*1e3);
          fprintf(cfid,'&end\n\n');
          fprintf(cfid,'&run_control &end\n\n');
          fprintf(cfid,'&twiss_output\n');
          fprintf(cfid,'    filename = %%s.twi,\n');
          fprintf(cfid,'    matched = 0,\n');
          fprintf(cfid,'    beta_x = %g,\n',Initial.x.Twiss.beta);
          fprintf(cfid,'    alpha_x = %g,\n',Initial.x.Twiss.alpha);
          fprintf(cfid,'    eta_x = %g,\n',Initial.x.Twiss.eta);
          fprintf(cfid,'    etap_x = %g,\n',Initial.x.Twiss.etap);
          fprintf(cfid,'    beta_y = %g,\n',Initial.y.Twiss.beta);
          fprintf(cfid,'    alpha_y = %g,\n',Initial.y.Twiss.alpha);
          fprintf(cfid,'    eta_y = %g,\n',Initial.y.Twiss.eta);
          fprintf(cfid,'    etap_y = %g,\n',Initial.y.Twiss.etap);
          fprintf(cfid,'&end\n');
          fclose(cfid);
          fprintf(fid,'BQ0: CHARGE, TOTAL=%g, ALLOW_TOTAL_CHANGE=1\n',Initial.Q);
          fprintf(fid,'TWSS0: TWISS, BETAX=%g, ALPHAX=%g, ETAX=%g, ETAXP=%g, BETAY=%g, ALPHAY=%g, ETAY=%g, ETAYP=%g\n',...
                      Initial.x.Twiss.beta,Initial.x.Twiss.alpha,Initial.x.Twiss.eta,Initial.x.Twiss.etap,...
                      Initial.y.Twiss.beta,Initial.y.Twiss.alpha,Initial.y.Twiss.eta,Initial.y.Twiss.etap) ;
          if isfield(BEAMLINE{1},'Coordi')
            fprintf(fid,'ICOORD: FLOOR, X=%g, Y=%g, Z=%g, THETA=%g, PHI=%g, PSI=%g\n',BEAMLINE{1}.Coordi,BEAMLINE{1}.Anglei);
          else
            fprintf(fid,'ICOORD: MARK\n');
          end
          fprintf(fid,'\n! ================================\n');
          fprintf(fid,sprintf('\n%s: LINE=(BQ0,TWSS0,ICOORD,%s',linename,BEAMLINE{1}.Name));
        otherwise
          error('Unknown deck type')
      end
      % - Write beamline
      nline=length(BEAMLINE{1}.Name)+length(linename)+8;
      if length(BEAMLINE)>1
        for ibl=2:length(BEAMLINE)
          if (nline+length(BEAMLINE{ibl}.Name)+2)>obj.maxcol
            fprintf(fid,sprintf(',&\n%s',BEAMLINE{ibl}.Name)); % end line if over max col width
            nline=length(BEAMLINE{ibl}.Name);
          else
            fprintf(fid,sprintf(',%s',BEAMLINE{ibl}.Name));
            nline=nline+length(BEAMLINE{ibl}.Name)+1;
          end
        end
      end
      fprintf(fid,')\n\n');
      if useline
        fprintf(fid,'\n!===============================\n!Line to use:\n');
        switch obj.deckType
          case 'XSIF'
            fprintf(fid,'USE, %s\n',linename);
          case 'BMAD'
            fprintf(fid,'use, %s\n',linename);
        end
      end
      fclose(fid);
    end
    function txt=XSIFRead(obj,filename)
      % Read xsif file, expand lines and call statements
      bdir=obj.basedir; % Assume all called files are relative to main xsif file
      if ~exist(fullfile(bdir,filename),'file')
        error('%s not found',filename)
      end
      fid=fopen(fullfile(bdir,filename),'r');
      % MAD commands etc to be ignored
      ignorecom={'ASSIGN' 'OPTION' 'SETPLOT' 'SPLIT' 'SURVEY' 'SAVEBETA' 'IBS' 'DYNAMIC' 'STATIC' 'OPTICS' 'SAVE' ...
        'BMPM' 'EXCITE' 'INCREMENT' 'ARCHIVE' 'RETRIEVE' 'PACKMEMORY' 'STATUS' 'POOLDUMP' 'POOLLOAD' ...
        'GETDISP' 'PUTDISP' 'GETKICK' 'PUTKICK' 'GETORBIT' 'PUTORBIT' 'USEKICK' 'USEMONITOR' 'MICADO' 'CORRECT' ...
        'RESPLOT' 'PLOT' 'TRACK' 'TWISS' 'TUNES' 'TABLE' 'STRING' 'LMDIF' 'MIGRAD' 'SIMPLEX' 'NOISE' 'OBSERVE' 'START' ...
        'TSAVE' 'RUN' 'CHROM' 'SHOW' 'PRINT' 'NORMAL' 'EPRINT' 'EOPT' 'EALIGN' 'EFCOMP' 'EFIELD' 'TITLE' 'BEAM'};
      for icom=1:length(ignorecom); ignorecom{icom}=sprintf('%s,',ignorecom{icom}); end
      txt={}; t_next=[]; sname=''; subtxt=[]; callstack={};
      while 1
        % Process next statement from previous line if seperated with ;
        if ~isempty(t_next)
          t=t_next;
        else % Or, Get new line and stop if last line
          try
            t=fgetl(fid);
          catch
            t=[];
          end
        end
        if ~ischar(t) % hit end of this file, if more on the stack, open that, else break out of this loop
          if isempty(callstack)
            break
          end
          try
            fclose(fid);
          catch
          end
           if ~exist(fullfile(bdir,callstack{1}),'file')
            error('%s not found',callstack{1})
          end
          fid=fopen(fullfile(bdir,callstack{1}),'r');
          callstack(1)=[];
          t=fgetl(fid);
        end
        % Remove comments and whitespace
        t=regexprep(regexprep(t,'!.*',''),'\s+','');
        if isempty(t)
          continue
        end
        % Concatinate line extensions
        while ~isempty(t) && t(end)=='&'
          tcont=fgetl(fid);
          if ~ischar(tcont)
            error('Deck file cannot end with continution (&) character!')
          end
          tcont=regexprep(regexprep(tcont,'!.*',''),'\s+','');
          if ~isempty(tcont)
            t=[t(1:end-1) tcont];
          end
        end
        % Parser cannot deal with special characters ; or , in quotes,
        % replace by string equivalents to be re-substituted later
        lquo=regexp(t,'"');
        if length(lquo)>=2
          for iquo=1:floor(length(lquo)/2)
            if contains(t(lquo(1+(iquo-1)*2):lquo(2+(iquo-1)*2)),',')
              newtxt=regexprep(t(lquo(1+(iquo-1)*2)+1:lquo(2+(iquo-1)*2)-1),',','__COMMA__');
              t=[t(1:lquo(1+(iquo-1)*2)) newtxt t(lquo(2+(iquo-1)*2):end)];
            end
            if contains(t(lquo(1+(iquo-1)*2):lquo(2+(iquo-1)*2)),';')
              newtxt=regexprep(t(lquo(1+(iquo-1)*2)+1:lquo(2+(iquo-1)*2)-1),';','__SEMICOLON__');
              t=[t(1:lquo(1+(iquo-1)*2)) newtxt t(lquo(2+(iquo-1)*2):end)];
            end
          end
        end
        % Deal with multiple commands on one line
        if contains(t,';')
          tl=regexp(t,';','once');
          t_next=t(tl+1:end);
          t=t(1:tl-1);
        else
          t_next=[];
        end
        % If there are MAD8 style MATRIX element definitions, re-format
        % them as XSIF style ones
        if contains(t,':MATR','IgnoreCase',true)
          t=regexprep(t,'rm\((\d),(\d)\)','R$1$2','ignorecase');
          t=regexprep(t,'tm\((\d),(\d),(\d)\)','T$1$2$3','ignorecase');
        end
        % If reach STOP or RETURN statement, ignore everything following
        if startsWith(t,{'STOP' 'RETURN'},'IgnoreCase',true)
          fclose(fid);
          continue
        end
        % Ignore any comment blocks
        if startsWith(t,'COMMENT','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDCOMMENT','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Replace constant declarations with assignments
        t=regexprep(t,':CONSTANT=',':='); t=regexprep(t,':constant=',':=');
        % Replace LIST's with LINE's
        t=regexprep(t,':LIST=',':LINE='); t=regexprep(t,':list=',':LINE=');
        % Replace set commands with := assignment operators
        if startsWith(t,'set,','IgnoreCase',true)
          t=regexprep(t,'^SET,','','ignorecase');
          t=regexprep(t,'=',':=');
          t=regexprep(t,',',':=');
        end
        % Flag SUBROUTINE blocks
        if startsWith(t,'ENDSUBROUTINE','IgnoreCase',true)
          sname='';
          continue
        elseif endsWith(t,'SUBROUTINE','IgnoreCase',true)
          sname=regexprep(t,':SUBROUTINE','');
          subtxt.(sname)={};
          continue
        end
        % Ignore MATCH blocks
        if endsWith(t,'MATCH','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDMATCH','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Ignore HARMON blocks
        if endsWith(t,'HARMON','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDHARM','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Remove lines featured in ignore list
        if startsWith(t,ignorecom,'IgnoreCase',true)
          while 1
            if ~ischar(t) || ~endsWith(t,'&')
              break
            else
              t=fgetl(fid);
            end
          end
          continue
        end
        % Replace any $ chars with _
        t=regexprep(t,'\$','_');
        % Expand called external files or just add line to txt stack
        if startsWith(t,'call','IgnoreCase',true)
          % try with filename parameter
          try
            cfn=regexprep(cell2mat(regexpi(t,'"(\S+\.?\S*)"','tokens','once')),'"','');
          catch
            cfn=[];
          end
          fprintf('Calling file: %s\n',cfn);
          if isempty(cfn) || ~exist(fullfile(bdir,cfn),'file')
            fclose(fid);
            error('Unfound filename in CALL statement: %s',fullfile(bdir,cfn))
          end
          % Add file to stack to read after done with this one
          callstack{end+1}=cfn;
        elseif isempty(sname) && isfield(subtxt,t) % expand out a subroutine call
          for isub=1:length(subtxt.(t))
            txt{end+1}=subtxt.(t){isub};
          end
        elseif ~isempty(sname) % store subroutine commands
          subtxt.(sname){end+1}=t;
        else
          txt{end+1}=t;
        end
      end
      try
        fclose(fid);
      catch
      end
    end
    function Initial=XSIFParse(obj,txt,linename,betaname,beamname)
      global BEAMLINE WF
      % MAD math functions and Matlab equivalents
      madmath={'sqrt(' 'log(' 'exp(' 'sin(' 'cos(' 'tan(' 'asin(' 'abs(' 'max(' 'min(' 'ranf(' 'gauss('} ;
      matmath={'sqrt(' 'log(' 'exp(' 'sin(' 'cos(' 'tan(' 'asin(' 'abs(' 'max(' 'min(' 'rand(' 'randn('} ;
      Initial=InitCondStruc();
      if isempty(txt)
        return
      end
      if exist('linename','var')
        linename=lower(linename);
      end
      if exist('betaname','var')
        betaname=lower(betaname);
      end
      if exist('beamname','var')
        beamname=lower(beamname);
      end
      uselist={}; % container for indicated beamlines, beta0, beam's to use
      % Default named parameters
      np.pi=pi;
      np.twopi=pi*2;
      np.emass=5.109989e-4;
      np.clight=299792458;
      np.e=2.7182818284590;
      np.degrad=57.295779513082323;
      np.raddeg=0.017453292519943;
      np.electron=-1;
      np.positron=1;
      % Pull out user-defined named parameters, also pull out line name if
      % there is one, also form element, parameter lists
      inp=[]; ndef=[]; firstcall=true; en=[];
      while ~isempty(ndef) || firstcall
        if firstcall
          prog=true;
        else
          prog=false;
        end
        firstcall=false;
        for itxt=1:length(txt)
          txt{itxt}=lower(txt{itxt});
          tok=regexp(txt{itxt},'^(\w+):?=(.*)','tokens','once');
          if length(tok)==2 && ~ismember(itxt,inp)
            expval=[];
            try
              % replace variable names
              if isempty(strfind(tok{2},'np.'))
                tok{2}=regexprep(tok{2},'([a-z|A-Z]\w*)','np.$1');
                tok{2}=regexprep(tok{2},'([\dn|\.])np\.(e[-?+?\d])','$1$2'); % ignore exponent case
                % Deal with math function case
                imatch=1;
                while contains(tok{2},madmath) && imatch<=length(madmath)
                  tok{2}=regexprep(tok{2},sprintf('np.%s',madmath{imatch}),matmath(imatch));
                  imatch=imatch+1;
                end
              end
              % deal with possible name[par] format
              if contains(tok{2},'[')
                tok{2}=regexprep(tok{2},'\[np\.(\w+)','[$1');
                tokex = regexp(tok{2},'np\.(\w+)\[(\w+)\]','tokens','once');
                if ~isfield(en.(tokex{1}),tokex{2}) % looking for keyword concatination
                  fn=fieldnames(en.(tokex{1}));
                  for ifn=1:length(fn)
                    if startsWith(string(tokex{2}),fn{ifn}) || startsWith(fn{ifn},tokex{2})
                      tok{2}=sprintf('en.%s.%s',tokex{1},fn{ifn});
                      break;
                    end
                  end
                else
                  tok{2}=regexprep(tok{2},'np\.(\w+)\[(\w+)\]','en.$1.$2');
                end
              end
              expval=eval(tok{2});
              ndef(ismember(ndef,itxt))=[];
            catch % if still cannot eval, flag and move on- resolve on next try, or fail
              if ismember(itxt,ndef) && ~prog
                error('Failed to resolve expression in line: %s',sprintf('%s := %s',tok{1},tok{2}))
              elseif ~ismember(itxt,ndef)
                ndef(end+1)=itxt;
                if isfield(np,tok{1})
                  np=rmfield(np,tok{1});
                end
              end
            end
            if ~isempty(expval)
              if isfield(np,tok{1})
                warning('Lucretia:DeckTool','parameter %s defined multiple times',tok{1});
              end
              np.(tok{1})=expval;
              prog=true;
              inp(end+1)=itxt;
            end
          elseif ~ismember(itxt,inp)
            % Store any use commands
            ln=cell2mat(regexpi(txt{itxt},'use,(\w+)','tokens','once'));
            if ~isempty(ln)
              uselist{end+1}=ln;
              inp(end+1)=itxt;
              continue;
            end
            % form element structure
            tok=regexp(txt{itxt},'^(\w+):(\w+),?(.*)','tokens','once');
            if length(tok)>=2 && ~ismember(itxt,inp) && ~strcmp(tok{2},'line')
              if isfield(en,tok{1}) && ~ismember(itxt,ndef)
                warning('Lucretia:DeckTool:MultiEleDeclare','Element %s defined multiple times',tok{1})
              end
              en.(tok{1}).Class=tok{2};
              if length(tok)>=3 && ~isempty(tok{3})
                par=split(tok{3},',');
                ndef(ismember(ndef,itxt))=[];
                for ipar=1:length(par)
                  pval=regexp(char(par(ipar)),'(\w+)=?(.*)','tokens','once');
                  % Re-substitute , and ; characters that are inside quotes
                  pval{2}=regexprep(pval{2},'__COMMA__',',','ignorecase');
                  pval{2}=regexprep(pval{2},'__SEMICOLON__',';','ignorecase');
                  if length(pval)==2 && ~isempty(pval{2})
                    % Replace quotes to '' so matlab eval parses them
                    % correctly
                    pval{2}=regexprep(pval{2},'"','''');
%                     pval{2}=regexprep(regexprep(pval{2},'''',''''''),'"','''''');
                    % Try evaluating parameter expression
                    try
                      % replace variable names
                      if isempty(strfind(pval{2},'np.')) && pval{2}(1)~=''''
                        pval{2}=regexprep(regexprep(pval{2},'([a-z|A-Z]\w*)','np.$1'),'np\.([a-z|A-Z]\w*\()','$1');
                        pval{2}=regexprep(pval{2},'([\dn|\.])np\.(e[-?+?\d])','$1$2'); % ignore exponent case
                      end
                      % Deal with math function case
                      imatch=1;
                      while contains(pval{2},madmath) && imatch<=length(madmath)
                        pval{2}=regexprep(pval{2},sprintf('np.%s',madmath{imatch}),matmath(imatch));
                        imatch=imatch+1;
                      end
                      % deal with possible name[par] format
                      if contains(pval{2},'[')
                        pval{2}=regexprep(pval{2},'\[np\.(\w+)','[$1');
                        pval{2}=regexprep(pval{2},'np\.(\w+)\[(\w+)\]','en.$1.$2');
                      end
                      % Evaluate parameter
                      en.(tok{1}).(pval{1})=eval(pval{2});
                    catch % if cannot eval, flag and move on- resolve on next try, or fail
                      if ~prog
                        error('Failed to resolve parameter in element: %s',tok{1})
                      elseif ~ismember(itxt,ndef)
                        ndef(end+1)=itxt;
                        continue
                      end
                    end
                  elseif ~isempty(pval)
                    en.(tok{1}).(pval{1})=[];
                  else
                    en.(tok{1})=[];
                  end
                end
              elseif length(tok)<3
                en.(tok{1})=[];
              end
              if ~ismember(itxt,ndef)
                inp(end+1)=itxt;
              end
            elseif length(tok)>1 && strcmp(tok{2},'line') && ~ismember(itxt,inp) % deal with line definitions seperately
              en.(tok{1}).Class='line';
              en.(tok{1}).linearg=regexprep(txt{itxt},'^.+=(.*)','$1');
            end
          end
        end
      end
%       txt(inp)=[];
      % Parse element list
      fn=fieldnames(en); lines=[]; ifn=1;
      plist={'MARK' 'DRIF' 'QUAD' 'SEXT' 'OCTU' 'MULT' 'SBEN' 'SOLE' 'LCAV' 'DRIF' 'HKIC' ...
            'VKIC' 'KICK' 'MONI' 'HMON' 'VMON' 'INST' 'PROF' 'WIRE' 'BLMO' 'SLMO' 'IMON' ...
            'COLL' 'MATR' 'BETA' 'BEAM' 'LINE' 'SIGM' 'RCOL' 'ECOL' 'SROT' 'ROLL' 'ZROT' 'YROT' 'GKIC'};
      while ifn<=length(fn) %for ifn=1:length(fn)
        % if no class field at this stage, then it is a marker
        if ~isfield(en.(fn{ifn}),'Class')
          en.(fn{ifn}).Class='marker';
        end
        % Inherting from another element?
        if length(en.(fn{ifn}).Class)<4 || ~ismember(upper(en.(fn{ifn}).Class(1:4)),plist)
          iobj = en.(en.(fn{ifn}).Class) ;
          en.(fn{ifn}).Class = iobj.Class ;
          fn2=fieldnames(iobj);
          for ifn2=1:length(fn2)
            if ~isfield(en.(fn{ifn}),fn2{ifn2})
              en.(fn{ifn}).(fn2{ifn2}) = iobj.(fn2{ifn2}) ;
            end
          end
        end
        switch upper(en.(fn{ifn}).Class(1:4))
          case 'DRIF'
            L=obj.collectPar(en.(fn{ifn}),'l');
            en.(fn{ifn}).LucretiaElement = DrifStruc( L, fn{ifn} ) ;
          case {'SBEN','RBEN'}
            [L,ANG,K1,K2,E1,E2,Tilt,H1,H2,Hgap,Fint,Hgap2,Fint2,Type]=obj.collectPar(en.(fn{ifn}),'l','angle','k1','k2','e1','e2','tilt','h1','h2','hgap','fint','hgapx','fintx','type');
            E=[E1,E2];
            H=[H1,H2];
            if L==0
              L=1e-9;
            end
            if isnan(Hgap2)
              Hgap=[Hgap,Hgap];
            else
              Hgap=[Hgap,Hgap2];
            end
            if isnan(Fint2)
              Fint=[Fint,Fint];
            else
              Fint=[Fint,Fint2];
            end
            if isnan(K1)
              BField=ANG/L;
            else
              BField=[ANG/L,K1];
            end
            if isempty(Tilt)
              Tilt=pi/2;
            end
            en.(fn{ifn}).LucretiaElement = SBendStruc( L, BField.*L, ANG, E, H, Hgap, Fint, Tilt, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            % Deal with design sextupole field content - later gets turned
            % into MULT elements either side of bend
            if ~isnan(K2) && abs(K2)>0
              if length(BField)==2
                BField=[BField,-K2*sign(ANG)];
              else
                BField=[BField,0,-K2*sign(ANG)];
              end
              en.(fn{ifn}).LucretiaElement.B = BField.*L ;
            end
          case 'QUAD'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k1','tilt','aperture','type') ;
            if L==0
              L=1e-9;
            end
            if isempty(Tilt)
              Tilt=pi/4;
            end
            if isnan(B); B=0; end
            en.(fn{ifn}).LucretiaElement = QuadStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'SEXT'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k2','tilt','aperture','type') ;
            if L==0
              L=1e-9;
            end
            if isempty(Tilt)
              Tilt=pi/6;
            end
            en.(fn{ifn}).LucretiaElement = SextStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'OCTU'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k3','tilt','aperture','type') ;
            if isempty(Tilt)
              Tilt=pi/8;
            end
            en.(fn{ifn}).LucretiaElement = OctuStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'MULT'
            [L, LRAD, TiltAll, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','lrad','tilt','aperture','type') ;
            B=[]; Tilt=[]; PIndx=[];
            mpn=fieldnames(en.(fn{ifn}));
            kv=regexp(mpn,'^k(\d+)l?','ignorecase','tokens','once');
            for ik=1:length(kv)
              if ~isempty(kv{ik})
                B(end+1)=en.(fn{ifn}).(mpn{ik});
                if isfield(en.(fn{ifn}),sprintf('t%c',kv{ik}{1}))
                  if isempty(en.(fn{ifn}).(sprintf('t%c',kv{ik}{1})))
                    Tilt(end+1)=pi/((str2double(kv{ik}{1})+1)*2)+TiltAll;
                  else
                    Tilt(end+1)=en.(fn{ifn}).(sprintf('t%c',kv{ik}{1}))+TiltAll;
                  end
                else
                  Tilt(end+1)=TiltAll;
                end
                PIndx(end+1)=str2double(kv{ik}{1});
              end
            end
            en.(fn{ifn}).LucretiaElement = MultStruc( L, B, Tilt, PIndx, [0 0], aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            if ~isnan(LRAD)
              en.(fn{ifn}).LucretiaElement.Lrad=LRAD;
            end
          case 'SOLE'
            [L, B, aper,tilt, Type] = obj.collectPar(en.(fn{ifn}),'l','ks','aperture','tilt','type') ;
            if L==0
              L=1e-9;
            end
            en.(fn{ifn}).LucretiaElement = SolenoidStruc( L, B, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Offset(6) = tilt ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'LCAV','TCAV'}
            [L, V,phi,freq,eloss,lfile,tfile,aper,nbin,tilt,Type] = obj.collectPar(en.(fn{ifn}),'l','deltae','phi0','freq','eloss','lfile','tfile','aperture','nbin','tilt','type');
            BinWidth=1/nbin; % BinWidth is fraction of sigma - default here to 0.1
            srwf_z=0; srwf_t=0;
            if ~any(isnan(lfile)) && ~isempty(lfile) && ~ismember(lfile,obj.parsedWFfiles) && exist(fullfile(obj.basedir,lfile),'file')
              [stat,W] = ParseSRWF( fullfile(obj.basedir,lfile), BinWidth ) ;
              if stat{1}~=1; error(stat{2}); end
              obj.parsedWFfiles{end+1}=lfile;
              if isempty(WF) || ~isfield(WF,'ZSR')
                WF.ZSR(1)=W;
              else
                WF.ZSR(end+1)=W;
              end
              srwf_z=length(WF.ZSR);
            elseif ~any(isnan(lfile)) && ~isempty(lfile) && ~ismember(lfile,obj.parsedWFfiles) && ~exist(fullfile(obj.basedir,lfile),'file')
              warning('Lucretia:DeckTool:nolfile','referenced lfile not found, skipping: %s',fullfile(obj.basedir,lfile))
            end
            if ~any(isnan(tfile)) && ~isempty(tfile) && ~ismember(lfile,obj.parsedWFfiles) && exist(fullfile(obj.basedir,tfile),'file')
              [stat,W] = ParseSRWF( fullfile(obj.basedir,tfile), BinWidth ) ;
              obj.parsedWFfiles{end+1}=tfile;
              if stat{1}~=1; error(stat{2}); end
              if isempty(WF) || ~isfield(WF,'TSR')
                WF.TSR(1)=W;
              else
                WF.TSR(end+1)=W;
              end
              srwf_t=length(WF.TSR);
            elseif ~any(isnan(tfile)) && ~isempty(tfile) && ~ismember(lfile,obj.parsedWFfiles) && ~exist(fullfile(obj.basedir,tfile),'file')
              warning('Lucretia:DeckTool:notfile','referenced tfile not found, skipping: %s',fullfile(obj.basedir,tfile))
            end
            if strcmpi(en.(fn{ifn}).Class(1:4),'TCAV')
              mode=1;
            else
              mode=0;
            end
            en.(fn{ifn}).LucretiaElement = RFStruc(  L, V, phi, freq, srwf_z, srwf_t, eloss, aper, fn{ifn}, mode ) ;
            en.(fn{ifn}).LucretiaElement.Offset(6) = tilt ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            if ~any(isnan(lfile)) && ~isempty(lfile)
              en.(fn{ifn}).LucretiaElement.LWFfile=lfile;
            end
            if ~any(isnan(tfile)) && ~isempty(tfile)
              en.(fn{ifn}).LucretiaElement.TWFfile=tfile;
            end
          case {'SROT','ROLL','ZROT','YROT','GKIC'}
            dpsi=0;
            [ang,dx,dtheta,dy,dphi,dz]=obj.collectPar(en.(fn{ifn}),'angle','dx','dxp','dy','dyp','dz');
            if ismember(upper(en.(fn{ifn}).Class(1:4)),{'SROT','ROLL','GKIC'})
              dpsi=ang;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'YROT')
              dtheta=ang;
            else
              dphi=ang;
            end
            en.(fn{ifn}).LucretiaElement = CoordStruc( dx, dtheta, dy, dphi, dz, dpsi, fn{ifn} );
          case {'HKIC','VKIC','KICK'}
            [L,KICK,HKICK,VKICK,TILT, Type]=obj.collectPar(en.(fn{ifn}),'l','kick','hkick','vkick','tilt','type');
            if strcmpi(en.(fn{ifn}).Class(1:4),'HKIC')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, KICK, TILT, 1, fn{ifn} ) ;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'VKIC')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, KICK, TILT, 2, fn{ifn} ) ;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'KICK')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, [HKICK,VKICK], TILT, 3, fn{ifn} ) ;
            end
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'ECOL','RCOL'}
            [L,xgap,ygap,Tilt,Type]=obj.collectPar(en.(fn{ifn}),'l','xsize','ysize','tilt','type') ;
            if strcmpi(en.(fn{ifn}).Class(1:4),'ECOL')
              en.(fn{ifn}).LucretiaElement = CollStruc( L, xgap, ygap, 'Ellipse', Tilt, fn{ifn} ) ;
            else
              en.(fn{ifn}).LucretiaElement = CollStruc( L, xgap, ygap, 'Rectangle', Tilt, fn{ifn} ) ;
            end
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'MATR'
            mf=fieldnames(en.(fn{ifn}));
            L=obj.collectPar(en.(fn{ifn}),'l');
            R=diag(ones(1,6)); T=[]; Tinds=[];
            for imf=1:length(mf)
              ind=regexpi(mf{imf},'^R(\d)(\d)','tokens','once');
              if ~isempty(ind)
                R(str2double(ind{1}),str2double(ind{2}))=en.(fn{ifn}).(mf{imf});
              end
              ind=regexpi(mf{imf},'^T(\d)(\d)(\d)','tokens','once');
              if ~isempty(ind)
                Tinds(end+1)=str2double(cell2mat(ind));
                T(end+1)=en.(fn{ifn}).(mf{imf});
              end
            end
            en.(fn{ifn}).LucretiaElement = TMapStruc( fn{ifn}, L ) ;
            en.(fn{ifn}).LucretiaElement.R = R ;
            if ~isempty(T)
              en.(fn{ifn}).LucretiaElement.T = T;
              en.(fn{ifn}).LucretiaElement.Tinds = Tinds;
            end
          case {'HMON','VMON','MONI'}
            [L, Type]=obj.collectPar(en.(fn{ifn}),'l','type') ;
            en.(fn{ifn}).LucretiaElement = BPMStruc( L, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'BLMO', 'PROF', 'WIRE', 'SLMO', 'IMON', 'INST'}
            [L, Type]=obj.collectPar(en.(fn{ifn}),'l','type') ;
            en.(fn{ifn}).LucretiaElement = InstStruc( L, upper(en.(fn{ifn}).Class(1:4)), fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'MARK'
            en.(fn{ifn}).LucretiaElement = MarkerStruc( fn{ifn} ) ;
          case 'BETA'
            % Initialization parameters set if using this BETA0
            if strcmpi(fn{ifn},betaname) || ismember(fn{ifn},uselist)
              [bx,ax,mx,by,ay,my,dx,dpx,dy,dpy,xoff,xpoff,yoff,ypoff,E0] = ...
                obj.collectPar(en.(fn{ifn}),'betx','alfx','mux','bety','alfy','muy','dx','dpx','dy','dpy','x','px','y','py','energy') ;
              if ~isnan(E0)
                Initial.Momentum=E0;
              end
              Initial.x.Twiss.beta=bx;
              Initial.x.Twiss.alpha=ax;
              Initial.x.Twiss.nu=mx;
              Initial.x.Twiss.eta=dx;
              Initial.x.Twiss.etap=dpx;
              Initial.x.pos=xoff;
              Initial.x.ang=xpoff;
              Initial.y.Twiss.beta=by;
              Initial.y.Twiss.alpha=ay;
              Initial.y.Twiss.nu=my;
              Initial.y.Twiss.eta=dy;
              Initial.y.Twiss.etap=dpy;
              Initial.y.pos=yoff;
              Initial.y.ang=ypoff;
            end
          case 'BEAM'
            if strcmpi(fn{ifn},beamname) || ismember(fn{ifn},uselist)
              [q,np,E0,gamma,ex,ey,exn,eyn,sigz,sige,nbunch] = ...
                obj.collectPar(en.(fn{ifn}),'charge','npart','energy','gamma','ex','ey','exn','eyn','sigt','sige','kbunch');
                if ~isnan(E0)
                  Initial.Momentum=E0;
                end
                if ~isnan(gamma)
                  Initial.Momentum=gamma*5.109989e-4;
                end
                gamma=Initial.Momentum/5.109989e-4;
                Initial.Q=q;
                if np*1.60217653e-19 > q
                  Initial.Q=np*1.60217653e-19;
                end
                if ~isnan(ex)
                  Initial.x.NEmit=ex*gamma;
                end
                if ~isnan(ey)
                  Initial.y.NEmit=ey*gamma;
                end
                if ~isnan(exn)
                  Initial.x.NEmit=exn;
                end
                if ~isnan(eyn)
                  Initial.y.NEmit=eyn;
                end
                if ~isnan(sigz)
                  Initial.sigz=sigz;
                end
                if ~isnan(sige)
                  Initial.SigPUncorrel=Initial.Momentum*sige;
                end
                if ~isnan(nbunch)
                  Initial.NBunch=nbunch;
                end
            end
          case 'LINE'
            if ~isempty(regexp(fn{ifn},'(','once'))
              error('Formal arguments in beamline definitions not currently supported')
            end
            % deal with any repeated element groups
            b1=regexp(en.(fn{ifn}).linearg,'\d+\*\(');
            while ~isempty(b1)
              b2=regexp(en.(fn{ifn}).linearg,'\)(,|\))');
              reptxt=[];
              nrep=str2double(regexp(en.(fn{ifn}).linearg(b1(1):end),'\d+','once','match'));
              for irep2=1:nrep
                reptxt=[reptxt cell2mat(regexp(en.(fn{ifn}).linearg(b1(1):b2(1)),'\((.+)\)','tokens','once'))];
                if irep2<nrep
                  reptxt(end+1)=',';
                end
              end
              en.(fn{ifn}).linearg=[en.(fn{ifn}).linearg(1:b1(1)-1) reptxt en.(fn{ifn}).linearg(b2(1)+1:end)];
              b1=regexp(en.(fn{ifn}).linearg,'\d+\*\(');
            end
            bl=split(regexprep(regexprep(en.(fn{ifn}).linearg,'^(',''),')$',''),',');
            lines.(fn{ifn})=bl;
          case 'SIGM'
            % just ignore this for now
          otherwise % look to see if user made class or treat as drift
            if strcmpi(en.(fn{ifn}).Class(1:4),'line')
              ifn=ifn+1;
              continue
            end
            if isfield(en,en.(fn{ifn}).Class)
              clfd=fieldnames(en.(fn{ifn}));
              tmp=en.(fn{ifn});
              en.(fn{ifn})=en.(en.(fn{ifn}).Class);
              for iclfd=1:length(clfd)
                if ~strcmp(clfd{iclfd},'Class')
                  en.(fn{ifn}).(clfd{iclfd}) = tmp.(clfd{iclfd}) ;
                end
              end
              continue
            end
            warning('Lucretia:DeckTool:ParseUnknownClass','Found an unknown Class names when parsing elements: %s, treating %s as drift or marker',en.(fn{ifn}).Class,fn{ifn});
            L=obj.collectPar(en.(fn{ifn}),'l');
            if L>0
              en.(fn{ifn}).LucretiaElement = DrifStruc( L, fn{ifn} ) ;
            else
              en.(fn{ifn}).LucretiaElement = MarkerStruc( fn{ifn} ) ;
            end
        end
        ifn=ifn+1;
      end
      % Parse beamline
      if isempty(lines)
        error('No beamlines found!')
      end
      lnames=fieldnames(lines);
      if ~ismember(linename,lnames) && ~any(ismember(lnames,uselist))
        error('No defined line to use')
      end
      % - expand all lines
      hasexpanded=false; firstcall=true;
      while hasexpanded || firstcall
        firstcall=false; hasexpanded=false;
        for iline=1:length(lnames)
          newline={}; dorev=[]; newlineind=[];
          for iele=1:length(lines.(lnames{iline}))
            elestr=char(lines.(lnames{iline})(iele));
            if elestr(1)=='-' && ismember(elestr(2:end),lnames)
              dorev(end+1)=true;
              newline{end+1}=elestr(2:end);
              newlineind(end+1)=iele;
            elseif ismember(elestr,lnames)
              dorev(end+1)=false;
              newline{end+1}=elestr;
              newlineind(end+1)=iele;
            end
          end
          if ~isempty(newline)
            hasexpanded=true;
            if ~ismember(newline{1},lnames)
              error('Line name %s not found',newline{1})
            end
            if newlineind(1)==1
              if dorev(1)
                exline=lines.(newline{1})(end:-1:1);
              else
                exline=lines.(newline{1});
              end
            else
              exline=lines.(lnames{iline})(1:newlineind(1)-1);
              if dorev(1)
                exline=[exline; lines.(newline{1})(end:-1:1)];
              else
                exline=[exline; lines.(newline{1})];
              end
            end
            if length(newline)>1
              for inewline=2:length(newline)
                if ~ismember(newline{inewline},lnames)
                  error('Line name %s not found',newline{inewline})
                end
                if newlineind(inewline)-1 >= newlineind(inewline-1)+1
                  exline=[exline; lines.(lnames{iline})(newlineind(inewline-1)+1:newlineind(inewline)-1)];
                end
                if dorev(inewline)
                  exline=[exline; lines.(newline{inewline})(end:-1:1)];
                else
                  exline=[exline; lines.(newline{inewline})];
                end
              end
            end
            if newlineind(end)<length(lines.(lnames{iline}))
              exline=[exline; lines.(lnames{iline})(newlineind(end)+1:end)];
            end
            lines.(lnames{iline})=exline;
          end
        end
      end
      % Form BEAMLINE from chosen line
      if ismember(linename,lnames)
        line=lines.(linename);
      else
        linesel=find(ismember(lnames,uselist));
        line=lines.(lnames{linesel(end)});
      end
      BEAMLINE={};
      for iline=1:length(line)
        % Check for repeated element command
        if ~isempty(regexp(char(line(iline)),'\d\*', 'once'))
          nrep=double(string(regexp(line(iline),'(\d+)\*','tokens','once')));
          line(iline)=regexprep(line(iline),'\d+\*','');
        else
          nrep=1;
        end
        if ~isfield(en,char(line(iline)))
          error('Line element not found: %s',line{iline})
        end
        if ~isfield(en.(char(line(iline))),'LucretiaElement')
          error('Error passing Element: %s',line(iline))
        end
        for irep=1:nrep
          BEAMLINE{end+1,1}=en.(char(line(iline))).LucretiaElement;
          BEAMLINE{end}.Name=upper(BEAMLINE{end}.Name);
        end
      end
      % Set correct B fields for chosen beam and lattice energy
      P=Initial.Momentum;
      Cb=1e9/2.99792458e8;       % rigidity conversion (T-m/GeV)
      for iele=1:length(BEAMLINE)
        if isfield(BEAMLINE{iele},'P')
          BEAMLINE{iele}.P=P;
        end
        if isfield(BEAMLINE{iele},'B') % Bfield stored as KL up to now, multiply by Brho to put in Lucretia units
          BEAMLINE{iele}.B=BEAMLINE{iele}.B.*(Cb*P);
        end
        if ismember(BEAMLINE{iele}.Class,{'LCAV','TCAV'})
          BEAMLINE{iele}.Egain=BEAMLINE{iele}.Volt * cosd(BEAMLINE{iele}.Phase) - BEAMLINE{iele}.L * Initial.Q * BEAMLINE{iele}.Kloss*1e-6 ;
          P=P+BEAMLINE{iele}.Egain*1e-3;
        end
      end
    end
    function txt=ElegantRead(obj,filenames)
      % Read Elegant files, expand lines and call statements
      % filename = {lte file, ele file} (1x2 char arrays)
      bdir=obj.basedir; % Assume all called files are relative to main lte file
      if ~iscell(filenames)
        filenames={filenames};
      end
      if ~exist(fullfile(bdir,filenames{1}),'file')
        error('%s not found',filenames{1})
      end
      ifile=1;
      fid=fopen(fullfile(bdir,filenames{1}),'r');
      % commands etc to be ignored
      ignorecom={};
      for icom=1:length(ignorecom); ignorecom{icom}=sprintf('%s,',ignorecom{icom}); end
      txt={}; t_next=[]; sname=''; subtxt=[]; callstack={};
      while 1
        % Process next statement from previous line if seperated with ;
        if ~isempty(t_next)
          t=t_next;
        else % Or, Get new line and stop if last line
          try
            t=fgetl(fid);
          catch
            t=[];
          end
        end
        if ~ischar(t) % hit end of this file, if more on the stack, open that, else break out of this loop
          if length(filenames)>1 && ifile==1
            fid=fopen(fullfile(bdir,filenames{2}),'r');
            continue
          end
          if isempty(callstack)
            break
          end
          try
            fclose(fid);
          catch
          end
           if ~exist(fullfile(bdir,callstack{1}),'file')
            error('%s not found',callstack{1})
          end
          fid=fopen(fullfile(bdir,callstack{1}),'r');
          callstack(1)=[];
          t=fgetl(fid);
        end
        % Remove comments and whitespace and quotes and string substitutions on element name, replace other " with '
        t=regexprep(regexprep(t,'!.*',''),'\s+','');
        if ~isempty(t) && t(1) == '"'
          t = regexprep(regexprep(t,'"','',1),'"','',1) ;
        end
        t=regexprep(t,'"','''');
        t=regexprep(t,'%s','ss');
        if isempty(t)
          continue
        end
        % Concatinate line extensions
        while ~isempty(t) && t(end)=='&'
          tcont=fgetl(fid);
          if ~ischar(tcont)
            error('Deck file cannot end with continution (&) character!')
          end
          tcont=regexprep(regexprep(tcont,'!.*',''),'\s+','');
          if ~isempty(tcont)
            t=[t(1:end-1) tcont];
          end
        end
        % Parser cannot deal with special characters ; or , in quotes,
        % replace by string equivalents to be re-substituted later
        lquo=regexp(t,'"');
        if length(lquo)>=2
          for iquo=1:floor(length(lquo)/2)
            if contains(t(lquo(1+(iquo-1)*2):lquo(2+(iquo-1)*2)),',')
              newtxt=regexprep(t(lquo(1+(iquo-1)*2)+1:lquo(2+(iquo-1)*2)-1),',','__COMMA__');
              t=[t(1:lquo(1+(iquo-1)*2)) newtxt t(lquo(2+(iquo-1)*2):end)];
            end
            if contains(t(lquo(1+(iquo-1)*2):lquo(2+(iquo-1)*2)),';')
              newtxt=regexprep(t(lquo(1+(iquo-1)*2)+1:lquo(2+(iquo-1)*2)-1),';','__SEMICOLON__');
              t=[t(1:lquo(1+(iquo-1)*2)) newtxt t(lquo(2+(iquo-1)*2):end)];
            end
          end
        end
        % Deal with multiple commands on one line
        if contains(t,';')
          tl=regexp(t,';','once');
          t_next=t(tl+1:end);
          t=t(1:tl-1);
        else
          t_next=[];
        end
        % If there are MAD8 style MATRIX element definitions, re-format
        % them as XSIF style ones
        if contains(t,':MATR','IgnoreCase',true)
          t=regexprep(t,'rm\((\d),(\d)\)','R$1$2','ignorecase');
          t=regexprep(t,'tm\((\d),(\d),(\d)\)','T$1$2$3','ignorecase');
        end
        % If reach STOP or RETURN statement, ignore everything following
        if startsWith(t,{'STOP' 'RETURN'},'IgnoreCase',true)
          fclose(fid);
          continue
        end
        % Ignore any comment blocks
        if startsWith(t,'COMMENT','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDCOMMENT','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Replace constant declarations with assignments
        t=regexprep(t,':CONSTANT=',':='); t=regexprep(t,':constant=',':=');
        % Replace LIST's with LINE's
        t=regexprep(t,':LIST=',':LINE='); t=regexprep(t,':list=',':LINE=');
        % Replace set commands with := assignment operators
        if startsWith(t,'set,','IgnoreCase',true)
          t=regexprep(t,'^SET,','','ignorecase');
          t=regexprep(t,'=',':=');
          t=regexprep(t,',',':=');
        end
        % Flag SUBROUTINE blocks
        if startsWith(t,'ENDSUBROUTINE','IgnoreCase',true)
          sname='';
          continue
        elseif endsWith(t,'SUBROUTINE','IgnoreCase',true)
          sname=regexprep(t,':SUBROUTINE','');
          subtxt.(sname)={};
          continue
        end
        % Ignore MATCH blocks
        if endsWith(t,'MATCH','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDMATCH','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Ignore HARMON blocks
        if endsWith(t,'HARMON','IgnoreCase',true)
          while 1
            if ~ischar(t) || startsWith(t,'ENDHARM','IgnoreCase',true)
              break
            else
              t=fgetl(fid);
              t=regexprep(regexprep(t,'!.*',''),'\s+','');
            end
          end
          continue
        end
        % Remove lines featured in ignore list
        if contains(t,ignorecom,'IgnoreCase',true)
          while 1
            if ~ischar(t) || ~endsWith(t,'&')
              break
            else
              t=fgetl(fid);
            end
          end
          continue
        end
        % Replace any $ chars with _
        t=regexprep(t,'\$','_');
        % Expand called external files or just add line to txt stack
        if startsWith(t,'call','IgnoreCase',true)
          % try with filename parameter
          try
            cfn=regexprep(cell2mat(regexpi(t,'"(\S+\.?\S*)"','tokens','once')),'"','');
          catch
            cfn=[];
          end
          fprintf('Calling file: %s\n',cfn);
          if isempty(cfn) || ~exist(fullfile(bdir,cfn),'file')
            fclose(fid);
            error('Unfound filename in CALL statement: %s',fullfile(bdir,cfn))
          end
          % Add file to stack to read after done with this one
          callstack{end+1}=cfn;
        elseif isempty(sname) && isfield(subtxt,t) % expand out a subroutine call
          for isub=1:length(subtxt.(t))
            txt{end+1}=subtxt.(t){isub};
          end
        elseif ~isempty(sname) % store subroutine commands
          subtxt.(sname){end+1}=t;
        else
          txt{end+1}=t;
        end
      end
      try
        fclose(fid);
      catch
      end
    end
    function Initial=ElegantParse(obj,txt,linename,betaname,beamname)
      global BEAMLINE WF
      % math functions and Matlab equivalents
      madmath={'sqrt(' 'log(' 'exp(' 'sin(' 'cos(' 'tan(' 'asin(' 'abs(' 'max(' 'min(' 'ranf(' 'gauss('} ;
      matmath={'sqrt(' 'log(' 'exp(' 'sin(' 'cos(' 'tan(' 'asin(' 'abs(' 'max(' 'min(' 'rand(' 'randn('} ;
      Initial=InitCondStruc();
      if isempty(txt)
        return
      end
      if exist('linename','var')
        linename=lower(linename);
      end
      if exist('betaname','var')
        betaname=lower(betaname);
      end
      if exist('beamname','var')
        beamname=lower(beamname);
      end
      uselist={}; % container for indicated beamlines, beta0, beam's to use
      % Default named parameters
      np.pi=pi;
      np.twopi=pi*2;
      np.emass=5.109989e-4;
      np.clight=299792458;
      np.e=2.7182818284590;
      np.degrad=57.295779513082323;
      np.raddeg=0.017453292519943;
      np.electron=-1;
      np.positron=1;
      % Pull out user-defined named parameters, also pull out line name if
      % there is one, also form element, parameter lists
      inp=[]; ndef=[]; firstcall=true; en=[];
      while ~isempty(ndef) || firstcall
        if firstcall
          prog=true;
        else
          prog=false;
        end
        firstcall=false;
        for itxt=1:length(txt)
          txt{itxt}=lower(txt{itxt});
          tok=regexp(txt{itxt},'^(\w+):?=(.*)','tokens','once');
          if length(tok)==2 && ~ismember(itxt,inp)
            expval=[];
            try
              % replace variable names
              if isempty(strfind(tok{2},'np.'))
                tok{2}=regexprep(tok{2},'([a-z|A-Z]\w*)','np.$1');
                tok{2}=regexprep(tok{2},'([\dn|\.])np\.(e[-?+?\d])','$1$2'); % ignore exponent case
                % Deal with math function case
                imatch=1;
                while contains(tok{2},madmath) && imatch<=length(madmath)
                  tok{2}=regexprep(tok{2},sprintf('np.%s',madmath{imatch}),matmath(imatch));
                  imatch=imatch+1;
                end
              end
              % deal with possible name[par] format
              if contains(tok{2},'[')
                tok{2}=regexprep(tok{2},'\[np\.(\w+)','[$1');
                tokex = regexp(tok{2},'np\.(\w+)\[(\w+)\]','tokens','once');
                if ~isfield(en.(tokex{1}),tokex{2}) % looking for keyword concatination
                  fn=fieldnames(en.(tokex{1}));
                  for ifn=1:length(fn)
                    if startsWith(string(tokex{2}),fn{ifn}) || startsWith(fn{ifn},tokex{2})
                      tok{2}=sprintf('en.%s.%s',tokex{1},fn{ifn});
                      break;
                    end
                  end
                else
                  tok{2}=regexprep(tok{2},'np\.(\w+)\[(\w+)\]','en.$1.$2');
                end
              end
              expval=eval(tok{2});
              ndef(ismember(ndef,itxt))=[];
            catch % if still cannot eval, flag and move on- resolve on next try, or fail
              if ismember(itxt,ndef) && ~prog
                error('Failed to resolve expression in line: %s',sprintf('%s := %s',tok{1},tok{2}))
              elseif ~ismember(itxt,ndef)
                ndef(end+1)=itxt;
                if isfield(np,tok{1})
                  np=rmfield(np,tok{1});
                end
              end
            end
            if ~isempty(expval)
              if isfield(np,tok{1})
                warning('Lucretia:DeckTool','parameter %s defined multiple times',tok{1});
              end
              np.(tok{1})=expval;
              prog=true;
              inp(end+1)=itxt;
            end
          elseif ~ismember(itxt,inp)
            % Store any use commands
            ln=cell2mat(regexpi(txt{itxt},'use,(\w+)','tokens','once'));
            if ~isempty(ln)
              uselist{end+1}=ln;
              inp(end+1)=itxt;
              continue;
            end
            % form element structure
            tok=regexp(txt{itxt},'^(\w+):(\w+),?(.*)','tokens','once');
            if length(tok)>=2 && ~ismember(itxt,inp) && ~strcmp(tok{2},'line')
              if isfield(en,tok{1}) && ~ismember(itxt,ndef)
                warning('Lucretia:DeckTool:MultiEleDeclare','Element %s defined multiple times',tok{1})
              end
              en.(tok{1}).Class=tok{2};
              if length(tok)>=3 && ~isempty(tok{3})
                par=split(tok{3},',');
                ndef(ismember(ndef,itxt))=[];
                for ipar=1:length(par)
                  pval=regexp(char(par(ipar)),'(\w+)=?(.*)','tokens','once');
                  % Re-substitute , and ; characters that are inside quotes
                  pval{2}=regexprep(pval{2},'__COMMA__',',','ignorecase');
                  pval{2}=regexprep(pval{2},'__SEMICOLON__',';','ignorecase');
                  if length(pval)==2 && ~isempty(pval{2})
                    % Replace quotes to '' so matlab eval parses them
                    % correctly
                    pval{2}=regexprep(pval{2},'"','''');
%                     pval{2}=regexprep(regexprep(pval{2},'''',''''''),'"','''''');
                    % Try evaluating parameter expression
                    try
                      % replace variable names
                      if isempty(strfind(pval{2},'np.')) && pval{2}(1)~=''''
                        pval{2}=regexprep(regexprep(pval{2},'([a-z|A-Z]\w*)','np.$1'),'np\.([a-z|A-Z]\w*\()','$1');
                        pval{2}=regexprep(pval{2},'([\dn|\.])np\.(e[-?+?\d])','$1$2'); % ignore exponent case
                      end
                      % Deal with math function case
                      imatch=1;
                      while contains(pval{2},madmath) && imatch<=length(madmath)
                        pval{2}=regexprep(pval{2},sprintf('np.%s',madmath{imatch}),matmath(imatch));
                        imatch=imatch+1;
                      end
                      % deal with possible name[par] format
                      if contains(pval{2},'[')
                        pval{2}=regexprep(pval{2},'\[np\.(\w+)','[$1');
                        pval{2}=regexprep(pval{2},'np\.(\w+)\[(\w+)\]','en.$1.$2');
                      end
                      % Evaluate parameter
                      en.(tok{1}).(pval{1})=eval(pval{2});
                    catch % if cannot eval, flag and move on- resolve on next try, or fail
                      if ~prog
                        error('Failed to resolve parameter in element: %s',tok{1})
                      elseif ~ismember(itxt,ndef)
                        ndef(end+1)=itxt;
                        continue
                      end
                    end
                  elseif ~isempty(pval)
                    en.(tok{1}).(pval{1})=[];
                  else
                    en.(tok{1})=[];
                  end
                end
              elseif length(tok)<3
                en.(tok{1})=[];
              end
              if ~ismember(itxt,ndef)
                inp(end+1)=itxt;
              end
            elseif length(tok)>1 && strcmp(tok{2},'line') && ~ismember(itxt,inp) % deal with line definitions seperately
              en.(tok{1}).Class='line';
              en.(tok{1}).linearg=regexprep(txt{itxt},'^.+=(.*)','$1');
            end
          end
        end
      end
%       txt(inp)=[];
      % Parse element list
      fn=fieldnames(en); lines=[]; ifn=1;
      plist={'MARK' 'DRIF' 'CSRD' 'CSRC' 'CSBE' 'QUAD' 'SEXT' 'OCTU' 'MULT' 'CSBE' 'SOLE' 'LCAV' 'DRIF' 'HKIC' ...
            'VKIC' 'KICK' 'MONI' 'HMON' 'VMON' 'INST' 'PROF' 'WIRE' 'BLMO' 'SLMO' 'IMON' 'WIGG' 'CWIG' 'WATC' 'IBSC' 'LSRM' 'CHAR' ...
            'COLL' 'MATR' 'BETA' 'BEAM' 'LINE' 'SIGM' 'RCOL' 'ECOL' 'SROT' 'ROLL' 'ZROT' 'YROT' 'GKIC' 'TWIS'};
      while ifn<=length(fn) %for ifn=1:length(fn)
        % if no class field at this stage, then it is a marker
        if ~isfield(en.(fn{ifn}),'Class')
          en.(fn{ifn}).Class='marker';
        end
        % Inherting from another element?
        if length(en.(fn{ifn}).Class)<4 || ~ismember(upper(en.(fn{ifn}).Class(1:4)),plist)
          iobj = en.(en.(fn{ifn}).Class) ;
          en.(fn{ifn}).Class = iobj.Class ;
          fn2=fieldnames(iobj);
          for ifn2=1:length(fn2)
            if ~isfield(en.(fn{ifn}),fn2{ifn2})
              en.(fn{ifn}).(fn2{ifn2}) = iobj.(fn2{ifn2}) ;
            end
          end
        end
        switch upper(en.(fn{ifn}).Class(1:4))
          case {'DRIF', 'CSRD', 'WIGG', 'CWIG', 'LSRM'}
            L=obj.collectPar(en.(fn{ifn}),'l');
            en.(fn{ifn}).LucretiaElement = DrifStruc( L, fn{ifn} ) ;
          case {'CSBE', 'CSRC'}
            [L,ANG,K1,K2,E1,E2,Tilt,H1,H2,Hgap,Fint,Hgap2,Fint2,Type]=obj.collectPar(en.(fn{ifn}),'l','angle','k1','k2','e1','e2','tilt','h1','h2','hgap','fint','hgapx','fintx','type');
            E=[E1,E2];
            H=[H1,H2];
            if L==0
              L=1e-9;
            end
            if isnan(Hgap2)
              Hgap=[Hgap,Hgap];
            else
              Hgap=[Hgap,Hgap2];
            end
            if isnan(Fint2)
              Fint=[Fint,Fint];
            else
              Fint=[Fint,Fint2];
            end
            if isnan(K1)
              BField=ANG/L;
            else
              BField=[ANG/L,K1];
            end
            if isempty(Tilt)
              Tilt=pi/2;
            end
            en.(fn{ifn}).LucretiaElement = SBendStruc( L, BField.*L, ANG, E, H, Hgap, Fint, Tilt, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            % Deal with design sextupole field content - later gets turned
            % into MULT elements either side of bend
            if ~isnan(K2) && abs(K2)>0
              if length(BField)==2
                BField=[BField,-K2*sign(ANG)];
              else
                BField=[BField,0,-K2*sign(ANG)];
              end
              en.(fn{ifn}).LucretiaElement.B = BField.*L ;
            end
          case 'QUAD'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k1','tilt','aperture','type') ;
            if L==0
              L=1e-9;
            end
            if isempty(Tilt)
              Tilt=pi/4;
            end
            if isnan(B); B=0; end
            en.(fn{ifn}).LucretiaElement = QuadStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'SEXT'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k2','tilt','aperture','type') ;
            if L==0
              L=1e-9;
            end
            if isempty(Tilt)
              Tilt=pi/6;
            end
            en.(fn{ifn}).LucretiaElement = SextStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'OCTU'
            [L, B, Tilt, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','k3','tilt','aperture','type') ;
            if isempty(Tilt)
              Tilt=pi/8;
            end
            en.(fn{ifn}).LucretiaElement = OctuStruc( L, B*L, Tilt, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'MULT'
            [L, LRAD, TiltAll, aper, Type] = obj.collectPar(en.(fn{ifn}),'l','lrad','tilt','aperture','type') ;
            B=[]; Tilt=[]; PIndx=[];
            mpn=fieldnames(en.(fn{ifn}));
            kv=regexp(mpn,'^k(\d+)l?','ignorecase','tokens','once');
            for ik=1:length(kv)
              if ~isempty(kv{ik})
                B(end+1)=en.(fn{ifn}).(mpn{ik});
                if isfield(en.(fn{ifn}),sprintf('t%c',kv{ik}{1}))
                  if isempty(en.(fn{ifn}).(sprintf('t%c',kv{ik}{1})))
                    Tilt(end+1)=pi/((str2double(kv{ik}{1})+1)*2)+TiltAll;
                  else
                    Tilt(end+1)=en.(fn{ifn}).(sprintf('t%c',kv{ik}{1}))+TiltAll;
                  end
                else
                  Tilt(end+1)=TiltAll;
                end
                PIndx(end+1)=str2double(kv{ik}{1});
              end
            end
            en.(fn{ifn}).LucretiaElement = MultStruc( L, B, Tilt, PIndx, [0 0], aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            if ~isnan(LRAD)
              en.(fn{ifn}).LucretiaElement.Lrad=LRAD;
            end
          case 'SOLE'
            [L, B, aper,tilt, Type] = obj.collectPar(en.(fn{ifn}),'l','ks','aperture','tilt','type') ;
            if L==0
              L=1e-9;
            end
            en.(fn{ifn}).LucretiaElement = SolenoidStruc( L, B, aper, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Offset(6) = tilt ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'LCAV','TCAV'}
            [L, V,phi,freq,eloss,lfile,tfile,aper,nbin,tilt,Type] = obj.collectPar(en.(fn{ifn}),'l','deltae','phi0','freq','eloss','lfile','tfile','aperture','nbin','tilt','type');
            BinWidth=1/nbin; % BinWidth is fraction of sigma - default here to 0.1
            srwf_z=0; srwf_t=0;
            if ~any(isnan(lfile)) && ~isempty(lfile) && ~ismember(lfile,obj.parsedWFfiles) && exist(fullfile(obj.basedir,lfile),'file')
              [stat,W] = ParseSRWF( fullfile(obj.basedir,lfile), BinWidth ) ;
              if stat{1}~=1; error(stat{2}); end
              obj.parsedWFfiles{end+1}=lfile;
              if isempty(WF) || ~isfield(WF,'ZSR')
                WF.ZSR(1)=W;
              else
                WF.ZSR(end+1)=W;
              end
              srwf_z=length(WF.ZSR);
            elseif ~any(isnan(lfile)) && ~isempty(lfile) && ~ismember(lfile,obj.parsedWFfiles) && ~exist(fullfile(obj.basedir,lfile),'file')
              warning('Lucretia:DeckTool:nolfile','referenced lfile not found, skipping: %s',fullfile(obj.basedir,lfile))
            end
            if ~any(isnan(tfile)) && ~isempty(tfile) && ~ismember(lfile,obj.parsedWFfiles) && exist(fullfile(obj.basedir,tfile),'file')
              [stat,W] = ParseSRWF( fullfile(obj.basedir,tfile), BinWidth ) ;
              obj.parsedWFfiles{end+1}=tfile;
              if stat{1}~=1; error(stat{2}); end
              if isempty(WF) || ~isfield(WF,'TSR')
                WF.TSR(1)=W;
              else
                WF.TSR(end+1)=W;
              end
              srwf_t=length(WF.TSR);
            elseif ~any(isnan(tfile)) && ~isempty(tfile) && ~ismember(lfile,obj.parsedWFfiles) && ~exist(fullfile(obj.basedir,tfile),'file')
              warning('Lucretia:DeckTool:notfile','referenced tfile not found, skipping: %s',fullfile(obj.basedir,tfile))
            end
            if strcmpi(en.(fn{ifn}).Class(1:4),'TCAV')
              mode=1;
            else
              mode=0;
            end
            en.(fn{ifn}).LucretiaElement = RFStruc(  L, V, phi, freq, srwf_z, srwf_t, eloss, aper, fn{ifn}, mode ) ;
            en.(fn{ifn}).LucretiaElement.Offset(6) = tilt ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
            if ~any(isnan(lfile)) && ~isempty(lfile)
              en.(fn{ifn}).LucretiaElement.LWFfile=lfile;
            end
            if ~any(isnan(tfile)) && ~isempty(tfile)
              en.(fn{ifn}).LucretiaElement.TWFfile=tfile;
            end
          case {'SROT','ROLL','ZROT','YROT','GKIC'}
            dpsi=0;
            [ang,dx,dtheta,dy,dphi,dz]=obj.collectPar(en.(fn{ifn}),'angle','dx','dxp','dy','dyp','dz');
            if ismember(upper(en.(fn{ifn}).Class(1:4)),{'SROT','ROLL','GKIC'})
              dpsi=ang;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'YROT')
              dtheta=ang;
            else
              dphi=ang;
            end
            en.(fn{ifn}).LucretiaElement = CoordStruc( dx, dtheta, dy, dphi, dz, dpsi, fn{ifn} );
          case {'HKIC','VKIC','KICK'}
            [L,KICK,HKICK,VKICK,TILT, Type]=obj.collectPar(en.(fn{ifn}),'l','kick','hkick','vkick','tilt','type');
            if strcmpi(en.(fn{ifn}).Class(1:4),'HKIC')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, KICK, TILT, 1, fn{ifn} ) ;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'VKIC')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, KICK, TILT, 2, fn{ifn} ) ;
            elseif strcmpi(en.(fn{ifn}).Class(1:4),'KICK')
              en.(fn{ifn}).LucretiaElement = CorrectorStruc(  L, [HKICK,VKICK], TILT, 3, fn{ifn} ) ;
            end
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'ECOL','RCOL'}
            [L,xgap,ygap,Tilt,Type]=obj.collectPar(en.(fn{ifn}),'l','xsize','ysize','tilt','type') ;
            if strcmpi(en.(fn{ifn}).Class(1:4),'ECOL')
              en.(fn{ifn}).LucretiaElement = CollStruc( L, xgap, ygap, 'Ellipse', Tilt, fn{ifn} ) ;
            else
              en.(fn{ifn}).LucretiaElement = CollStruc( L, xgap, ygap, 'Rectangle', Tilt, fn{ifn} ) ;
            end
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case 'MATR'
            mf=fieldnames(en.(fn{ifn}));
            L=obj.collectPar(en.(fn{ifn}),'l');
            R=diag(ones(1,6)); T=[]; Tinds=[];
            for imf=1:length(mf)
              ind=regexpi(mf{imf},'^R(\d)(\d)','tokens','once');
              if ~isempty(ind)
                R(str2double(ind{1}),str2double(ind{2}))=en.(fn{ifn}).(mf{imf});
              end
              ind=regexpi(mf{imf},'^T(\d)(\d)(\d)','tokens','once');
              if ~isempty(ind)
                Tinds(end+1)=str2double(cell2mat(ind));
                T(end+1)=en.(fn{ifn}).(mf{imf});
              end
            end
            en.(fn{ifn}).LucretiaElement = TMapStruc( fn{ifn}, L ) ;
            en.(fn{ifn}).LucretiaElement.R = R ;
            if ~isempty(T)
              en.(fn{ifn}).LucretiaElement.T = T;
              en.(fn{ifn}).LucretiaElement.Tinds = Tinds;
            end
          case {'HMON','VMON','MONI'}
            [L, Type]=obj.collectPar(en.(fn{ifn}),'l','type') ;
            en.(fn{ifn}).LucretiaElement = BPMStruc( L, fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'BLMO', 'PROF', 'WIRE', 'SLMO', 'IMON', 'INST'}
            [L, Type]=obj.collectPar(en.(fn{ifn}),'l','type') ;
            en.(fn{ifn}).LucretiaElement = InstStruc( L, upper(en.(fn{ifn}).Class(1:4)), fn{ifn} ) ;
            en.(fn{ifn}).LucretiaElement.Type=Type;
          case {'MARK','WATC','IBSC','CHAR'}
            en.(fn{ifn}).LucretiaElement = MarkerStruc( fn{ifn} ) ;
          case 'TWIS'
            % Initialization parameters set if using this BETA0
            en.(fn{ifn}).LucretiaElement = MarkerStruc( fn{ifn} ) ;
            if strcmpi(fn{ifn},betaname) || ismember(fn{ifn},uselist)
              [bx,ax,mx,by,ay,my,dx,dpx,dy,dpy,xoff,xpoff,yoff,ypoff,E0] = ...
                obj.collectPar(en.(fn{ifn}),'betax','alphax','mux','betay','alphay','muy','dx','dpx','dy','dpy','x','px','y','py','energy') ;
              if ~isnan(E0)
                Initial.Momentum=E0;
              end
              Initial.x.Twiss.beta=bx;
              Initial.x.Twiss.alpha=ax;
              Initial.x.Twiss.nu=mx;
              Initial.x.Twiss.eta=dx;
              Initial.x.Twiss.etap=dpx;
              Initial.x.pos=xoff;
              Initial.x.ang=xpoff;
              Initial.y.Twiss.beta=by;
              Initial.y.Twiss.alpha=ay;
              Initial.y.Twiss.nu=my;
              Initial.y.Twiss.eta=dy;
              Initial.y.Twiss.etap=dpy;
              Initial.y.pos=yoff;
              Initial.y.ang=ypoff;
            end
          case 'BEAM'
            if strcmpi(fn{ifn},beamname) || ismember(fn{ifn},uselist)
              [q,np,E0,gamma,ex,ey,exn,eyn,sigz,sige,nbunch] = ...
                obj.collectPar(en.(fn{ifn}),'charge','npart','energy','gamma','ex','ey','exn','eyn','sigt','sige','kbunch');
                if ~isnan(E0)
                  Initial.Momentum=E0;
                end
                if ~isnan(gamma)
                  Initial.Momentum=gamma*5.109989e-4;
                end
                gamma=Initial.Momentum/5.109989e-4;
                Initial.Q=q;
                if np*1.60217653e-19 > q
                  Initial.Q=np*1.60217653e-19;
                end
                if ~isnan(ex)
                  Initial.x.NEmit=ex*gamma;
                end
                if ~isnan(ey)
                  Initial.y.NEmit=ey*gamma;
                end
                if ~isnan(exn)
                  Initial.x.NEmit=exn;
                end
                if ~isnan(eyn)
                  Initial.y.NEmit=eyn;
                end
                if ~isnan(sigz)
                  Initial.sigz=sigz;
                end
                if ~isnan(sige)
                  Initial.SigPUncorrel=Initial.Momentum*sige;
                end
                if ~isnan(nbunch)
                  Initial.NBunch=nbunch;
                end
            end
          case 'LINE'
            if ~isempty(regexp(fn{ifn},'(','once'))
              error('Formal arguments in beamline definitions not currently supported')
            end
            % deal with any repeated element groups
            b1=regexp(en.(fn{ifn}).linearg,'\d+\*\(');
            while ~isempty(b1)
              b2=regexp(en.(fn{ifn}).linearg,'\)(,|\))');
              reptxt=[];
              nrep=str2double(regexp(en.(fn{ifn}).linearg(b1(1):end),'\d+','once','match'));
              for irep2=1:nrep
                reptxt=[reptxt cell2mat(regexp(en.(fn{ifn}).linearg(b1(1):b2(1)),'\((.+)\)','tokens','once'))];
                if irep2<nrep
                  reptxt(end+1)=',';
                end
              end
              en.(fn{ifn}).linearg=[en.(fn{ifn}).linearg(1:b1(1)-1) reptxt en.(fn{ifn}).linearg(b2(1)+1:end)];
              b1=regexp(en.(fn{ifn}).linearg,'\d+\*\(');
            end
            bl=split(regexprep(regexprep(en.(fn{ifn}).linearg,'^(',''),')$',''),',');
            lines.(fn{ifn})=bl;
          case 'SIGM'
            % just ignore this for now
          otherwise % look to see if user made class or treat as drift
            if strcmpi(en.(fn{ifn}).Class(1:4),'line')
              ifn=ifn+1;
              continue
            end
            if isfield(en,en.(fn{ifn}).Class)
              clfd=fieldnames(en.(fn{ifn}));
              tmp=en.(fn{ifn});
              en.(fn{ifn})=en.(en.(fn{ifn}).Class);
              for iclfd=1:length(clfd)
                if ~strcmp(clfd{iclfd},'Class')
                  en.(fn{ifn}).(clfd{iclfd}) = tmp.(clfd{iclfd}) ;
                end
              end
              continue
            end
            warning('Lucretia:DeckTool:ParseUnknownClass','Found an unknown Class names when parsing elements: %s, treating %s as drift or marker',en.(fn{ifn}).Class,fn{ifn});
            L=obj.collectPar(en.(fn{ifn}),'l');
            if L>0
              en.(fn{ifn}).LucretiaElement = DrifStruc( L, fn{ifn} ) ;
            else
              en.(fn{ifn}).LucretiaElement = MarkerStruc( fn{ifn} ) ;
            end
        end
        ifn=ifn+1;
      end
      % Parse beamline
      if isempty(lines)
        error('No beamlines found!')
      end
      lnames=fieldnames(lines);
      if ~ismember(linename,lnames) && ~any(ismember(lnames,uselist))
        error('No defined line to use')
      end
      % - expand all lines
      hasexpanded=false; firstcall=true;
      while hasexpanded || firstcall
        firstcall=false; hasexpanded=false;
        for iline=1:length(lnames)
          newline={}; dorev=[]; newlineind=[];
          for iele=1:length(lines.(lnames{iline}))
            elestr=char(lines.(lnames{iline})(iele));
            if elestr(1)=='-' && ismember(elestr(2:end),lnames)
              dorev(end+1)=true;
              newline{end+1}=elestr(2:end);
              newlineind(end+1)=iele;
            elseif ismember(elestr,lnames)
              dorev(end+1)=false;
              newline{end+1}=elestr;
              newlineind(end+1)=iele;
            end
          end
          if ~isempty(newline)
            hasexpanded=true;
            if ~ismember(newline{1},lnames)
              error('Line name %s not found',newline{1})
            end
            if newlineind(1)==1
              if dorev(1)
                exline=lines.(newline{1})(end:-1:1);
              else
                exline=lines.(newline{1});
              end
            else
              exline=lines.(lnames{iline})(1:newlineind(1)-1);
              if dorev(1)
                exline=[exline; lines.(newline{1})(end:-1:1)];
              else
                exline=[exline; lines.(newline{1})];
              end
            end
            if length(newline)>1
              for inewline=2:length(newline)
                if ~ismember(newline{inewline},lnames)
                  error('Line name %s not found',newline{inewline})
                end
                if newlineind(inewline)-1 >= newlineind(inewline-1)+1
                  exline=[exline; lines.(lnames{iline})(newlineind(inewline-1)+1:newlineind(inewline)-1)];
                end
                if dorev(inewline)
                  exline=[exline; lines.(newline{inewline})(end:-1:1)];
                else
                  exline=[exline; lines.(newline{inewline})];
                end
              end
            end
            if newlineind(end)<length(lines.(lnames{iline}))
              exline=[exline; lines.(lnames{iline})(newlineind(end)+1:end)];
            end
            lines.(lnames{iline})=exline;
          end
        end
      end
      % Form BEAMLINE from chosen line
      if ismember(linename,lnames)
        line=lines.(linename);
      else
        linesel=find(ismember(lnames,uselist));
        line=lines.(lnames{linesel(end)});
      end
      BEAMLINE={};
      for iline=1:length(line)
        % Check for repeated element command
        if ~isempty(regexp(char(line(iline)),'\d\*', 'once'))
          nrep=double(string(regexp(line(iline),'(\d+)\*','tokens','once')));
          line(iline)=regexprep(line(iline),'\d+\*','');
        else
          nrep=1;
        end
        if ~isfield(en,char(line(iline)))
          error('Line element not found: %s',line{iline})
        end
        if ~isfield(en.(char(line(iline))),'LucretiaElement') && string(en.(char(line(iline))).Class)~="line"
          error('Error passing Element: %s',line(iline))
        end
        for irep=1:nrep
          if string(en.(char(line(iline))).Class)=="line"
            for il=1:length(lines.(char(line(iline))))
              BEAMLINE{end+1,1}=en.(lines.(char(line(iline))){il}).LucretiaElement;
              BEAMLINE{end}.Name=upper(BEAMLINE{end}.Name);
            end
          else
            BEAMLINE{end+1,1}=en.(char(line(iline))).LucretiaElement;
            BEAMLINE{end}.Name=upper(BEAMLINE{end}.Name);
          end
        end
      end
      % Set correct B fields for chosen beam and lattice energy
      P=Initial.Momentum;
      Cb=1e9/2.99792458e8;       % rigidity conversion (T-m/GeV)
      for iele=1:length(BEAMLINE)
        if isfield(BEAMLINE{iele},'P')
          BEAMLINE{iele}.P=P;
        end
        if isfield(BEAMLINE{iele},'B') % Bfield stored as KL up to now, multiply by Brho to put in Lucretia units
          BEAMLINE{iele}.B=BEAMLINE{iele}.B.*(Cb*P);
        end
        if ismember(BEAMLINE{iele}.Class,{'LCAV','TCAV'})
          BEAMLINE{iele}.Egain=BEAMLINE{iele}.Volt * cosd(BEAMLINE{iele}.Phase) - BEAMLINE{iele}.L * Initial.Q * BEAMLINE{iele}.Kloss*1e-6 ;
          P=P+BEAMLINE{iele}.Egain*1e-3;
        end
      end
    end
    function SetProps(obj)
      global BEAMLINE
      classList=properties(obj);
      grp.B={'QUAD' 'SEXT' 'OCTU' 'SOLENOID' 'SBEN' 'XCOR' 'YCOR' 'XYCOR' 'MULT'};
      grp.L=classList(~ismember(classList,{'MARK' 'COORD'}));
      grp.Lrad={'MULT'};
      grp.Angle={'SBEN'};
      grp.Tilt={'QUAD' 'SEXT' 'OCTU' 'MULT' 'SBEN' 'XCOR' 'YCOR' 'XYCOR'};
      grp.T={'TMAP'};
      grp.U={'TMAP'};
      grp.V={'TMAP'};
      grp.W={'TMAP'};
      grp.R={'TMAP'};
      grp.Change={'COORD'};
      grp.Freq={'LCAV' 'TCAV'};
      grp.Kloss={'LCAV' 'TCAV'};
      grp.Phase={'LCAV' 'TCAV'};
      grp.Volt={'LCAV' 'TCAV'};
      grp.FINT={'SBEN'};
      grp.HGAP={'SBEN'};
      grp.EdgeCurvature={'SBEN'};
      grp.EdgeAngle={'SBEN'};
      grp.P={'LCAV' 'TCAV'};
      grp.aper={'QUAD' 'SEXT' 'OCTU' 'MULT' 'SOLENOID' 'LCAV' 'COLL'  'TCAV'};
      grp.ISR={'SBEN'};
      grp.LSC={'LCAV' 'DRIF'};
      grp.CSR={'SBEN' 'DRIF'};
      grpList=fieldnames(grp);
      for iclass=1:length(classList)
        obj.(classList{iclass})=[];
        eleList=findcells(BEAMLINE,'Class',classList{iclass});
        if ~isempty(eleList)
          obj.(classList{iclass}).index=zeros(1,length(eleList));
          for iele=1:length(eleList)
            obj.(classList{iclass}).index(iele)=eleList(iele);
            for igrp=1:length(grpList)
              if ismember(classList{iclass},grp.(grpList{igrp}))
                try
                  if ismember(grpList{igrp},{'CSR','LSC','ISR'})
                    if isfield(BEAMLINE{eleList(iele)},'TrackFlag') && isfield(BEAMLINE{eleList(iele)}.TrackFlag,grpList{igrp})
                      obj.(classList{iclass}).(grpList{igrp}){iele}=BEAMLINE{eleList(iele)}.TrackFlag.(grpList{igrp});
                    else
                      obj.(classList{iclass}).(grpList{igrp}){iele}=0;
                    end
                  else
                    obj.(classList{iclass}).(grpList{igrp}){iele}=BEAMLINE{eleList(iele)}.(grpList{igrp});
                  end
                catch
                end
              end
            end
          end
        end
      end
    end
  end
  methods
    function WriteSpreadsheet(obj,filename,boreoffset,regionID)
      %WRITESPREADSHEET generate master spreadsheet for BEAMLINE lattice, including separate sheet for each Class of elements
      %WriteSpreadsheet(filename [,boreoffset,spotsize,regionID])
      % boreoffset : added distance to physical bore aperture (e.g. including beam pipe wall thickness), default = 0
      % regionID : Structure with region names as field names, each region has 'id1' and 'id2' fields with BEAMLINE indices defining region of BEAMLINE array
      global BEAMLINE PS GIRDER WF
      if exist(filename,'file'); delete(filename); end
      BL0=BEAMLINE; PS0=PS; GIR0=GIRDER; WF0=WF;
      if ~exist('boreoffset','var') || isempty(boreoffset)
        boreoffset=0;
      end
      % - All Sheet
      id = 1:length(BEAMLINE) ;
      nod = ~ismember(id,findcells(BEAMLINE,'Class','DRIF'));
      id = id(nod) ;
      names = arrayfun(@(x) BEAMLINE{x}.Name,id,'UniformOutput',false);
      itid = findcells(BEAMLINE,'Type') ; tid=ismember(id,itid) ;
      types = repmat("UNKNOWN",length(id),1) ;
      types(tid) = arrayfun(@(x) string(BEAMLINE{x}.Type),id(tid));
      L = zeros(length(id),1) ;
      iLid = findcells(BEAMLINE,'L') ; Lid=ismember(id,iLid) ;
      L(Lid) = arrayfun(@(x) BEAMLINE{x}.L,id(Lid));
      P = arrayfun(@(x) BEAMLINE{x}.P,id);
      S = arrayfun(@(x) BEAMLINE{x}.S,id);
      Xi = arrayfun(@(x) BEAMLINE{x}.Coordi(1),id);
      Yi = arrayfun(@(x) BEAMLINE{x}.Coordi(2),id);
      Zi = arrayfun(@(x) BEAMLINE{x}.Coordi(3),id);
      Xf = arrayfun(@(x) BEAMLINE{x}.Coordf(1),id);
      Yf = arrayfun(@(x) BEAMLINE{x}.Coordf(2),id);
      Zf = arrayfun(@(x) BEAMLINE{x}.Coordf(3),id);
      XPi = arrayfun(@(x) BEAMLINE{x}.Anglei(1),id);
      YPi = arrayfun(@(x) BEAMLINE{x}.Anglei(2),id);
      ZPi = arrayfun(@(x) BEAMLINE{x}.Anglei(3),id);
      XPf = arrayfun(@(x) BEAMLINE{x}.Anglef(1),id);
      YPf = arrayfun(@(x) BEAMLINE{x}.Anglef(2),id);
      ZPf = arrayfun(@(x) BEAMLINE{x}.Anglef(3),id);
      classes = arrayfun(@(x) BEAMLINE{x}.Class,id,'UniformOutput',false);
      regions=repmat("NONE",length(id),1);
      if exist('regionID','var') && ~isempty(regionID)
        fn=fieldnames(regionID);
        for ifn=1:length(fn)
          if regionID.(fn{ifn}).id2<=length(BEAMLINE)
            regions(id>=regionID.(fn{ifn}).id1 & id<=regionID.(fn{ifn}).id2) = string(fn{ifn}) ;
          else
            BEAMLINE=BL0; PS=PS0; GIRDER=GIR0; WF=WF0;
            error('Bad region definition')
          end
        end
      end
      for iele=1:length(id)
        BEAMLINE{id(iele)}.Region = regions(iele) ;
      end
      T=table(id(:),names(:),regions(:),classes(:), types(:),L(:),P(:),S(:),Xi(:),Yi(:),Zi(:),XPi(:),YPi(:),ZPi(:),Xf(:),Yf(:),Zf(:),XPf(:),YPf(:),ZPf(:),'VariableNames',...
        {'Model ID'; 'Model Name'; 'Region Name'; 'Class'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      writetable(T,filename,'Sheet','All');
      % - COLL Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','COLL')) ;
      if any(sid)
        geom = arrayfun(@(x) BEAMLINE{x}.Geometry,id(sid),'UniformOutput',false) ;
        xgap = arrayfun(@(x) BEAMLINE{x}.aper(1),id(sid));
        ygap = arrayfun(@(x) BEAMLINE{x}.aper(2),id(sid));
        xoff = arrayfun(@(x) BEAMLINE{x}.Offset(1),id(sid));
        yoff = arrayfun(@(x) BEAMLINE{x}.Offset(3),id(sid));
        T_COLL=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',geom(:),xgap(:),ygap(:),xoff(:),yoff(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Geometry';'X Gap [m]';'Y Gap [m]';'X Offset [m]';'Y Offset [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - COR Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','*COR')) ;
      if any(sid)
        dim = repmat("X",sum(sid),1) ;
        dim(ismember(id(sid),findcells(BEAMLINE,'Class','YCOR'))) = 'Y' ;
        dim(ismember(id(sid),findcells(BEAMLINE,'Class','XYCOR'))) = 'XY' ;
        tilt = rad2deg(arrayfun(@(x) BEAMLINE{x}.Tilt,id(sid))) ;
        T_COR=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',dim(:),tilt(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Dimension';'Tilt [deg]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - BPM Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','MONI')) ;
      if any(sid)
        T_MONI=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - WIRE Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','WIRE')) ;
      if any(sid)
        T_WIRE=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - PROF Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','PROF')) ;
      if any(sid)
        T_PROF=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - IMON Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','IMON')) ;
      if any(sid)
        T_IMON=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - BLMO Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','BLMO')) ;
      if any(sid)
        T_BLMO=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - INST Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','INST')) ;
      if any(sid)
        T_INST=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % - MARK Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','MARK')) ;
      if any(sid)
        T_MARK=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
      end
      % -- Unsplit magnets for other Sheets
      obj.unsplitMags();
      id = 1:length(BEAMLINE) ;
      nod = ~ismember(id,findcells(BEAMLINE,'Class','DRIF'));
      id = id(nod) ;
      names = regexprep(arrayfun(@(x) BEAMLINE{x}.Name,id,'UniformOutput',false),'_.*$','') ;
      itid = findcells(BEAMLINE,'Type') ; tid=ismember(id,itid) ;
      types = repmat("UNKNOWN",length(id),1) ;
      types(tid) = arrayfun(@(x) string(BEAMLINE{x}.Type),id(tid));
      L = zeros(length(id),1) ;
      iLid = findcells(BEAMLINE,'L') ; Lid=ismember(id,iLid) ;
      L(Lid) = arrayfun(@(x) BEAMLINE{x}.L,id(Lid));
      P = arrayfun(@(x) BEAMLINE{x}.P,id);
      S = arrayfun(@(x) BEAMLINE{x}.S,id);
      Xi = arrayfun(@(x) BEAMLINE{x}.Coordi(1),id);
      Yi = arrayfun(@(x) BEAMLINE{x}.Coordi(2),id);
      Zi = arrayfun(@(x) BEAMLINE{x}.Coordi(3),id);
      Xf = arrayfun(@(x) BEAMLINE{x}.Coordf(1),id);
      Yf = arrayfun(@(x) BEAMLINE{x}.Coordf(2),id);
      Zf = arrayfun(@(x) BEAMLINE{x}.Coordf(3),id);
      XPi = arrayfun(@(x) BEAMLINE{x}.Anglei(1),id);
      YPi = arrayfun(@(x) BEAMLINE{x}.Anglei(2),id);
      ZPi = arrayfun(@(x) BEAMLINE{x}.Anglei(3),id);
      XPf = arrayfun(@(x) BEAMLINE{x}.Anglef(1),id);
      YPf = arrayfun(@(x) BEAMLINE{x}.Anglef(2),id);
      ZPf = arrayfun(@(x) BEAMLINE{x}.Anglef(3),id);
      regions=repmat("NONE",length(id),1);
      for iele=1:length(id)
        regions(iele)=BEAMLINE{id(iele)}.Region;
      end
      % - LCAV Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','LCAV')) ;
      if any(sid)
        id_this = id(sid) ;
        names_this = names(sid) ;
        regions_this = regions(sid) ;
        types_this = types(sid) ;
        P_this = P(sid) ;
        S_this = S(sid) ;
        freq = arrayfun(@(x) BEAMLINE{x}.Freq,id_this) ;
        volt = arrayfun(@(x) BEAMLINE{x}.Volt,id_this) ;
        phase = arrayfun(@(x) BEAMLINE{x}.Phase,id_this) ;
        egain = arrayfun(@(x) BEAMLINE{x}.Egain, id_this) ;
        kloss = arrayfun(@(x) BEAMLINE{x}.Kloss,id_this) ;
        L_this = arrayfun(@(x) BEAMLINE{x}.L,id_this) ;
        Xi_this = Xi(sid) ;
        Yi_this = Yi(sid) ;
        Zi_this = Zi(sid) ;
        XPi_this = XPi(sid) ;
        YPi_this = YPi(sid) ;
        ZPi_this = ZPi(sid) ;
        Xf_this = Xf(sid) ;
        Yf_this = Yf(sid) ;
        Zf_this = Zf(sid) ;
        XPf_this = XPf(sid) ;
        YPf_this = YPf(sid) ;
        ZPf_this = ZPf(sid) ;
        isel=true(length(id_this),1);
        for istruc=1:length(id_this)
          ele = id_this(istruc) ;
          if isfield(BEAMLINE{ele},'Slices')
            if BEAMLINE{ele}.Slices(end) ~= ele
              isel(istruc)=false;
              volt(istruc+1) = volt(istruc+1) + volt(istruc) ;
              egain(istruc+1) = egain(istruc+1) + egain(istruc) ;
              kloss(istruc+1) = kloss(istruc+1) + kloss(istruc) ;
              L_this(istruc+1) = L_this(istruc+1) + L_this(istruc) ;
            end
          end
        end
        T=table(id_this(isel)',names_this(isel)',regions_this(isel),types_this(isel),L_this(isel)',P_this(isel)',S_this(isel)',freq(isel)',volt(isel)',phase(isel)',egain(isel)',kloss(isel)',Xi_this(isel)',Yi_this(isel)',Zi_this(isel)',XPi_this(isel)',YPi_this(isel)',ZPi_this(isel)',Xf_this(isel)',Yf_this(isel)',Zf_this(isel)',XPf_this(isel)',YPf_this(isel)',ZPf_this(isel)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Freq [MHz]';'Voltage [MV]';'Phase [deg]';'EGAIN [MV]';'Kloss [V/C/m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
        writetable(T,filename,'Sheet','LCAV');
      end
      % - SBEN Sheet
      sid=ismember(id,findcells(BEAMLINE,'Class','SBEN')) ;
      if any(sid)
        Zlen = arrayfun(@(x) BEAMLINE{x}.Coordf(3),id(sid)) - arrayfun(@(x) BEAMLINE{x}.Coordi(3),id(sid)) ;
        gap = arrayfun(@(x) max(BEAMLINE{x}.HGAP),id(sid)) ;
        fint = arrayfun(@(x) max(BEAMLINE{x}.FINT),id(sid)) ;
        tilt = rad2deg(arrayfun(@(x) BEAMLINE{x}.Tilt,id(sid))) ;
        ang = rad2deg(arrayfun(@(x) BEAMLINE{x}.Angle,id(sid))) ;
        e1 = arrayfun(@(x) BEAMLINE{x}.EdgeCurvature(1),id(sid)) ;
        e2 = arrayfun(@(x) BEAMLINE{x}.EdgeCurvature(2),id(sid)) ;
        BL = arrayfun(@(x) BEAMLINE{x}.B(1),id(sid)) ;
        B = arrayfun(@(x) BEAMLINE{x}.B(1)/BEAMLINE{x}.L,id(sid)) ;
        n=0; K1=zeros(1,sum(sid)); g=K1; gl=K1;
        for iele=id(sid)
          n=n+1;
          if length(BEAMLINE{iele}.B)==1
            K1(n) = 0 ;
            g(n) = 0 ;
            gl(n) = 0 ;
          else
            Brho=physConsts.clight./(BEAMLINE{iele}.P*1e9);
            K1(n) = Brho * ( BEAMLINE{iele}.B(2) / BEAMLINE{iele}.L ) ;
            gl(n) = BEAMLINE{iele}.B(2) ;
            g(n) = gl(n) / BEAMLINE{iele}.L ;
          end
        end
        T=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',Zlen(:),gap(:),fint(:),tilt(:),ang(:),e1(:),e2(:),BL(:),B(:),K1(:),gl(:),g(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Z Length [m]';'Gap [m]';'Field Integral';'Tilt [deg]';'Bend Angle [deg]';'E1 [deg]';'E2 [deg]';'BL [T.m]';'B [T]';'K1 [1/m^2]';'GL [T]';'G [T/m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
        writetable(T,filename,'Sheet','SBEN');
      end
      % - QUAD Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','QUAD')) ;
      if any(sid)
        bore = arrayfun(@(x) max(BEAMLINE{x}.aper),id(sid)) + boreoffset ;
        tilt = rad2deg(arrayfun(@(x) BEAMLINE{x}.Tilt,id(sid))) ;
        Brho = physConsts.clight./(arrayfun(@(x) BEAMLINE{x}.P,id(sid)).*1e9);
        g = arrayfun(@(x) BEAMLINE{x}.B/BEAMLINE{x}.L,id(sid)) ;
        gl = arrayfun(@(x) BEAMLINE{x}.B,id(sid)) ;
        K1 = Brho .* gl ;
        T=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',bore(:),tilt(:),K1(:),gl(:),g(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Bore [m]';'Tilt [deg]';'K1 [1/m^2]';'GL [T]';'G [T/m]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
        writetable(T,filename,'Sheet','QUAD');
      end
      % - SEXT Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','SEXT')) ;
      if any(sid)
        bore = arrayfun(@(x) max(BEAMLINE{x}.aper),id(sid)) + boreoffset ;
        tilt = rad2deg(arrayfun(@(x) BEAMLINE{x}.Tilt,id(sid))) ;
        g = arrayfun(@(x) BEAMLINE{x}.B/BEAMLINE{x}.L,id(sid)) ;
        gl = arrayfun(@(x) BEAMLINE{x}.B,id(sid)) ;
        Brho = physConsts.clight./(arrayfun(@(x) BEAMLINE{x}.P,id(sid)).*1e9);
        K2 = 0.5 .* Brho .* gl ;
        T=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',bore(:),tilt(:),K2(:),gl(:),g(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'Bore [m]';'Tilt [deg]';'K2 [1/m^3]';'G''L [T/m]';'G'' [T/m^2]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
        writetable(T,filename,'Sheet','SEXT');
      end
      % - SOLENOID Sheet
      sid = ismember(id,findcells(BEAMLINE,'Class','SOLENOID')) ;
      if any(sid)
        g = arrayfun(@(x) BEAMLINE{x}.B/BEAMLINE{x}.L,id(sid)) ;
        gl = arrayfun(@(x) BEAMLINE{x}.B,id(sid)) ;
        Brho = physConsts.clight./(arrayfun(@(x) BEAMLINE{x}.P,id(sid)).*1e9);
        KS = 0.5 .* Brho .* gl ;
        T=table(id(sid)',names(sid)',regions(sid),types(sid),L(sid),P(sid)',S(sid)',KS(:),gl(:),g(:),Xi(sid)',Yi(sid)',Zi(sid)',XPi(sid)',YPi(sid)',ZPi(sid)',Xf(sid)',Yf(sid)',Zf(sid)',XPf(sid)',YPf(sid)',ZPf(sid)','VariableNames',...
          {'Model ID'; 'Model Name'; 'Region Name'; 'Engineering Type';'Path Length [m]';'E [GeV]';'S [m]';'KS [1/m]';'BL [T/m]';'B'' [T]';'X Coord (init) [m]';'Y Coord (init) [m]';'Z Coord (init) [m]';'X Angle (init) [rad]';'Y Angle (init) [rad]';'Z Angle (init) [rad]';'X Coord (fin) [m]';'Y Coord (fin) [m]';'Z Coord (fin) [m]';'X Angle (fin) [rad]';'Y Angle (fin) [rad]';'Z Angle (fin) [rad]'}) ;
        writetable(T,filename,'Sheet','SOLENOID');
      end
      % - Type Sheet
      ttab = obj.MagnetTypeData ;
      % -- Write out other table sheets
      if exist('T_COLL','var'); writetable(T_COLL,filename,'Sheet','COLL'); end
      if exist('T_COR','var'); writetable(T_COR,filename,'Sheet','COR'); end
      if exist('T_MONI','var'); writetable(T_MONI,filename,'Sheet','MONI'); end
      if exist('T_WIER','var'); writetable(T_WIRE,filename,'Sheet','WIRE'); end
      if exist('T_PROF','var'); writetable(T_PROF,filename,'Sheet','PROF'); end
      if exist('T_IMON','var'); writetable(T_IMON,filename,'Sheet','IMON'); end
      if exist('T_BLMO','var'); writetable(T_BLMO,filename,'Sheet','BLMO'); end
      if exist('T_INST','var'); writetable(T_INST,filename,'Sheet','INST'); end
      if exist('T_MARK','var'); writetable(T_MARK,filename,'Sheet','MARK'); end
      if ~isempty(ttab); writetable(ttab,filename,'Sheet','MagnetTypes'); end
      BEAMLINE=BL0; PS=PS0; GIRDER=GIR0; WF=WF0;
    end
    function restoreLucretiaData(obj)
      % Restore original Lucretia data structures
      global BEAMLINE PS GIRDER
      BEAMLINE=obj.BEAMLINE;
      PS=obj.PS;
      GIRDER=obj.GIRDER;
      obj.SetProps();
    end
    function splitMags(obj)
      global BEAMLINE PS
      splitClass={'QUAD' 'SEXT' 'OCTU' 'MULT' 'SBEN' 'SOLENOID' 'XCOR' 'YCOR'};
      if obj.doSplitDrifts
        splitClass{end+1}='DRIF';
      end
      ibl=0;
      while ibl<length(BEAMLINE)
        ibl=ibl+1;
        % Only split magnets which aren't already split
        if ismember(BEAMLINE{ibl}.Class,splitClass) && (~isfield(BEAMLINE{ibl},'Slices') || length(BEAMLINE{ibl}.Slices)==1)
          if isfield(BEAMLINE{ibl},'B')
            BEAMLINE{ibl}.B=BEAMLINE{ibl}.B./2;
          end
          BEAMLINE{ibl}.L=BEAMLINE{ibl}.L/2;
          if isfield(BEAMLINE{ibl},'Lrad')
            BEAMLINE{ibl}.Lrad=BEAMLINE{ibl}.Lrad/2;
          end
          if strcmp(BEAMLINE{ibl}.Class,'SBEN')
            BEAMLINE{ibl}.Angle=BEAMLINE{ibl}.Angle/2;
            if length(BEAMLINE{ibl}.EdgeAngle)==1
              BEAMLINE{ibl}.EdgeAngle=[BEAMLINE{ibl}.EdgeAngle BEAMLINE{ibl}.EdgeAngle];
            end
            if length(BEAMLINE{ibl}.EdgeCurvature)==1
              BEAMLINE{ibl}.EdgeCurvature=[BEAMLINE{ibl}.EdgeCurvature BEAMLINE{ibl}.EdgeCurvature];
            end
            if length(BEAMLINE{ibl}.FINT)==1
              BEAMLINE{ibl}.FINT=[BEAMLINE{ibl}.FINT BEAMLINE{ibl}.FINT];
            end
          end
          BEAMLINE=[BEAMLINE(1:ibl); BEAMLINE(ibl); BEAMLINE(ibl+1:end)];
          if isfield(BEAMLINE{ibl},'PS') && BEAMLINE{ibl}.PS>0
            PS(BEAMLINE{ibl}.PS).Element=[ibl ibl+1];
          end
          BEAMLINE{ibl}.Slices=[ibl ibl+1]; BEAMLINE{ibl+1}.Slices=[ibl ibl+1];
          if strcmp(BEAMLINE{ibl}.Class,'SBEN')
            BEAMLINE{ibl}.EdgeAngle(2)=0;
            BEAMLINE{ibl+1}.EdgeAngle(1)=0;
            BEAMLINE{ibl}.EdgeCurvature(2)=0;
            BEAMLINE{ibl+1}.EdgeCurvature(1)=0;
            BEAMLINE{ibl}.FINT(2)=0;
            BEAMLINE{ibl+1}.FINT(1)=0;
%             BEAMLINE{ibl}.Name=sprintf('%s_1',BEAMLINE{ibl}.Name);
%             BEAMLINE{ibl+1}.Name=sprintf('%s_2',BEAMLINE{ibl+1}.Name);
          end
          ibl=ibl+1;
        end
      end
      SetElementSlices(1,length(BEAMLINE)); SetElementBlocks(1,length(BEAMLINE));
      SetSPositions(1,length(BEAMLINE),BEAMLINE{1}.S);
      if isfield(BEAMLINE{1},'Coordi')
        SetFloorCoordinates(1,length(BEAMLINE),[BEAMLINE{1}.Coordi BEAMLINE{1}.Anglei]);
      end
      obj.SetProps(); % Initialize object properties from Lucretia globals
    end
    function unsplitMags(obj)
      global BEAMLINE PS GIRDER
      slinds=findcells(BEAMLINE,'Slices');
      if ~isempty(slinds)
        SetElementSlices(1,length(BEAMLINE));
      end
      splitClass={'QUAD' 'SEXT' 'OCTU' 'MULT' 'SBEN' 'SOLENOID'};
      rmele=[];
      for ibl=1:length(BEAMLINE)
        if isfield(BEAMLINE{ibl},'Slices') && ismember(BEAMLINE{ibl}.Class,splitClass)
          sliceind=BEAMLINE{ibl}.Slices;
          for ib=1:length(BEAMLINE{ibl}.B)
            BEAMLINE{ibl}.B(ib) = sum(arrayfun(@(x) BEAMLINE{x}.B(ib),sliceind)) ;
          end
          BEAMLINE{ibl}.L = sum(arrayfun(@(x) BEAMLINE{x}.L,sliceind)) ;
          if isfield(BEAMLINE{ibl},'Lrad')
            BEAMLINE{ibl}.Lrad = sum(arrayfun(@(x) BEAMLINE{x}.Lrad,sliceind)) ;
          end
          if strcmp(BEAMLINE{ibl}.Class,'SBEN')
            BEAMLINE{ibl}.Angle = sum(arrayfun(@(x) BEAMLINE{x}.Angle,sliceind)) ;
            if length(BEAMLINE{sliceind(end)}.EdgeAngle)==2
              BEAMLINE{ibl}.EdgeAngle(2) = BEAMLINE{sliceind(end)}.EdgeAngle(2) ;
            end
            if length(BEAMLINE{sliceind(end)}.EdgeCurvature)==2
              BEAMLINE{ibl}.EdgeCurvature(2) = BEAMLINE{sliceind(end)}.EdgeCurvature(2) ;
            end
            if length(BEAMLINE{sliceind(end)}.FINT)==2
              BEAMLINE{ibl}.FINT(2) = BEAMLINE{sliceind(end)}.FINT(2) ;
            end
          end
          BEAMLINE{ibl}=rmfield(BEAMLINE{ibl},'Slices');
          if isfield(BEAMLINE{ibl},'PS') && BEAMLINE{ibl}.PS
            PS(BEAMLINE{ibl}.PS).Element=ibl;
          end
          if isfield(BEAMLINE{ibl},'Girder') && BEAMLINE{ibl}.Girder
            GIRDER(BEAMLINE{ibl}.Girder).Element=ibl;
          end
          rmele=[rmele sliceind(2:end)];
        end
      end
      BEAMLINE(rmele)=[];
      SetElementBlocks(1,length(BEAMLINE));
      SetElementSlices(1,length(BEAMLINE));
      SetSPositions(1,length(BEAMLINE),BEAMLINE{1}.S);
      if isfield(BEAMLINE{1},'Coordi')
        SetFloorCoordinates(1,length(BEAMLINE),[BEAMLINE{1}.Coordi BEAMLINE{1}.Anglei]);
      end
      obj.SetProps();
    end
  end
  methods(Static)
    function FlipLattice
      %FLIPDECK Flip the order of the BEAMLINE lattice
      global BEAMLINE GIRDER
      % Flip the order of all elements
      BEAMLINE=flip(BEAMLINE);
      SetSPositions(1,length(BEAMLINE),0);
      if any(cellfun(@(x) isfield(x,'Slices'),BEAMLINE))
        SetElementSlices(1,length(BEAMLINE));
      end
      if any(cellfun(@(x) isfield(x,'Block'),BEAMLINE))
        SetElementBlocks(1,length(BEAMLINE));
      end
      for iele=1:length(BEAMLINE)
        if isfield(BEAMLINE{iele},'Coordi')
          ci=BEAMLINE{iele}.Coordi;
          ai=BEAMLINE{iele}.Anglei;
          BEAMLINE{iele}.Coordi=BEAMLINE{iele}.Coordf;
          BEAMLINE{iele}.Anglei=BEAMLINE{iele}.Anglef;
          BEAMLINE{iele}.Coordf=ci;
          BEAMLINE{iele}.Anglef=ai;
          BEAMLINE{iele}.Anglei(1)=BEAMLINE{iele}.Anglei(1)+pi;
          BEAMLINE{iele}.Anglef(1)=BEAMLINE{iele}.Anglef(1)+pi;
        end
      end
      % Re-order bend magnet slice information
      for iele=findcells(BEAMLINE,'Class','SBEN')
        BEAMLINE{iele}.EdgeAngle=flip(BEAMLINE{iele}.EdgeAngle);
        BEAMLINE{iele}.HGAP=flip(BEAMLINE{iele}.HGAP);
        BEAMLINE{iele}.FINT=flip(BEAMLINE{iele}.FINT);
        BEAMLINE{iele}.EdgeCurvature=flip(BEAMLINE{iele}.EdgeCurvature);
      end
      % Setup correct parameters for RF elements
      for iele=findcells(BEAMLINE,'Egain')
        BEAMLINE{iele}.Egain=-BEAMLINE{iele}.Egain;
        BEAMLINE{iele}.Phase=BEAMLINE{iele}.Phase+180;
      end
      % Flip required Offset parameters
      for iele=findcells(BEAMLINE,'Offset')
        BEAMLINE{iele}.Offset([1 2 5 6]) = -BEAMLINE{iele}.Offset([1 2 5 6]) ;
      end
      if ~isempty(GIRDER)
        for igir=1:length(GIRDER)
          GIRDER{igir}.Offset([1 2 5 6]) = -GIRDER{igir}.Offset([1 2 5 6]) ;
        end
      end
    end
    function SliceMags(sdist)
      %SLICEMAGS Split up all magnet elements in BEAMLINE so that each slice is less than sdist [m]
      global BEAMLINE PS
      for ips=1:length(PS)
        if PS(ips).Element(1)<iend
          RenormalizePS(ips);
          for iele=PS(ips).Element
            if PS(ips).Ampl==0
              BEAMLINE{iele}.B=BEAMLINE{iele}.B*0;
            end
            BEAMLINE{iele}.PS=0;
          end
        end
      end
      BL_new={};
      for iele=1:length(BEAMLINE)
        if isfield(BEAMLINE{iele},'Girder')
          BEAMLINE{iele}.Girder=0;
        end
        if isfield(BEAMLINE{iele},'Block')
          BEAMLINE{iele}=rmfield(BEAMLINE{iele},'Block');
        end
        if isfield(BEAMLINE{iele},'Slices')
          BEAMLINE{iele}=rmfield(BEAMLINE{iele},'Slices');
        end
        if isfield(BEAMLINE{iele},'L') && BEAMLINE{iele}.L>sdist && ~strcmp(BEAMLINE{iele}.Class,'TMAP')
          nsplit=ceil(BEAMLINE{iele}.L/sdist);
          BL=BEAMLINE{iele};
          BL.L=BL.L/nsplit;
          if isfield(BL,'B')
            BL.B=BL.B./nsplit;
          end
          if isfield(BL,'Volt')
            BL.Volt=BL.Volt./nsplit;
            BL.Egain=BL.Egain./nsplit;
          end
          if strcmp(BL.Class,'SBEN')
            BL.Angle=BL.Angle./nsplit;
            if length(BL.EdgeAngle)==1
              BL.EdgeAngle=ones(1,2).*BL.EdgeAngle;
            end
            if length(BL.HGAP)==1
              BL.HGAP=ones(1,2).*BL.HGAP;
            end
            if length(BL.FINT)==1
              BL.FINT=ones(1,2).*BL.FINT;
            end
            if length(BL.EdgeCurvature)==1
              BL.EdgeCurvature=ones(1,2).*BL.EdgeCurvature;
            end
          end
          for isplit=1:nsplit
            if isfield(BL,'EdgeAngle') && isplit==1
              BL.EdgeAngle=[BEAMLINE{iele}.EdgeAngle(1) 0];
              BL.HGAP=[BEAMLINE{iele}.HGAP(1) 0];
              BL.FINT=[BEAMLINE{iele}.FINT(1) 0];
              BL.EdgeCurvature=[BEAMLINE{iele}.EdgeCurvature(1) 0];
            elseif isfield(BL,'EdgeAngle') && isplit==nsplit
              BL.EdgeAngle=[0 BEAMLINE{iele}.EdgeAngle(2)];
              BL.HGAP=[0 BEAMLINE{iele}.HGAP(2)];
              BL.FINT=[0 BEAMLINE{iele}.FINT(2)];
              BL.EdgeCurvature=[0 BEAMLINE{iele}.EdgeCurvature(2)];
            elseif isfield(BL,'EdgeAngle')
              BL.EdgeAngle=[0 0];
              BL.HGAP=[0 0];
              BL.FINT=[0 0];
              BL.EdgeCurvature=[0 0];
            end
            BL_new{end+1}=BL;
          end
        else
          BL_new{end+1}=BEAMLINE{iele};
        end
      end
      BEAMLINE=BL_new(:);
      SetSPositions(1,length(BEAMLINE),BEAMLINE{1}.S);
    end
    function T=ReadMadTwiss(filename)
      fid=fopen(filename);
      fgetl(fid); fgetl(fid);
      T.Name={};
      while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        disp(tline)
      end
      fclose(fid);
    end
    function tab = MagnetTypeData(ind1,ind2)
      global BEAMLINE
      if ~exist('ind1','var')
        ind1=1; ind2=length(BEAMLINE);
      end
      SetElementSlices(ind1,ind2);
      mele=findcells(BEAMLINE,'B',[],ind1,ind2);
      types=string([]); minB=[]; maxB=[]; classes=types; count=[];
      for imag=mele
        if ~isfield(BEAMLINE{imag},'Type')
          BEAMLINE{imag}.Type = sprintf('UNKNOWN_%s',BEAMLINE{imag}.Class) ;
        end
        if isfield(BEAMLINE{imag},'Slices') && imag~=BEAMLINE{imag}.Slices(end)
          continue
        end
        Bact = GetTrueStrength(imag,1) ;
        if isempty(types) || ~ismember(string(BEAMLINE{imag}.Type),types)
          types(end+1)=string(BEAMLINE{imag}.Type);
          classes(end+1)=string(BEAMLINE{imag}.Class);
          count(end+1)=1;
          minB(end+1)=Bact(1); maxB(end+1)=Bact(1);
        else
          itype=ismember(types,string(BEAMLINE{imag}.Type));
          count(itype)=count(itype)+1;
          if Bact(1) < minB(itype)
            minB(itype) = Bact(1) ;
          elseif Bact(1) > maxB(itype)
            maxB(itype) = Bact(1) ;
          end
        end
      end
      tab=table(types(:),classes(:),count(:),minB(:),maxB(:),'VariableNames',{'Type','Class','Count','MinB [T.m^(1-n)]','MaxB [T.m^(1-n)]'});
    end
    function AssignCSR
      %ASSIGNCSR Assign CSR tags to DRIFTs downstream of bends
      global BEAMLINE
      ele=sort([findcells(BEAMLINE,'Class','SBEN') findcells(BEAMLINE,'Class','DRIF')]); 
      ndrift=0;
      for iele=ele
        if strcmp(BEAMLINE{iele}.Class,'SBEN')
          % - how many downstream drifts to be set to CSRDRIFT type?
          if isfield(BEAMLINE{iele}.TrackFlag,'CSR_DriftSplit') && BEAMLINE{iele}.TrackFlag.CSR_DriftSplit>0
            R=BEAMLINE{iele}.L/BEAMLINE{iele}.Angle;
            % - if no design bunch length provided, assume 1mm rms
            if ~isfield(BEAMLINE{iele}.TrackFlag,'SIGZDES')
              lDecay=3*(24*1e-3*R^2)^(1/3);
            else
              lDecay=3*(24*BEAMLINE{iele}.TrackFlag.SIGZDES*R^2)^(1/3);
            end
            ndrift=0;
            if iele<length(BEAMLINE)
              lsum=0; lsumd=0;
              for iele2=iele+1:length(BEAMLINE)
                if isfield(BEAMLINE{iele2},'L')
                  lsum=lsum+BEAMLINE{iele2}.L;
                end
                if strcmp(BEAMLINE{iele2}.Class,'DRIF')
                  lsumd=lsumd+BEAMLINE{iele2}.L;
                  ndrift=ndrift+1;
                  dz=lsumd/BEAMLINE{iele}.TrackFlag.CSR_DriftSplit;
                  if lsum>lDecay
                    break
                  end
                end
              end
            end
          end
        elseif ndrift>0
          ndrift=ndrift-1;
          BEAMLINE{iele}.TrackFlag.CSR=dz;
        end
      end
    end
  end
  methods(Static,Access=private)    
    function varargout=collectPar(ele,varargin)
      npar=nargin-1;
      if npar<1; varargout{1}=[]; return; end
      if nargout~=npar; error('collectPar format error'); end
      for ipar=1:npar
        if isfield(ele,varargin{ipar})
          varargout{ipar}=ele.(varargin{ipar});
        else
          switch varargin{ipar}
            case {'l','angle','e1','e2','tilt','h1','h2','hgap','fint','k2','deltae','phi0',...
                'eloss','dx','dxp','dy','dyp','dz','kick','hkick','vkick','dpx','dpy','x','y','px','py',...
                'alphax','alphay','alpfx','alphy','mux','muy'}
              varargout{ipar}=0;
            case {'k1','hgapx','fintx','lrad','lfile','tfile','energy','gamma',...
                'ex','ey','et','exn','eyn','sigt','sige'}
              varargout{ipar}=NaN;
            case {'freq','xsize','ysize','betax','betay','betx','bety','npart','kbunch'}
              varargout{ipar}=1;
            case 'type'
              varargout{ipar}='UNKNOWN';
            case 'nbin'
              varargout{ipar}=10;
            case 'aperture'
              if isfield(ele,'aper')
                varargout{ipar}=ele.aper;
              else
                varargout{ipar}=1;
              end
            case 'charge'
              varargout{ipar}=1.60217662e-19;
            otherwise
              error('collectPar: unknown parameter type: %s',varargin{ipar})
          end
        end
      end
    end
    function [L,K0L,K1L,K2L,K3L]=getProps(ele)
      global BEAMLINE PS
      L=0; K0L=0; K1L=0; K2L=0; K3L=0;
      clight=2.99792458e8; % speed of light (m/sec)
      Cb=1e9/clight;       % rigidity conversion (T-m/GeV)
      if isfield(BEAMLINE{ele},'L')
        L=BEAMLINE{ele}.L;
      end
      if isfield(BEAMLINE{ele},'B')
        if strcmp(BEAMLINE{ele}.Class,'SBEN')
          K0L=BEAMLINE{ele}.B(1);
          if length(BEAMLINE{ele}.B)>1
            K1L=BEAMLINE{ele}.B(2);
          end
          if length(BEAMLINE{ele}.B)>2
            K2L=BEAMLINE{ele}.B(3);
          end
          if length(BEAMLINE{ele}.B)>3
            K3L=BEAMLINE{ele}.B(4);
          end
        elseif strcmp(BEAMLINE{ele}.Class,'QUAD')
          K1L=BEAMLINE{ele}.B(1);
          if length(BEAMLINE{ele}.B)>1
            K2L=BEAMLINE{ele}.B(2);
          end
          if length(BEAMLINE{ele}.B)>2
            K3L=BEAMLINE{ele}.B(3);
          end
        elseif strcmp(BEAMLINE{ele}.Class,'SEXT')
          K2L=BEAMLINE{ele}.B;
        elseif strcmp(BEAMLINE{ele}.Class,'OCTU')
          K2L=BEAMLINE{ele}.B;
        elseif strcmp(BEAMLINE{ele}.Class,'MULT')
          pI=BEAMLINE{ele}.PoleIndex;
          for ipole=1:length(pI)
            if pI(ipole)==0
              K0L=BEAMLINE{ele}.B(ipole);
            elseif pI(ipole)==1
              K1L=BEAMLINE{ele}.B(ipole);
            elseif pI(ipole)==2
              K2L=BEAMLINE{ele}.B(ipole);
            elseif pI(ipole)==3
              K3L=BEAMLINE{ele}.B(ipole);
            end
          end
        end
      end
      K0L=K0L/(Cb*BEAMLINE{ele}.P);
      K1L=K1L/(Cb*BEAMLINE{ele}.P);
      K2L=K2L/(Cb*BEAMLINE{ele}.P);
      K3L=K3L/(Cb*BEAMLINE{ele}.P);
      % K0L=K0L*2; K1L=K1L*2; K2L=K2L*2; K3L=K3L*2;
      % if ~strcmp(BEAMLINE{ele}.Class,'LCAV')
      %   L=L*2;
      % end
      if isnan(K0L); K0L=0; end
      if isnan(K1L); K1L=0; end
      if isnan(K2L); K2L=0; end
      if isnan(K3L); K3L=0; end
      if isfield(BEAMLINE{ele},'PS') && BEAMLINE{ele}.PS>0
        psampl=PS(BEAMLINE{ele}.PS).Ampl;
        K0L=K0L*psampl; K1L=K1L*psampl; K2L=K2L*psampl; K3L=K3L*psampl;
      end
    end
  end
end