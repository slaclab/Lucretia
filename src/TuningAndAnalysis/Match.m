classdef Match < handle & physConsts
  %MATCH Lucretia beam matching using GetTwiss and TrackThru
  %  SOME OPTIONS REQUIRE OPTIMIZATION TOOLBOX AND/OR GLOBAL OPTIMISATION
  %  TOOLBOX
  %
  %Useage example below (see 'doc Match' for complete list of properties and methods):
  %Create object:
  %M=Match;
  %
  %Assign Lucretia beam if tracking needed (matching S,T or U):
  %M.beam=Beam;
  %
  %Assign Lucretia Initial structure if twiss matching needed
  % (anything other than S,T or U):
  %  M.initStruc=InitialStructure;
  %
  %Assign initial track/twiss point corresponding to above:
  %M.iInitial=findcells(BEAMLINE,'Name','InitialMarkerName');
  %
  %Choose an optimizer to use:
  %M.optim='lsqnonlin';
  %  Supported optimizers are:
  %    * fminsearch (supported in standard Matlab)
  %    * fmincon (requires optimization toolbox)
  %    * lsqnonlin (requries optimization toolbox)
  %    * fgoalattain (requires optimization toolbox)
  %    * moga (requires global optimization and optimization toolboxes)
  %    * global (requires global optimization and optimization toolboxes)
  %  The recommended optimizers are lsqnonlin for twiss optimization only
  %  and fminsearch or fmincon for tracking-based optimization (S, T or U).
  %  The Genetic algorithm requires a lot of tuning of its input parameters
  %  to give sensible answers, you will need to play with the gaoptimset options
  %  inside the doMatch method.
  %
  %Add optimization variables
  %  addVariable(M,'PS',BEAMLINE{quadEle}.PS,'Ampl',0.8,1.2);
  %  Available variable types are: PS,BEAMLINE,GIRDER,KLYSTRON
  %  You can use any available field of the supported types that contains a
  %  scalar quantity. Repeat above for each required variable. The last 2
  %  input parameters define the lower and upper bounds for this variable.
  %  For the constrained optimizers the final variable values are
  %  guaranteed to be within this range, for the unconstrained ones this
  %  range is used as part of the weighting system to restrain the allowed
  %  movement of this variable.
  %
  %Add constraints
  %  addMatch(M,beamlineElement1,'alpha_y',0,1e-4);
  %  addMatch(M,beamlineElement2,'Sigma',35e-9^2,1e-9^2,'33');
  %  Add a constraint for the optimizer at the given BEAMLINE element
  %  number. Available constraint types are: alpha_x, beta_x, eta_x,
  %  etap_x, nu_x, NEmit_x, S, T, U. Also _y for vertical twiss parameters.
  %  For S, T, U (second moment, and second and third-order beam
  %  correlations) also supply correlation matrix elements as last
  %  argument. e.g. '33' for sigma_33. '3126' for U3126 etc... Third
  %  argument is required match value. Forth argument is tolerance, the
  %  matcher will stop when all added constraints are closer than this
  %  value to the desired match value.
  %  Repeat for all required match parameters
  %
  %display(M) [or simply type 'M']
  %  Look at the list of assigned variables and constraints and the current
  %  match data
  %
  %  M.doMatch;
  %  Perform the matching
  %
  %display(M)
  %  See how it did
  %
  % (Note: to use the T/U match conditions you need the "polyfitn" tools
  % supplied in the Lucretia/src/utils/PolyfitnTools directory)
  %
  % See also:
  %  InitCondStruc MakeBeam6dGauss findcells fminsearch fmincon lsqnonlin
  %  fgoalattain
  %
  % Reference page in Help browser for list of accessible properties and
  % methods:
  %   <a href="matlab:doc Match">doc Match</a>
  
  properties
    beam % Beam used for tracking (should be Lucretia Beam format used by TrackThru)
    initStruc % Lucretia Initial structure at iInitial BEAMLINE location
    iInitial % Initial beamline index (corresponding to beam)
    optim='lsqnonlin'; % Optimizer to use (options are 'gradDrop', 'fminsearch', 'fmincon', 'lsqnonlin', 'global', 'moga', 'fgoalattain')
    verbose=false; % if set true, then display match conditions as optimiser runs
    optimDisplayRate=10; % How often to update the status display if verbose set to true (seconds)
    optimDisplay='final'; % what to pass to the optimiser display parameter (e.g. 'off', 'final', 'iter')
    useParallel=false; % Use parallel processing for optimisation. Requires parallel toolbox and fmincon or moga or fgoalattain algrithm use.
    storeOptim=false; % Store all optimisation step data
    linstep=10; % percentage of total range for each variable to use when forming linMat with optim='lscov'
    nLookupSteps=50; % number of random samples of all variable phase space to generate "lookup table"
    lookupSaveLoc='matchLookupData'; % [directory+]file to save computed lookup table (not saved if empty or empty string)
    DL
    createLookupNdims=1; % =1 if just want to use 1-d lookup, >1 uses higher-order polynomial fits
    useFitData=false; % Use fit data instead of Twiss calculation / tracking
    matchWeights=[]; % Vector of weights for match entries
    userFitFun % function handle for user supplied fitting routine (must take Lucretia beam structure as only argument)
    userOutputFn % User mimiser output function
    MaxFunEvals=100000; % Max number of function evaluations to allow when performing optimization
  end
  properties(SetAccess=protected)
    matchType={}; % Cell array of match type descriptors (see allowedMatchTypes property)
    matchTypeQualifiers={}; % Qualifiers for given types of match (e.g. '13' tp set Sigma_13, '166' to set T_166), cell array matching matchType size
    matchVals % The values for the match constraints specified
    matchInd % Beamline vector of indices defining match points
    varType={}; % Cell array of variable types (see allowedVariableTypes property)
    varIndex={}; % Cell array of indexes pertaining to varType (if cell entry is vector, then tie together multiple elements using same varField)
    varField={}; % Field of requested variable (must be existing field of requested variable type), cell array of same dimension as varType
    varFieldIndex=[]; % Index of field (same length as varField array
    varLimits % array of high/low limits for variables [2,length(obj.varType)]
    BUMP=[]; % BUMP information (array) - use defineBump method to add
    linMat % Linear response matrix for use in optim='lscov' (form with createLinMatrix method)
    lookupMat
    lookupVarVals
    lookupCnstVals
    optimExitFlag
  end
  properties(Dependent)
    dotrack
    dotwiss
    varVals
    iMatch % unique list of required match points
    optimVals % match values returned from the optimizer
    Ix % horizontal Initial vals for GetTwiss
    Iy % vertical Initial vals for GetTwiss
    optimData % contraint, variable and optimStop values at each iteration step
    ipLastBeam % the last tracked beam sizes (RMS)
  end
  properties(Constant)
    allowedMatchTypes={'alpha_x' 'alpha_y' 'beta_x' 'beta_y' 'NEmit_x' 'NEmit_y' 'eta_x' 'Qloss' ...
      'etap_x' 'eta_y' 'etap_y' 'nu_x' 'nu_y' 'modnu_x' 'modnu_y' 'SigmaGauss' 'SigmaFit' 'User' 'Sigma' 'R' ...
      'T' 'U' 'Waist_x' 'Waist_y' 'Disp_x' 'Disp_y' 'SumBeta' 'AbsDiffBeta' 'MinBeta' 'MaxBeta' 'W_x' 'W_y' 'Wsum_x' 'Wsum_y'};
    allowedVariableTypes={'PS' 'GIRDER' 'KLYSTRON' 'BEAMLINE' 'BUMP' 'INITIAL'};
    allowedOptimMethods={'gradDrop' 'fminsearch' 'fmincon' 'lsqnonlin' 'moga' 'global' 'fgoalattain'};
  end
  
  methods
    function obj=Match(DL)
      % Match - checks for optimization toolbox upon object creation
      vout=ver;
      if any(ismember({vout(:).Name},'Optimization Toolbox'))
        obj.optim='lsqnonlin';
      else
        obj.optim='fminsearch';
      end
      if exist('DL','var')
        obj.DL=DL;
      end
    end
    function set.useParallel(obj,val)
      if val
        v=ver;
        if ~any(arrayfun(@(x) strcmp(v(x).Name,'Parallel Computing Toolbox'),1:length(v)))
          error('Must have parallel computing toolbox to use parallel option')
        end
        if ~ismember(obj.optim,{'fmincon','moga','fgoalattain'}) %#ok<MCSUP>
          error('Must use one of the algorithms: ''fmincon'',''moga'',''fgoalattain'' to use parallel option')
        end
      end
      obj.useParallel=val;
    end
    function SetMatchVal(obj,name,newval)
      ival=ismember(name,obj.matchType);
      if isempty(ival)
        error('No match type exists with name %s',name);
      end
      obj.matchVals(ival)=newval;
    end
    function addMatch(obj,mind,type,vals,tol,typeQual)
      % ADDMATCH - add a match condition to the problem set
      % addMatch(obj,type,weight,vals,tol [,typeQual])
      % type = string or cell list of strings, see allowedMatchTypes property for available list
      % weight = double, weighting function for constraints
      % vals = double, desired match values
      % typeQual = qualifier (optional) string or cell vector of strings or
      %            double or cell of doubles
      %            (e.g. '13' to set Sigma_13, '166' to set T_166)
      % tol = tolerance for this match condition
      % ... All above can be scalar or vectors of same length to either add
      %     a single match condition or multiple
      if ~exist('type','var') || any(~ismember(type,obj.allowedMatchTypes))
        error('Must choose from available list of match types')
      end
      if ~exist('mind','var') || (iscell(type) && length(mind)~=length(type))
        error('Must supply BEAMLINE index or vector of indices for this match constraint (length same as number of type declarations)')
      end
      if ~exist('vals','var') || ~isnumeric(vals) || (iscell(type) && length(vals)~=length(type))
        error('Must supply match value(s) ''var'' scalar double or vector of same length as type')
      end
      if ~exist('tol','var') || ~isnumeric(tol) || (iscell(type) && length(tol)~=length(type))
        error('Must supply match value(s) ''tol'' scalar double or vector of same length as type')
      end
      if iscell(type)
        nadd=length(type);
      else
        nadd=1;
      end
      if exist('typeQual','var') && iscell(typeQual) && iscell(type) && length(typeQual) ~= length(type)
        error('If supply typeQual, must be either scalar cell or vector same length as ''type''')
      elseif exist('typeQual','var') && iscell(typeQual)
        obj.matchTypeQualifiers(end+1:end+nadd)=typeQual;
      elseif exist('typeQual','var')
        obj.matchTypeQualifiers{end+1}=typeQual;
      elseif ~exist('typeQual','var')
        for iadd=1:nadd
          obj.matchTypeQualifiers{end+1}='';
        end
      end
      if iscell(type)
        obj.matchType(end+1:end+nadd)=type;
      else
        obj.matchType{end+1}=type;
      end
      obj.matchWeights(end+1:end+nadd)=tol;
      obj.matchVals(end+1:end+nadd)=vals;
      obj.matchInd(end+1:end+nadd)=mind;
      % linear response matrix no longer any good, clear it
      obj.linMat=[];
    end
    function defineBump(obj,type,corPsInds,bumpInd)
      % DEFINEBUMP - add a BUMP definition
      % defineBump(obj,type,corPsInds,bumpInd)
      %
      % currently only support type='COR3' (three corrector dipole magnet
      % bump)
      % corPsInds = vector of 3 PS indices (must point to XCOR or
      % YCOR class elements)
      % bumpInd = BEAMLINE index of target for bump (must be between first
      % 2 correctors)
      % Creates obj.BUMP(end+1) with fields: 'corPS' 'bumpIND' 'type'
      %                                      'dim' 'coefs' 'val'
      global PS BEAMLINE
      if ~exist('type','var') || ~exist('type','var') || ~exist('corPsInds','var') || ~exist('bumpInd','var')
        error('some required defineBump method inputs not specified')
      end
      if ~strcmp(type,'COR3')
        error('Only type=''COR3'' supported for now')
      end
      if any(corPsInds>length(PS)) || any(corPsInds<1)
        error('corPsInds must be valid indices of the PS global array')
      end
      if bumpInd>PS(corPsInds(2)).Element(1) || bumpInd<(PS(corPsInds(1)).Element(end)+1)
        error('bumpInd must be a BEAMLINE element between first two specified correctors')
      end
      obj.BUMP(end+1).type=type;
      obj.BUMP(end).corPS=corPsInds;
      obj.BUMP(end).bumpIND=bumpInd;
      if strcmp(BEAMLINE{PS(corPsInds(1)).Element(1)}.Class,'XCOR')
        obj.BUMP(end).dim=1;
      elseif strcmp(BEAMLINE{PS(corPsInds(1)).Element(1)}.Class,'YCOR')
        obj.BUMP(end).dim=3;
      else
        error('Corrector elements must consist of class XCOR or YCOR')
      end
      bumpcoefs = obj.getBumpCoef(corPsInds, bumpInd, obj.BUMP(end).dim) ;
      obj.BUMP(end).coefs=bumpcoefs;
      obj.BUMP(end).val=0;
    end
    function addVariable(obj,type,typeIndex,field,lowLimit,highLimit,fieldIndex)
      % ADDVARIABLE - add a variable to match problem set
      % addVariable(obj,type,typeIndex,field,lowLimit,highLimit [,fieldIndex])
      % type = string, or cell vector of strings: see allowedVariableTypes property for allowed options
      % typeIndex = integer
      % field = string, or cell of strings length of type: fieldname of type
      % lowLimit = lower bound on possible vals
      % highLimit = upper bound on possible vals
      % fieldIndex (optional) = double, index of field
      % ... All above can be scalar or vectors of same length for single or
      %     multiple variable declarations
      if ~exist('type','var') || any(~ismember(type,obj.allowedVariableTypes))
        error('Must supply variable type(s) from allowed list')
      end
      if ~exist('typeIndex','var') || (iscell(type) && length(typeIndex)~=length(type))
        error('Must supply typeIndex integer or vector of integers same length as type')
      end
      if ~exist('field','var')  || (iscell(type) && length(field)~=length(type))
        error('Must supply field name or cell of field names same length as type')
      end
      if ~exist('lowLimit','var') || (iscell(type) && length(lowLimit)~=length(type))
        error('Must supply low limit to variable or vector of the same length as type')
      end
      if ~exist('highLimit','var') || (iscell(type) && length(highLimit)~=length(type))
        error('Must supply high limit to variable or vector of the same length as type')
      end
      if iscell(type)
        nadd=length(type);
      else
        nadd=1;
      end
      if exist('fieldIndex','var') && iscell(type) && length(fieldIndex)~=length(type)
        error('If supplying fieldIndex, must be of same length as number of types supplied')
      elseif exist('fieldIndex','var')
        obj.varFieldIndex(end+1:end+nadd)=fieldIndex;
      elseif ~exist('fieldIndex','var')
        obj.varFieldIndex(end+1:end+nadd)=1;
      end
      if iscell(type)
        obj.varType(end+1:end+nadd)=type;
      else
        obj.varType{end+1}=type;
      end
      if iscell(typeIndex)
        obj.varIndex(end+1:end+nadd)=typeIndex;
      else
        obj.varIndex{end+1}=typeIndex;
      end
      if iscell(field)
        obj.varField(end+1:end+nadd)=field;
      else
        obj.varField{end+1}=field;
      end
      obj.varLimits(1,end+1:end+nadd)=lowLimit;
      obj.varLimits(2,end-nadd+1:end)=highLimit;
      % linear response matrix no longer any good, clear it
      obj.linMat=[];
    end
    function disp(obj)
%       try
        fprintf('Lucretia match using optimizer: %s\n',obj.optim)
        fprintf('-----------------------------------------\n')
        fprintf('Beam sizes at match points:\n')
        ov=obj.optimVals;
        v=obj.minFunc('getvals');
        try
          for im=1:length(obj.iMatch)
            fprintf('%d: sigma_x= %g sigma_y = %g\n',obj.iMatch(im),v.bs.x(im),v.bs.y(im))
          end
        catch
          disp('...No beam data...');
        end
        fprintf('-----------------------------------------\n')
        fprintf('Constraints / Desired Vals / Current Vals / Weights\n')
        fprintf('\n')
        for itype=1:length(obj.matchType)
          if ischar(obj.matchTypeQualifiers{itype})
            fprintf('%d: %s (%s) / %g / %g / %g\n',obj.matchInd(itype),obj.matchType{itype},obj.matchTypeQualifiers{itype},...
              obj.matchVals(itype),ov(itype),obj.matchWeights(itype))
          else
            fprintf('%d: %s (%s) / %g / %g / %g\n',obj.matchInd(itype),obj.matchType{itype},num2str(obj.matchTypeQualifiers{itype}),...
              obj.matchVals(itype),ov(itype),obj.matchWeights(itype))
          end
        end
        fprintf('\n')
        fprintf('-----------------------------------------\n')
        fprintf('Variables / Val / Low / High\n')
        fprintf('\n')
        curVals=obj.varVals;
        for itype=1:length(obj.varType)
          indstr=' ';
          for ind=obj.varIndex{itype}
            indstr=[indstr num2str(ind) ' '];
          end
          fprintf('%s(%s).%s(%d) / %g / %g / %g\n',obj.varType{itype},indstr,obj.varField{itype},...
            obj.varFieldIndex(itype),curVals(itype),obj.varLimits(1,itype),obj.varLimits(2,itype))
        end
        fprintf('============================================\n')
        fprintf('============================================\n')
%       catch
%         disp('--==--==--==--==--==--==')
%         disp('!!! Error in data !!!')
%         disp('--==--==--==--==--==--==')
%       end
    end
    function doMatch(obj)
      % Perform programmed match procedure
      
      % Check structure good for fitting
      if obj.dotrack && isempty(obj.beam)
        error('Must supply Lucretia beam for tracking')
      end
      if obj.dotwiss && isempty(obj.initStruc)
        error('Must supply Lucretia Initial structure for Twiss propagation')
      end
      % --- Perform matching
      % Set global options
      if strcmp(obj.optim,'moga')
        opts=optimset('Display',obj.optimDisplay,'MaxFunEvals',obj.MaxFunEvals,'MaxIter',100000,'TolX',1e-5,'TolFun',1e-5,...
          'OutputFcn',@(x,optimValues,state) optimOutFun(obj,x,optimValues,state));
      elseif strcmp(obj.optim,'lsqnonlin')
        opts=optimset('Display',obj.optimDisplay,'MaxFunEvals',obj.MaxFunEvals,'MaxIter',100000,'TolX',1e-5,'TolFun',1e-5,...
          'OutputFcn',@(x,optimValues,state) optimOutFun(obj,x,optimValues,state));
      elseif strcmp(obj.optim,'fmincon') || strcmp(obj.optim,'global')
        opts=optimset('Display',obj.optimDisplay,'OutputFcn',@(x,optimValues,state) optimOutFun(obj,x,optimValues,state),'MaxFunEvals',obj.MaxFunEvals,...
          'MaxIter',100000,'TolX',1e-3,'TolFun',1e-1,'UseParallel','never','Algorithm','active-set');
      else
        opts=optimset('Display',obj.optimDisplay,'OutputFcn',@(x,optimValues,state) optimOutFun(obj,x,optimValues,state),'MaxFunEvals',obj.MaxFunEvals,...
          'MaxIter',100000,'TolX',1e-6,'TolFun',1e-6,'UseParallel','never');
      end
      
      % Output function supplied?
      if ~isempty(obj.userOutputFn)
        opts.OutputFcn=obj.userOutputFn;
      end
      
      % Perform fit
      % - normalise variables
      varW=(obj.varLimits(2,:)-obj.varLimits(1,:)); varVals=obj.varVals;
      switch obj.optim
        case 'global'
          gs = GlobalSearch;
          problem = createOptimProblem('fmincon','x0',varVals./varW,'objective',@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss),...
            'lb', obj.varLimits(1,:)./varW,'ub',obj.varLimits(2,:)./varW,'options',opts);
          x = run(gs,problem) ;
        case 'moga'
          x = gamultiobj(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss,varW,varVals),length(varVals),[],[],[],[],...
            obj.varLimits(1,:)./varW,obj.varLimits(2,:)./varW,opts);
        case 'lsqnonlin'
          [x , ~, ~, eflag]=lsqnonlin(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss,varW,varVals),...
            varVals./varW,obj.varLimits(1,:)./varW,obj.varLimits(2,:)./varW,opts);
          obj.optimExitFlag=eflag;
        case 'fminsearch'
          [x , ~, eflag]=fminsearch(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss),varVals./varW,opts);
          obj.optimExitFlag=eflag;
        case 'fmincon'
          x=fmincon(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss),varVals./varW,[],[],[],[],...
            obj.varLimits(1,:)./varW,obj.varLimits(2,:)./varW,[],opts);
%           x = simulannealbnd(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss),varVals./varW,obj.varLimits(1,:)./varW,obj.varLimits(2,:)./varW,opts);
        case 'fgoalattain'
          x=fgoalattain(@(x) minFunc(obj,x,obj.dotrack,obj.dotwiss),varVals,obj.matchVals,obj.matchWeights,...
            [],[],[],[],obj.varLimits(1,:),obj.varLimits(2,:),[],opts);
        case 'gradDrop'
          cnstr=obj.optimVals; outTol=abs(cnstr-obj.matchVals)>obj.matchWeights;
          t0=clock;
          while any(outTol)
            hTol=cnstr>obj.matchVals; corVal=zeros(size(cnstr));
            corVal(outTol&hTol)=corVal(outTol&hTol)-obj.matchWeights(outTol&hTol);
            corVal(outTol&~hTol)=corVal(outTol&~hTol)+obj.matchWeights(outTol&~hTol);
            obj.createLinMatrix;
            obj.varVals = obj.varVals + (obj.linMat\corVal')' ;
            cnstr=obj.optimVals; outTol=abs(cnstr-obj.matchVals)>obj.matchWeights;
            if obj.verbose && etime(clock,t0) > obj.optimDisplayRate
              obj.display
              t0=clock;
            end
          end
          return
        otherwise
          error('Unknown or unsupported optimizer');
      end
      if strcmp(obj.optim,'fgoalattain') || strcmp(obj.optim,'lscov')
        obj.varVals=x;
      else
        obj.varVals=x.*varW;
      end
    end
    function out=get.iMatch(obj)
      out=unique(obj.matchInd);
    end
    function out=get.optimData(obj)
      out=minFunc(obj,'getvals');
    end
    function out=get.dotrack(obj)
      % If want to use fit data instead of tracking, indicate this by
      % returning negative value for dotrack property
      if obj.useFitData
        % must have some data
        if ~isempty(obj.lookupMat)
          out=-1;
          return
        end
      end
      % Check for match requirements to see if tracking is required
      if any(ismember(obj.matchType,{'Sigma' 'Qloss' 'SigmaGauss' 'SigmaFit' 'User' 'T' 'U' 'NEmit_x' 'NEmit_y' 'Waist_x' 'Waist_y' 'Disp_x' 'Disp_y'}))
        out=1;
      else
        out=0;
      end
    end
    function out=get.dotwiss(obj)
      % Check for match requirements to see if twiss calculation is required
      if any(ismember(obj.matchType,{'alpha_x' 'alpha_y' 'beta_x' 'beta_y' 'eta_x' 'etap_x' 'eta_y' 'etap_y' 'nu_x' 'nu_y' 'modnu_x' 'modnu_y' 'R' 'SumBeta' 'AbsDiffBeta'}))
        out=1;
        if any(ismember(obj.matchType,{'W_x' 'W_y' 'Wsum_x' 'Wsum_y'}))
          out=2;
        end
      elseif any(ismember(obj.matchType,{'W_x' 'W_y' 'Wsum_x' 'Wsum_y'}))
        out=2;
      else
        out=0;
      end
    end
    function vals=get.optimVals(obj)
      varW=(obj.varLimits(2,:)-obj.varLimits(1,:));
      if strcmp(obj.optim,'fgoalattain') || (strcmp(obj.optim,'lscov'))
        minFunc(obj,obj.varVals,obj.dotrack,obj.dotwiss);
      else
        minFunc(obj,obj.varVals./varW,obj.dotrack,obj.dotwiss);
      end
      v=obj.minFunc('getvals');
      vals=v.func;
    end
    function vals=get.ipLastBeam(obj)
      v=obj.minFunc('getvals');
      vals=v.bs.Beam;
    end
    function vals=get.varVals(obj)
      global BEAMLINE PS GIRDER KLYSTRON
      for itype=1:length(obj.varType)
        ind=obj.varIndex{itype}(1);
        switch obj.varType{itype}
          case 'BEAMLINE'
            vals(itype)=BEAMLINE{ind}.(obj.varField{itype})(obj.varFieldIndex(itype));
          case 'PS'
            vals(itype)=PS(ind).(obj.varField{itype})(obj.varFieldIndex(itype));
          case 'GIRDER'
            vals(itype)=GIRDER{ind}.(obj.varField{itype})(obj.varFieldIndex(itype));
          case 'KLYSTRON'
            vals(itype)=KLYSTRON(ind).(obj.varField{itype})(obj.varFieldIndex(itype));
          case 'BUMP'
            vals(itype)=obj.BUMP(ind).val;
          case 'INITIAL'
            vals(itype)=obj.initStruc.(obj.varField{itype}(end)).Twiss.(obj.varField{itype}(1:end-1));
        end
      end
    end
    function set.optim(obj,val)
      if ismember(val,obj.allowedOptimMethods)
        obj.optim=val;
      else
        error('Not a supported optimisation method')
      end
    end
    function set.varVals(obj,x)
      % Set variables
      global BEAMLINE PS GIRDER KLYSTRON
      for itype=1:length(obj.varType)
        for ind=obj.varIndex{itype}
          switch obj.varType{itype}
            case 'BEAMLINE'
              if strcmp(BEAMLINE{ind}.Class,'SBEN') && strcmp(obj.varField{itype},'B')
                B0=BEAMLINE{ind}.B(1); brat=x(itype)/B0;
                BEAMLINE{ind}.Angle=BEAMLINE{ind}.Angle*brat;
                BEAMLINE{ind}.EdgeAngle=BEAMLINE{ind}.EdgeAngle.*brat;
                BEAMLINE{ind}.B=BEAMLINE{ind}.B.*brat;
              else
                BEAMLINE{ind}.(obj.varField{itype})(obj.varFieldIndex(itype))=x(itype);
              end
            case 'PS'
              PS(ind).(obj.varField{itype})(obj.varFieldIndex(itype))=x(itype);
              if strcmp(obj.varField{itype},'SetPt'); PSTrim(ind); end;
            case 'GIRDER'
              GIRDER{ind}.(obj.varField{itype})(obj.varFieldIndex(itype))=x(itype);
              if strcmp(obj.varField{itype},'MoverSetPt'); MoverTrim(ind); end;
            case 'KLYSTRON'
              KLYSTRON(ind).(obj.varField{itype})(obj.varFieldIndex(itype))=x(itype);
              if strcmp(obj.varField{itype},'AmplSetPt') || strcmp(obj.varField{itype},'PhaseSetPt')
                KlystronTrim(ind);
              end
            case 'BUMP'
              bsize=x(itype)-obj.BUMP(ind).val;
              obj.applyBump(ind,obj.varField{itype},bsize);
            case 'INITIAL'
              obj.initStruc.(obj.varField{itype}(end)).Twiss.(obj.varField{itype}(1:end-1))=x(itype);
          end
        end
      end
    end
    function createLinMatrix(obj)
      % Create linear response matrix between variables and constraints
      iVals=obj.varVals; % get initial variable values
      iConst=obj.optimVals; % get initial constraint values
      % get 3 point measurements around inital variable values
      stepVals=(obj.varLimits(1,:)-obj.varLimits(2,:))/obj.linstep;
      lval=cell(1,length(iVals));
      for vval=1:length(iVals)
        newVals=iVals; newVals(vval)=iVals(vval)-stepVals(vval);
        obj.varVals=newVals;
        lval{vval}=obj.optimVals;
      end
      hval=cell(1,length(iVals));
      for vval=1:length(iVals)
        newVals=iVals; newVals(vval)=iVals(vval)+stepVals(vval);
        obj.varVals=newVals;
        hval{vval}=obj.optimVals;
      end
      % Set variables back to initial values
      obj.varVals=iVals;
      % fit slopes and form response matrix
      for vval=1:length(iVals)
        for cval=1:length(iConst)
          p=polyfit(iVals(vval)+[-stepVals(vval) 0 stepVals(vval)],...
            [lval{vval}(cval) iConst(cval) hval{vval}(cval)],1) ;
          obj.linMat(cval,vval)=p(1);
        end
      end
    end
    function createLookup(obj)
      if obj.createLookupNdims>1
        iVals=obj.varVals; % get initial variable values
        iConst=obj.optimVals; % get initial constraint values
        lookupMat=zeros(obj.nLookupSteps,length(iConst)+length(iVals));
        lookupCount=1;
        Mt=obj;
        while lookupCount<obj.nLookupSteps
          spmd
            v=Mt.varLimits(1,:)+rand(size(Mt.varVals)).*(Mt.varLimits(2,:)-Mt.varLimits(1,:));
            Mt.varVals=v;
            c=Mt.optimVals;
          end
          lookupMat(lookupCount:lookupCount+length(c)-1,:)=[cell2mat(c(:)) cell2mat(v(:))];
          lookupCount=lookupCount+length(c);
        end
        sz=size(lookupMat);
        if sz(1)>obj.nLookupSteps
          obj.lookupMat=lookupMat(1:obj.nLookupSteps,:);
        else
          obj.lookupMat=lookupMat;
        end
        vvals=[]; %#ok<NASGU>
      else
        nscan=obj.nLookupSteps;
        ivals=obj.varVals;
        varlims=obj.varLimits;
        lookupMat=zeros(length(ivals),nscan,length(obj.optimVals));
        iConst=obj.optimVals;
        M=obj;
        varCount=0;
        vvals=cell(1,length(ivals));
        while varCount<length(ivals)
          spmd % ivar=1:length(ivals)
            ivar=labindex+varCount;
            if ivar <= length(ivals)
              vvals_p=linspace(varlims(1,ivar),varlims(2,ivar),nscan);
              for ival=1:length(vvals_p)
                vals=ivals; vals(ivar)=vvals_p(ival);
                M.varVals=vals;
                lookupMat_p(ival,:)=M.optimVals-iConst;
              end
              M.varVals=ivals;
              filledVals=labindex+varCount;
            else
              filledVals=0;
            end
          end
          fvals=cell2mat(filledVals(:));
          thisInd=fvals(fvals~=0);
          varCount=varCount+length(thisInd);
          for iind=1:length(filledVals)
            if filledVals{iind}
              lookupMat(filledVals{iind},:,:)=lookupMat_p{iind};
              vvals{filledVals{iind}}=vvals_p{iind};
            end
          end
        end
        obj.lookupMat=lookupMat;
        obj.lookupVarVals=vvals;
        obj.lookupCnstVals=iConst;
      end
      save(obj.lookupSaveLoc,'lookupMat','iConst','vvals') ;
    end
    function loadLookup(obj)
      ld=load(obj.lookupSaveLoc,'lookupMat','iConst','vvals') ;
      obj.lookupMat=ld.lookupMat;
      obj.lookupVarVals=ld.vvals;
      obj.lookupCnstVals=ld.iConst;
    end
    function rmVariable(obj,ind)
      obj.varType(ind)=[];
      obj.varFieldIndex(ind)=[];
      obj.varIndex(ind)=[];
      obj.varField(ind)=[];
      obj.varLimits(:,ind)=[];
      obj.linMat=[];
      obj.minFunc('clear');
    end
    function rmMatch(obj,ind)
      obj.matchTypeQualifiers(ind)=[];
      obj.matchType(ind)=[];
      obj.matchWeights(ind)=[];
      obj.matchVals(ind)=[];
      obj.matchInd(ind)=[];
      obj.linMat=[];
    end
  end
  methods(Access=protected)
    function applyBump(obj,bumpInd,field,val)
      % APPLYBUMP - change the size of a closed orbit bump
      % applyBump(obj,bumpInd,field,val)
      %
      % bumpInd = index into obj.BUMP bump definition
      % field = 'SetPt' or 'Ampl' for the PS elements of the bump
      % val = relative change in bump amplitude
      global PS BEAMLINE
      B=obj.BUMP(bumpInd);
      % ensure PS in radian units
      unitChange=false;
      for ips=B.corPS
        unit=obj.Cb*BEAMLINE{PS(ips).Element(1)}.P/length(PS(ips).Element);
        if BEAMLINE{PS(ips).Element(1)}.B~=unit
          Bch=unit/BEAMLINE{PS(ips).Element(1)}.B;
          BEAMLINE{PS(ips).Element(1)}.B=unit;
          unitChange=true;
        end
      end
      if unitChange
        PS(ips).Ampl=PS(ips).Ampl*(1/Bch);
        PS(ips).SetPt=PS(ips).SetPt*(1/Bch);
      end
      for icor=1:length(obj.BUMP(bumpInd).corPS)
        PS(obj.BUMP(bumpInd).corPS(icor)).(field)=PS(obj.BUMP(bumpInd).corPS(icor)).(field)+val*obj.BUMP(bumpInd).coefs(icor) ;
      end
      if strcmp(field,'SetPt')
        PSTrim(obj.BUMP(bumpInd).corPS) ;
      end
      obj.BUMP(bumpInd).val=obj.BUMP(bumpInd).val+val;
    end
    function [stop,options,optchanged]=gaOptimOutFun(obj,options,~,~,~)
      optchanged=false;
      stop = optimOutFun(obj);
    end
    function stop = optimOutFun(obj,~,optimval,~)
      persistent lastUpdate
      stop=false;
      if ~strcmp(obj.optim,'fgoalattain')
        if (isfield(optimval,'fval') && optimval.fval<1) || (isfield(optimval,'resnorm') && optimval.resnorm<1)
          stop=true;
        end
      end
      if obj.verbose && (isempty(lastUpdate) || etime(clock,lastUpdate)>obj.optimDisplayRate)
        display(obj);
        lastUpdate=clock;
      end
    end
    function F = doLookup(obj,x)
      sz=size(obj.lookupMat);
      F=obj.lookupCnstVals;
      for ix=1:length(x)
        for ic=1:sz(3)
          F(ic)=F(ic)+interp1(obj.lookupVarVals{ix},squeeze(obj.lookupMat(ix,:,ic)),x(ix));
        end
      end
    end
    function F = minFunc(obj,x,dotrack,dotwiss,varW,varVals)
      % The minimizer function
      % dotrack=-1 : use lookup table
      persistent lastvals bs
      
      % just get last match values
      if isequal(x,'getvals')
        F.func=lastvals;
        F.bs=bs;
        return
      elseif isequal(x,'clear')
        lastvals=[];
        bs=[];
        return
      end
      
      % Undo variable normalisation
      if ~strcmp(obj.optim,'fgoalattain')
        vm=(obj.varLimits(2,:)-obj.varLimits(1,:));
        x=x.*vm;
      end
      try
        % If using lookup table, do that now and exit
        if dotrack<0
          F=obj.doLookup(x);
          lastvals=F;

          if strcmp(obj.optim,'lsqnonlin') || strcmp(obj.optim,'moga')
            F=((F-obj.matchVals)./obj.matchWeights);
            if exist('varW','var')
              F(end+1:end+length(varW))=(x-varVals)./varW;
            end
          elseif ~strcmp(obj.optim,'fgoalattain')
            F=sum(((F-obj.matchVals)./obj.matchWeights).^2);
          end
          return
        end

        % Set the variables
        obj.varVals=x;
        
        % Get twiss in correct format
        if dotwiss && ~(dotrack<0)
          Ix.beta=obj.initStruc.x.Twiss.beta;
          Ix.alpha=obj.initStruc.x.Twiss.alpha;
          Ix.eta=obj.initStruc.x.Twiss.eta;
          Ix.etap=obj.initStruc.x.Twiss.etap;
          Ix.nu=obj.initStruc.x.Twiss.nu;
          Iy.beta=obj.initStruc.y.Twiss.beta;
          Iy.alpha=obj.initStruc.y.Twiss.alpha;
          Iy.eta=obj.initStruc.y.Twiss.eta;
          Iy.etap=obj.initStruc.y.Twiss.etap;
          Iy.nu=obj.initStruc.y.Twiss.nu;
        end
        
        % Set the variables
        obj.varVals=x;

        % Get the data (do the tracking)
        F=zeros(1,length(obj.matchType));
        for itrack=1:length(obj.iMatch)
          try
            if itrack==1
              if dotrack
                [~, beamout]=TrackThru(obj.iInitial,obj.iMatch(1),obj.beam,1,1,0);
                stop=beamout.Bunch.stop;
%                 if stat{1}~=1; error('Error in tracking in minimiser function: %s',stat{2}); end;
              end
              if dotwiss
                if dotwiss>1
                  [stat, T]=GetTwissW(obj.iInitial,obj.iMatch(1),Ix,Iy);
                else
                  [stat, T]=GetTwiss(obj.iInitial,obj.iMatch(1),Ix,Iy);
                end
                if stat{1}~=1; error('Error in tracking in minimiser function: %s',stat{2}); end;
              end
            else
              if dotrack
                [~, beamout]=TrackThru(obj.iMatch(itrack-1),obj.iMatch(itrack),beamout,1,1,0);
                stop=beamout.Bunch.stop;
%                 if stat{1}~=1; error('Error in tracking in minimiser function: %s',stat{2}); end;
              end
              if dotwiss
                %             Ix.beta=T.betax; Ix.alpha=T.alphax; Ix.eta=T.etax; Ix.etap=T.etapx; Ix.nu=T.nux;
                %             Iy.beta=T.betay; Iy.alpha=T.alphay; Iy.eta=T.etay; Iy.etap=T.etapy; Iy.nu=T.nuy;
                if dotwiss>1
                  [stat, T]=GetTwissW(obj.iInitial,obj.iMatch(itrack),Ix,Iy);
                else
                  [stat, T]=GetTwiss(obj.iInitial,obj.iMatch(itrack),Ix,Iy);
                end
                if stat{1}~=1; error('Error in tracking in minimiser function: %s',stat{2}); end;
              end
            end
          catch
            F=ones(size(F)).*1e10;
            continue
          end
          % Extract the data
          for itype=1:length(obj.matchType)
            if obj.matchInd(itype)~=obj.iMatch(itrack); continue; end;
            switch obj.matchType{itype}
              case 'SumBeta'
                F(itype)=T.betax(end)+T.betay(end);
              case 'AbsDiffBeta'
                F(itype)=abs(T.betax(end)-T.betay(end));
              case 'MinBeta'
                F(itype)=min([T.betax(end),T.betay(end)]);
              case 'MaxBeta'
                F(itype)=max([T.betax(end),T.betay(end)]);
              case 'R'
                [~,R]=RmatAtoB(obj.iInitial,obj.iMatch(itrack));
                F(itype)=R(str2double(obj.matchTypeQualifiers{itype}(1)),str2double(obj.matchTypeQualifiers{itype}(2)));
              case 'alpha_x'
                F(itype)=T.alphax(end);
              case 'alpha_y'
                F(itype)=T.alphay(end);
              case 'beta_x'
                F(itype)=T.betax(end);
              case 'beta_y'
                F(itype)=T.betay(end);
              case 'eta_x'
                F(itype)=T.etax(end);
              case 'eta_y'
                F(itype)=T.etay(end);
              case 'etap_x'
                F(itype)=T.etapx(end);
              case 'etap_y'
                F(itype)=T.etapy(end);
              case 'W_x'
                F(itype)=T.Wx(end);
              case 'W_y'
                F(itype)=T.Wy(end);
              case 'Wsum_x'
                F(itype)=sum(T.Wx);
              case 'Wsum_y'
                F(itype)=sum(T.Wy);
              case 'modnu_x'
                if length(obj.matchTypeQualifiers{itype})>1
                  F(itype)=mod(T.nux(end)-obj.matchTypeQualifiers{itype}(1),obj.matchTypeQualifiers{itype}(2));
                else
                  F(itype)=mod(T.nux(end),obj.matchTypeQualifiers{itype});
                end
              case 'modnu_y'
                if length(obj.matchTypeQualifiers{itype})>1
                  F(itype)=mod(T.nuy(end)-obj.matchTypeQualifiers{itype}(1),obj.matchTypeQualifiers{itype}(2));
                else
                  F(itype)=mod(T.nuy(end),obj.matchTypeQualifiers{itype});
                end
              case 'nu_x'
                F(itype)=T.nux(end);
              case 'nu_y'
                F(itype)=T.nuy(end);
              case 'NEmit_x'
                if ~exist('nx','var')
                  [nx,ny] = GetNEmitFromBeam( beamout ,1);
                end
                F(itype)=nx;
              case 'NEmit_y'
                if ~exist('ny','var')
                  [nx,ny] = GetNEmitFromBeam( beamout ,1);
                end
                F(itype)=ny;
              case 'Qloss'
                F(itype) = sum(beamout.Bunch.stop~=0)/length(beamout.Bunch.Q) ;
              case 'Sigma'
                if ~exist('S','var')
                  S=cov(beamout.Bunch.x(:,~stop)');
                end
                F(itype)=S(str2double(obj.matchTypeQualifiers{itype}(1)),str2double(obj.matchTypeQualifiers{itype}(2)));
              case 'SigmaGauss'
                if ~exist('fitCoef','var') || dim~=str2double(obj.matchTypeQualifiers{itype}(1))
                  dim=str2double(obj.matchTypeQualifiers{itype}(1));
                  [fitTerm,fitCoef,bsizecor] = beamTerms(dim,beamout);
                end
                F(itype)=bsizecor(end);
              case 'SigmaFit'
                dim=str2double(obj.matchTypeQualifiers{itype}(1));
                nbin=max([length(beamout.Bunch.Q)/100 100]);
                [ fx , bc ] = hist(beamout.Bunch.x(dim,~stop),nbin);
                [~, q] = gauss_fit(bc,fx) ;
                F(itype)=abs(q(4));
              case 'User'
                F(itype) = obj.userFitFun(beamout);
              case {'T' 'U'}
                if ~exist('fitCoef','var') || dim~=str2double(obj.matchTypeQualifiers{itype}(1))
                  dim=str2double(obj.matchTypeQualifiers{itype}(1));
                  [fitTerm,fitCoef] = beamTerms(dim,beamout);
                end
                term=zeros(1,6);
                for iterm=2:length(obj.matchTypeQualifiers{itype})
                  term(str2double(obj.matchTypeQualifiers{itype}(iterm)))=term(str2double(obj.matchTypeQualifiers{itype}(iterm)))+1;
                end
                F(itype)=fitCoef(arrayfun(@(x) isequal(fitTerm(x,:),term),1:length(fitTerm)));
              case {'Waist_x' 'Waist_y'}
                S=cov(beamout.Bunch.x(:,~stop)');
                R=diag(ones(1,6));L=zeros(6,6);L(1,2)=1;L(3,4)=1;
                if strcmp(obj.matchType{itype},'Waist_x')
                  F(itype)=fminsearch(@(x) minWaist(obj,x,R,L,S,1),0,optimset('Tolx',1e-6,'TolFun',0.1e-6^2));
                else
                  F(itype)=fminsearch(@(x) minWaist(obj,x,R,L,S,3),0,optimset('Tolx',1e-6,'TolFun',0.1e-6^2));
                end
              case {'Disp_x' 'Disp_y'}
                [Tx, Ty]=GetUncoupledTwissFromBeamPars(beamout,1);
                if strcmp(obj.matchType{itype},'Disp_x')
                  F(itype)=Tx.eta;
                else
                  F(itype)=Ty.eta;
                end
            end
          end

          % Beamsize data
          if dotrack
            bs.Beam=beamout;
            bs.x(itrack)=std(beamout.Bunch.x(1,~stop));
            bs.y(itrack)=std(beamout.Bunch.x(3,~stop));
            bs.z(itrack)=std(beamout.Bunch.x(5,~stop));
          else
            bs.Beam=[];
            bs.x(itrack)=sqrt((obj.initStruc.x.NEmit/(obj.initStruc.Momentum/0.511e-3))*T.betax(end));
            bs.y(itrack)=sqrt((obj.initStruc.y.NEmit/(obj.initStruc.Momentum/0.511e-3))*T.betay(end));
            bs.z=obj.initStruc.sigz;
          end
        end
      catch
        F=ones(size(obj.matchVals)).*1e60;
      end

      % Subtract desired values so minimiser does the correct thing
      % And apply weights
      lastvals=F;
      
      if strcmp(obj.optim,'lsqnonlin') || strcmp(obj.optim,'moga')
        F=((F-obj.matchVals)./obj.matchWeights);
        if exist('varW','var')
          F(end+1:end+length(varW))=(x-varVals)./varW;
        end
      elseif ~strcmp(obj.optim,'fgoalattain')
        F=sum(((F-obj.matchVals)./obj.matchWeights).^2);
      end
      if any(isnan(F)) || any(isinf(F))
        F=ones(size(F)).*1e60;
      end
    end
    function chi2 = minWaist(~,x,R,L,sig,dir)
      newsig=(R+L.*x(1))*sig*(R+L.*x(1))';
      chi2=newsig(dir,dir)^2;
    end
  end
  methods(Static)
    function bumpcoefs = getBumpCoef(bumpcorinds, bumppos, bumptype)
      
      % calculate coefficients for 3-corrector bump (bumpcorinds are PS indices)
      % NOTE - bumppos must be between corrector 1 and 2
      % bumptype =1:4 = x/x'/y/y'
      
      global BEAMLINE PS
      
      % correctors aren't split in half ... use a half-corrector-length drift to get
      % (approximately) to each corrector's center
      L=sum(arrayfun(@(x) BEAMLINE{x}.L,PS(bumpcorinds(1)).Element)); % corrector 1 length
      cormat1=eye(6);
      cormat1(1,2)=L/2;cormat1(3,4)=L/2; % half-length drift
      L=sum(arrayfun(@(x) BEAMLINE{x}.L,PS(bumpcorinds(2)).Element)); % corrector 2 length
      cormat2=eye(6);
      cormat2(1,2)=L/2;cormat2(3,4)=L/2; % half-length drift
      L=sum(arrayfun(@(x) BEAMLINE{x}.L,PS(bumpcorinds(3)).Element)); % corrector 3 length
      cormat3=eye(6);
      cormat3(1,2)=L/2;cormat3(3,4)=L/2; % half-length drift
      
      % corrector 1 to bumppos
      [stat,Amat]=RmatAtoB(PS(bumpcorinds(1)).Element(end)+1,bumppos); % corrector 1 exit to bumppos
      if (stat{1}~=1),error(stat{2}),end
      Amat=Amat*cormat1; % corrector 1 center to bumppos
      
      % corrector 1 to corrector 3
      [stat,Bmat]=RmatAtoB(PS(bumpcorinds(1)).Element(end)+1,PS(bumpcorinds(3)).Element(1)-1); % corrector 1 exit to corrector 3 entrance
      if (stat{1}~=1),error(stat{2}),end
      Bmat=cormat3*Bmat*cormat1; % corrector 1 center to corrector 3 center
      
      % corrector 2 to corrector 3
      [stat,Cmat]=RmatAtoB(PS(bumpcorinds(2)).Element(end)+1,PS(bumpcorinds(3)).Element(1)-1); % corrector 2 exit to corrector 3 entrance
      if (stat{1}~=1),error(stat{2}),end
      Cmat=cormat3*Cmat*cormat2; % corrector 2 center to corrector 3 center
      
      switch bumptype
        case 1
          bumpmat = [...
            Amat(1,2) 0         0;
            Bmat(1,2) Cmat(1,2) 0;
            Bmat(2,2) Cmat(2,2) 1];
        case 2
          bumpmat = [...
            Amat(2,2) 0         0;
            Bmat(1,2) Cmat(1,2) 0;
            Bmat(2,2) Cmat(2,2) 1];
        case 3
          bumpmat = [...
            Amat(3,4) 0         0;
            Bmat(3,4) Cmat(3,4) 0;
            Bmat(4,4) Cmat(4,4) 1];
        case 4
          bumpmat = [...
            Amat(4,4) 0         0;
            Bmat(3,4) Cmat(3,4) 0;
            Bmat(4,4) Cmat(4,4) 1];
      end
      
      bumpcoefs = bumpmat \ [1; 0; 0];
    end
  end
end

