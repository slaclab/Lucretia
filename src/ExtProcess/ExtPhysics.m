classdef ExtPhysics < handle
  % Details of Physics processes to apply to EXT processes
  
  properties(SetAccess=private)
    supportedProcessPhysics = {'msc','eIoni','eBrem','annihil','SynRad','phot','compt','conv','Rayl','muIoni','muBrems','muPairProd','AnnihiToMuPair','GammaToMuPair','ee2hadr','electronNuclear','positronNuclear','photonNuclear','muonNuclear','Decay'} ;
    supportedParticleTypes = {'gamma' 'e-' 'e+' 'mu-' 'mu+'} ;
  end
  properties(Dependent)
    processSelection
    particleCuts
    RandSeed
  end
  properties(Access=private)
    processSelectionVals
    particleCutVals=[1 1 1 NaN NaN]; % mm
    fRandSeed
  end
  properties
    Edep=0; % Total energy deposited in element / J
  end
  
  methods
    function obj=ExtPhysics()
      pv=false(length(obj.supportedParticleTypes),length(obj.supportedProcessPhysics));
      pv(1,[6:9 14 18])=true;
      pv(2,[1:3 5 16])=true;
      pv(3,[1:5 13 15 17])=true;
      pv(4,[1 10:12 19 20])=true;
      pv(5,[1 10:12 19 20])=true;
      obj.processSelectionVals=pv;
      obj.fRandSeed=uint32(ceil(rand*1e5));
    end
    function val=get.RandSeed(obj)
      val=obj.fRandSeed;
    end
    function set.RandSeed(obj,val)
      obj.fRandSeed=uint32(abs(val));
    end
    function val=get.processSelection(obj)
      val=obj.processSelectionVals;
    end
    function val=get.particleCuts(obj)
      val = obj.particleCutVals(~isnan(obj.particleCutVals));
    end
    function SetParticleCut(obj,ptype,cut)
      % SetParticleCuts(ptype,cut)
      %  ptype = char from supportedParticleTypes
      %  cut = particle cut in mm
      partID=ismember(obj.supportedParticleTypes,ptype);
      if ~any(partID) || any(isnan(obj.particleCutVals(partID)))
        error('Unsupported Particle Type or cut not available for this type: %s',ptype)
      end
      if cut<0
        error('Cut > 0')
      end
      obj.particleCutVals(partID)=cut;
    end
    function SelectProcess(obj,ptype,proc,select)
      % SetProcessSelection(ptype,proc)
      % ptype = char or cell of char listing supportedParticleTypes (can
      %  supply 'all')
      % proc = char or cell of char listing supportedProcessPhysics (can
      %  supply 'all')
      % select = true or false
      if isequal(ptype,'all')
        ptype=obj.supportedParticleTypes;
      end
      if isequal(proc,'all')
        proc=obj.supportedProcessPhysics;
      end
      if ~iscell(ptype)
        ptype={ptype};
      end
      if ~iscell(proc)
        proc={proc};
      end
      for ipart=1:length(ptype)
        for iproc=1:length(proc)
          procID=ismember(obj.supportedProcessPhysics,proc{iproc});
          partID=ismember(obj.supportedParticleTypes,ptype{ipart});
          if ~any(procID) || ~any(partID)
            error('Unsupported Process and/or Particle Type: %s / %s',proc{iproc},ptype{ipart})
          end
          obj.processSelectionVals(partID,procID)=logical(select);
        end
      end
    end
  end
  
end

