classdef ExtMesh < handle
  % Generate and manage data mesh for statistics relating to external
  % processes
  
  properties(Constant, Hidden)
    meshParticleTypes={'e-' 'mu-' 'nu_e' 'nu_mu' 'nu_tau' 'tau-' 'e+' 'mu+' 'anti_nu_e' 'anti_nu_mu' 'anti_nu_tau' 'tau+' ...
      'pi0' 'pi+' 'pi-' 'kaon0' 'anti_kaon0' 'kaon+' 'kaon-' ...
      'neutron' 'ani_neutron' 'proton' 'anti-proton' ...
      'gamma' ...
      'all'};
    meshWeightTypes={'ke' 'mom' 'Edep' 'N'};
  end
  properties(SetAccess=private)
    meshInfo % container for mesh definition information, fields= type, ptypes, weight, s1, s2, s3, n1, n2, n3
    meshEntries % Number of defined meshes
  end
  properties
    meshData={}; % container for filled mesh data
  end
  methods
    function obj=ExtMesh()
      obj.meshEntries=uint8(0);
    end
    function AddMesh(obj,ptypes,weight,nx,ny,nz)
      % Get mesh ID
      % AddMesh(ptypes,weight,nx,ny,nz)
      % ptypes: char of particle type to bin data in mesh OR cell array of
      %   multiple particle types (see meshParticleTypes property)
      % weight: choose to weight binning by kinetic energy ('ke') (GeV), track
      %   momentum ('mom') (GeV/c), energy deposited in mesh cell (J) ('Edep') or just
      %   count tracks in cell ('N')
      % nx, ny, nz: number of cells in x, y and z dimensions (mesh is
      %   distributed as a box around element volume (max radius =
      %   aperture+thickness parameters)
      %     (Sizes are in m, energies in GeV)
      if isempty(obj.meshInfo)
        mID=1;
      else
        mID=length(obj.meshInfo)+1;
      end
      % Check particle types
      if ~exist('ptypes','var') || (~iscell(ptypes) && ~ischar(ptypes))
        error('Incorrect argument format')
      end
      if ~iscell(ptypes)
        ptypes={ptypes};
      end
      if ~all(ismember(ptypes,obj.meshParticleTypes))
        error('Unsupported particle types in list')
      end
      % Check weight type argument
      if ~exist('weight','var') || ~ischar(weight)
        error('Incorrect argument format')
      end
      if ~ismember(weight,obj.meshWeightTypes)
        error('Unsupported weight type')
      end
      % Parse mesh geometry
      obj.meshInfo(mID).n1=uint32(nx);
      obj.meshInfo(mID).n2=uint32(ny);
      obj.meshInfo(mID).n3=uint32(nz);
      % Parse other inputs
      obj.meshInfo(mID).weight=int32(bin2dec(num2str(ismember(obj.meshWeightTypes,weight))));
      obj.meshInfo(mID).ptypes=int32(bin2dec(strrep(num2str(ismember(obj.meshParticleTypes,ptypes)),' ','')));
      % Initialize mesh data
      obj.meshData{mID}=zeros(obj.meshInfo(mID).n1,obj.meshInfo(mID).n2,obj.meshInfo(mID).n3);
      % Increment mesh counter
      obj.meshEntries=obj.meshEntries+1;
    end
    function DeleteMesh(obj,mID)
      % DeleteMesh - delete a previously created mesh mID is array entry in
      %              meshInfo
      if length(obj.meshInfo)<mID
        error('Mesh ID does not exist')
      end
      obj.meshInfo(mID)=[];
      obj.meshData(mID)=[];
      % Decrement mesh counter
      obj.meshEntries=obj.meshEntries-1;
    end
  end
  
end

