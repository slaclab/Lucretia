classdef ExtGeometry < handle
  %EXTGEOMETRY - Class to descibe geometry required for operation of
  %external processes (e.g. for GEANT4)
  
  properties(Constant, Hidden)
    allowedGeometryTypes={'Rectangle','Ellipse','Tapered','GDML'};
  end
  properties(SetAccess=protected)
    allowedMaterials={'Vacuum','User1','User2','User3'};
  end
  properties(SetAccess=protected)
    GeometryType='Ellipse'; % type of Geometry (e.g. 'Rectangle', 'Ellipse' collimator or 'GDML')
    AperX=1; % Inside half-aperture of shape (m) - Horizontal dimension
    AperY=1; % Inside half-aperture of shape (m) - Vertical dimension
    AperX2=0; % Secondary aperture definition (m) (for Tapered type)
    AperY2=0; % Secondary aperture definition (m) (for Tapered type)
    AperX3=0; % Secondary aperture definition (m) (for Tapered type)
    AperY3=0; % Secondary aperture definition (m) (for Tapered type)
    CollDX=1; % Collimator half-width (m) (for Tapered type)
    CollDY=1; % Collimator half-height (m) (for Tapered type)
    CollLen2=0; % Aux collimator length (for Tapered type)
    Thickness=1; % Geometry thickness (m) - defines half-aperture of World box
    Material='Vacuum'; % material type
    Material2='Vacuum'; % secondary material type
    VacuumMaterial='Vacuum'; % material to associate with the vacuum interior to the defined aperture
    UserMaterial=[] ; % Container to hold user-defined materials
    MaterialPressure=0; % 0 = use STP
    MaterialTemperature=0; % 0 = use STP
    Material2Pressure=0; % 0 = use STP
    Material2Temperature=0; % 0 = use STP
  end
  properties
    GDMLFile='None.gdml'; % Use GDML input file
  end
  
  methods
    function obj=ExtGeometry(varargin)
      % Check inputs
      if mod(nargin,2)
        error('Must supply property, value pairs as creation arguments')
      end
      % parse inputs
      for iarg=1:2:nargin
        if isprop(ExtGeometry,varargin{iarg})
          obj.(varargin{iarg})=varargin{iarg+1};
        else
          error('No such property: %s',varargin{iarg})
        end
      end
      try
        obj.checkExtGeometryProps();
      catch ME
        error('Error constructing geometry:\n%',ME.message)
      end
      % Initialise UserMaterial structure
      for id=1:3
        obj.UserMaterial(id).Density=1e-19;
        obj.UserMaterial(id).Pressure=0;
        obj.UserMaterial(id).Temperature=0;
        obj.UserMaterial(id).State='Gas';
        obj.UserMaterial(id).NumComponents=1;
        obj.UserMaterial(id).Element(1).Name='Hydrogen';
        obj.UserMaterial(id).Element(1).Symbol='H';
        obj.UserMaterial(id).Element(1).Z=1.0;
        obj.UserMaterial(id).Element(1).A=1.0;
        obj.UserMaterial(id).Element(1).FractionMass=1.0;
      end
    end
    function SetUserMaterial(obj,id,density,pressure,temperature,state,numComponents)
      % SetUserMaterial(id,density,pressure,temperature,state,numComponents)
      %  Set the User material properties
      %   id = id of UserMaterial (integer from 1 to 3), reference using 'User1', 'User2' or 'User3' in material name fields
      %   density / g/cm^3
      %   pressure / pascals (0=STP)
      %   temperature / Kelvin (0=STP)
      %   state = 'Solid', 'Liquid' or 'Gas'
      %   numComponents = number of elemental components making up this material
      if id<0 || id>3
        error('Max 3 id slots for UserMaterial: %s',evalc('help ExtGeometry.SetUserMaterial'))
      end
      if density<=0 || pressure<0 || temperature<0 || numComponents<=0
        error('density, numComponents must be >0 (P,T >=0): %s',evalc('help ExtGeometry.SetUserMaterial'))
      end
      if ~ismember(state,{'Solid','Liquid','Gas'})
        error('state must be ''Solid'',''Liquid'' or ''Gas'': %s',evalc('help ExtGeometry.SetUserMaterial'))
      end
      obj.UserMaterial(id).Density=density;
      obj.UserMaterial(id).Pressure=pressure;
      obj.UserMaterial(id).Temperature=temperature;
      obj.UserMaterial(id).State=state;
      obj.UserMaterial(id).NumComponents=floor(numComponents);
      for iele=1:floor(numComponents)
        obj.UserMaterial(id).Element(iele).Name='Hydrogen';
        obj.UserMaterial(id).Element(iele).Symbol='H';
        obj.UserMaterial(id).Element(iele).Z=1.0;
        obj.UserMaterial(id).Element(iele).A=1.0;
        obj.UserMaterial(id).Element(iele).FractionMass=1/floor(numComponents);
      end
    end
    function SetMaterialPressure(obj,id,pressure)
      if pressure<0
        error('Set pressure >=0')
      end
      if ~ismember(id,[1 2])
        error('id= 1 or 2');
      end
      if id==1
        obj.MaterialPressure=pressure;
      else
        obj.Material2Pressure=pressure;
      end
    end
    function SetMaterialTemperature(obj,id,temp)
      if temp<0
        error('Set temp >=0')
      end
      if ~ismember(id,[1 2])
        error('id= 1 or 2');
      end
      if id==1
        obj.MaterialTemperature=temp;
      else
        obj.Material2Temperature=temp;
      end
    end
    function SetUserMaterialElements(obj,id,names,symbols,Z,A,fractionMass)
      % SetUserMaterialElement(id,names,symbols,Z,A,fractionMass)
      %  Set elemental components of User Material
      %    id = id of UserMaterial (from 1 to 3)
      %    names = cell array of elemental names (length == UserMaterial(id).NumComponents)
      %    symbols = cell array of elemental symbol strings (length == UserMaterial(id).NumComponents)
      %    Z = vector of atomic numbers (length == UserMaterial(id).NumComponents)
      %    A = vector of atomic masses / g/mole (length == UserMaterial(id).NumComponents)
      %    fractionMass = vector of mass fractions (must sum to 1 and length == UserMaterial(id).NumComponents)
      if id<0 || id>3
        error('Max 3 id slots for UserMaterial')
      end
      if length(names)~=obj.UserMaterial(id).NumComponents || length(names)~=length(symbols) || ...
          length(names)~=length(Z) || length(names)~=length(A) || sum(fractionMass)~=1 || ~iscell(names) || ...
          ~iscell(symbols)
        error('Badly specified parameters: %s',evalc('help ExtGeometry.SetUserMaterialElement'))
      end
      for iele=1:obj.UserMaterial(id).NumComponents
        obj.UserMaterial(id).Element(iele).Name=names{iele};
        obj.UserMaterial(id).Element(iele).Symbol=symbols{iele};
        obj.UserMaterial(id).Element(iele).Z=Z(iele);
        obj.UserMaterial(id).Element(iele).A=A(iele);
        obj.UserMaterial(id).Element(iele).FractionMass=fractionMass(iele);
      end
    end
    function SetAper(obj,val1,val2)
      if ~exist('val1','var') || ~exist('val2','var')
        error('Must provide 2 values for AperX and AperY')
      end
      if any([val1 val2]<0)
        error('Apertures must be >=0')
      end
      obj.AperX=val1;
      obj.AperY=val2;
    end
    function SetAper2(obj,val1,val2)
      if ~exist('val1','var') || ~exist('val2','var')
        error('Must provide 2 values for AperX2 and AperY2')
      end
      if any([val1 val2]<0)
        error('Apertures must be >=0')
      end
      if any([val1 val2]>1)
        error('Apertures must be <=1')
      end
      obj.AperX2=val1;
      obj.AperY2=val2;
    end
    function SetAper3(obj,val1,val2)
      if ~exist('val1','var') || ~exist('val2','var')
        error('Must provide 2 values for AperX3 and AperY3')
      end
      if any([val1 val2]<0)
        error('Apertures must be >=0')
      end
      if any([val1 val2]>1)
        error('Apertures must be <=1')
      end
      obj.AperX3=val1;
      obj.AperY3=val2;
    end
    function SetGeometryType(obj,type)
      if ~exist('type','var') || ~ischar(type)
        error('Must provide Geometry Type String')
      end
      if ~any(strcmp(type,obj.allowedGeometryTypes))
        disp('Allowed Types:')
        disp(obj.allowedGeometryTypes)
        error('Must choose from list of allowed Geometry types ^')
      end
      obj.GeometryType=type;
    end
    function SetMaterial(obj,material)
      if ~any(strcmp(material,obj.allowedMaterials))
        error('Material not found in database');
      end
      obj.Material=material;
    end
    function SetVacuumMaterial(obj,material)
      if ~any(strcmp(material,obj.allowedMaterials))
        error('Material not found in database');
      end
      obj.VacuumMaterial=material;
    end
    function SetMaterial2(obj,material)
      if ~any(strcmp(material,obj.allowedMaterials))
        error('Material not found in database');
      end
      obj.Material2=material;
    end
    function SetThickness(obj,val)
      if ~exist('val','var') || val<=0
        error('Must provide thickness parameter >0 (m)')
      end
      try
        obj.SetCollDX(obj.CollDX);
      catch
        obj.CollDX=val;
        if strcmp(obj.GeometryType,'Tapered')
          disp('WARNING: Setting thickness to < CollDX, changing CollDX=Thickness')
        end
      end
      try
        obj.SetCollDY(obj.CollDY);
      catch
        obj.CollDY=val;
        if strcmp(obj.GeometryType,'Tapered')
          disp('WARNING: Setting thickness to < CollDY, changing CollDY=Thickness')
        end
      end
      obj.Thickness=val;
    end
    function SetCollDX(obj,val)
      if ~exist('val','var') || val<=0 || val>obj.Thickness
        error('Must provide DX parameter >0 & <= obj.Thickness (m)')
      end
      obj.CollDX=val;
    end
    function SetCollDY(obj,val)
      if ~exist('val','var') || val<=0 || val>obj.Thickness
        error('Must provide DY parameter >0 & <= obj.Thickness (m)')
      end
      obj.CollDY=val;
    end
    function SetCollLen2(obj,val)
      if val<0 || val>1
        error('CollLen2 parameter must be [0:1] (fraction of element length)');
      end
      obj.CollLen2=val;
    end
    function checkExtGeometryProps(obj)
      obj.SetGeometryType(obj.GeometryType);
      obj.SetAper(obj.AperX,obj.AperY);
      obj.SetAper2(obj.AperX2,obj.AperY2);
      obj.SetAper3(obj.AperX3,obj.AperY3);
      obj.SetThickness(obj.Thickness);
      obj.SetMaterial(obj.Material);
      obj.SetMaterial2(obj.Material2);
      obj.SetCollDX(obj.CollDX);
      obj.SetCollDY(obj.CollDY);
    end
  end
  
end

