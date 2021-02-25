classdef ExtTunnelGeometry < handle
  % ExtTunnelGeometry - class for definining tunnel geometry surrounding a
  % BEAMLINE element. Supports cube world volume of given material, and a
  % main and service tunnel within. Geometry descriptions are provided by
  % the GEANT4 description of a generic trapezoid:
  % http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html#sect.Geom
  properties(SetAccess=protected)
    TunnelHeight=0;
    TunnelWidth=0;
    TunnelMaterial='None';
    BeamTunnelGeom=zeros(1,10);
    BeamTunnelPos=[0 0];
    ServiceTunnelGeom=zeros(1,10);
    ServiceTunnelPos=[0 0];
  end
  
  methods
    function SetTunnelHeight(obj,height)
      obj.TunnelHeight=height;
    end
    function SetTunnelWidth(obj,wid)
      obj.TunnelWidth=wid;
    end
    function SetTunnelMaterial(obj,mat)
      obj.TunnelMaterial=mat;
    end
    function SetBeamTunnelGeom(obj,trapArgs)
      % trapArgs: [theta,phi,dy1,dx1,dx2,alp1,dy2,dx3,dx4,alp2] (dz given
      % by element length) - see http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html
      if ~exist('trapArgs','var') || length(trapArgs)~=10
        error('Incorrect geometry definition')
      end
      obj.BeamTunnelGeom=trapArgs;
    end
    function SetBeamTunnelPos(obj,pos)
      % pos: [x,y]
      if ~exist('pos','var') || length(pos)~=2
        error('Incorrect geometry position')
      end
      obj.BeamTunnelPos=pos;
    end
    function SetServiceTunnelGeom(obj,trapArgs)
      % trapArgs: [theta,phi,dy1,dx1,dx2,alp1,dy2,dx3,dx4,alp2] (dz given
      % by element length) - see http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html
      if ~exist('trapArgs','var') || length(trapArgs)~=10
        error('Incorrect geometry definition')
      end
      obj.ServiceTunnelGeom=trapArgs;
    end
    function SetServiceTunnelPos(obj,pos)
      % pos: [x,y]
      if ~exist('pos','var') || length(pos)~=2
        error('Incorrect geometry position')
      end
      obj.ServiceTunnelPos=pos;
    end
  end
end

