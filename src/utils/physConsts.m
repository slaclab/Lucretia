classdef(Abstract) physConsts < handle
  % Definition of physics constants
  
  properties(Constant)
    emass=0.51099906e-3; % electron rest mass (GeV)
    pmass=0.9382720813; % proton rest mass (GeV)
    eQ=1.602176462e-19;  % |electron charge| (C)
    clight=2.99792458e8; % speed of light (m/sec)
    Cb=1e9/2.99792458e8; % rigidity conversion (T-m/GeV)
    Z0=376.730313667; % free space impedence (Ohms)
    mu0=4e-7*pi; % Vacuum permeability (H/m)
  end
end