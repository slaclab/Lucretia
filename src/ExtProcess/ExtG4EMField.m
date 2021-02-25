classdef ExtG4EMField < handle & ExtEMField
  %EXTG4EMFIELD - GEANT4 implementation of EM fields
  
  properties(Constant,Hidden)
    StepMethods={'SimpleRunge','ClassicalRK4','CashKarpRKF45','ExplicitEuler','ImplicitEuler','SimpleHeum'};
    DefaultStepMethod='ClassicalRK4';
    Interpolators={'nearest','linear','cubic'};
    DefaultInterpolator='linear';
  end
  properties % EM tracking parameters (See GEANT4 manual) 0=use internal defaults
    DeltaOneStep=1e-5;
    DeltaIntersection=1e-5;
    DeltaChord=1e-5;
    EpsMin=1e-7;
    EpsMax=1e-5;
  end
  
end

