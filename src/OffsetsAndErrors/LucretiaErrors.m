classdef LucretiaErrors < handle
  %LUCRETIAERRORS Class for storage of error/tolerance values for Lucretia tracking simulations
  properties
    dX_init = 0 % Initial horizontal position error on cathode [m]
    dY_init = 0 % Initial vertical position error on cathode [m]
    Gun_rfphase = 0 % Phase error to apply to RF source at Gun [deg S-band]
    Gun_volts = 0 % Fractional error on supplied RF voltage from Gun
    Source_time = 0 % Source emission timing error [s]
    Source_charge = 0 % Fractional error in source charge emission
    Gun_solenoid_offs = [0 0] % Gun solenoid transverse offset error [m] [x,y]
    Gun_solenoid_rot = [0 0 0] % Gun solenoid rotation error [rad] [x,y,z]
    Gun_solenoid_volts = 0 % Fractional error on gun solenoid strength
    L0_phase = 0 % Phase error on L0 structures [deg S-band]
    L0_volt = 0 % Relative voltage error on L0 structures
    InjQuad_k = 0 % Relative error on injector quads
  end
end