classdef ExtPrimaryParticles < handle
  %EXTPRIMARYPARTICLES - Data associated with primary particles handed over
  %to EXT processes
  
  properties
    regeneratedID % List of particles stopped by Lucretia tracking that EXT process re-started
    TrackingData_x % tracking data points (x,y,z) recorded by EXT process if requested with TrackStoreMax>0
    TrackingData_y % data is stored in 2d array: 
    TrackingData_z % [bunch ID(1:nmacro_particles), track point ID(1:max # points recorded {<=TrackStoreMax})]
    TrackingDataPointer % array of ray IDs with pointers to last index in TrackingData for that ray
    PrimaryTrackTime % total track time in element for primary ray / ns
  end
  
  methods
    function obj = ExtPrimaryParticles(maxpart)
      if ~exist('maxpart','var')
        maxpart=0;
      end
      obj.regeneratedID=uint32(zeros(1,maxpart));
      obj.TrackingDataPointer=uint32(zeros(1,maxpart));
    end
  end
  
end

