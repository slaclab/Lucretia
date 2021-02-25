classdef FlApp < handle
  %FLAPP Template class for Floodland application classes
  % - All intended Floodland applications must be a subclass of FlApp
  
  properties(Abstract,Constant)
    appName % Application name
  end
  
  methods(Abstract)
    handle=guiMain(src,event) % Main gui for application
  end
  
end

