classdef ExtEMField < handle
  %EXTEMFIELD - Data class to handle electromagnetic field descriptions for EXT processes
  
  properties(Constant,Abstract,Hidden)
    StepMethods
    DefaultStepMethod
    Interpolators
    DefaultInterpolator
  end
  properties(SetAccess=protected)
    StepMethod ;
    Interpolator ;
    EnableEM=int8(0);
    FieldBox=[0,0,0,0];
  end
  properties
    StepPrec = 1e-5;
    Bx=0;
    By=0;
    Bz=0;
    Ex=0;
    Ey=0;
    Ez=0;
  end
  methods
    function obj=ExtEMField()
      obj.SetInterpolator('default');
      obj.SetStepMethod('default');
    end
    function EnableFieldCalc(obj)
      obj.EnableEM=int8(1);
    end
    function DisableFieldCalc(obj)
      obj.EnableEM=int8(0);
    end
    function SetFieldBox(obj,xmin,xmax,ymin,ymax)
      obj.FieldBox=[xmin xmax ymin ymax];
    end
    function SetInterpolator(obj,interp)
      switch interp
        case 'default'
          obj.Interpolator=obj.DefaultInterpolator;
        case obj.Interpolators
          obj.Interpolator=interp;
        otherwise
          error('Unsupported interpolation method')
      end
    end
    function SetStepMethod(obj,method)
      switch method
        case 'default'
          obj.StepMethod=obj.DefaultStepMethod;
        case obj.StepMethods
          obj.StepMethod=method;
        otherwise
          error('Unsupported stepper method')
      end
    end
  end
  
end