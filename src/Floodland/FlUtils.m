classdef FlUtils < handle
  %FLUTILS General utility methods for use with Floodland classes
  %
  % getIndex - index referencing for Lucretia lattice arrays and Floodland
  %            objects
  properties
    Index % index references into BEAMLINE
  end
  properties(Constant)
    Cb = 3.335640952 ; % T.m / GeV
    clight = 299792458 ; % speed of light [m/s]
  end
  
  methods
    function ret=getIndex(obj,prop,propval,propval2)
      % ret=getIndex(obj,prop,propval,propval2)
      %   Search for properties and property values in Floodland objects
      %   e.g.
      %   FlUtilsObj.getIndex(Index,123)
      %     - Get the object index which points to the BEAMLINE index 123
      %   FlUtilsObj.getIndex(BEAMLINE,'Name','SomeBeamlineElementName')
      %     - Find BEAMLINE index with requested name
      %   FlUtilsObj.getIndex(KLYSTRON,'V',3)
      %     - Find FlIndex KLSTRON index with 3V
      %
      % Note, PS,GIRDER,KLSYTON searches only work with FlIndex objects
      global BEAMLINE PS KLYSTRON GIRDER
      if isprop(obj,prop)
        if exist('propval2','var') && strcmp(prop,'PS')
          if isfield(PS,propval)
            ret=arrayfun(@(x) isequal(PS(x).(propval),propval2),obj.PS);
          else
            try
              ret=arrayfun(@(x) isequal(BEAMLINE{PS(x).Element(1)}.(propval),propval2),obj.PS);
            catch
              error('No matching property field found in PS or BEAMLINE')
            end
          end
        elseif exist('propval2','var') && strcmp(prop,'KLYSTRON')
          if isfield(KLYSTRON,propval)
            ret=arrayfun(@(x) isequal(KLYSTRON(x).(propval),propval2),obj.KLYSTRON);
          else
            try
              ret=arrayfun(@(x) isequal(BEAMLINE{KLYSTRON(x).Element(1)}.(propval),propval2),obj.KLYSTRON);
            catch
              error('No matching property field found in KLYSTRON or BEAMLINE')
            end
          end
        elseif exist('propval2','var') && strcmp(prop,'GIRDER')
          if isfield(GIRDER,propval)
            ret=arrayfun(@(x) isequal(GIRDER{x}.(propval),propval2),obj.GIRDER);
          else
            try
              ret=arrayfun(@(x) isequal(BEAMLINE{GIRDER{x}.Element(1)}.(propval),propval2),obj.GIRDER);
            catch
              error('No matching property field found in GIRDER or BEAMLINE')
            end
          end
        elseif ischar(propval)
          if iscell(obj.(prop))
            ret=cellfun(@(x) strcmp(x,propval),obj.(prop));
          else
            ret=arrayfun(@(x) strcmp(x,propval),obj.(prop));
          end
        elseif strcmp(prop,'Index')
          if iscell(obj.(prop))
            ret=cellfun(@(x) x==propval,obj.(prop));
          else
            ret=arrayfun(@(x) x==propval,obj.(prop));
          end
        else
          if iscell(obj.(prop))
            ret=cellfun(@(x) isequal(x,propval),obj.(prop));
          else
            ret=cellfun(@(x) isequal(x,propval),obj.(prop));
          end
        end
      else
        if ischar(propval)
          ret=cellfun(@(x) isfield(x,prop)&&strcmp(x.(prop),propval),BEAMLINE);
          ret=ismember(obj.Index,find(ret));
        else
          ret=cellfun(@(x) isfield(x,prop)&&isequal(x.(prop),propval),BEAMLINE);
          ret=ismember(obj.Index,find(ret));
        end
      end
    end
  end
  
end

