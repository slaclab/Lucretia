global BEAMLINE GIRDER
 
 if ( exist('GIRDER','var') && length(GIRDER)>0 )
   for gnum=1:length(GIRDER)
     for elenum=GIRDER{gnum}.Element(1):GIRDER{gnum}.Element(end)
       if ~isfield(BEAMLINE{elenum},'Girder')
         BEAMLINE{elenum}.Girder = gnum;
         BEAMLINE{elenum}.L2AMLadded = true;
       end
     end
   end
 end


