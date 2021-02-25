function FixThinSkewQuads()

global BEAMLINE

id=findcells(BEAMLINE,'Name','MKEX1') ;
id=[id,findcells(BEAMLINE,'Name','MQM6R')] ;
id=[id,findcells(BEAMLINE,'Name','MQM7R')] ;
id=[id,findcells(BEAMLINE,'Name','MBS1X')] ;
id=[id,findcells(BEAMLINE,'Name','MBS2X')] ;
id=[id,findcells(BEAMLINE,'Name','MBS3X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS1X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS2X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS3X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS4X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS5X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS6X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS7X')] ;
id=[id,findcells(BEAMLINE,'Name','MQS8X')] ;

for n=id
  BEAMLINE{n}.Tilt=pi/4;
  BEAMLINE{n}.PoleIndex=1;
  BEAMLINE{n}.B=0;
end

id=findcells(BEAMLINE,'Name','DEC6');
id=[id,findcells(BEAMLINE,'Name','DEC4')] ;
for n=id
  BEAMLINE{n}.B=0;
  BEAMLINE{n}.Tilt=0;
  BEAMLINE{n}.PoleIndex=1;
end