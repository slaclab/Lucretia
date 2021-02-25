function PosRet = GetRealOff( beam0, Group, mag_ind )
% PosRet = GetRealOff( beam0, Group, mag_ind )
% Get actual offset of Magnet center to beam
global BEAMLINE

% Get magnet bpm reference
bpm_ind=Group.(Group.AllMag{mag_ind}.Type).Bpm(Group.AllMag{mag_ind}.Pointer).Ind;
bpm_han=Group.(Group.AllMag{mag_ind}.Type).Bpm(Group.AllMag{mag_ind}.Pointer).Han;

% First set resolution to zero in magnet BPM
init_bpmres = BEAMLINE{bpm_ind}.Resolution;
BEAMLINE{bpm_ind}.Resolution=0;

% Track the beam
[stat,beamout,instdata] = TrackThru( 1, length(BEAMLINE), beam1, 1, 1, 0 );if stat{1}~=1; error(stat{2:end}); end;

% Get reported beam position in BPM
bpm_reading=[instdata{1}(bpm_han).x instdata{1}(bpm_han).y];

% Get beam position with respect to girder
beampos=bpm_reading-BEAMLINE{bpm_ind}.ElecOffset+[BEAMLINE{bpm_ind}.Offset(1) BEAMLINE{bpm_ind}.Offset(3)];

% Get Magnet position with respect to girder
magpos=BEAMLINE{Group.(Group.AllMag{mag_ind}.Type).dB.ClusterList(Group.AllMag{mag_ind}.Pointer).index(1)}.Offset;
magpos=[magpos(1) magpos(3)];

% Return actual beam position with respect to magnet entrance
PosRet = beampos - magpos ;

% Restore original BPM resolution
BEAMLINE{bpm_ind}.Resolution = init_bpmres ;

return