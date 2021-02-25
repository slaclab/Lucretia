function [S, x, y, S_beam, x_beam, y_beam] = getabs(ind_range,instdata)
% [S, x_mag, y_mag] = getabs(ind_range)
% Get absolute co-ordinates for magnets and beam at BPMs

global BEAMLINE GIRDER

cind=0;
S=zeros(1,length(ind_range));
x=zeros(1,length(ind_range));
y=zeros(1,length(ind_range));
han=zeros(1,length(ind_range));
for ind=ind_range
  cind=cind+1;
  han(cind)=ind;
  if isfield(BEAMLINE{ind},'Offset')
    S(cind)=BEAMLINE{ind}.S;
    x(cind)=BEAMLINE{ind}.Offset(1);
    y(cind)=BEAMLINE{ind}.Offset(3);
  end % if offset
  if isfield(BEAMLINE{ind},'girder')
    if length(GIRDER{BEAMLINE{ind}.Girder}.MoverPos)>3
      posInd=[1 3];
    else
      posInd=[1 2];
    end % get posInd
    x(cind)=x(cind)+GIRDER{BEAMLINE{ind}.Girder}.MoverPos(posInd(1)) + ...
    GIRDER{BEAMLINE{ind}.Girder}.Offset(1);
    y(cind)=y(cind)+GIRDER{BEAMLINE{ind}.Girder}.MoverPos(posInd(2)) + ...
    GIRDER{BEAMLINE{ind}.Girder}.Offset(3);
  end % if girder
  if isfield(BEAMLINE{ind},'ElecOffset')
    x(cind)=x(cind)-BEAMLINE{ind}.ElecOffset(1);
    y(cind)=y(cind)-BEAMLINE{ind}.ElecOffset(2);
  end % if bpm
end % for ind

if exist('instdata','var') && ~isempty(instdata)
  cind=0;
  for ind=find([instdata{1}.Index]>0)
    cind=cind+1;
    index=instdata{1}(ind).Index-1+ind_range(1);
    S_beam(cind)=instdata{1}(ind).S; %#ok<AGROW>
    x_beam(cind)=x(han==index) + instdata{1}(ind).x ; %#ok<AGROW>
    y_beam(cind)=y(han==index) + instdata{1}(ind).y ; %#ok<AGROW>
  end % for ind=1:length(instdata{1})
end % if instdata given