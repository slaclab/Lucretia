function ModifyBeamline(elemno,cmd,varargin)
% MODIFYBEAMLINE - Add or delete an element from the BEAMLINE global
% array and correctly re-adjust all linked indexing
%
% ModifyBeamline(elemno,'delete')
%  - Remove BEAMLINE element elemno
%
% ModifyBeamline(elemno,'add',NewElementStruc)
%  - Add a new beamline element structure at location elemno, shifting all
%  elements elemno:end downstream
% e.g.
%   ModifyBeamline(123,'add',CorrectorStruc(0.2,0.1,0,1,'TestCor'))
%     - adds a new corrector, length 0.2m at BEAMLINE array location 123
global BEAMLINE PS KLYSTRON GIRDER

% Check args
if elemno<1 || elemno>length(BEAMLINE)+1
  error('elemno out of range')
end
if ~exist('cmd','var') || ~ischar(cmd)
  error('Must supply command argument')
end

% Adding or deleting an element?
if strcmpi(cmd,'delete')
  dele=-1;
elseif strcmpi(cmd,'add')
  dele=1;
else
  error('Unknown command')
end
if strcmpi(cmd,'delete')
  elelist=elemno+1:length(BEAMLINE);
elseif strcmpi(cmd,'add')
  elelist=elemno:length(BEAMLINE);
end
% If adding an element, needs to be passed as third argument
if strcmpi(cmd,'add')
  if nargin<3
    error('Incorrect arguments')
  end
  newele=varargin{1};
end
% Need to change BEAMLINE co-ordinates if deleting an element with length
% or adding a new element with length
dL=0;
if strcmpi(cmd,'delete') && isfield(BEAMLINE{elemno},'L')
  dL=-BEAMLINE{elemno}.L;
elseif strcmpi(cmd,'add') && isfield(newele,'L')
  dL=newele.L;
end
% Change co-ordinates of downstream elements
icoord=[];
if abs(dL)>0
  clist=elelist(ismember(elelist,findcells(BEAMLINE,'S',[],elelist(1),elelist(end))));
  for iele=clist
    BEAMLINE{iele}.S=BEAMLINE{iele}.S+dL;
  end
  clist=elelist(ismember(elelist,findcells(BEAMLINE,'Coordi',[],elelist(1),elelist(end))));
  if ~isempty(clist)
    icoord=[BEAMLINE{elemno}.Coordi BEAMLINE{elemno}.Anglei];
  end
end
% change blocks and slices
Blocks=findcells(BEAMLINE,'Block');
Slices=findcells(BEAMLINE,'Slices');
if strcmpi(cmd,'delete')
  for iele=Blocks
    if any(BEAMLINE{iele}.Block==elemno)
      BEAMLINE{iele}.Block(ismember(BEAMLINE{iele}.Block,elemno))=[];
    end
    if any(BEAMLINE{iele}.Block>elemno)
      BEAMLINE{iele}.Block(BEAMLINE{iele}.Block>elemno)=...
        BEAMLINE{iele}.Block(BEAMLINE{iele}.Block>elemno)-1;
    end
    if ~isempty(BEAMLINE{iele}.Block) && BEAMLINE{iele}.Block(1)<=elemno && BEAMLINE{iele}.Block(end)>=elemno
      newele.Block=BEAMLINE{iele}.Block;
    end
  end
  for iele=Slices
    if any(BEAMLINE{iele}.Slices==elemno)
      BEAMLINE{iele}.Slices(ismember(BEAMLINE{iele}.Slices,elemno))=[];
    end
    if any(BEAMLINE{iele}.Slices>elemno)
      BEAMLINE{iele}.Slices(BEAMLINE{iele}.Slices>elemno)=...
        BEAMLINE{iele}.Slices(BEAMLINE{iele}.Slices>elemno)-1;
    end
  end
elseif strcmpi(cmd,'add')
  for iele=Blocks
    if any(BEAMLINE{iele}.Block>=elemno)
      BEAMLINE{iele}.Block(BEAMLINE{iele}.Block>=elemno)=...
        BEAMLINE{iele}.Block(BEAMLINE{iele}.Block>=elemno)+1;
    end
  end
  for iele=Slices
    if any(BEAMLINE{iele}.Slices>=elemno)
      BEAMLINE{iele}.Slices(BEAMLINE{iele}.Slices>=elemno)=...
        BEAMLINE{iele}.Slices(BEAMLINE{iele}.Slices>=elemno)+1;
    end
  end
end
% change PS listing(s)
if strcmpi(cmd,'delete') && isfield(BEAMLINE{elemno},'PS')
  ips=BEAMLINE{elemno}.PS;
  if length(PS)>=ips && ips>0
    PS(ips).Element=PS(ips).Element(~ismember(PS(ips).Element,elemno));
  end
end
pslist=arrayfun(@(x) BEAMLINE{x}.PS,...
  findcells(BEAMLINE,'PS',[],elelist(1),elelist(end)));
if ~isempty(PS)
  for ips=pslist(pslist>0)
    PS(ips).Element=PS(ips).Element+dele;
  end
end
% change KLYSTRON listing(s)
if strcmpi(cmd,'delete') && isfield(BEAMLINE{elemno},'Klystron')
  ikly=BEAMLINE{elemno}.Klystron;
  if length(KLYSTRON)>=ikly && ikly>0
    KLYSTRON(ikly).Element=...
      KLYSTRON(ikly).Element(~ismember(KLYSTRON(ikly).Element,elemno));
  end
end
klist=arrayfun(@(x) BEAMLINE{x}.Klystron,...
  findcells(BEAMLINE,'Klystron',[],elelist(1),elelist(end)));
if ~isempty(KLYSTRON)
  for ikly=klist(klist>0)
    KLYSTRON(ikly).Element=KLYSTRON(ikly).Element+dele;
  end
end
% change GIRDER listing(s)
if strcmpi(cmd,'delete') && isfield(BEAMLINE{elemno},'GIRDER')
  igir=BEAMLINE{elemno}.GIRDER;
  if length(GIRDER)>=igir && igir>0
    GIRDER{igir}.Element=...
      GIRDER{igir}.Element(~ismember(GIRDER{igir}.Element,elemno));
  end
end
glist=arrayfun(@(x) BEAMLINE{x}.GIRDER,...
  findcells(BEAMLINE,'GIRDER',[],elelist(1),elelist(end)));
if ~isempty(GIRDER)
  for igir=glist(glist>0)
    GIRDER{igir}.Element=GIRDER{igir}.Element+dele;
  end
end
% Actually add or delete the element
[rw,col]=size(BEAMLINE);
if col>rw
  BEAMLINE=BEAMLINE';
end
if strcmpi(cmd,'delete')
  BEAMLINE(elemno)=[];
elseif strcmpi(cmd,'add')
  newele.P=BEAMLINE{elemno}.P;
  newele.S=BEAMLINE{elemno}.S;
  if elemno==length(BEAMLINE)+1
    BEAMLINE{end+1}=BEAMLINE{end};
    BEAMLINE{end-1}=newele;
  elseif elemno==1
    BEAMLINE=[newele; BEAMLINE];
  else
    BEAMLINE=[BEAMLINE(1:elemno-1); newele; BEAMLINE(elemno:end)];
  end
end
% Re-calc floor coords
if ~isempty(icoord)
  stat=SetFloorCoordinates(elemno,length(BEAMLINE),icoord);
  if stat{1}~=1
    error('Error configuring floor coordinates: %s',stat{2})
  end
end