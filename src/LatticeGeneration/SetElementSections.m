function [stat,sectionnames] = SetElementSections(istart,iend)
% SETELEMENTSECTIONS Use "BEG*" and "END*" Marker names in BEAMLINE to set
%                    lattice sections
%
% [stat,sectionnames] = SetElementSections( [istart, iend] ) 
%   Apply Section fields to each BEAMLINE element within BEG/END marker blocks
%     sectionnames = return string list of applied section names (if any)
%     Use BEAMLINE elements istart:iend (or entire BEAMLINE array if not given)
%     Also stores section data in SECTION global array
%
% Return status:  +1 if successful, 0 if no sections or bad "BEG*" "END*" setup etc.
%
% Version date:  16-May-2024

%==========================================================================
global BEAMLINE SECTION

if ~exist('istart','var')
  istart=1;
end
if ~exist('iend','var')
  iend=length(BEAMLINE);
end
% Clear existing section data
SECTION=[];
for iele=istart:iend
  if isfield(BEAMLINE{iele},'Section')
    BEAMLINE{iele}=rmfield(BEAMLINE{iele},'Section');
  end
end
stat = 1 ;
sectionnames=string([]);

sbeg=findcells(BEAMLINE,'Name','BEG*',istart,iend);
send=findcells(BEAMLINE,'Name','END*',istart,iend); endnames=arrayfun(@(x) BEAMLINE{x}.Name,send,'UniformOutput',false);
for iele=1:length(sbeg)
  secname=regexprep(BEAMLINE{sbeg(iele)}.Name,'^BEG','');
  ie2=ismember(endnames,sprintf('END%s',secname));
  if any(ie2)
    sectionnames(end+1)=secname;
    SECTION(end+1).Name=secname;
    SECTION(end).Element=[sbeg(iele), send(ie2)]; SECTION(end).Element=SECTION(end).Element(1:2);
    for ibl=SECTION(end).Element(1):SECTION(end).Element(2)
      BEAMLINE{ibl}.Section=length(SECTION);
    end
  end
end
if isempty(SECTION)
  stat=0;
end