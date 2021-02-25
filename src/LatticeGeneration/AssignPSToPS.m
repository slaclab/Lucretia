function stat = AssignPSToPS( psList, newPS )
% ASSIGNPSTOPS - Assign list of PSs to a specified or new PS
%  stat = AssignPSToPS( psList [,newPS] )
%   psList = vector of existing PSs
%   newPS = new PS number to asign (if omitted, assign length(PS)+1)
%
% --- Version date: 11-March-2012

global PS

% Check inputs
stat = InitializeMessageStack( ) ;
if any(~ismember(psList,1:length(PS)))
  stat{1} = 0 ;
  stat = AddMessageToStack(stat,...
    'PS does not exist') ;
end

% Make new PS
if exist('newPS','var')
  thisPS=newPS;
else
  thisPS=length(PS)+1;
end
PS(thisPS).Ampl=0;
PS(thisPS).SetPt=0;
PS(thisPS).Step=0;
PS(thisPS).Element=[];
PS(thisPS).dAmpl=0;
PS(thisPS).PSElement=psList;
PS(thisPS).PS=0;

% Set PS fields in psList to point to this master PS
for ips=psList
  PS(ips).PS=thisPS;
end