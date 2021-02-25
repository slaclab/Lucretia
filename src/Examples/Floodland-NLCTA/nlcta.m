% function nlcta
% NLCTA - startup script for Lucretia:Floodland NLCTA applications
global BEAMLINE KLYSTRON PS GIRDER %#ok<NUSED>
disp('Performing FS setup...')

% Read in lattice from XSIF file, make some groupings, setup sim and
% hardware environment and output Floodland, Instrument and Hardware index objects
[FL INSTR FLI]=loadlat();

% Load status from last run if any
try
  status=FlMenu.statusLoad('nlcta_status','BEAMLINE','simStat','INSTR','FLI','FL','FB','SR');
catch
  status=[];
end

% Look to see if any changes to BEAMLINE since last run
% (see if length or names have changed)
blNames=cellfun(@(x) x.Name,BEAMLINE,'UniformOutput',false);
if ~isempty(status) && ( ~isequal(blNames,cellfun(@(x) x.Name,status.BEAMLINE,'UniformOutput',false)) || ...
    ~isequal(FLI.INDXnames,status.FLI.INDXnames) || ~isequal(INSTR.Index,status.INSTR.Index) )
  resp=questdlg('Model or hardware data has changed since last time, suggest not loading past config','OK - use defaults','Load anyway','OK - use defaults');
  if strcmp(resp,'OK - use defaults')
    FlMenu.statusLoadBlock('nlcta_status');
    status=[];
  end
end

% Turn live?
if ~isempty(status)
  FL=status.FL;
  INSTR=status.INSTR;
  FLI=status.FLI;
else
  FL.issim=true;
end

% Setup feedback objects, read in last ones from file if there
if ~isempty(status)
  FB=status.FB;
else
  INSTR.clearData;
  FB=FlFeedback(FL,INSTR,FLI);
  FB.useInstr=true(size(FB.useInstr)); % use all
  fbdev={'X810372T' 'X810480T' 'Y810391T' 'Y810480T' 'ACCL380A'};
  FB.useCntrl=ismember(FB.INDXnames,fbdev);
end

% Save/restore program
if ~isempty(status)
  SR=status.SR;
else
  SR=FlSaveRestore(FL,FLI);
end

% Generate Floodland Menu object
menu=FlMenu(FL);
menu.guiTitle='NLCTA Lucretia Apps';

% Add applications
menu.addApp(FB);
menu.addApp(SR);

% Launch menu
uiwait(menu.guiMain);

% Save status
simStat=FL.issim;
FlMenu.statusSave('nlcta_status',BEAMLINE,simStat,INSTR,FLI,FL,FB,SR);