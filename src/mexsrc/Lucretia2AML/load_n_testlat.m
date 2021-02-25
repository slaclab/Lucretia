function load_n_testlat(latfile)

% Check inputs
if ~exist('latfile','var')
  error('You must specify a lattice file.')
elseif ~(exist(latfile,'file') || exist([latfile '.mat'],'file'))
  error(['Cannot find ' latfile ' on the search path.'])
end

global BEAMLINE

% Load the lattice
load(latfile)

% BEAMLINE *has* to exist.  Error if not present.
if ~exist('BEAMLINE','var')
  error([latfile ' should contain a BEAMLINEcell array.'])
elseif ~iscell(BEAMLINE)
  error('BEAMLINE must be a cell array.')
end

