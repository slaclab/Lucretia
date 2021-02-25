%================================================================
%  Simple FODO line with ground motion
%  MAT-LIAR simulation
%     Two fodo lines pointing to each other
%
%  A.S. Dec.21, 2001                                             
%================================================================

%  Initialize LIAR with FODO beamline
init_fodo100
%
% Number of iterations in time
niter=64;
%
% Settings for getting the beam parameters.
[ierr,sbpm1] = mat_liar('MAT_INFO_BPMS');
sbpm2 = sbpm1;
nbpm=length(sbpm1);
bpm_indx=1:1:nbpm;
mat_liar('MAT_INIT_XBPM',nbpm,bpm_indx);
mat_liar('MAT_INIT_YBPM',nbpm,bpm_indx);
% Empty arrays for storing the data.
bpmrx1=zeros(nbpm,niter+1);
bpmry1=zeros(nbpm,niter+1);
beamx1=zeros(nbpm,niter+1);
beamy1=zeros(nbpm,niter+1);
mislx1=zeros(nbpm,niter+1);
misly1=zeros(nbpm,niter+1);
bpmrx2=zeros(nbpm,niter+1);
bpmry2=zeros(nbpm,niter+1);
beamx2=zeros(nbpm,niter+1);
beamy2=zeros(nbpm,niter+1);
mislx2=zeros(nbpm,niter+1);
misly2=zeros(nbpm,niter+1);
%
% Set up the ground motion model 
mat_liar('define_gm_model,');
mat_liar('pwkfile = ''gm_model_B.data''');
mat_liar('print_pwk_param');
%
%
% Set up BPM resolution, no other errors.
mat_liar('error_gauss_bpm, name = ''*'','); % Name to be matched
mat_liar('x_sigma = 0.e-6,  '); % Sigma misalignment
mat_liar('x_cut   = 3,      '); % Cut for Gaussian  
mat_liar('y_sigma = 0.e-6,  '); % Sigma misalignment
mat_liar('y_cut   = 3,      '); % Cut for Gaussian
mat_liar('resol = 1.0e-15,  '); % bpm res, meters
mat_liar('reset = .t.       '); % reset previous misalignm.
%
% Random seed for GM model
sdstr=['seed = ',num2str(round(abs(randn(1,1)*1000000)))];
%
%   Beamline 1 
%
liarstr= ['seed_gm_random_gen, ',sdstr];
mat_liar(liarstr);
%
mat_liar('prepare_gm_harmonics');
mat_liar('set_gm_abs_time, timeabs = 0  ');
mat_liar('gm_move, dt = 0.0, reset=.t. ');
%
mat_liar('  track, dolog = .t. ');
% Get initial beam parameters and logbook, without misalignment
mat_liar('logbook, process = .t.,');
mat_liar('print = .t.');
[iss,bpmrx1(:,1),bpmry1(:,1)] = mat_liar('MAT_GET_BPMR',nbpm); % BPM reading (includes random resolution)
[iss,beamx1(:,1),beamy1(:,1)] = mat_liar('MAT_GET_BPMS',nbpm); % absolute position of the beam
[iss,mislx1(:,1),misly1(:,1)] = mat_liar('MAT_GET_BPMD',nbpm); % misalignments of BPMs
%
for iter=1:niter
    mat_liar('gm_move, dt = 0.01 , sbeg= -2560, flips = .f. '); 
    mat_liar('increase_gm_abs_time, dtime = 0.01  '); 
    mat_liar('  track, dolog = .t. ');
    [iss,bpmrx1(:,iter+1),bpmry1(:,iter+1)] = mat_liar('MAT_GET_BPMR',nbpm);
    [iss,beamx1(:,iter+1),beamy1(:,iter+1)] = mat_liar('MAT_GET_BPMS',nbpm);
    [iss,mislx1(:,iter+1),misly1(:,iter+1)] = mat_liar('MAT_GET_BPMD',nbpm);
end
%
%   Beamline 2
%
liarstr= ['seed_gm_random_gen, ',sdstr];
mat_liar(liarstr);
%
% (it may not be necessary to repeat prepare harmonics ...  will fix
%
mat_liar('prepare_gm_harmonics');
mat_liar('set_gm_abs_time, timeabs = 0  ');
mat_liar('gm_move, dt = 0.0, reset=.t. ');
%
mat_liar('  track, dolog = .t. ');
% Get initial beam parameters and logbook, without misalignment
mat_liar('logbook, process = .t.,');
mat_liar('print = .t.');
[iss,bpmrx2(:,1),bpmry2(:,1)] = mat_liar('MAT_GET_BPMR',nbpm); % BPM reading (includes random resolution)
[iss,beamx2(:,1),beamy2(:,1)] = mat_liar('MAT_GET_BPMS',nbpm); % absolute position of the beam
[iss,mislx2(:,1),misly2(:,1)] = mat_liar('MAT_GET_BPMD',nbpm); % misalignments of BPMs
%
for iter=1:niter
    mat_liar('gm_move, dt = 0.01 , sbeg=  2560, flips = .t. '); 
    mat_liar('increase_gm_abs_time, dtime = 0.01  '); 
    mat_liar('  track, dolog = .t. ');
    [iss,bpmrx2(:,iter+1),bpmry2(:,iter+1)] = mat_liar('MAT_GET_BPMR',nbpm);
    [iss,beamx2(:,iter+1),beamy2(:,iter+1)] = mat_liar('MAT_GET_BPMS',nbpm);
    [iss,mislx2(:,iter+1),misly2(:,iter+1)] = mat_liar('MAT_GET_BPMD',nbpm);
end
%
%
% Save the logbook
mat_liar('logbook, process = .t.,');
mat_liar('print = .t.');
%
%

% Save output in Matlab format
%clear filename ierr iss liarstr iter pip strname
save gm_two_fodo_save
%
