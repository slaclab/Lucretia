clear all; close all

disp(' ')
disp(' ')

sparse_beam = 1 ; % Set to 1 for a sparse representation, or 0 for 10k macro-particles.

randn('state',0) ;
statall = InitializeMessageStack( ) ;

[stat,Initial]=XSIFToLucretia('tesla_linac.xsif');
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
global BEAMLINE GIRDER PS KLYSTRON WF; %#ok<NUSED,NUSED>

Initial_offs=Initial;
Initial_offs.y.pos=5e-6;

disp('...Generating Twiss parameters...')
[stat,Twiss] = GetTwiss(1,length(BEAMLINE),Initial.x.Twiss,Initial.y.Twiss) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
if (stat{1} == 1)
    figure(10) ;
    subplot(3,1,1) ;
    plot(Twiss.S/1000,Twiss.betax) ;
    ylabel('\beta_x [m]') ;
    title('Twiss parameters of TESLA Linac')
    subplot(3,1,2) ;
    plot(Twiss.S/1000,Twiss.betay) ;
    ylabel('\beta_y [m]') ;
    subplot(3,1,3)
    plot(Twiss.S/1000,Twiss.etax) ;
    ylabel('\eta_x [m]') ;
    xlabel('S position [km]') ;
else
    disp('Problem in GetTwiss!  Halting execution!') ;
    return ;
end

disp('...setting element slices...')
stat = SetElementSlices( 1, length(BEAMLINE) ) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...setting alignment blocks...')
stat = SetElementBlocks( 1, length(BEAMLINE) ) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...girderising the linac...')
stat = SetGirderByBlock(1,length(BEAMLINE),0);
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...setting independent power supplies...')
stat = SetIndependentPS( 1, length(BEAMLINE) ) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...setting 24 structures per klystron...')
stat = SetKlystrons( 1, length(BEAMLINE), 24  ) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...moving physics units from magnets to power supplies...')
stat = MovePhysicsVarsToPS(1:length(PS)) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...moving physics units from structures to klystrons...')
stat = MovePhysicsVarsToKlystron(1:length(KLYSTRON)) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...Setting BPM resolution to zero...')
bpmlist = findcells(BEAMLINE,'Class','MONI') ;
for count = 1:length(bpmlist)
    BEAMLINE{bpmlist(count)}.Resolution = 0 ;
end

if sparse_beam==1
    disp('...generating "sparse" 21 x 21 beam...')
    beam_onaxis = MakeBeam6DSparse(Initial,3,21,21) ;
    disp('...generating "sparse" 21 x 21 offset beam...')
    beam_offs = MakeBeam6DSparse(Initial_offs,3,21,21) ;
else
    disp('...generating 10k rays in 6D Gaussian distribution...')
    beam_onaxis = MakeBeam6DGauss(Initial,10000,5,0) ;
    disp('...generating offset bunch -- 10k rays in 6D Gaussian distribution...')
    beam_offs = MakeBeam6DGauss(Initial_offs,10000,5,0) ;
end

disp('...Setting tracking flags...')
SetTrackFlags('LRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMData',1,1,length(BEAMLINE)) ;
SetTrackFlags('GetSBPMData',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMBeamPars',1,1,length(BEAMLINE)) ;
SetTrackFlags('ZMotion',0,1,length(BEAMLINE)) ;

disp(' ')
disp('...wakefields off...')
SetTrackFlags('SRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',0,1,length(BEAMLINE)) ;
disp('...tracking the on-axis beam...')
[stat,beamout_nowake,data_nowake] = TrackThru(1,length(BEAMLINE),beam_onaxis,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
disp('...tracking the offset beam...')
[stat,beamout_offs_nowake,data_offs_nowake] = TrackThru(1,length(BEAMLINE),beam_offs,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_nowake=zeros(1,length(data_nowake{1}));
y_pos_offs_nowake=zeros(1,length(data_nowake{1}));
for bpm_num=1:length(data_nowake{1})
    y_pos_nowake(bpm_num)=data_nowake{1}(bpm_num).y;
    y_pos_offs_nowake(bpm_num)=data_offs_nowake{1}(bpm_num).y;
end

figure(1);subplot(421);plot(y_pos_nowake*1e6,'-bx');hold on;plot(y_pos_offs_nowake*1e6,'-rx')
title('5 \mum Betatron Orbit, wakes off')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthEast')

[S_nowake,nx_nowake,ny_nowake,nt_nowake] = GetNEmitFromBPMData(data_nowake{1});
[S_offs_nowake,nx_offs_nowake,ny_offs_nowake,nt_offs_nowake] = GetNEmitFromBPMData(data_offs_nowake{1});
figure(1);subplot(425);plot(ny_nowake*1e9,'-bx');hold on;plot(ny_offs_nowake*1e9,'-rx')
title('Normalised y emittance, 5 \mum Betatron Orbit, wakes off')
xlabel('BPM Index');ylabel('Emittance / nm')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthWest')

disp(' ')
disp('...transverse wakes on, longitudinal remain off...')
SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',0,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranonlongoff,data_tranonlongoff] = TrackThru(1,length(BEAMLINE),beam_onaxis,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
disp('...tracking the offset beam...')
[stat,beamout_offs_tranonlongoff,data_offs_tranonlongoff] = TrackThru(1,length(BEAMLINE),beam_offs,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranonlongoff=zeros(1,length(data_tranonlongoff{1}));
y_pos_offs_tranonlongoff=zeros(1,length(data_tranonlongoff{1}));
for bpm_num=1:length(data_tranonlongoff{1})
    y_pos_tranonlongoff(bpm_num)=data_tranonlongoff{1}(bpm_num).y;
    y_pos_offs_tranonlongoff(bpm_num)=data_offs_tranonlongoff{1}(bpm_num).y;
end

figure(1);subplot(422);plot(y_pos_tranonlongoff*1e6,'-bx');hold on;plot(y_pos_offs_tranonlongoff*1e6,'-rx')
title('5 \mum Betatron Orbit, transverse wake on, longitudinal off')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthEast')

[S_tranonlongoff,nx_tranonlongoff,ny_tranonlongoff,nt_tranonlongoff] = GetNEmitFromBPMData(data_tranonlongoff{1});
[S_offs_tranonlongoff,nx_offs_tranonlongoff,ny_offs_tranonlongoff,nt_offs_tranonlongoff] = GetNEmitFromBPMData(data_offs_tranonlongoff{1});
figure(1);subplot(426);plot(ny_tranonlongoff*1e9,'-bx');hold on;plot(ny_offs_tranonlongoff*1e9,'-rx')
title('Normalised y emittance, 5 \mum Betatron Orbit, transverse wake on, longitudinal off')
xlabel('BPM Index');ylabel('Emittance / nm')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthWest')

disp(' ')
disp('...transverse wakes back off, longitudinal on...')
SetTrackFlags('SRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranofflongon,data_tranofflongon] = TrackThru(1,length(BEAMLINE),beam_onaxis,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
disp('...tracking the offset beam...')
[stat,beamout_offs_tranofflongon,data_offs_tranofflongon] = TrackThru(1,length(BEAMLINE),beam_offs,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranofflongon=zeros(1,length(data_tranofflongon{1}));
y_pos_offs_tranofflongon=zeros(1,length(data_tranofflongon{1}));
for bpm_num=1:length(data_tranofflongon{1})
    y_pos_tranofflongon(bpm_num)=data_tranofflongon{1}(bpm_num).y;
    y_pos_offs_tranofflongon(bpm_num)=data_offs_tranofflongon{1}(bpm_num).y;
end

figure(1);subplot(423);plot(y_pos_tranofflongon*1e6,'-bx');hold on;plot(y_pos_offs_tranofflongon*1e6,'-rx')
title('5 \mum Betatron Orbit, transverse wake off, longitudinal on')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthEast')

[S_tranofflongon,nx_tranofflongon,ny_tranofflongon,nt_tranofflongon] = GetNEmitFromBPMData(data_tranofflongon{1});
[S_offs_tranofflongon,nx_offs_tranofflongon,ny_offs_tranofflongon,nt_offs_tranofflongon] = GetNEmitFromBPMData(data_offs_tranofflongon{1});
figure(1);subplot(427);plot(ny_tranofflongon*1e9,'-bx');hold on;plot(ny_offs_tranofflongon*1e9,'-rx')
title('Normalised y emittance, 5 \mum Betatron Orbit, transverse wake off, longitudinal on')
xlabel('BPM Index');ylabel('Emittance / nm')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthWest')

disp(' ')
disp('...transverse and longitudinal wakes on...')
SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranonlongon,data_tranonlongon] = TrackThru(1,length(BEAMLINE),beam_onaxis,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
disp('...tracking the offset beam...')
[stat,beamout_offs_tranonlongon,data_offs_tranonlongon] = TrackThru(1,length(BEAMLINE),beam_offs,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranonlongon=zeros(1,length(data_tranonlongon{1}));
y_pos_offs_tranonlongon=zeros(1,length(data_tranonlongon{1}));
for bpm_num=1:length(data_tranonlongon{1})
    y_pos_tranonlongon(bpm_num)=data_tranonlongon{1}(bpm_num).y;
    y_pos_offs_tranonlongon(bpm_num)=data_offs_tranonlongon{1}(bpm_num).y;
end

figure(1);subplot(424);plot(y_pos_tranonlongon*1e6,'-bx');hold on;plot(y_pos_offs_tranonlongon*1e6,'-rx')
title('5 \mum Betatron Orbit, transverse wake on, longitudinal on')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthEast')

[S_tranonlongon,nx_tranonlongon,ny_tranonlongon,nt_tranonlongon] = GetNEmitFromBPMData(data_tranonlongon{1});
[S_offs_tranonlongon,nx_offs_tranonlongon,ny_offs_tranonlongon,nt_offs_tranonlongon] = GetNEmitFromBPMData(data_offs_tranonlongon{1});
figure(1);subplot(428);plot(ny_tranonlongon*1e9,'-bx');hold on;plot(ny_offs_tranonlongon*1e9,'-rx')
title('Normalised y emittance, 5 \mum Betatron Orbit, transverse wake on, longitudinal on')
xlabel('BPM Index');ylabel('Emittance / nm')
legend('Design trajectory','5 \mum initial vertical offset','Location','NorthWest')

disp('Done!')