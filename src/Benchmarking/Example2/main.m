clear all; close all

disp(' ')
disp(' ')

sparse_beam = 0 ; % Set to 1 for a sparse representation, or 0 for 10k macro-particles.

randn('state',0) ;
statall = InitializeMessageStack( ) ;

[stat,Initial]=XSIFToLucretia('tesla_linac.xsif');
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;
global BEAMLINE GIRDER PS KLYSTRON WF; %#ok<NUSED>

disp('...setting element slices...')
[stat] = SetElementSlices( 1, length(BEAMLINE) ) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

disp('...setting alignment blocks...')
[stat] = SetElementBlocks( 1, length(BEAMLINE) ) ;
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

% set BPM resolution to zero
disp('...Setting BPM resolution to zero...')
bpmlist = findcells(BEAMLINE,'Class','MONI') ;
for count = 1:length(bpmlist)
    BEAMLINE{bpmlist(count)}.Resolution = 0 ;
end

disp('...finding cavities, quads, and ycors...')
cavlist=findcells(BEAMLINE,'Name','CAV');
cavaplist=findcells(BEAMLINE,'Name','CAVAP');
quadlist=findcells(BEAMLINE,'Class','QUAD');
ycorlist=findcells(BEAMLINE,'Class','YCOR');

disp('...loading misalignment files...')
quadmisalign=dlmread('quadmisalign.txt');
bpmmisalign=dlmread('bpmmisalign.txt');
cavmisalign=dlmread('cavmisalign.txt');
cavapmisalign=dlmread('cavapmisalign.txt');

disp('...misaligning the linac...')
count=0;
for element_num=quadlist
    count=count+1;
    BEAMLINE{element_num}.Offset(1)=quadmisalign(count,2);
    BEAMLINE{element_num}.Offset(2)=quadmisalign(count,4);
    BEAMLINE{element_num}.Offset(3)=quadmisalign(count,3);
    BEAMLINE{element_num}.Offset(4)=quadmisalign(count,5);
    BEAMLINE{element_num}.Offset(6)=quadmisalign(count,6);
end
count=0;
for element_num=bpmlist
    count=count+1;
    BEAMLINE{element_num}.Offset(1)=bpmmisalign(count,2);
    BEAMLINE{element_num}.Offset(2)=bpmmisalign(count,4);
    BEAMLINE{element_num}.Offset(3)=bpmmisalign(count,3);
    BEAMLINE{element_num}.Offset(4)=bpmmisalign(count,5);
    BEAMLINE{element_num}.Offset(6)=bpmmisalign(count,6);
end
count=0;
for element_num=cavlist
    count=count+1;
    BEAMLINE{element_num}.Offset(1)=cavmisalign(count,2);
    BEAMLINE{element_num}.Offset(2)=cavmisalign(count,4);
    BEAMLINE{element_num}.Offset(3)=cavmisalign(count,3);
    BEAMLINE{element_num}.Offset(4)=cavmisalign(count,5);
    BEAMLINE{element_num}.Offset(6)=cavmisalign(count,6);
end
count=0;
for element_num=cavaplist
    count=count+1;
    BEAMLINE{element_num}.Offset(1)=cavapmisalign(count,2);
    BEAMLINE{element_num}.Offset(2)=cavapmisalign(count,4);
    BEAMLINE{element_num}.Offset(3)=cavapmisalign(count,3);
    BEAMLINE{element_num}.Offset(4)=cavapmisalign(count,5);
    BEAMLINE{element_num}.Offset(6)=cavapmisalign(count,6);
end

disp('...loading the corrector settings...')
ycor_settings=dlmread('nick23p4_misxy_ycor_1.txt');

disp('...applying the corrector settings...')
count=0;
for element_num=ycorlist
    count=count+1;
    PS(BEAMLINE{element_num}.PS).Ampl=ycor_settings(count,2);
end

disp('...Setting tracking flags...')
SetTrackFlags('LRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMData',1,1,length(BEAMLINE)) ;
SetTrackFlags('GetSBPMData',0,1,length(BEAMLINE)) ;
SetTrackFlags('GetBPMBeamPars',1,1,length(BEAMLINE)) ;
SetTrackFlags('ZMotion',0,1,length(BEAMLINE)) ;

if sparse_beam==1
    disp('...generating "sparse" 21 x 21 beam...')
    beamin = MakeBeam6DSparse(Initial,3,21,21) ;
else
    disp('...generating 10k rays in 6D Gaussian distribution...')
    beamin = MakeBeam6DGauss(Initial,10000,5,0) ;
end

disp(' ')
disp('...wakefields off...')
SetTrackFlags('SRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',0,1,length(BEAMLINE)) ;
disp('...tracking the on-axis beam...')
[stat,beamout_nowake,data_nowake] = TrackThru(1,length(BEAMLINE),beamin,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_nowake=zeros(1,length(data_nowake{1}));
for bpm_num=1:length(data_nowake{1})
    y_pos_nowake(bpm_num)=data_nowake{1}(bpm_num).y;
end

figure(1);subplot(421);plot(y_pos_nowake*1e6,'-bx');
title('Linac misaligned, DFS steered, wakes off')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')

[S_nowake,nx_nowake,ny_nowake,nt_nowake] = GetNEmitFromBPMData(data_nowake{1});
figure(1);subplot(425);plot(ny_nowake*1e9,'-bx');
title('Normalised y emittance, Linac misaligned, DFS steered, wakes off')
xlabel('BPM Index');ylabel('Emittance / nm')

disp(' ')
disp('...transverse wakes on, longitudinal remain off...')
SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',0,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranonlongoff,data_tranonlongoff] = TrackThru(1,length(BEAMLINE),beamin,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranonlongoff=zeros(1,length(data_tranonlongoff{1}));
for bpm_num=1:length(data_tranonlongoff{1})
    y_pos_tranonlongoff(bpm_num)=data_tranonlongoff{1}(bpm_num).y;
end

figure(1);subplot(422);plot(y_pos_tranonlongoff*1e6,'-bx');
title('Linac misaligned, DFS steered, transverse wake on, longitudinal off')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')

[S_tranonlongoff,nx_tranonlongoff,ny_tranonlongoff,nt_tranonlongoff] = GetNEmitFromBPMData(data_tranonlongoff{1});
figure(1);subplot(426);plot(ny_tranonlongoff*1e9,'-bx');
title('Normalised y emittance, Linac misaligned, DFS steered, transverse wake on, longitudinal off')
xlabel('BPM Index');ylabel('Emittance / nm')

disp(' ')
disp('...transverse wakes back off, longitudinal on...')
SetTrackFlags('SRWF_T',0,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranofflongon,data_tranofflongon] = TrackThru(1,length(BEAMLINE),beamin,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranofflongon=zeros(1,length(data_tranofflongon{1}));
for bpm_num=1:length(data_tranofflongon{1})
    y_pos_tranofflongon(bpm_num)=data_tranofflongon{1}(bpm_num).y;
end

figure(1);subplot(423);plot(y_pos_tranofflongon*1e6,'-bx');
title('Linac misaligned, DFS steered, transverse wake off, longitudinal on')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')

[S_tranofflongon,nx_tranofflongon,ny_tranofflongon,nt_tranofflongon] = GetNEmitFromBPMData(data_tranofflongon{1});
figure(1);subplot(427);plot(ny_tranofflongon*1e9,'-bx');
title('Normalised y emittance, Linac misaligned, DFS steered, transverse wake off, longitudinal on')
xlabel('BPM Index');ylabel('Emittance / nm')

disp(' ')
disp('...transverse and longitudinal wakes on...')
SetTrackFlags('SRWF_T',1,1,length(BEAMLINE)) ;
SetTrackFlags('SRWF_Z',1,1,length(BEAMLINE)) ;

disp('...tracking the on-axis beam...')
[stat,beamout_tranonlongon,data_tranonlongon] = TrackThru(1,length(BEAMLINE),beamin,1,1) ;
statall = AddStackToStack(statall,stat) ;
statall{1} = [statall{1} stat{1}] ;

y_pos_tranonlongon=zeros(1,length(data_tranonlongon{1}));
for bpm_num=1:length(data_tranonlongon{1})
    y_pos_tranonlongon(bpm_num)=data_tranonlongon{1}(bpm_num).y;
end

figure(1);subplot(424);plot(y_pos_tranonlongon*1e6,'-bx');
title('Linac misaligned, DFS steered, transverse wake on, longitudinal on')
xlabel('BPM Index');ylabel('Vertical Orbit / \mum')

[S_tranonlongon,nx_tranonlongon,ny_tranonlongon,nt_tranonlongon] = GetNEmitFromBPMData(data_tranonlongon{1});
figure(1);subplot(428);plot(ny_tranonlongon*1e9,'-bx');
title('Normalised y emittance, Linac misaligned, DFS steered, transverse wake on, longitudinal on')
xlabel('BPM Index');ylabel('Emittance / nm')

disp('Done!')