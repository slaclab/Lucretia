function Gp = matgp(nSeed,Bunch1,Bunch2,nb,repRate)
% Run GUINEA-PIG with provided Lucretia bunches
% -- Inputs are Model seed and 2 Lucretia bunch files (Beam.Bunch.x)
% -- Gp structure returned gives lumi, beam-beam kick and % macroparticles
% that fell out of calculation range (should be small)
% nb=number of bunches/train
global VERB

% Run guinea-pig job in it's own directory
gpdir=sprintf('/tmp/GP_%d',nSeed);
if ~exist(gpdir,'dir')
  mkdir(gpdir);
end
% Change to GP directory and copy over required files
letdir=pwd;
copyfile('acc.dat',gpdir);
cd(gpdir)
system(sprintf('ln -fs %s/gp++',letdir));

% Set grid sizes for GUINEA-PIG calculation
xsize=ceil(max([std(Bunch1(1,:)) std(Bunch2(1,:))])*1e9);
ysize=ceil(max([std(Bunch1(3,:)) std(Bunch2(3,:))])*1e9);
xoffset=abs(mean(Bunch1(1,:))-mean(Bunch2(1,:)))/1e-9;
yoffset=abs(mean(Bunch1(3,:))-mean(Bunch2(3,:)))/1e-9;
offsets=([xoffset xoffset yoffset yoffset]./2).*1e-9;

% Center bunches in grid
for f=[1 3]
  if mean(Bunch1(f,:)) > mean(Bunch2(f,:))
    Bunch1(f,:)=Bunch1(f,:)-mean(Bunch1(f,:))+offsets(f);
    Bunch2(f,:)=Bunch2(f,:)-mean(Bunch2(f,:))-offsets(f);
  else
    Bunch1(f,:)=Bunch1(f,:)-mean(Bunch1(f,:))-offsets(f);
    Bunch2(f,:)=Bunch2(f,:)-mean(Bunch2(f,:))+offsets(f);
  end % if b1>b2
end % for f

accFile=fopen('acc.dat','r');
tline={};
while 1
  tline{length(tline)+1} = fgetl(accFile); %#ok<AGROW>
  if ~ischar(tline{end}), break, end
  if length(tline)==6
    tline{end}=sprintf('n_m=%d;',max([length(Bunch1) length(Bunch2)])); %#ok<AGROW>
  elseif length(tline)==7
    tline{end}='n_x=256;'; %#ok<AGROW>
  elseif length(tline)==8
    tline{end}='n_y=512;'; %#ok<AGROW>
  elseif length(tline)==9
    tline{end}=['cut_x=',num2str(4*xsize+2*xoffset),';']; %#ok<AGROW>
  elseif length(tline)==10
    tline{end}=['cut_y=',num2str(4*ysize+4.5*yoffset),';']; %#ok<AGROW>
  elseif length(tline)==11
    tline{end}=['rndm_seed=',num2str(nSeed),';']; %#ok<AGROW>
  end
end
fclose(accFile);
accFile=fopen('acc.dat','w');
for iLine=1:length(tline)
  if ischar(tline{iLine}); fprintf(accFile,[tline{iLine},'\n']); end;
end
fclose(accFile);

% Error checking
if length(Bunch1)<6 || length(Bunch2)<6; error('Not enough particles in Bunch files!'); end;

% If bunches have different N, chop excess from larger bunch
lBunch1=Bunch1(:,1:min([length(Bunch1) length(Bunch2)]));
lBunch2=Bunch2(:,1:min([length(Bunch1) length(Bunch2)]));

% Convert Lucretia Bunches into GUINEA-PIG files
% Lucretia Bunch format: Bunch=[6,N]; [x,x',y,y',z,E] / m,rad,GeV
% GP Bunch format: gpBunch=[N,6]; [E,x,y,z,x',y'] / um,urad,GeV
gpBunch1=[lBunch1(6,:)' lBunch1(1,:)' lBunch1(3,:)' lBunch1(5,:)' lBunch1(2,:)' lBunch1(4,:)'];
gpBunch2=[lBunch2(6,:)' lBunch2(1,:)' lBunch2(3,:)' lBunch2(5,:)' lBunch2(2,:)' lBunch2(4,:)'];
% Order files by z;
[~,I]=sort(gpBunch1(:,4)); gpBunch1=gpBunch1(I,:);
[~,I]=sort(gpBunch2(:,4)); gpBunch2=gpBunch2(I,:);
gpBunch1(:,2:6)=gpBunch1(:,2:6).*1e6;
gpBunch2(:,2:6)=gpBunch2(:,2:6).*1e6;
% Write out GP files
save 'electron.ini' gpBunch1 -ASCII
save 'positron.ini' gpBunch2 -ASCII

% Run GUINEA-PIG
if exist(['acc_',num2str(nSeed),'.out'],'var'); delete( ['acc_',num2str(nSeed),'.out'] ); end ;
evalc(['!./gp++ nominal par acc_',num2str(nSeed),'.out']);

if ~exist('nb','var') || isempty(nb); nb=1312; end;
if ~exist('repRate','var') || isempty(repRate); repRate=5; end;
lcf=nb*repRate*1e-4;
  
% Read in GP output file
if ~exist(['acc_',num2str(nSeed),'.out'],'file')
  error('GUINEA PIG program failed')
end
gpOut=textread(['acc_',num2str(nSeed),'.out'],'%s'); %#ok<DTXTRD>
Gp.lumi=str2double(gpOut{find(cellfun(@(x) isequal(x,'lumi_ee'),gpOut))+2})*lcf;
Gp.missPart=str2double(gpOut(find(cellfun(@(x) isequal(x,'miss'),gpOut))+2))*100; Gp.missPart=max(Gp.missPart);
Gp.kick.x(1)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vx.1'),gpOut))+2})*1e-6;
Gp.kick.x(2)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vx.2'),gpOut))+2})*1e-6;
Gp.kick.y(1)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vy.1'),gpOut))+2})*1e-6;
Gp.kick.y(2)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vy.2'),gpOut))+2})*1e-6;
if VERB==2
  fprintf('%% Particles out of grid: %2.2f\n',Gp.missPart)
  fprintf('Kick (e- direction) x,y (urad): %.2g %.2g urad\n',Gp.kick.x(1)*1e6,Gp.kick.y(1)*1e6)
  fprintf('Kick (e+ direction) x,y (urad): %.2g %.2g urad\n',Gp.kick.x(2)*1e6,Gp.kick.y(2)*1e6)
  fprintf('Total Lumi: %g cm^-2 s^-1\n',Gp.lumi)
  fprintf('\n----------------------------------------------------------------\n')
else
  if Gp.missPart>5
    warning('Missing Particles in GP -> %2.2f %%\n',Gp.missPart) %#ok<WNTAG>
    fprintf('Offsets (nm): %g %g %g %g\n',offsets*2)
  end
end

% Get any stored secondaries
if exist('photon.dat','file')
  Gp.photons=importdata('photon.dat');
end
if exist('secondaries.dat','file')
  Gp.pairs=importdata('secondaries.dat');
end

% Change back to working directory and cleanup temp directory
cd(letdir)
rmdir(gpdir,'s');
