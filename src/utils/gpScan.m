sigma_x=655; %nm
sigma_y=5.7; %nm
nSeed=1;
VERB=2;
% kick_x=[];kick_y=[];lumi=[];
offsets=5000:5000;
% loop over offsets
for offset=offsets
  % Grid size and Ngrids
  % Grid size and Ngrids
  xoffset=0;
  yoffset=offset;
  gridSize=3*sigma_x+xoffset;
  nGrids=2^nextpow2(ceil((gridSize/sigma_x)*5));
  if nGrids<16
    nGrids=16;
  elseif nGrids>128
    nGrids=128;
  end

  accFile=fopen('acc.dat','r');
  tline={};
  while 1
    tline{length(tline)+1} = fgetl(accFile); %#ok<AGROW>
    if ~ischar(tline{end}), break, end
    if length(tline)==6
      tline{end}=['n_x=',num2str(nGrids),';']; %#ok<AGROW>
    elseif length(tline)==7
      tline{end}='n_y=128;'; %#ok<AGROW>
    elseif length(tline)==8
      tline{end}=['cut_x=',num2str(3*sigma_x+xoffset),';']; %#ok<AGROW>
    elseif length(tline)==9
      tline{end}=['cut_y=',num2str(13*sigma_y+3.5*yoffset),';']; %#ok<AGROW>
    elseif length(tline)==10
      tline{end}=['rndm_seed=',num2str(nSeed),';']; %#ok<AGROW>
    elseif length(tline)==11
      tline{end}=['offset_x=',num2str(xoffset/2),';']; %#ok<AGROW>
    elseif length(tline)==12
      tline{end}=['offset_y=',num2str(yoffset/2),';']; %#ok<AGROW>
    end
  end
  fclose(accFile);
  accFile=fopen('acc.dat','w');
  for iLine=1:length(tline)
    if ischar(tline{iLine}); fprintf(accFile,[tline{iLine},'\n']); end;
  end
  fclose(accFile);

 % Run GUINEA-PIG
  evalc('!gp++ nominal par acc.out');

  lcf=2820*5*1e-4;
  % Read in GP output file
  gpOut=textread('acc.out','%s');
  Gp.lumi=str2double(gpOut{find(cellfun(@(x) isequal(x,'lumi_ee'),gpOut))+2})*lcf;
  Gp.missPart=str2double(gpOut{find(cellfun(@(x) isequal(x,'miss.1'),gpOut))+2})*100;
  Gp.kick.x(1)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vx.1'),gpOut))+2})*1e-6;
  Gp.kick.x(2)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vx.2'),gpOut))+2})*1e-6;
  Gp.kick.y(1)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vy.1'),gpOut))+2})*1e-6;
  Gp.kick.y(2)=str2double(gpOut{find(cellfun(@(x) isequal(x,'bpm_vy.2'),gpOut))+2})*1e-6;
  if VERB==2
    fprintf(1,'%% Particles out of grid: %2.2f\n',Gp.missPart)
    fprintf(1,'Kick (e- direction) x,y (urad): %.2g %.2g urad\n',Gp.kick.x(1)*1e6,Gp.kick.y(1)*1e6)
    fprintf(1,'Kick (e+ direction) x,y (urad): %.2g %.2g urad\n',Gp.kick.x(2)*1e6,Gp.kick.y(2)*1e6)
    fprintf(1,'Total Lumi: %g cm^-2 s^-1\n',Gp.lumi)
    fprintf(1,'\n----------------------------------------------------------------\n')
  else
    if Gp.missPart>5
      warning('Missing Particles in GP -> %2.2f %%\n',Gp.missPart) %#ok<WNTAG>
      fprintf('Offsets (nm): %g %g %g %g\n',offsets*2)
    end
  end
end % offset loop