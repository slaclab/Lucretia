
% open the XTFF file

  fid=fopen(fname);
  if (fid==-1)
    error(['  Failed to open ',fname])
  end

% read in the header ... check that XTFF file is a twiss file

  line=fgetl(fid);
  xtff=line(9:16);
  if (~strcmp(xtff,'  SURVEY'))
    error(['  Unexpected XTFF type (',xtff,') encountered ... abort'])
  end

% read in the run title

  tt=deblank(fgetl(fid));

% read in the INITIAL data

  line=fgetl(fid);  % first line
  K=line(1:4);
  N=line(5:20);
  L=str2double(line(21:32));
  p1=str2double(line(33:48));
  p2=str2double(line(49:64));
  p3=str2double(line(65:80));
  A=str2double(line(81:96));
  T=line(98:113);
  E=str2double(line(115:130));
  line=fgetl(fid);  % second line
  p4=str2double(line(1:16));
  p5=str2double(line(17:32));
  p6=str2double(line(33:48));
  p7=str2double(line(49:64));
  p8=str2double(line(65:80));
  FDN=blanks(24);
  line=fgetl(fid);  % third line
  x=str2double(line(1:16));
  y=str2double(line(17:32));
  z=str2double(line(33:48));
  suml=str2double(line(49:64));
  line=fgetl(fid);  % fourth line
  yaw=str2double(line(1:16));
  pitch=str2double(line(17:32));
  roll=str2double(line(33:48));
  P=[p1,p2,p3,p4,p5,p6,p7,p8];
  coor=[x,y,z,yaw,pitch,roll];
  S=suml;

% read in the data ... break at end of the file

  while(1)
    line=fgetl(fid);
    if (isempty(line)), break, end
    K=[K;line(1:4)];
    N=[N;line(5:20)];
    L=[L;str2double(line(21:32))];
    p1=str2double(line(33:48));
    p2=str2double(line(49:64));
    p3=str2double(line(65:80));
    A=[A;str2double(line(81:96))];
    T=[T;line(98:113)];
    E=[E;str2double(line(115:130))];
    line=fgetl(fid);
    if (length(line)<105), line=[line,blanks(105-length(line))]; end
    p4=str2double(line(1:16));
    p5=str2double(line(17:32));
    p6=str2double(line(33:48));
    p7=str2double(line(49:64));
    p8=str2double(line(65:80));
    FDN=[FDN;line(82:105)];
    line=fgetl(fid);
    x=str2double(line(1:16));
    y=str2double(line(17:32));
    z=str2double(line(33:48));
    suml=str2double(line(49:64));
    line=fgetl(fid);
    yaw=str2double(line(1:16));
    pitch=str2double(line(17:32));
    roll=str2double(line(33:48));
    P=[P;p1,p2,p3,p4,p5,p6,p7,p8];
    coor=[coor;x,y,z,yaw,pitch,roll];
    S=[S;suml];
  end

% close the XTFF file

  fclose(fid);