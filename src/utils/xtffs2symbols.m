%function dummy=xtffs2symbols(infile,outfile,S0)
%
% xtffs2symbols(infile,outfile,S0)
%
% Create an MD "Symbols Format" spreadsheet file from an XTFF survey file.
%
% INPUTs:
%
%  infile  = XTFF survey file name
%  outfile = MD "Symbols Format" spreadsheet file name
%  S0      = (optional; default=0) initial value for "S" column in outfile
%
%  example:  xtffs2symbols('lcls10nov03_survey.tape','lcls10nov03.txt');
%
% NOTEs:
%
% - supported MAD keywords: LCAV,SBEN,QUAD,HKIC,VKIC,MONI,PROF,WIRE,BLMO,IMON,
%   INST,RCOL,SOLE
% - non-chicane ("normal") bends must be split in half in order to compute the
%   steel center coordinates correctly

%if (nargin<3),S0=0;end

keyi=['QUAD';'HKIC';'VKIC';'MONI';'PROF';'WIRE'; ...
      'BLMO';'IMON';'INST';'RCOL';'SOLE';'ECOL'];
keyo=['QUAD';'XCOR';'YCOR';'BPM ';'PROF';'WIRE'; ...
      'BLMO';'TORO';'INST';'COLL';'SOLE';'ECOL'];

% NOTE: re-order x,y,z,theta,phi,psi -> z,x,y,psi,phi,theta
% (for Solid-Edge convention - Aug. 2, 2005, P. Emma)

ii=[3,1,2,6,5,4];

% read MAD survey tape file

[tt,K,N,L,P,A,T,E,FDN,coor,S]=xtffs2mat(infile);

% special handling for LCAV elements

dlist=[];
id=find_name('LCAV',K);
if (~isempty(id))
  nc=6; % first nc characters in name define device
  name=blanks(nc);
  defn=0;
  for m=1:length(id)
    n=id(m);
    if (strcmp(N(n,1:nc),name))
      leff=leff+L(n);
      Eend=E(n);
      Send=S(n);
      Cend=coor(n,ii);
    else
      if (defn)
        temp.keyw='LCAV';
        temp.name=name;
        temp.engt=engt;
        temp.leff=leff;
        temp.aper=aper;
        temp.parm=parm;
        temp.E=mean([Ebeg;Eend]);
        temp.S=mean([Sbeg;Send]);
        temp.C=mean([Cbeg;Cend]);
        temp.id=iddv;
        dlist=[dlist;temp];
      end
      name=N(n,1:nc);
      engt=T(n,:);
      leff=L(n);
      aper=0;
      parm=zeros(1,8);
      Ebeg=E(n-1);
      Sbeg=S(n-1);
      Cbeg=coor(n-1,ii);
      Eend=E(n);
      Send=S(n);
      Cend=coor(n,ii);
      iddv=n;
      defn=1;
    end
  end
  temp.keyw='LCAV';
  temp.name=name;
  temp.engt=engt;
  temp.leff=leff;
  temp.aper=aper;
  temp.parm=parm;
  temp.E=mean([Ebeg;Eend]);
  temp.S=mean([Sbeg;Send]);
  temp.C=mean([Cbeg;Cend]);
  temp.id=iddv;
  dlist=[dlist;temp];
end

% special handling for SBEN elements

id=find_name('SBEN',K);
if (~isempty(id))
  nc=16;
  name=blanks(nc);
  defn=0;
  for m=1:length(id)
    n=id(m);
    if (strcmp(N(n,1:nc),name))
      leff=leff+L(n);
      parm(1)=parm(1)+P(n,1);
      parm(6)=P(n,6);
      parm(8)=P(n,8);
      Eend=E(n);
      Send=S(n);
      Cend=coor(n,ii);
    else
      if (defn)
        temp.keyw='BEND';
        temp.name=name;
        temp.engt=engt;
        temp.leff=leff;
        temp.aper=aper;
        temp.parm=parm;
        temp.E=mean([Ebeg;Eend]);
        temp.S=mean([Sbeg;Send]);
        e1=parm(5);
        e2=parm(6);
        if ((e1==0)&(e2~=0))
          temp.C=[mean([Cbeg(1);Cend(1)]), ...
                  mean([Cbeg(2);Cend(2)]), ...
                  mean([Cbeg(3);Cend(3)]), ...
                  Cbeg(4:6)];
        elseif ((e1~=0)&(e2==0))
          temp.C=[mean([Cbeg(1);Cend(1)]), ...
                  mean([Cbeg(2);Cend(2)]), ...
                  mean([Cbeg(3);Cend(3)]), ...
                  Cend(4:6)];
        else
          temp.C=[(Cbeg(1)+Cend(1)+2*Cmid(1))/4, ...
                  (Cbeg(2)+Cend(2)+2*Cmid(2))/4, ...
                  (Cbeg(3)+Cend(3)+2*Cmid(3))/4, ...
                  Cmid(4:6)];
        end
        temp.id=iddv;
        dlist=[dlist;temp];
      end
      nc=length(deblank(N(n,:)))-1; % remove last character of name
      name=N(n,1:nc);
      engt=T(n,:);
      leff=L(n);
      aper=A(n);
      parm=P(n,:);
      Ebeg=E(n-1);
      Sbeg=S(n-1);
      Cbeg=coor(n-1,ii);
      Cmid=coor(n,ii);
      Eend=E(n);
      Send=S(n);
      Cend=coor(n,ii);
      iddv=n;
      defn=1;
    end
  end
  temp.keyw='BEND';
  temp.name=name;
  temp.engt=engt;
  temp.leff=leff;
  temp.aper=aper;
  temp.parm=parm;
  temp.E=mean([Ebeg;Eend]);
  temp.S=mean([Sbeg;Send]);
  e1=parm(5);
  e2=parm(6);
  if ((e1==0)&(e2~=0))
    temp.C=[mean([Cbeg(1);Cend(1)]), ...
            mean([Cbeg(2);Cend(2)]), ...
            mean([Cbeg(3);Cend(3)]), ...
            Cbeg(4:6)];
  elseif ((e1~=0)&(e2==0))
    temp.C=[mean([Cbeg(1);Cend(1)]), ...
            mean([Cbeg(2);Cend(2)]), ...
            mean([Cbeg(3);Cend(3)]), ...
            Cend(4:6)];
  else
    temp.C=[(Cbeg(1)+Cend(1)+2*Cmid(1))/4, ...
            (Cbeg(2)+Cend(2)+2*Cmid(2))/4, ...
            (Cbeg(3)+Cend(3)+2*Cmid(3))/4, ...
            Cmid(4:6)];
  end
  temp.id=iddv;
  dlist=[dlist;temp];
end

% process elements by keyword

for k=1:length(keyi)
  id=find_name(keyi(k,:),K);
  if (~isempty(id))
    nc=16; % first nc characters in name define device
    name=blanks(nc);
    defn=0;
    for m=1:length(id)
      n=id(m);
      if (strcmp(N(n,1:nc),name))
        leff=leff+L(n);
        Eend=E(n);
        Send=S(n);
        Cend=coor(n,ii);
      else
        if (defn)
          temp.keyw=keyo(k,:);
          temp.name=name;
          temp.engt=engt;
          temp.leff=leff;
          temp.aper=aper;
          temp.parm=parm;
          temp.E=mean([Ebeg;Eend]);
          temp.S=mean([Sbeg;Send]);
          temp.C=mean([Cbeg;Cend]);
          temp.id=iddv;
          dlist=[dlist;temp];
        end
        name=N(n,1:nc);
        engt=T(n,:);
        leff=L(n);
        aper=A(n);
        parm=P(n,:);
        Ebeg=E(n-1);
        Sbeg=S(n-1);
        Cbeg=coor(n-1,ii);
        Eend=E(n);
        Send=S(n);
        Cend=coor(n,ii);
        iddv=n;
        defn=1;
      end
    end
    temp.keyw=keyo(k,:);
    temp.name=name;
    temp.engt=engt;
    temp.leff=leff;
    temp.aper=aper;
    temp.parm=parm;
    temp.E=mean([Ebeg;Eend]);
    temp.S=mean([Sbeg;Send]);
    temp.C=mean([Cbeg;Cend]);
    temp.id=iddv;
    dlist=[dlist;temp];
  end
end

Nd=length(dlist);
iddv=zeros(Nd,1);
for n=1:Nd
  iddv(n)=dlist(n).id;
end
[temp,id]=sort(iddv);

fmt='%5d, ,%-4s,%-16s,%-16s';
for n=1:18
  fmt=[fmt,',%17.9e'];
end
fmt=[fmt,'\n'];
fid=fopen(outfile,'w');
for m=1:Nd
  n=id(m);
  fprintf(fid,fmt,dlist(n).id,dlist(n).keyw,dlist(n).name,dlist(n).engt, ...
    dlist(n).leff,dlist(n).aper,dlist(n).parm,dlist(n).E,dlist(n).S, ...
    dlist(n).C);
end
fclose(fid);
disp(['   Output file written: ' outfile])

xlswrite('test.xls',dlist)
