function stat=FlSETfile(setfile)
% stat=FlSETfile(setfile);
%
% Write magnet currents corresponding to present default FS optics to a
% copy of the specified SET-file ... use BH1R main current from setfile to
% determine the beam energy and scale the magnet setpoints accordingly
%
% Example:
%
%   stat=FlSETfile('set10may20_1122.dat');
%
% will create a new SET-file (set10may20_1122_new.dat) containing all of
% the original data from setfile, but with EXT and FF quadrupole and
% sextupole set-currents replaced by energy-scaled values corresponding to
% the present default FS optics and the energy implied by the BH1R
% set-current value from setfile

global FL BEAMLINE PS

debug=0;

% get Ibh1r from SET-file ... compute beam energy
cmd=['grep " BH1R " ',setfile];
[stat,r]=system(cmd);
if (stat~=0)
  error('cmd execution error: %s',cmd)
end
C=textscan(r,'%s %f %f %f %f');
Ibh1r=C{2};
energy=DR_energy(1,Ibh1r); % GeV

% get pointers to magnets (EXT+FF QUADS+SEXTs)
id1=FL.SimModel.extStart;
id2=length(BEAMLINE);
id=sort([findcells(BEAMLINE,'Class','QUAD',id1,id2), ...
         findcells(BEAMLINE,'Class','SEXT',id1,id2)]);
id=id(1:2:end);
name=cellfun(@(x) x.Name,BEAMLINE(id),'UniformOutput',false);
% QF13X is powered by QF15X in series ... no SET-file entry
id=strmatch('QF13X',name);name(id)=[];

% open input and output files
[fp,fn,fe]=fileparts(setfile);
outfile=fullfile(fp,strcat(fn,'_new',fe));
fidi=fopen(setfile,'rt');
fido=fopen(outfile,'wt');

% process magnets in SET-file
Nm=0;
done=false(size(name)); % magnet has been processed
while (true)
  s=fgetl(fidi);
  if (~ischar(s)),break,end
  if (~strcmp(s(1),'!'))
    C=textscan(s,'%s');
    if (strcmp(C{1}(1),'FFMAG'))
      C=textscan(s,'%s %s %f %f %f %f');
      psname=C{2};
    else
      C=textscan(s,'%s %f %f %f %f');
      psname=C{1};
    end
    id=strmatch(psname,name);
    if (~isempty(id))
      Nm=Nm+1;
      done(id)=true;
      % compute desired magnet current (scale to new SET-file energy)
      iel=findcells(BEAMLINE,'Name',name{id});iel=iel(1);
      ips=BEAMLINE{iel}.PS;
      B=(BEAMLINE{iel}.B*PS(ips).Ampl)*(energy/BEAMLINE{iel}.P);
      conv=FL.HwInfo.PS(ips).conv;
      I=interp1(conv(2,:),conv(1,:),abs(B),'linear');
      % check for wrong FF matching quad polarity
      if (~isempty(regexp(name{id},'QM1[1-6]FF')))
        pval=lcaGet(FL.HwInfo.PS(ips).polarity);
        if (sign(B)~=pval)
          fprintf(1,'FlSETfile: %s polarity?\n',name{id});
        end
      end
      % write new current into SET-file record
      ic=strfind(s,'.');ic=ic(1)+[-3,3];
      snew=s;
      snew(ic(1):ic(2))=sprintf('%7.3f',I);
      if (debug)
        fprintf(1,'%2d (old): %s\n',Nm,s);
        fprintf(1,'%2d (new): %s\n',Nm,snew);
      end
      s=snew;
    end
  end
  fprintf(fido,'%s\n',s);
end
fclose(fidi);
fclose(fido);

if (Nm~=length(name))
  fprintf(1,'FlSETfile: Nmag=%d , Nset=%d\n',length(name),Nm);
  id=find(done==false);
  for n=1:length(id)
    fprintf(1,'  v%s\n',name{id(n)});
  end
  stat=0;
else
  stat=1;
end

end
