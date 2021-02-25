function Nray=dim2turtle(infile,outfile)
%
% Nray=dim2turtle(infile,outfile);
%
% Converts DIMAD output rays to TURTLE input rays
%
% INPUTs:
%
%   infile  = name of DIMAD output rays file
%   outfile = name of TURTLE input rays file
%
% OUTPUT:
%
%   Nray = number of rays processed

ray0=load(infile);
ray0=ray0(:,2:7);
[Nray,dummy]=size(ray0);

fid=fopen(outfile,'wt');
for n=1:Nray
  s=[];
  for m=1:6
    s=[s,strrep(sprintf('%13.4e',ray0(n,m)),'e-0','E-')];
  end
  fprintf(fid,'%s\n',s);
end
fclose(fid);

figure
subplot(321),hist(1e3*ray0(:,1),50),xlabel('X (mm)')
subplot(322),hist(1e3*ray0(:,2),50),xlabel('PX (mr)')
subplot(323),hist(1e3*ray0(:,3),50),xlabel('Y (mm)')
subplot(324),hist(1e3*ray0(:,4),50),xlabel('PY (mr)')
subplot(325),hist(1e3*ray0(:,5),50),xlabel('dL (mm)')
subplot(326),hist(1e2*ray0(:,6),50),xlabel('dP (%)')
