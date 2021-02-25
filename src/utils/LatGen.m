function [lat_out] = LatGen(modelname)
% Generate output structure of lattice info

[tt,type,lat_out.name,L,P,A,T,E,FDN,twss,orbt,S]=xtfft2mat(modelname);

lat_out.isbpm=zeros(length(type),1);
lat_out.isquad=zeros(length(type),1);
lat_out.isdrift=zeros(length(type),1);
lat_out.issext=zeros(length(type),1);
lat_out.isoct=zeros(length(type),1);
lat_out.ismark=zeros(length(type),1);
lat_out.isother=zeros(length(type),1);
lat_out.s=S;

for ind=1:length(type)
  switch type(ind,:)
    case 'MONI'
      lat_out.isbpm(ind)=1;
    case 'QUAD'
      lat_out.isquad(ind)=1;
    case 'DRIF'
      lat_out.isdrift(ind)=1;
    case 'SEXT'
      lat_out.issext(ind)=1;
    case 'OCTU'
      lat_out.isoct(ind)=1;
    case 'MARK'
      lat_out.ismark(ind)=1;
    otherwise
      lat_out.isother(ind)=1;
  end
  lat_out.group_members{ind}=find(lat_out.s(ind)==lat_out.s);
end

% Twiss params
twiss_names={'mux','betx','alfx','dx','dpx','muy','bety','alfy','dy','dpy'};
for f=1:length(twiss_names)
    evalc(['lat_out.twiss.',twiss_names{f},'=twss(:,',num2str(f),')']);
end

return