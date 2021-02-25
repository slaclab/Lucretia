function gpdata=guineapig0(gdir,wdir)
%
% gpdata=guineapig0(gdir,wdir);
%
% Run GuineaPig assuming files electron.ini, positron.ini, and acc.dat already
% exist in the specified working directory.
%
% INPUTs:
%
%   gdir = path to folder containing GuineaPig executable
%   wdir = path to folder containing input files
%
% OUTPUT:
%
%   gpdata = GuineaPig output (see the user's guide)
%
%            [n_m,lumi_fine;lumi_ee;lumi_ee_high; ...
%             bpm_vx.1;bpm_sig_vx.1;bpm_vy.1;bpm_sig_vy.1; ...
%             bpm_vx.2;bpm_sig_vx.2;bpm_vy.2;bpm_sig_vy.2];
%             

% make gdir path DOS compatible

gdir=strrep(gdir,'/','\'); % DOS format
if (strcmp(gdir(end),'\'))
  gdir(end)=[];
end

disp(' ')

% cd to the working directory

wd=cd;
cd(wdir)

% get accelerator name, computational parameter set name, and number
% of particles (n_m) from acc.dat

aname=[];
pname=[];
fid=fopen('acc.dat','r');
while (1)
  temp=fgetl(fid);
  if (~ischar(temp)),break,end
  n=strfind(temp,'$ACCELERATOR::');
  if (~isempty(n))
    n1=n+length('$ACCELERATOR::');
    n2=length(temp);
    aname=deblank(temp(n1:n2));
    aname=fliplr(deblank(fliplr(aname)));
  end
  n=strfind(temp,'$PARAMETERS::');
  if (~isempty(n))
    n1=n+length('$PARAMETERS::');
    n2=length(temp);
    pname=deblank(temp(n1:n2));
    pname=fliplr(deblank(fliplr(pname)));
  end
  if (strfind(temp,'n_m='))
    n1=strfind(temp,'=')+1;
    n2=strfind(temp,';')-1;
    n_m=str2num(temp(n1:n2));
  end
end
fclose(fid);
if (isempty(aname)|isempty(pname))
  cd(wd)
  error('Can''t find $ACCELERATOR:: and/or $PARAMETERS:: in acc.dat')
end

% run GuineaPig

disp(['   running GuineaPig (',int2str(n_m),' macroparticles) ...'])

cmd=[gdir,'\guinea ',aname,' ',pname,' gp.out > gp.log'];
[iss,result]=dos(cmd);
if (iss)
  cd(wd)
  error('guinea failed')
end

% extract results from acc.out

disp('   extracting results ...')

cmd=[gdir,'\read_gp_out < gp.out > gp_out.m'];
[iss,result]=dos(cmd);
if (iss)
  cd(wd)
  error('read_gp_out failed')
end

gp_out
gpdata=[n_m;lumi_fine;lumi_ee;lumi_ee_high; ...
  bpm_vx_1;bpm_sig_vx_1;bpm_vy_1;bpm_sig_vy_1; ...
  bpm_vx_2;bpm_sig_vx_2;bpm_vy_2;bpm_sig_vy_2];

% cd back to starting directory

cd(wd)
disp(' ')
