function gpdata=guineapig(erfile,prfile,afile)
%
% gpdata=guineapig(erfile,prfile,afile);
%
% Run GuineaPig starting from two matLIAR (DIMAD) output rays files
% (use "add_charge=.t." in LIAR "meas_part" command to get particle
% charge information) and a GuineaPig "acc.dat" template file.
%
% INPUTs
%
%   erfile = matLIAR (DIMAD) "electron" output rays file name
%   prfile = matLIAR (DIMAD) "positron" output rays file name
%   afile  = GuineaPig "acc.dat" template file name
%            (see acc_nm_*.dat files in GuineaPig directory)
%
% OUTPUT
%
%   gpdata = GuineaPig output (see the user's guide)
%
%            [n_m,lumi_fine;lumi_ee;lumi_ee_high; ...
%             bpm_vx.1;bpm_sig_vx.1;bpm_vy.1;bpm_sig_vy.1; ...
%             bpm_vx.2;bpm_sig_vx.2;bpm_vy.2;bpm_sig_vy.2];
%             

% set these paths for your particular system

gdir='D:\guineapig';      % GuineaPig directory
wdir='D:\guineapig\work'; % working directory

% make input filespecs DOS compatible

erfiled=strrep(erfile,'/','\'); % DOS format
prfiled=strrep(prfile,'/','\'); % DOS format
afiled=strrep(afile,'/','\');   % DOS format
disp(' ')

% copy matLIAR (DIMAD) output ray files to working directory

disp('   copying files ...')

cmd=['copy ',erfiled,' ',wdir,'\electronc.ini'];
[iss,result]=dos(cmd);
if (iss)
  error(['DOS copy error (',int2str(iss),'): ',erfile])
end
cmd=['copy ',prfiled,' ',wdir,'\positronc.ini'];
[iss,result]=dos(cmd);
if (iss)
  error(['DOS copy error (',int2str(iss),'): ',prfile])
end

% copy GuineaPig "acc.dat" template file to working directory

cmd=['copy ',gdir,'\',afiled,' ',wdir,'\acc_nm.dat'];
[iss,result]=dos(cmd);
if (iss)
  error(['DOS copy error (',int2str(iss),'): ',afile])
end

% cd to the working directory

wd=cd;
cd(wdir)

% run Andrei's GuineaPig prep program

disp('   preparing GuineaPig input files ...')

cmd=[gdir,'\prepare_gp'];
[iss,result]=dos(cmd);
if (iss)
  cd(wd)
  error('prepare_gp failed')
end

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
