
% aidainit initializes a Matlab session for using Aida.
%
% ==============================================================
%
%  Name:  aidainit
%
%  Rem:   Aida requires a classpath, certain import statements, and
%         an instantiated Err singleton, to operate. aidasetup does
%         the first two of those. This M-file script additionally 
%         intantiates an Err object with a reasonable session name 
%         (used in cmlog 'Sys' column). The Err singleton 
%         can be acquired by the Matlab session after calling
%         this M-file by calling Err.getInstance() giving no-arg. 
%
%         If you want more control over name used in
%         Err.getInstance(name), use aidasetup instead, whcih does
%         not include the Err.getInstance part.
%
%  Usage: aidainit
%
%  Side:  Sets aidainitdone. If aidainitdone is 1, this script will
%         not re-execute since the Err singleton has already been
%         set.
%
%  Auth:  06-Apr-2005, Greg White (greg): 
%  Rev:   
%
%--------------------------------------------------------------
% Mods: (Latest to oldest)
%         09-May-2005, Greg White (greg)
%         Removed import statements, since those are part of
%         aidasetup.m. That will work as long as aidasetup is a
%         script not a function.
%
%============================================================== 

aidasetup

global aidainitdone
if isempty(aidainitdone)

  % Error handling. User M-files may get the error object so
  % constructed by just calling, for instance err = Err.getInstance()
  % (without an argument), that will return the singleton.
  %
  if strcmp(computer,'PCWIN') == 1
    cmlogsys = 'PCMatlab';
  else
    user = getenv('USER');
    cmlogsys = strcat(user,'sMatlab');
  end
  try
    Err.getInstance(cmlogsys);  
  catch
    disp 'Err instance already instantiated'
  end
    
  % Tidy up and remember that we've done this
  %
  aidainitdone = 1;
  disp 'Aida client initialization completed';

end

