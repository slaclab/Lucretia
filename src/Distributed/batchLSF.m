classdef(Abstract) batchLSF < handle
  % BATCHLSF Abstract Class for management of jobs over LSF batch scheduler using
  % compiled matlab code (requires Matlab Compiler toolbox)
  % To Use:
  %   Generate a subclass which inherits from batchLSF and implements
  %   Abstract methods and properties. Put code to execute in submitFunc
  %   which can define additional static input arguments but must have iseed
  %   as the first argument to provide the instance seed # and must have 
  %   a single output argument which is used to return data
  %   (can be any allowed matlab data class).
  %   Set other properties as required and submit jobs using run()
  %   method to compile and run code, or separately use prepare() and
  %   submit() methods.
  %   NB: any constructor method must be able to function with no input arguments
  
  properties(Abstract)
    jobName % String to identify job queue entries and data files
    dataDirectory % Directory to store output data file: <jobName>_<iseed>.mat. Path should be absolute.
    runDirectory % Directory to compile and execute code. Path should be absolute.
    tmpDirectory % Temporary directory for generating cache files
    batchQueueName % Name of batch queue to submit jobs to
    mcrRoot % Location of MCR runtime directory
  end
  
  properties
    maxfails=3; % max number of job seed submission failures to tolerate
    pollTime=120; % Delay between polling of LSF bacth queue / s
    wallTime=120; % Max wall clock time on submission engine machine / min
    supportFiles={}; % List of files to embed with run time object (e.g. mat files required)
    soLibs={}; % List of directories to search for any shared object libraries associated mex files may need
    envVars={}; % environment variable definitions to pass to run time environment - must pass as pairs of strings
    nseed=1; % Number of seeds to run
    runArgs=[]; % run arguments (added to -R switch in bsub command)
    autocollect=true; % automatically return submitFunc() method return data to this object or do manually with collectData() method
    storeJobOutput=false; % keep screen output of jobs?
    storeErrOutput=false; % keep error output stream?
    maxsimrun=1024; % Max # jobs to be simultaneously running
    submitWaitTime=1; % wait time between submission commands / s
    crashCleanup=false; % Set true to auto-delete ~/matlab_crash_dump files
  end
  
  properties%(SetAccess=private)
    retdata={}; % Holder for returned data from batch jobs
  end
  
  methods(Abstract)
    storedData=submitFunc(obj,iseed) % returned data in storedData matlab data element is saved in <jobName>_<iseed>.mat
  end
  
  methods
    function obj=batchLSF()
    end
    function run(obj)
      % compile (if autocompile property set) and submit job for running on cluster
      obj.prepare();
      obj.submit();
    end
    function prepare(obj)
      % Prepare exe, wrapper file and submit scripts
      obj.argCheck();
      obj.generateRunFunction();
      obj.compile();
    end
    function cleanData(obj)
      % clear previous job data
      delete(fullfile(obj.dataDirectory,sprintf('%s_*.mat',obj.jobName)));
    end
    function collectData(obj)
      % collect data from previously run job
      obj.retdata=cell(1,obj.nseed);
      for icoll=1:obj.nseed
        try
          ld=load(fullfile(obj.dataDirectory,sprintf('%s_%d',obj.jobName,icoll)));
          obj.retdata{icoll}=ld.outdata;
        catch
          obj.retdata{icoll}=[];
        end
      end
    end
    function data=testRun(obj)
      % TESTRUN - run compiled object once locally to test it works
      d0=pwd;
      cd(obj.runDirectory);
      rstr=sprintf('%s.sh %s 1',fullfile(obj.runDirectory,sprintf('run_%s',obj.jobName)),num2str(ceil(sum(clock).*100000)));
      sret=system(rstr);
      if sret~=0
        cd(d0)
        error('System error executing compiled binary')
      end
      obj.collectData();
      data=obj.retdata{1};
      cd(d0)
    end
    function submit(obj)
      % SUBMIT - submit jobs to batch queue and wait for all jobs to
      % complete running or fail
      obj.argCheck();
      obj.retdata=cell(1,obj.nseed);
      nfail=zeros(1,obj.nseed);
      jcomplete=[];
      jcollected=[];
      jfailed=[];
      hassub=zeros(1,obj.nseed);
      t0=clock;
      while length(jcomplete)<obj.nseed
        % Get list of completed or already running jobs
        [~,jrunstr]=system('bjobs -w');
        jrun=cellfun(@(x) str2double(x{1}),regexp(jrunstr,sprintf('%s_(\\d+)',obj.jobName),'tokens'));
        d=dir(fullfile(obj.dataDirectory,sprintf('%s*.mat',obj.jobName)));
        if ~isempty(d)
          jcomplete=arrayfun(@(x) str2double(cell2mat(regexp(d(x).name,'_(\d+)','tokens','once'))),1:length(d));
          jcomplete(isnan(jcomplete))=[];
        end
%         npend=length(regexp(jrunstr,'PEND'));
        nrun=length(jrun);
        % Submit required jobs
        if length(jcomplete)~=obj.nseed
          for iseed=1:obj.nseed
            if ~ismember(iseed,jrun) && ~ismember(iseed,jcomplete) && ~ismember(iseed,jfailed) && nrun<obj.maxsimrun
              % Setup error and output destinations
              if obj.storeErrOutput
                edest=fullfile(obj.dataDirectory,sprintf('%s_jErrOutput_%d.txt',obj.jobName,iseed));
              else
                edest='/dev/null';
              end
              if ~obj.storeJobOutput
                odest='/dev/null';
              else
                odest=fullfile(obj.dataDirectory,sprintf('%s_joutput_%d.txt',obj.jobName,iseed));
              end
              % Submit Job
              pause(obj.submitWaitTime)
              hassub(iseed)=hassub(iseed)+1;
              nrun=nrun+1;
              % -cwd "/scratch/jobcwd/%U/%J_%I" myjob
              % -E "pre_exec_command [arguments ...]" 
              if isempty(obj.runArgs)
                rstr=sprintf('bsub -q %s -cwd %s -oo %s -eo %s -J %s_%d -W %d -C 0 %s.sh %s %d',...
                obj.batchQueueName,obj.runDirectory,odest,edest,obj.jobName,iseed,obj.wallTime,...
                fullfile(obj.runDirectory,sprintf('run_%s',obj.jobName)),num2str(ceil(sum(clock).*100000)),iseed);
              else
                rstr=sprintf('bsub -q %s -cwd %s -oo %s -eo %s -J %s_%d -W %d -C 0 -R %s %s.sh %s %d',...
                obj.batchQueueName,obj.runDirectory,odest,edest,obj.jobName,iseed,obj.wallTime,obj.runArgs,...
                fullfile(obj.runDirectory,sprintf('run_%s',obj.jobName)),num2str(ceil(sum(clock).*100000)),iseed);
              end
              stat=system(rstr);
              if stat~=0
                warning('Error status reported from bsub for iseed= %d!',iseed)
              end
              % Keep track of # of job failures
              if hassub(iseed)>1
                nfail(iseed)=nfail(iseed)+1;
                if nfail(iseed)>=obj.maxfails
                  jfailed(end+1)=iseed;
                  warning('Max tries exeeded for iseed= %d!',iseed)
                end
              end
            end
          end
        end
        % Test to see if done
        if (length(jfailed)+length(jcomplete))>=obj.nseed
          disp('All jobs complete.')
          if ~isempty(jfailed)
            disp('Some exeeded maxfail limit: iseed=')
            disp(jfailed)
          end
          break
        end
        % autocollect data?
        if obj.autocollect
          jcoll=jcomplete(~ismember(jcomplete,jcollected));
          if ~isempty(jcoll)
            for icoll=jcoll
              jcollected(end+1)=icoll; %#ok<*AGROW>
              try
                ld=load(fullfile(obj.dataDirectory,sprintf('%s_%d',obj.jobName,icoll)));
                obj.retdata{icoll}=ld.outdata;
              catch
                obj.retdata{icoll}=[];
              end
            end
          end
        end
        % Report progress
        fprintf('# Completed: %d / %d (rate = %.2f / hr)\n',length(jcomplete),obj.nseed,length(jcomplete)/(etime(clock,t0)/3600))
        [val,ind]=max(nfail);
        fprintf('Max failed jobs: %d (iseed=%d)\n',val,ind)
        disp('================')
        % wait time
        pause(obj.pollTime)
      end
    end
    function loadData(obj,savedData)
      obj.retdata=savedData;
    end
  end
  
  methods(Access=protected)
    function argCheck(obj)
%       a=methods(obj,'-full');
%       nsubarg=length(regexp(regexp(a{cellfun(@(x) ~isempty(x),regexp(a,'submitFunc','once'))},'(.+$','match','once'),',','once'));
      if ~isempty(obj.envVars)
        if mod(length(obj.envVars),2)
          error('Must have pairs of arguments for envVars property')
        end
      end
      if ~strcmp(obj.runDirectory(end),'\') && ~strcmp(obj.runDirectory(end),'/')
        obj.runDirectory=sprintf('%s%c',obj.runDirectory,filesep);
      end
      if ~strcmp(obj.dataDirectory(end),'\') && ~strcmp(obj.dataDirectory(end),'/')
        obj.dataDirectory=sprintf('%s%c',obj.dataDirectory,filesep);
      end
    end
    function generateRunFunction(obj)
      % GENERATERUNFUNCTION
      % Write wrapper function for compiling for submission to cluster
      fname=sprintf('xXx_%s_xXx.m',obj.jobName);
      fid=fopen(fullfile(obj.runDirectory,fname),'w');
      if ~fid
        error('Failed to open temp file for writing job submission wrapper function')
      end
%       thisClassName=class(obj);
      line=sprintf('function ret=xXx_%s_xXx(iseed',obj.jobName);
      fprintf(fid,'%s)\n',line);
      fprintf(fid,'%% >> Auto generated wrapper function for job submission by batchLSF object <<\n');
      fprintf(fid,'ret=0;\n');
      fprintf(fid,'iseed=str2double(iseed);\n');
      BO=obj;
      save(fullfile(obj.runDirectory,'xXx__subObj__xXx.mat'),'BO');
%       fprintf(fid,'BO=%s;\n',thisClassName);
      fprintf(fid,'load xXx__subObj__xXx.mat BO\n');
      line='outdata=submitFunc(BO,iseed';
      fprintf(fid,'%s);\n',line);
      fprintf(fid,'lockfile=fullfile(''%s'',''batchLSFLockFile'');\n',obj.dataDirectory);
      fprintf(fid,'ptime=rand*2;\n');
      fprintf(fid,'while exist(lockfile,''file'')\n');
      fprintf(fid,'  d=dir(lockfile);\n');
      fprintf(fid,'  if isempty(d) || (now-d.datenum > 0.01)\n');
      fprintf(fid,'    break\n');
      fprintf(fid,'  end\n');
      fprintf(fid,'  pause(ptime);\n');
      fprintf(fid,'end\n');
      fprintf(fid,'try\n');
      fprintf(fid,'  system(sprintf(''touch %%s'',lockfile))\n');
      fprintf(fid,'  save(fullfile(''%s'',sprintf(''%s_%%d.mat'',iseed)),''outdata'');\n',obj.dataDirectory,obj.jobName);
      fprintf(fid,'catch ME\n');
      fprintf(fid,'  delete(lockfile);\n');
      fprintf(fid,'  rethrow(ME)\n');
      fprintf(fid,'end\n');
      fprintf(fid,'delete(lockfile);\n');
      fprintf(fid,'close all\n');
      fprintf(fid,'return\n');
      fclose(fid);
      % Make system run script
      tmpdir=sprintf('%s/%s_$tmpid',obj.tmpDirectory,obj.jobName);
      fid=fopen(fullfile(obj.runDirectory,sprintf('run_%s.sh',obj.jobName)),'w');
      fprintf(fid,'#!/bin/bash\n');
      fprintf(fid,'MCRROOT="%s"\n',obj.mcrRoot);
      fprintf(fid,'tmpid=$1\n');
      fprintf(fid,'shift\n');
      fprintf(fid,'mkdir %s\n',tmpdir);
      fprintf(fid,'export MCR_CACHE_ROOT=%s\n',tmpdir);
      fprintf(fid,'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.:${MCRROOT}/runtime/glnxa64\n');
      fprintf(fid,'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/bin/glnxa64\n');
      fprintf(fid,'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/os/glnxa64\n');
      fprintf(fid,'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/opengl/lib/glnxa64\n');
      if ~isempty(obj.soLibs)
        for ilib=1:length(obj.soLibs)
          fprintf(fid,sprintf('LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:%s\n',obj.soLibs{ilib}));
        end
      end
      fprintf(fid,'export LD_LIBRARY_PATH;\n');
      if ~isempty(obj.envVars)
        for ivar=1:2:length(obj.envVars)
          fprintf(fid,sprintf('export %s=%s\n',obj.envVars{ivar},obj.envVars{ivar+1}));
        end
      end
      fprintf(fid,'args=\n');
      fprintf(fid,'while [ $# -gt 0 ]; do\n');
      fprintf(fid,'  token=$1\n');
      fprintf(fid,'  args="${args} \"${token}\"" \n');
      fprintf(fid,'  shift\n');
      fprintf(fid,'done\n');
      fprintf(fid,'cp -r %s/* %s\n',obj.runDirectory,tmpdir);
      fprintf(fid,'cd %s\n',tmpdir);
      fprintf(fid,'eval "\"%s\"" $args\n',sprintf('./xXx_%s_xXx',obj.jobName));
      fprintf(fid,'cd ~\n');
      fprintf(fid,'rm -rf %s\n',tmpdir);
      if obj.crashCleanup
        fprintf(fid,'rm -f ~/matlab_crash_dump.*');
      end
      fclose(fid);
      system(sprintf('chmod a+x %s',fullfile(obj.runDirectory,sprintf('run_%s.sh',obj.jobName))));
    end
    function compile(obj)
      % COMPILE
      % Compile job to be run on cluster
      mccOPTS=[];
      if ~isempty(obj.supportFiles)
        for ifile=1:length(obj.supportFiles)
          mccOPTS=[mccOPTS ' -a ' obj.supportFiles{ifile}];
        end
      end
      if exist(sprintf('xXx_%s_xXx',obj.jobName),'file')
        delete(sprintf('xXx_%s_xXx',obj.jobName));
      end
      evalc(sprintf('mcc -I . -I %s %s -m %s',obj.runDirectory,mccOPTS,sprintf('xXx_%s_xXx.m',obj.jobName)));
      if ~strcmp(obj.runDirectory,'.') && ~strcmp(obj.runDirectory,'./') && ~strcmp(obj.runDirectory,pwd)
        movefile(sprintf('xXx_%s_xXx',obj.jobName),obj.runDirectory)
      end
    end
  end
  
end

