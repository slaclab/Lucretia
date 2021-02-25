classdef LSF_Example < handle & batchLSF % Your Matlab class must inherit these superclasses
  % To run:
  % >> L = LSF_Example(10); % instantiate object, requested 10 seeds
  % (Good idea to always test job before submitting):
  % >> L.submitFunc(1) % Test code runs natively on Matlab for first seed
  % >> L.prepare() % Compiles code and generates scripts for job submission
  % >> L.testRun() % Tests running compiled code with seed # 1 on local computer - check output as expected
  % >> L.submit() % Submits all jobs to LSF (text output indicates progress and when jobs have completed)
  % >> L.analyze() % look at results
  % (Or, if you are bold, a shortcut to compile and then submit in one step after making L object is):
  % >> L.run() % Compiles code, makes submission script files and then submits all jobs to LSF
  properties
    tmpDirectory='/scratch'; % /scratch is defined on all public batch machines @ SLAC
    jobName='TestLSF'; % Name to use for job submission on LSF queue
    dataDirectory='/nfs/slac/g/nlcrd/u01/whitegr/test'; % Where resources files to be stored on running job (should already exist)
    runDirectory='/nfs/slac/g/nlcrd/u01/whitegr/test'; % where the files to run the code are temporarily stored (should already exist)
    batchQueueName='short'; % Name of LSF queue to run on (what gets appended to "bsub -q ..." command at submission time)
    mcrRoot='/nfs/slac/g/nlcrd/u01/whitegr/mcr/v96'; % Location of Matlab runtime libraries (must be on shared filesystem visible to batch nodes)
    % Add any other user-defined properties required below...
    myprop=42;
  end
  methods
    function obj=LSF_Example(nseed) % Conventionally, write constructor to take a single argument as number of seeds
      if nargin==0 % Constuctor must run with no input arguments for execution time processing
        return
      end
      obj.submitWaitTime=0.1; % Time between each submission command (if too short then errors occur)
      obj.maxfails=5; % Max number of times to allow a single job seed to fail (at other times auto-re-submission occurs)
      obj.wallTime=5; % Max run time reported to scheduler (in minutes)
      obj.nseed=nseed; % Must set this property to declare how many jobs to submit
      % Copy any supporting files (.mat data files etc) to runDirectory here
      % copyfile(myfile,obj.dataDirecory) ...
      % You can also set any additional shared object dependencies here (if you have mex files that require them for example)
        % obj.soLibs={'mylib1.so' 'mylib2.so'};
        
      % Add any other user-defined constructor actions below...

    end
    function data=submitFunc(obj,iseed) % Must supply this method which takes just run-time seed ID as input (use other class properties and methods to provide additional input)
      
      % This is where you put code to run on batch machines, and return any seed-dependent results in data variable (can be any matlab variable class)
      rng(iseed); % e.g. initialize random number generator to this seed instance
      data = rand * obj.myprop ; % data for this seed
    end
    function analyze(obj) % Optionally write additional User methods, e.g. to analyze data
      obj.collectData(); % By default, all data is returned automatically to calling workspace, but this command forces all returned data to be returned (if Matlab session aborted and resumed for example)
      % Loop though all seeds and do something with computed results
      for iseed=1:obj.nseed
        jobdata = obj.retdata{iseed} ; % All data available in this property as cell array with [1,obj.nseed] dimension
        fprintf('Seed # %d, data= %g\n',iseed,jobdata)
      end
    end
  end
end
