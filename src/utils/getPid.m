function pid=getPid(iocName,doWait)
if ~exist('doWait','var') || doWait
  nCount=20;
else
  nCount=1;
end
tryCount=0; pid=[];
while isempty(regexp(evalc('!ps am'),['/' iocName], 'once')) && tryCount<nCount
  pause(0.5)
  tryCount=tryCount+1;
end
if tryCount<nCount
  pid=regexp(evalc(['!ps am | grep "/',iocName,'"']),'(\d+)','tokens');
end