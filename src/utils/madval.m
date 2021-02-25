function s=madval(rval)

if (rval==0)
  s='0.0';
else
  aval=abs(rval);
  if ((aval>0.01)&(aval<100))
    iexp=0;
  else
    iexp=fix(log10(aval));
    rval=rval*10^(-iexp);
  end
  s=sprintf('%20.12f',rval);
  for n=20:-1:10
    if (~strcmp(s(n),'0')),break,end
    s(n)=[];
  end
  if (iexp~=0),s=[s,sprintf('E%d',iexp)];end
  s=fliplr(deblank(fliplr(s)));
end
    