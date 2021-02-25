function L=choldc(a)
[nrow,ncol]=size(a);
if (nrow~=ncol),error('a must be square'),end
n=nrow;
temp=a;
p=zeros(n,1);
for i=1:n
   for j=1:n
      sum=a(i,j);
      for k=(i-1):-1:1
         sum=sum-a(i,k)*a(j,k);
      end
      if (i==j)
         if (sum<=0),error(sprintf('diagonal %d <=0 (%12.4e)',i,sum)),end
         p(i)=sqrt(sum);
      else
         a(j,i)=sum/p(i);
      end
   end
end
L=zeros(n,n);
for i=1:n
   for j=1:n
      if (i==j)
         L(i,j)=p(i);
      else
         L(i,j)=a(i,j);
      end
   end
end