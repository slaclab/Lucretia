function password_generate(N)

uchar='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
if (nargin==0),N=1;end
rand('state',sum(100*clock))
for n=1:N
  id=1+round((length(uchar)-1)*rand(8,1));
  disp(['   ' uchar(id)])
end
