Cb=33.35640952;
out=[];
for n=1:length(L)
  if (strcmp(K(n,:),'QUAD'))
    B=Cb*E(n)*P(n,2)*A(n);
    g=0.1*B/A(n);
    out=[out;N(n,:),T(n,:),sprintf(' %9.5f %6.1f',B,g)];
  end
end