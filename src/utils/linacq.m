qdat=load('c:/nlc/elin1/quad.dat');
isect=qdat(:,1);
iquad=qdat(:,2);
Bq=qdat(:,3);
Eq=qdat(:,4);
Lq=qdat(:,5);

Cb=33.35640952;
E0=Eq(1);
L=6.26983;
ax=-0.3026;
mux0=103.4272;
ay=-0.3105;
muy0=107.4554;

ans=prompt('x, y, or all?','xya','a');
if (strcmp(ans,'x'))
   idf=find((isect>0)&(isect<5)&(Bq>0));
   Ef=zeros(size(idf));
   Bf=zeros(size(idf));
   Bff=zeros(size(idf));
   for m=1:length(idf)
      n=idf(m);
      E=Eq(n);
      B=Bq(n);
      Ef(m)=E;
      Bf(m)=B;
      Bff(m)=2*Cb*E*sind(0.5*mux0*(E/E0)^ax)/L;
   end
   plot(Ef,Bf,'o',Ef,Bff,'--')
elseif (strcmp(ans,'y'))
   idd=find((isect>0)&(isect<5)&(Bq<0));
   Ed=zeros(size(idd));
   Bd=zeros(size(idd));
   Bdf=zeros(size(idd));
   for m=1:length(idd)
      n=idd(m);
      E=Eq(n);
      B=-Bq(n);
      Ed(m)=E;
      Bd(m)=B;
      Bdf(m)=2*Cb*E*sind(0.5*muy0*(E/E0)^ay)/L;
   end
   plot(Ed,-Bd,'o',Ed,-Bdf,'--')
else
   id=find((isect>0)&(isect<5));
   for m=1:length(id)
      n=id(m);
      E=Eq(n);
      B=Bq(n);
      if (B>0)
         Bf=2*Cb*E*sind(0.5*mux0*(E/E0)^ax)/L;
      else
         Bf=-2*Cb*E*sind(0.5*muy0*(E/E0)^ay)/L;
      end
      qname=sprintf('QQ%02d%02d',isect(n),iquad(n));
      disp(['  ' qname sprintf('  %11.5f  %11.5f',B,Bf)])
   end
end

