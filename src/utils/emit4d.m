
% measured beam matrix from ATF EXT-line (16-DEC-1998)

sigm=[ 0.207608947052626E-07 -0.358805225003625E-08  0.188994869419206E-08 -0.356578805894645E-10
      -0.358805225003625E-08  0.764725490081904E-09 -0.140978531590826E-09 -0.321355499854593E-11
       0.188994869419206E-08 -0.140978531590826E-09  0.691984845950062E-09 -0.536925665775580E-10
      -0.356578805894645E-10 -0.321355499854593E-11 -0.536925665775580E-10  0.660205889063416E-11];
   
% use Fartouk method to compute intrinsic emittances e1 and e2
   
J=[ 0  1  0  0
   -1  0  0  0
    0  0  0  1
    0  0 -1  0];
t2=trace((sigm*J)^2);
t4=trace((sigm*J)^4);
e1=sqrt((-t2+sqrt(-t2^2+4*t4))/4);
e2=sqrt((-t2-sqrt(-t2^2+4*t4))/4);
E=diag([e1,e1,e2,e2]);

use_TRANSPORT_R=1;
if (use_TRANSPORT_R)
   
%  define R such that R*sigm*R'=E (R comes from TRANSPORT calculation ... it's symplectic)
   
   R=[-0.63401  -2.51765   0.00729   0.01820
      -0.20181  -2.38372  -0.01762  -0.48600
      -0.12719  -0.47053   0.59079   4.47856
       0.08112   0.32544  -0.27743  -0.40500];

%  recompute R such that R*E*R'=sigm

   R=inv(R);
else
   
%  get eigenvalues D and right eigenvectors R of sigm such that R'*sigm*R=D

   [R,D]=eig(sigm);

%  compute normalization matrix C such that if Rbar=C*R', Rbar*sigm*Rbar'=E

   C=sqrt(E*inv(D));
   Rbar=C*R';

%  recompute R such that R*E*R'=sigm (unfortunately, this R is not symplectic)

   R=inv(Rbar);
end

% begin decomposition of R

[XR,D]=eig(R);
[XRt,Dt]=eig(R');