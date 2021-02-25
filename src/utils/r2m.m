function M=r2m(R)
%
% M=r2m(R)
%
%   R is a 2x2 R-matrix
%   M is a 3x3 Twiss parameter transport matrix
%
M=eye(2);
M(1,1)=R(1,1)^2;
M(1,2)=-2*R(1,1)*R(1,2);
M(1,3)=R(1,2)^2;
M(2,1)=-R(1,1)*R(2,1);
M(2,2)=R(1,1)*R(2,2)+R(1,2)*R(2,1);
M(2,3)=-R(1,2)*R(2,2);
M(3,1)=R(2,1)^2;
M(3,2)=-2*R(2,1)*R(2,2);
M(3,3)=R(2,2)^2;
M=M/det(R);
