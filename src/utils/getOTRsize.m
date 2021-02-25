function [stat,data]=getOTRsize(otrNum)
%
% [stat,sig11,sig33,sig31,theta,sigx,sigy]=getOTRsize(otrNum);
%
% otrNum        = OTR number (1=OTR0X,2=OTR1X,3=OTR2X,4=OTR3X)
%
% data.sig11    = mean semi-major axis from ellipse fit (um^2)
% data.sig11err = rms semi-major axis from ellipse fit (um^2)
% data.sig33    = mean semi-minor axis from ellipse fit (um^2)
% data.sig33err = rms semi-minor axis from ellipse fit (um^2)
% data.sig13    = mean x-y correlation from ellipse fit (um^2)
% data.sig13err = rms x-y correlation from ellipse fit (um^2)
% data.projx    = mean projected horizontal size from gaussian fit (um)
% data.projxerr = rms projected horizontal size from gaussian fit (um)
% data.projy    = mean projected vertical size from gaussian fit (um)
% data.projyerr = rms projected vertical size from gaussian fit (um)
% data.theta    = mean x-y tilt angle of ellipse (deg)
% data.sigx     = mean projected horizontal size from ellipse fit(um)
% data.sigy     = mean projected vertical size from ellipse fit(um)

pvlist={ ...
  sprintf('mOTR:procData%d:sigma11',otrNum); ...     % mean semi-major axis (um^2)
  sprintf('mOTR:procData%d:sigma11err',otrNum); ...; % rms semi-major axis (um^2)
  sprintf('mOTR:procData%d:sigma33',otrNum); ...     % mean semi-minor axis (um^2)
  sprintf('mOTR:procData%d:sigma33err',otrNum); ...  % rms semi-major axis (um^2)
  sprintf('mOTR:procData%d:sigma13',otrNum); ...     % mean x-y correlation (um^2)
  sprintf('mOTR:procData%d:sigma13err',otrNum); ...; % rms x-y correlation (um^2)
  sprintf('mOTR:procData%d:projx',otrNum); ...       % mean horizontal projection (um^2)
  sprintf('mOTR:procData%d:projxerr',otrNum); ...    % rms horizontal projection (um^2)
  sprintf('mOTR:procData%d:projy',otrNum); ...       % mean vertical projection (um^2)
  sprintf('mOTR:procData%d:projyerr',otrNum); ...    % rms vertical projection (um^2)
  sprintf('mOTR:procData%d:ict',otrNum); ...         % mean ict reading during measurements (1e10)
  sprintf('mOTR:procData%d:icterr',otrNum); ...      % std ict reading during measurements (1e10)
};

try
  d=lcaGet(pvlist);
  sig11=d(1);sig11err=d(2);
  sig33=d(3);sig33err=d(4);
  sig13=d(5);sig13err=d(6);
  projx=d(7);projxerr=d(8);
  projy=d(9);projyerr=d(10);
  ict=d(11);icterr=d(12);
catch
  stat{1}=-1;stat{2}='getOTRsize: lcaGet failed';
  return
end

theta=0.5*atan2d(2*sig13,sig11-sig33); % tilt angle (deg)
b=[ ...
  sig11-sig13*sind(2*theta); ...
  sig33+sig13*sind(2*theta); ...
];
A=[ ...
  cosd(theta)^2,sind(theta)^2
  sind(theta)^2,cosd(theta)^2
];
x=A\b;
sigx=sqrt(x(1));
sigy=sqrt(x(2));

data.sig11=sig11;
data.sig11err=sig11err;
data.sig33=sig33;
data.sig33err=sig33err;
data.sig13=sig13;
data.sig13err=sig13err;
data.projx=projx;
data.projxerr=projxerr;
data.projy=projy;
data.projyerr=projyerr;
data.theta=theta;
data.sigx=sigx;
data.sigy=sigy;
data.ict=ict;
data.icterr=icterr;

stat{1}=1;

end