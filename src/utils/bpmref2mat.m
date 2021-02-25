function [tt,micr,unit,name,x,y,tmit,stat]=bpmref2mat(fname)
%
% [tt,micr,unit,name,x,y,tmit,stat]=bpmref2mat(fname);
%
% Download BPM data from a config file
%
% INPUT:
%
%   fname = BPM config file name
%           (e.g. 'SLC_CNF:[BPM_P2BPMLER]BPM_P2BPMLER$1080.CNF' for NORMAL,
%            or   'SCR_CNF:[BPM_P2BPMLER]BPM_P2BPMLER$12200.CNF' for SCRATCH)
%
% OUTPUTs:
%
%   tt   = config title
%   micr = list of BPM micros (e.g. 'PR02')
%   unit = list of BPM units (e.g. 3017)
%   name = list of BPM names (e.g. 'M773017T')
%   x    = x data
%   y    = y data
%   tmit = tmit data
%   stat = decimal stat data (converted from hexadecimal)

if (isempty(findstr('[BPM_',fname))),error([fname,' is not a BPM config file']),end
fid=fopen(fname,'r');
if (fid==-1),error([fname,' not found']),end

tt=[];
micr=[];
unit=[];
name=[];
x=[];
y=[];
tmit=[];
stat=[];

while (1)
  s=fgetl(fid);
  if (~ischar(s)),break,end
  s=deblank(s);
  if (strmatch(' !TITLE=',s))
    tt=s(9:end);
  elseif (strmatch(' %BPMS,',s))
    if (length(s)<65)
      disp(' ')
      disp(sprintf('   *** line="%s"',s))
      disp(sprintf('   *** length=%d',length(s)))
      disp(' ')
      error('BPM config file error')
    end
    micro=s(8:11);
    bitid=micr2bitid(micro);
    micr=[micr;micro];
    unit=[unit;str2num(s(15:18))];
    name=[name;sprintf('M%02d%sT',bitid,s(15:18))];
    x=[x;str2num(s(20:32))];
    y=[y;str2num(s(34:46))];
    tmit=[tmit;str2num(s(48:60))];
    stat=[stat;hex2dec(s(62:65))];
  end
end

fclose(fid);

return

function bitid=micr2bitid(micr)
%
% bitid=micr2bitid(micr)
%

% microname/bitid data from ref_dbsfile:microname.dat (16-JUL-1998)
 
bitids=[    0 ;    1 ;    2 ;    3 ;    4 ;    5 ;    6 ;    7 ;    8 ; ...
            9 ;   10 ;   11 ;   12 ;   13 ;   14 ;   15 ;   16 ;   17 ; ...
           18 ;   19 ;   20 ;   21 ;   22 ;   23 ;   24 ;   25 ;   26 ; ...
           27 ;   28 ;   29 ;   30 ;   31 ;   32 ;   33 ;   34 ;   35 ; ...
           36 ;   37 ;   38 ;   39 ;   40 ;   41 ;   42 ;   43 ;   44 ; ...
           45 ;   46 ;   47 ;   48 ;   49 ;   50 ;   51 ;   52 ;   53 ; ...
           54 ;   55 ;   56 ;   57 ;   58 ;   59 ;   60 ;   61 ;   62 ; ...
           63 ;   64 ;   66 ;   67 ;   68 ;   69 ;   70 ;   71 ;   72 ; ...
           73 ;   74 ;   75 ;   76 ;   77 ;   78 ;   79 ;   80 ;   81 ; ...
           82 ;   83 ;   84 ;   85 ;   86 ;   87 ;   88 ;   90 ;   91 ; ...
           92 ;   93 ;   94 ;   95 ;   96 ;   97 ;   98 ;   99 ;  100 ; ...
          103 ;  104 ;  106 ;  151 ;  152 ;  153 ;  154 ;  155 ;  156 ; ...
          157 ;  158 ;  159 ;  160 ;  161 ;  256 ;  257 ;  258 ;  259 ; ...
          260 ;  261 ;  262 ;  263 ;  264 ;  265 ;  266 ];
 
micrs= ['LI00';'LI01';'LI02';'LI03';'LI04';'LI05';'LI06';'LI07';'LI08'; ...
        'LI09';'LI10';'LI11';'LI12';'LI13';'LI14';'LI15';'LI16';'LI17'; ...
        'LI18';'LI19';'LI20';'LI21';'LI22';'LI23';'LI24';'LI25';'LI26'; ...
        'LI27';'LI28';'LI29';'LI30';'LI31';'DR01';'DR02';'DR03';'FB31'; ...
        'FB30';'MP00';'CL01';'MP01';'DR11';'DR12';'DR13';'CA01';'CA02'; ...
        'CA03';'CA11';'CA12';'CA13';'FB73';'FF01';'FF11';'FB69';'MC00'; ...
        'LI32';'NP25';'FG00';'EP01';'EP02';'EP05';'FB88';'PT01';'TL01'; ...
        'TL00';'RP01';'XX00';'XX01';'XX02';'XX03';'XX04';'XX05';'XX06'; ...
        'XX07';'XX08';'XX09';'XX10';'PR02';'PR04';'PR06';'XD01';'TA01'; ...
        'TA02';'PI00';'PI01';'PI11';'PR08';'PR10';'AM00';'LI33';'LI34'; ...
        'CB00';'CB01';'CB02';'AB01';'LC00';'IG00';'PR01';'PR12';'LI35'; ...
        'BD01';'RF00';'PR13';'PC00';'PC01';'PC02';'PC03';'PC04';'PC05'; ...
        'PC06';'PC99';'MV01';'PX00';'PX01';'CW01';'CW02';'CW03';'CW04'; ...
        'CW05';'CW06';'CW07';'CW08';'CW09';'FBUS';'TEST'];

N=length(micrs);
n=0;
bitid=[];
while ((isempty(bitid))&(n<N))
   n=n+1;
   if (strcmp(micrs(n,:),upper(micr)))
      bitid=bitids(n);
   end
end

return
