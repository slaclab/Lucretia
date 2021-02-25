function dummy=umagtols_table(N,mtols,fn)
%
% umagtols_table(N,mtols,fn);
%
% Make a table of magnet tolerances.
%
% Inputs:
%
%   N     = element names
%   mtols = table of tolerances (results from umagtols)
%   fn    = (optional) file name ... if not provided, table written to terminal

file=exist('fn');

idb=find(mtols(:,1)==0);
if (~isempty(idb))
  Nb=length(idb);
  if (file)
    fprintf(fn,'NAME     | ROLL/mrad |   dB/B0  |   b1/b0  |   b2/b0  |   b4/b0  | r0/mm \n');
    fprintf(fn,'=========================================================================\n');
  else
    disp('NAME     | ROLL/mrad |   dB/B0  |   b1/b0  |   b2/b0  |   b4/b0  | r0/mm ');
    disp('=========================================================================');
  end
  for m=1:Nb
    n=idb(m);
    id=mtols(n,2);
    str=N(id,1:8);
    if (mtols(n,7)>0)
      if (mtols(n,7)>10)
        str=[str,blanks(3),'   >10   '];
      else
        str=[str,blanks(3),sprintf('%9.5f',1e3*mtols(n,7))];
      end
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,8))];
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,9))];
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,10))];
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,11))];
      str=[str,blanks(3),sprintf('%5.0f',1e3*mtols(n,12))];
    end
    if (file)
      fprintf(fn,[str,'\n']);
    else
      disp(str)
    end
  end
end

idq=find(mtols(:,1)==1);
if (~isempty(idq))
  Nq=length(idq);
  if (file)
    fprintf(fn,'\n')
    fprintf(fn,'NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  |   b2/b1  |   b5/b1  | r0/mm \n');
    fprintf(fn,'============================================================================================================\n');
  else
    disp(' ')
    disp('NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  |   b2/b1  |   b5/b1  | r0/mm ');
    disp('============================================================================================================');
  end
  for m=1:Nq
    n=idq(m);
    id=mtols(n,2);
    str=N(id,1:8);
    if (mtols(n,7)>0)
      if (mtols(n,7)>10)
        str=[str,blanks(3),'   >10   '];
      else
        str=[str,blanks(3),sprintf('%9.5f',1e3*mtols(n,7))];
      end
      if (mtols(n,3)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,3))];
      if (mtols(n,4)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,4))];
      if (mtols(n,5)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,5))];
      if (mtols(n,6)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(4),sprintf(fmt,1e6*mtols(n,6))];
      str=[str,blanks(4),sprintf('%8.1e',mtols(n,8))];
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,9))];
      str=[str,blanks(3),sprintf('%8.1e',mtols(n,10))];
      str=[str,blanks(3),sprintf('%5.0f',1e3*mtols(n,12))];
    end
    if (file)
      fprintf(fn,[str,'\n']);
    else
      disp(str)
    end
  end
end

ids=find(mtols(:,1)==2);
if (~isempty(ids))
  Ns=length(ids);
  if (file)
    fprintf(fn,'\n')
    fprintf(fn,'NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  \n');
    fprintf(fn,'==============================================================================\n');
  else
    disp(' ')
    disp('NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  ');
    disp('==============================================================================');
  end
  for m=1:Ns
    n=ids(m);
    id=mtols(n,2);
    str=N(id,1:8);
    if (mtols(n,7)>0)
      if (mtols(n,7)>10)
        str=[str,blanks(3),'   >10   '];
      else
        str=[str,blanks(3),sprintf('%9.5f',1e3*mtols(n,7))];
      end
      if (mtols(n,3)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,3))];
      if (mtols(n,4)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,4))];
      if (mtols(n,5)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,5))];
      if (mtols(n,6)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(4),sprintf(fmt,1e6*mtols(n,6))];
      str=[str,blanks(4),sprintf('%8.1e',mtols(n,8))];
    end
    if (file)
      fprintf(fn,[str,'\n']);
    else
      disp(str)
    end
  end
end

ido=find(mtols(:,1)==3);
if (~isempty(ido))
  No=length(ido);
  if (file)
    fprintf(fn,'\n')
    fprintf(fn,'NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  \n');
    fprintf(fn,'==============================================================================\n');
  else
    disp(' ')
    disp('NAME     | ROLL/mrad |   dX/um  |   dY/um  | sig_dX/um | sig_dY/um |   dB/B0  ');
    disp('==============================================================================');
  end
  for m=1:No
    n=ido(m);
    id=mtols(n,2);
    str=N(id,1:8);
    if (mtols(n,7)>0)
      if (mtols(n,7)>10)
        str=[str,blanks(3),'   >10   '];
      else
        str=[str,blanks(3),sprintf('%9.5f',1e3*mtols(n,7))];
      end
      if (mtols(n,3)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,3))];
      if (mtols(n,4)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,4))];
      if (mtols(n,5)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(3),sprintf(fmt,1e6*mtols(n,5))];
      if (mtols(n,6)>=0.01)
        fmt='%8.1e';
      else
        fmt='%8.3f';
      end
      str=[str,blanks(4),sprintf(fmt,1e6*mtols(n,6))];
      str=[str,blanks(4),sprintf('%8.1e',mtols(n,8))];
    end
    if (file)
      fprintf(fn,[str,'\n']);
    else
      disp(str)
    end
  end
end
