function [ BPM ] = get_bpms( BPM, LAT, cmd1, cmd2 )
% initialise BPM structure with pointers to readout channels

% Setup BPM resolutions
if ~isfield(BPM,'res'); BPM.res=0; end;

cmd1=lower(cmd1);
    
    switch cmd1
        
        case 'init'
            lfile=fopen(LAT.name,'r');
			bpmcnt=1;
			pat='^MONI(\w*).*';
			while 1
                tline=fgetl(lfile);
                if ~ischar(tline); break; end;
                s=[];f=[];t=[];
                [s,f,t]=regexp(tline,pat);
                if f
                    evalc(['BPM.name{bpmcnt}=tline(t{1}(1):t{1}(2));']);
                    bname=tline(t{1}(1):t{1}(2));
                    evalc(['BPM.x.handle(bpmcnt)=matmerlin(''get_r_channel'',''BPM.',bname,'.X'');']);
                    evalc(['BPM.y.handle(bpmcnt)=matmerlin(''get_r_channel'',''BPM.',bname,'.Y'');']);
                    bpmcnt=bpmcnt+1;
                end % if f
			end % while 1
      BPM.x.align=zeros(size(BPM.x.handle));
      BPM.y.align=zeros(size(BPM.y.handle));
			fclose(lfile);
        
        case 'read_all'
                for n=1:length(BPM.x.handle)
                    BPM.x.val(n)= matmerlin('read_channel',BPM.x.handle(n))+(randn.*BPM.res)+BPM.x.align(n);
                    BPM.y.val(n)= matmerlin('read_channel',BPM.y.handle(n))+(randn.*BPM.res)+BPM.y.align(n);
                end
            
        case 'read_single'
                ind = strmatch(lower(cmd2),lower(BPM.name),'exact');
                if ~ind; error('get_bpms: no such BPM!'); end;
                BPM.x.val = matmerlin('read_channel',BPM.x.handle(ind))+(randn.*BPM.res)+BPM.x.align(n);
                BPM.y.val = matmerlin('read_channel',BPM.y.handle(ind))+(randn.*BPM.res)+BPM.y.align(n);
            
        otherwise
            error('Invalid get_bpms command!');
            
    end % Switch
return