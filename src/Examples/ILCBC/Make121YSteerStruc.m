function steerstruc = Make121YSteerStruc( istart, iend, nycperregion )
%
% generate structures in support of 1-to-1 steering over a selected
% beamline region
%

%=======================================================================

  global BEAMLINE PS ;

  bpms = findcells(BEAMLINE,'Class','MONI',istart,iend) ;
  ycor = findcells(BEAMLINE,'Class','YCOR',istart,iend) ;
  if (length(bpms) ~= length(ycor))
    error('Unequal numbers of BPMs and correctors in selected region') ;
  end
  ncorrect = length(ycor) ;
  
% segment the linac based on the selected number of ycors per region

  NRegion = floor((ncorrect-1)/nycperregion) ;
  qstart = floor(1+linspace(0,NRegion,NRegion+1)*nycperregion) ;
  qstop = qstart + nycperregion - 1 ;
  qstop = min(qstop,ncorrect) ;
  NRegion = NRegion + 1 ;
  
  steerstruc = [] ;

% loop over regions
  
  for count = 1:NRegion
    string = ['...generating data for region ',num2str(count),'...'] ;
    disp(string) ;
% generate the list of BPMs and correctors

    bpmno = linspace(qstart(count),qstop(count),qstop(count)-qstart(count)+1) ;
    ycorno = bpmno ;
    nbpm = length(bpmno) ;
    ycorps = zeros(1,nbpm) ;
    
% generate the lists of elements 

    bpmelem = bpms(bpmno) ;
    ycorelem = ycor(ycorno) ;
    lastelem = bpmelem(nbpm) ;
      
% generate the appropriate matrix (the one that generates BPM values from
% corrector values) 

    xfermat = zeros(nbpm) ;
    
    for ccount = 1:nbpm
      yce = ycorelem(ccount) ;
      ycorps(ccount) = BEAMLINE{yce}.PS ;
      for bcount = ccount:nbpm
        bpe = bpmelem(bcount) ;
        [stat,R] = RmatAtoB(yce,bpe) ;  
        xfermat(bcount,ccount) = R(3,4) ;
      end
    end
    
% attach the values to the data structure

    steerstruc(count).bpmno = bpmno ;
    steerstruc(count).ycorno = ycorno ;
    steerstruc(count).bpmelem = bpmelem ;
    steerstruc(count).ycorelem = ycorelem ;
    steerstruc(count).ycorps = ycorps ;
    steerstruc(count).lastelem = lastelem ;
    steerstruc(count).xfermat = xfermat ;
    
  end
        
        