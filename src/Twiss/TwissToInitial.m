function InitialStruc=TwissToInitial(Twiss,elemno,InitialStruc)
% InitialStruc = TwissToInitial(Twiss,elemno,InitialStruc)
%  Copy twiss structure fields into Initial structure

if ~isfield(Twiss,'alphax') || elemno<1 || elemno>length(Twiss.alphax) || ~isfield(InitialStruc,'x') || ~isfield(InitialStruc.x,'Twiss')
  error('Incorrect input parameters')
end
InitialStruc.Momentum=Twiss.P(elemno);
InitialStruc.x.Twiss.beta=Twiss.betax(elemno); InitialStruc.y.Twiss.beta=Twiss.betay(elemno);
InitialStruc.x.Twiss.alpha=Twiss.alphax(elemno); InitialStruc.y.Twiss.alpha=Twiss.alphay(elemno); 
InitialStruc.x.Twiss.eta=Twiss.etax(elemno); InitialStruc.y.Twiss.eta=Twiss.etay(elemno); 
InitialStruc.x.Twiss.etap=Twiss.etapx(elemno); InitialStruc.y.Twiss.etap=Twiss.etapy(elemno);
InitialStruc.x.Twiss.nu=Twiss.nux(elemno); InitialStruc.y.Twiss.nu=Twiss.nuy(elemno);