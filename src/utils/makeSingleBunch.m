function BeamOut = makeSingleBunch(BeamIn)
% Make single particle bunch with mean values from provided bunch
BeamOut=BeamIn;
BeamOut.Bunch.x=mean(BeamIn.Bunch.x,2);
BeamOut.Bunch.Q=sum(BeamIn.Bunch.Q);
BeamOut.Bunch.stop=0;