[stat,beamout,instdata] = TrackThru( 1, length(BEAMLINE), beam1, 1, 1, 0 );if stat{1}~=1; error(stat{2:end}); end;
ip_bpm=model.ip_bpm;
for ihan=1:ip_bpm
  [nx(ihan),ny(ihan),nt] = GetNEmitFromSigmaMatrix( instdata{1}(ihan).P, instdata{1}(ihan).sigma );
  [nx_norm(ihan),ny_norm(ihan),nt_norm] = GetNEmitFromSigmaMatrix( instdata{1}(ihan).P, instdata{1}(ihan).sigma, 'normalmode' );
  bx_calc(ihan)=sqrt(model.Twiss.betax(Handles.BPM(ihan)).*(nx(ihan)./(250/0.511e-3)));
  by_calc(ihan)=sqrt(model.Twiss.betay(Handles.BPM(ihan)).*(ny(ihan)./(250/0.511e-3)));
  bx_meas(ihan)=sqrt(instdata{1}(ihan).sigma(1,1));
  by_meas(ihan)=sqrt(instdata{1}(ihan).sigma(3,3));
end