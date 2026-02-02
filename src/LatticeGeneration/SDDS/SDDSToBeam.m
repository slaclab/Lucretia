function Beam = SDDSToBeam(BeamFile,Q)

s=sddsload(BeamFile);
Beam = CreateBlankBeam(1,length(s.column.x.page1),1,1) ;
Beam.Bunch.x(1,:) = s.column.x.page1' ;
Beam.Bunch.x(2,:) = s.column.xp.page1' ;
Beam.Bunch.x(3,:) = s.column.y.page1' ;
Beam.Bunch.x(4,:) = s.column.yp.page1' ;
Beam.Bunch.x(5,:) = s.column.t.page1' .* physConsts.clight ;
Beam.Bunch.x(6,:) = s.column.p.page1' .* physConsts.emass ;
Beam.Bunch.Q = ones(size(Beam.Bunch.Q)) .* Q/length(Beam.Bunch.Q) ;