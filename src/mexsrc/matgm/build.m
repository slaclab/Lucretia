!rm *.mod
!rm *.o
mex -v -c definitions_mod.f control_mod.f
mex -v -c xsif_size_pars.f xsif_elem_pars.f
mex -v -c lattice_mod.F gm_model_mod.f
mex -v -c gm_model.f
mex -v matgm_mex.F gm_model.o gm_model_mod.o control_mod.o lattice_mod.o
!mv matgm_mex.mex* ../../GroundMotion/
