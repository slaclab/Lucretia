struct magmake {
  const char * Name;
  double L;
  double B;
  double dB;
  double Tilt;
  double PS;
  double Girder;
  double Offset[6];
  double Aper;
} ;

struct beamdef {
  double DesignBeamP;
  double BunchPopulation;
} ;

mxArray* UAPNode2Lucretia (UAPNode*, beamdef);
mxArray* make_beambeam(UAPNode*, beamdef);
mxArray* make_bend(UAPNode*, beamdef);
mxArray* make_custom(UAPNode*, beamdef);
mxArray* make_electrickicker(UAPNode*, beamdef);
mxArray* make_kicker(UAPNode*, beamdef);
mxArray* make_marker(UAPNode*, beamdef);
mxArray* make_match(UAPNode*, beamdef);
mxArray* make_mult(UAPNode*, beamdef);
mxArray* make_oct(UAPNode*, beamdef);
mxArray* make_patch(UAPNode*, beamdef);
mxArray* make_quad(UAPNode*, beamdef);
mxArray* make_rfcav(UAPNode*, beamdef);
mxArray* make_sext(UAPNode*, beamdef);
mxArray* make_solenoid(UAPNode*, beamdef);
mxArray* make_taylor(UAPNode*, beamdef);
mxArray* make_thinmult(UAPNode*, beamdef);
mxArray* make_wiggler(UAPNode*, beamdef);
mxArray* make_drift(UAPNode*, beamdef);
magmake make_magnet(UAPNode*, beamdef);
double getUAPinfo(UAPNode*, string);

