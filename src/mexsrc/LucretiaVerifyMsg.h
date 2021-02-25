/* file containing various stuff useful in the generation of
   messages during Lucretia lattice verification */

	const char* zwf = "WF.ZSR" ;
	const char* twf = "WF.TSR" ;

/* standard words commonly used: */

	const char* KlysStr = "Klystron" ;
	const char* PSStr = "PS" ;
	const char* GirderStr = "Girder" ;
	const char* ElemStr = "Element" ;
	const char* ErrStr = "Error" ;
	const char* WarnStr = "Warning" ;
	const char* InfoStr = "Info" ;
	const char* reqdStr = "required" ;
	const char* optStr = "optional" ;
	const char* TLRStr = "WF.TLR" ;
	const char* TLRErrStr = "WF.TLRErr" ;

	const char* Keywd ;
	const char* parname ;

/* standard messages */

/* missing parameter */

	const char* MissPar = "%s: %s %d, %s parameter %s missing" ;

/* invalid length */

	const char* BadLen = "%s: %s %d, parameter %s length incorrect" ;

/* points to non-existent table entry */

	const char* NoSuch = "%s: %s %d points at non-existent %s %d" ;

/* inconsistent pointers */

	const char* Inconsis = "%s: inconsistency between %s %d and %s %d" ;

/* zero elements */

	const char* ZeroElt = "%s: %s %d points at zero elements" ;