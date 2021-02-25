/* LucretiaVersionProto.h :
   Contains prototypes for the functions which return Lucretia version
   character strings for the Common and Physics subsystems.  They are
   put here instead of in LucretiaCommon.h / LucretiaPhysics.h because 
   LucretiaMatlab.c needs them, and I don't want to make LucretiaMatlab.c
   use every .h file in the universe. */

/* AUTH: PT, 03-aug-2004 */
/* MOD: 
                         */

#define LUCRETIA_VERSION_PROTO

/* return the LucretiaCommon version string */

char* LucretiaCommonVersion( ) ;

/* return the LucretiaPhysics version string */

char* LucretiaPhysicsVersion( ) ;

