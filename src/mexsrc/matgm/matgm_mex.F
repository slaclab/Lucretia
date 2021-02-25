#include "fintrf.h"
	SUBROUTINE    mexFunction (nlhs,plhs,nrhs,prhs)
* mex interface to Andrei Seryi Ground motion model
* inputs: array of s co-ordinates (m); GM model (A,B or C); seed #; dt (s)
* outputs: dx and dy for given co-ordinates
* all input error checking to be done in matlab interface m file

	           USE GM_PARAMETERS
	           USE GM_HARM_PREPARE
	           USE TEST_GM_LINE
	           USE GM_RANDOM_GEN
                   USE GM_TRANSFUNCTION
 
	    IMPLICIT NONE
      mwpointer mxcreatedoublematrix
      INTEGER*4 MAX_S, iseed
      PARAMETER( MAX_S=1000000 )

C                   Matlab gateway function variables

      INTEGER     nlhs,nrhs,nseed,ID
      mwpointer   plhs(*), prhs(*)

C                   Referenced FUNCTIONs

      mwsize     mxGetM,mxGetN, imodel, size, m ,n, num_s
	    CHARACTER INFILE*256, GMTFFILE*80

	      real*8 s(MAX_S), dx(MAX_S), dy(MAX_S), test, aNUMBER
        real*8 ran1_gm
        integer*4 i, ierr
        real*8 tfid_x(MAX_S), tfid_y(MAX_S), seed, dt, which_model
        INTEGER ID_TFx(MAX_S), ID_TFy(MAX_S), idtfx, idtfy
        mwpointer is_pr, iseed_pr, idt_pr, imodel_pr, idxout_pr
        mwpointer idyout_pr
        mwpointer mxGetPr

      

C     Get input s array.
        m = mxGetM(prhs(1))
        n = mxGetN(prhs(1))
C     Should be vector input.
      	if(n.gt.1) then
          call mexErrMsgTxt('Should pass s as row vector')
      	endif
	     num_s=m
	      if (num_s .gt. MAX_S) then
          CALL mexErrMsgTxt ('Max S array size exceeded')
        end if
        is_pr = mxGetPr(prhs(1))
        call mxCopyPtrToReal8(is_pr, s, num_s)
      

c     Get seed number
        m = mxGetM(prhs(2))
        n = mxGetN(prhs(2))
c     Should be scalar
       if (m.ne.1 .or. n.ne.1) then
	       call mexErrMsgTxt('Should pass seed as a scalar')
       end if
       iseed_pr = mxGetPr(prhs(2))
       call  mxCopyPtrToReal8(iseed_pr, seed, 1)
       nseed=seed
C   Get dt
      m = mxGetM(prhs(3))
      n = mxGetN(prhs(3))
      size = m*n;
C     Should be scalar
      if (m.ne.1 .or. n.ne.1) then
        call mexErrMsgTxt('Should pass dt as a scalar')
      end if
      idt_pr = mxGetPr(prhs(3))
      call mxCopyPtrToReal8(idt_pr, dt, size)
      imodel_pr = mxGetPr(prhs(4))
      call  mxCopyPtrToReal8(imodel_pr, which_model, size)
      imodel=which_model
C     Get Transfer function list if passed
      if (nrhs>4) then
        m = mxGetM(prhs(5))
        n = mxGetN(prhs(5))
C       Should be vector input.
        if(n.gt.1) then
          call mexErrMsgTxt('Should pass ID_TFx as row vector')
        endif
        is_pr = mxGetPr(prhs(5))
        call mxCopyPtrToReal8(is_pr, tfid_x, m)
        m = mxGetM(prhs(6))
        n = mxGetN(prhs(6))
C       Should be vector input.
        if(n.gt.1) then
          call mexErrMsgTxt('Should pass ID_TFy as row vector')
        endif
        is_pr = mxGetPr(prhs(6))
        call mxCopyPtrToReal8(is_pr, tfid_y, m)
        m = mxGetM(prhs(7))
        n = mxGetN(prhs(7))
C       Should be scalar
        if (m.ne.1 .or. n.ne.1) then
          call mexErrMsgTxt('Should pass nTF as a scalar')
        end if
        idt_pr = mxGetPr(prhs(7))
        call  mxCopyPtrToReal8(idt_pr, aNUMBER, 1)
        NUMBER_OF_GMTF = aNUMBER
      endif

C     Create matrix for the return argument.
      plhs(1) = mxCreateDoubleMatrix(num_s, 1, 0)
      plhs(2) = mxCreateDoubleMatrix(num_s, 1, 0)
      idxout_pr = mxGetPr(plhs(1))
      idyout_pr = mxGetPr(plhs(2))

C     Generate Ground motion
      if (nseed>0) then
        do iseed=1,nseed
          test = ran1_gm(idum)
        enddo
        select case (imodel)
          case (1)
            call READ_PWK_PARAM('gm_model_A.data',IERR)
          case (2)
            call READ_PWK_PARAM('gm_model_B.data',IERR)
          case (3)
            call READ_PWK_PARAM('gm_model_C.data',IERR)
          case (4)
            call READ_PWK_PARAM('gm_model_K.data',IERR) 
          case (5)
            call READ_PWK_PARAM('gm_model_ATF_v1a.data',IERR)
          case (6)
            call READ_PWK_PARAM('gm_model_ATF_v1b.data',IERR)
          case default
            call mexPrintf('WARNING: USING DEFAULT GM PARAMETERS!')
        end select
        if (NUMBER_OF_GMTF>0) then
          call ALLOCATE_GM_TF(NUMBER_OF_GMTF)
        endif
        call PREPARE_HARMONICS(0,ierr)
        call PREPARE_SYSTHAR(0,ierr)
        if (NUMBER_OF_GMTF>0) then
          do ID=1,NUMBER_OF_GMTF
            GMTF(ID)%tf_defined_by_file  =  .true.
            GMTFFILE='tfData_'//CHAR(48+ID)//'.dat'
            call PREPARE_GM2ELT_TF(ID,GMTFFILE,IERR)
          enddo
        endif
      endif
        do i=1,num_s
        if (NUMBER_OF_GMTF>0) then
          idtfx=int(tfid_x(i))
          idtfy=int(tfid_y(i))
        else
          idtfx=0;
          idtfy=0;
        endif
        call DXDY_PWK(0,dt,s(i),dx(i),dy(i),idtfx,idtfy,0,0,0,0)
      end do
C    Copy dx and dy to output variables
      call mxCopyReal8ToPtr(dx, idxout_pr, num_s)
      call mxCopyReal8ToPtr(dy, idyout_pr, num_s)
	RETURN
	END
