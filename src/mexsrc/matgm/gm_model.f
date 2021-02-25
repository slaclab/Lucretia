*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
*
*-----------------------------------------------------------------------
*                     Ground motion model @ LIAR
*                            Andrei Seryi
*                       Seryi@SLAC.Stanford.EDU 
*            http://www.slac.stanford.edu/~seryi/gm/model/
*                        Revision January,  2002
*                        Revision December, 2001                  
*                        Revision February, 2000
*                  based on my program written in 1996 
************************************************************************
*
*                              main routines
*
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
*-----------------------------------------------------------------------
*        The ground motion model allows to 
*        compute horizontal x(t,s) and vertical y(t,s)       
*        position of the ground at a given time t and in a given 
*        longitudinal position s, assuming that at time t=0 we had
*        x(0,s)=0 and y(0,s)=0.  The values x(t,s) and y(t,s) will 
*        be computed using the same power spectrum P(w,k), however
*        they are independent. Parameters of approximation of 
*        the P(w,k) (PWK) can be chosen to model quiet or noisy
*        place. PWK now can include ATL and waves-like motion. 
*        The model also uses Q(k) spectrum that describes 
*        systematic motion.
*
*        Units are seconds and meters (unless noted otherwise)
*       
*-----------------------------------------------------------------------
*        The program needs the file "model.data"  with parameters
*        of the ground motion model. 
*
*-----------------------------------------------------------------------
*                      How it works.
*      We assumed that we know power spectrum of ground motion P(w,k).
*      Then we assume that using only finite number of  
*      harmonics we can model ground motion in some limited range of
*      t and s. As one of inputs we have these ranges: minimum and 
*      maximum time Tmin and Tmax and minimum and maximum distance
*      Smin and Smax. We will define then the range of important
*      frequencies wmin to wmax and wave-numbers kmin to kmax.
*      Our harmonics we will distribute equidistantly in logariphmic
*      sense, that is, for example, k_{i+1}/k_{i} is fixed.
*      Number of steps defined Np from the input file, for
*      example Np=50. A single harmonic characterized by 
*      its amplitude am_{ij}, frequency w_{i}, wave number k_{j}
*      and phase phi_{ij}. Total number of harmonics is Np*Np.
*      The amplitudes of harmonics are defined once at the beginning 
*      from integral over surface (w_{i+1}:w_{i})(k_{j+1}:k_{j}) 
*      on P(w,k). Phases phi_{ij} are also defined once at the 
*      beginning by random choice. The resulting x(t,s) will be 
*      given by double sums:
C A.S. Apr-28-2002 : coefficient 0.5 changed to 0.707
*      x(t,s) = 0.707 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * sin(w_{i} t) 
*                                                * sin(k_{j} s + phi_{ij})
*             + 0.707 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * (cos(w_{i} t)-1)
*                                                * sin(k_{j} s + psi_{ij})
*      This choise of formula ensure x(t,s) = 0 at t=0.
*      The last sinus is presented in the program in other form
*      sin(k_{j} s + phi_{ij}) = sin(k_{j} s) cos(phi_{ij}) 
*                                     + cos(k_{j} s) sin(phi_{ij})
*      So, we store not phases phi_{ij}, but cos(phi_{ij}) and 
*      sin(phi_{ij})
*      The same for y(t,s) but with different phases phi_{ij}.
*
*      Systematic motion is treated in a similar fashion:      
*     "x(t,s)" = "x given by P(w,k)" + Function(time)* "given by Q(k)"
*      where Q(k) is the power spectrum of the systematic 
*      comnponent. Here "Function(time)" is either linear,
*      or initialy sqrt(t) and eventualy exp(-t)
*-----------------------------------------------------------------------
*
*      Recent changes:
*      1) Systematic motion added. Program rewritten in Fortran 90,
*         that allowed dynamical allocations of arrays.
*
*
*
*-----------------------------------------------------------------------
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine XY_PWK(t,s,x,y)
c this subroutine is used only in tests, rewrite tests without it
	USE GM_PARAMETERS
	USE GM_HARMONICS
	USE GM_HARMONICS_SYST
	USE GM_HARM_PREPARE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* it computes x and y positions of a single element that has
* longitudinal coordinate s, at a given time t (positive)
*
c      parameter (NH=250)
* arrays of: amplitude  am, frequency (omega) wh, wave number kh
c      real*8 am(NH,NH),wh(NH),kh(NH)
* arrays to store values sin(w_{i} t) and cos(w_{i} t)
* we will use only sinus, but cosine we need to calculate sinus
* for the new time t using values saved at time told
c      real*8 sw(NH),cw(NH)
c      real*8 dskx(NH,NH),dckx(NH,NH),dsky(NH,NH),dcky(NH,NH)
c      real*8 sk(NH),ck(NH)
*
c      real*8 ams(NH),khs(NH)
c      real*8 dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
*
c      common/harmonics/am,wh,kh,sw,cw,dskx,dckx,dsky,dcky
c      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys

c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1
* told is the time t of previous use of this subroutine

c      common/timeold/told 
	
          if(t.ne.told) then 
*
* we will calculate sin(w_{i} t) only if time has been changed since
* previous call, otherwise will used stored in the array sw values.
* it increases speed because this subroutine called Nelem times
* with the same t
*
           do i=1,Np
	qdsin=sin( (t-told)*wh(i) )
	qdcos=cos( (t-told)*wh(i) )
	 qsin=sw(i)*qdcos + cw(i)*qdsin
	 qcos=cw(i)*qdcos - sw(i)*qdsin
          sw(i)=qsin
          cw(i)=qcos
           end do
          end if      
      told=t

* we calculate sin(k_{j} s) at each step. This can be avoided 
*for the cost of two arrays (NH*NELMX). But this array can be
* very big. What is better?
*
           do j=1,Np
            sk(j)=sin(s*kh(j))
            ck(j)=cos(s*kh(j))
           end do
           
* clear variables, start of double sums 
      x=0.
      y=0.
        do i=1,Np
         do j=1,Np         
           sinkx=sk(j)*dckx(i,j)+ck(j)*dskx(i,j)
           sinky=sk(j)*dcky(i,j)+ck(j)*dsky(i,j)
C A.S. Apr-28-2002 : coeff 0.5 changed to 0.707
           x=x + am(i,j) * ( sw(i)*sinkx + (cw(i)-1.0)*sinky )*0.707
           y=y + am(i,j) * ( sw(i)*sinky + (cw(i)-1.0)*sinkx )*0.707
         end do
        end do

* add systematic components

* we calculate sin(k_{j} s) at each step. This can be avoided 
*for the cost of two arrays (NH*NELMX). 
*
           do j=1,Np
            sk(j)=sin(s*khs(j))
            ck(j)=cos(s*khs(j))
           end do

	   f_vs_t = a_settlement(t,tau_syst,tgap_syst,iwhat_syst)

         do j=1,Np         
           sinkx=sk(j)*dckxs(j)+ck(j)*dskxs(j)
           sinky=sk(j)*dckys(j)+ck(j)*dskys(j)
c           x=x + ams(j) * t * sinkx
c           y=y + ams(j) * t * sinky
           x=x + ams(j) * f_vs_t * sinkx
           y=y + ams(j) * f_vs_t * sinky
         end do

      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine DXDY_PWK(tabs,dt,s,dx,dy,idtfx,idtfy,idnx,idny,imx,imy)
c A.S. 01/28/02 added ground motion transfer function
	USE GM_PARAMETERS
	USE GM_HARMONICS
	USE GM_HARMONICS_SYST
	USE GM_HARM_PREPARE
	USE GM_TRANSFUNCTION
	USE GM_TECHNICAL_NOISE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* it computes x and y positions of a single element that has
* longitudinal coordinate s, at a given time t (positive)
*
c      parameter (NH=250)
* arrays of: amplitude  am, frequency (omega) wh, wave number kh
c      real*8 am(NH,NH),wh(NH),kh(NH)
* arrays to store values sin(w_{i} t) and cos(w_{i} t)
* we will use only sinus, but cosine we need to calculate sinus
* for the new time t using values saved at time told
c      real*8 sw(NH),cw(NH)
c      real*8 dskx(NH,NH),dckx(NH,NH),dsky(NH,NH),dcky(NH,NH)
c      real*8 sk(NH),ck(NH)
*
c      real*8 ams(NH),khs(NH)
c      real*8 dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
*
c      common/harmonics/am,wh,kh,sw,cw,dskx,dckx,dsky,dcky
c      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys

c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1
* told is the time t of previous use of this subroutine

c      common/timeold/told 

c	if(tabs.eq.0.0.and.dt.eq.0.0) then
c	 write(deflun,*)told,dtold
c	 write(deflun,*)'sw=',sw(1),sw(2)
c	 write(deflun,*)'cw=',cw(1),cw(2)
c	 write(deflun,*)'swdt=',swdt(1),swdt(2)
c	 write(deflun,*)'cwdt=',cwdt(1),cwdt(2)
c	end if
	

          if(tabs.ne.told) then 
*
* we will calculate sin(w_{i} t) only if time has been changed since
* previous call, otherwise will used stored in the array sw values.
* it increases speed because this subroutine called Nelem times
* with the same t
*
           do i=1,Np
	qdsin=sin( (tabs-told)*wh(i) )
	qdcos=cos( (tabs-told)*wh(i) )
	 qsin=sw(i)*qdcos + cw(i)*qdsin
	 qcos=cw(i)*qdcos - sw(i)*qdsin
          sw(i)=qsin
          cw(i)=qcos
           end do

c -- calc for tech noise -- start
	if(TECH_NOISE_EXIST) then
	do ID=1,NUMBER_OF_TECHNOISE
	 if(TCHN(ID)%ID_NOISE.ne.0) then
           do i=1,Np
	qdsin=sin( (tabs-told)*TCHN(ID)%wh_n(i) )
	qdcos=cos( (tabs-told)*TCHN(ID)%wh_n(i) )
	qsin=TCHN(ID)%sw_n(i)*qdcos + TCHN(ID)%cw_n(i)*qdsin
	qcos=TCHN(ID)%cw_n(i)*qdcos - TCHN(ID)%sw_n(i)*qdsin
          TCHN(ID)%sw_n(i)=qsin
          TCHN(ID)%cw_n(i)=qcos
          end do
	 end if ! end of if(TCHN(ID)%ID_NOISE.ne.0)
	end do ! end do ID=1,NUMBER_OF_TECHNOISE
	end if ! end of if(TECH_NOISE_EXIST)
c -- calc for tech noise -- end 

          end if       ! end of if(tabs.ne.told)
      told=tabs


*
* similar for dt, if it does not change, do not recalculate 
*
          if(dt.ne.dtold) then 
           do i=1,Np
	qsin=sin( dt*wh(i) )
	qcos=cos( dt*wh(i) )
          swdt(i)=qsin
          cwdt(i)=qcos
           end do

c -- calc for tech noise -- start
	if(TECH_NOISE_EXIST) then
	do ID=1,NUMBER_OF_TECHNOISE
	 if(TCHN(ID)%ID_NOISE.ne.0) then
           do i=1,Np
	qsin=sin( dt*TCHN(ID)%wh_n(i) )
	qcos=cos( dt*TCHN(ID)%wh_n(i) )
          TCHN(ID)%swdt_n(i)=qsin
          TCHN(ID)%cwdt_n(i)=qcos
          end do
	 end if ! end of if(TCHN(ID)%ID_NOISE.ne.0)
	end do ! end do ID=1,NUMBER_OF_TECHNOISE
	end if ! end of if(TECH_NOISE_EXIST)
c -- calc for tech noise -- end 

          end if   ! end of if(dt.ne.dtold)  
      dtold=dt


* we calculate sin(k_{j} s) at each step. This can be avoided 
*for the cost of two arrays (NH*NELMX). But this array can be very big. 
 
           do j=1,Np
            sk(j)=sin(s*kh(j))
            ck(j)=cos(s*kh(j))
           end do
           
* clear variables, start of double sums 
c      x=0.
c      y=0.
	dx=0.
	dy=0.

        do i=1,Np
	   dsds = sw(i)*cwdt(i)+cw(i)*swdt(i)-sw(i)
	   dcdc = cw(i)*cwdt(i)-sw(i)*swdt(i)-cw(i)

C A.S. 01/28/02  -- change for transfer function
	dsds_x = dsds
	dcdc_x = dcdc
	dsds_y = dsds
	dcdc_y = dcdc
	if(TF_EXIST) then
	 if(IDTFx.ne.0) then
	  dcdc_x = dcdc*GMTF(IDTFx)%TF_RE(i) + dsds*GMTF(IDTFx)%TF_IM(i)
	  dsds_x = dsds*GMTF(IDTFx)%TF_RE(i) - dcdc*GMTF(IDTFx)%TF_IM(i)
	 end if
	 if(IDTFy.ne.0) then
	  dcdc_y = dcdc*GMTF(IDTFy)%TF_RE(i) + dsds*GMTF(IDTFy)%TF_IM(i)
	  dsds_y = dsds*GMTF(IDTFy)%TF_RE(i) - dcdc*GMTF(IDTFy)%TF_IM(i)
	 end if
	end if  ! end of if(TF_EXIST)

C A.S. 01/29/02  -- change for tech noises
	if(TECH_NOISE_EXIST) then

	 if(IDNx.ne.0) then
	  dsds_nx = TCHN(IDNx)%sw_n(i)*TCHN(IDNx)%cwdt_n(i)
     >   +TCHN(IDNx)%cw_n(i)*TCHN(IDNx)%swdt_n(i) -TCHN(IDNx)%sw_n(i)
	  dcdc_nx = TCHN(IDNx)%cw_n(i)*TCHN(IDNx)%cwdt_n(i)
     >   -TCHN(IDNx)%sw_n(i)*TCHN(IDNx)%swdt_n(i) -TCHN(IDNx)%cw_n(i)
	 end if
	 if(IDNy.ne.0) then 
	  dsds_ny = TCHN(IDNy)%sw_n(i)*TCHN(IDNy)%cwdt_n(i)
     >   +TCHN(IDNy)%cw_n(i)*TCHN(IDNy)%swdt_n(i) -TCHN(IDNy)%sw_n(i)
	  dcdc_ny = TCHN(IDNy)%cw_n(i)*TCHN(IDNy)%cwdt_n(i)
     >   -TCHN(IDNy)%sw_n(i)*TCHN(IDNy)%swdt_n(i) -TCHN(IDNy)%cw_n(i)
	 end if

C A.S. Apr-28-2002  allow TF * noise or not
	if(TF_EXIST) then
	 if(imx.ne.0.and.IDTFx.ne.0.and.IDNx.ne.0) then
	  dcdcnx = dcdc_nx
	  dsdsnx = dsds_nx
	  dcdc_nx=dcdcnx*GMTF(IDTFx)%TF_RE(i)+dsdsnx*GMTF(IDTFx)%TF_IM(i)
	  dsds_nx=dsdsnx*GMTF(IDTFx)%TF_RE(i)-dcdcnx*GMTF(IDTFx)%TF_IM(i)
	 end if
	 if(imy.ne.0.and.IDTFy.ne.0.and.IDNy.ne.0) then
	  dcdcny = dcdc_ny
	  dsdsny = dsds_ny
	  dcdc_ny=dcdcny*GMTF(IDTFy)%TF_RE(i)+dsdsny*GMTF(IDTFy)%TF_IM(i)
	  dsds_ny=dsdsny*GMTF(IDTFy)%TF_RE(i)-dcdcny*GMTF(IDTFy)%TF_IM(i)
	 end if
	end if  ! end of if(TF_EXIST)
C A.S. Apr-28-2002   end change

	end if  ! end of if(TECH_NOISE_EXIST)


c ground motion given by P(w,k)
         do j=1,Np         
           sinkx=sk(j)*dckx(i,j)+ck(j)*dskx(i,j)
           sinky=sk(j)*dcky(i,j)+ck(j)*dsky(i,j)
c           x=x + am(i,j) * ( sw(i)*sinkx + (cw(i)-1.0)*sinky )*0.707
c           y=y + am(i,j) * ( sw(i)*sinky + (cw(i)-1.0)*sinkx )*0.707
C A.S. Apr-28-2002 : coeff 0.5 changed to 0.707
           dx=dx + am(i,j) * ( dsds_x*sinkx + dcdc_x*sinky )*0.707
           dy=dy + am(i,j) * ( dsds_y*sinky + dcdc_y*sinkx )*0.707
         end do


c add tech noises


	if(TECH_NOISE_EXIST) then
	 if(IDNx.ne.0) then
        sinn=TCHN(IDNx)%ds_n(i)
        cosn=TCHN(IDNx)%dc_n(i)
C A.S. Apr-28-2002 : coefficient 0.5 changed by 0.707
        dx=dx + TCHN(IDNx)%am_n(i)*(dsds_nx*sinn+dcdc_nx*cosn)*0.707
	 end if
	 if(IDNy.ne.0) then 
        sinn=TCHN(IDNy)%ds_n(i)
        cosn=TCHN(IDNy)%dc_n(i)
C A.S. Apr-28-2002 : coefficient 0.5 changed by 0.707
        dy=dy + TCHN(IDNy)%am_n(i)*(dsds_ny*sinn+dcdc_ny*cosn)*0.707
	 end if
	end if  ! end of if(TECH_NOISE_EXIST)

      end do  ! end of  do i=1,Np (which is Nw)


* add systematic components

* we calculate sin(k_{j} s) at each step. This can be avoided 
*for the cost of two arrays (NH*NELMX). 
*
           do j=1,Np
            sk(j)=sin(s*khs(j))
            ck(j)=cos(s*khs(j))
           end do

	 df_vs_t = a_settlement(tabs+dt,tau_syst,tgap_syst,iwhat_syst)
     >          -a_settlement(tabs   ,tau_syst,tgap_syst,iwhat_syst)

         do j=1,Np         
           sinkx=sk(j)*dckxs(j)+ck(j)*dskxs(j)
           sinky=sk(j)*dckys(j)+ck(j)*dskys(j)
ccc           x=x + ams(j) * t * sinkx
ccc           y=y + ams(j) * t * sinky
c           dx=dx + ams(j) * dt * sinkx
c           dy=dy + ams(j) * dt * sinky
           dx=dx + ams(j) * df_vs_t * sinkx
           dy=dy + ams(j) * df_vs_t * sinky
         end do

      return
      end
                 
            
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine READ_PWK_PARAM(INFILE,IERR)
* read parameters of P(w,k) from the file 
	USE GM_PARAMETERS
	USE CONTROL_MOD
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 

c	TYPE (GM_CORRECTED_ATL) our_atl
c	TYPE (GM_WAVE) our_waves(3)
c	TYPE (GM_SYSTEMATIC) our_syst
c      TYPE (GM_REGION) our_region

c      common/filejustread/inewparam,inewparams
	character*50 chardum
      character*(*) INFILE
      
c      inewparam=0

	write(deflun,*)'  '
	write(deflun,*)
     &    '   Reading parameters of the ground motion model   '
	write(deflun,*)
     &    '   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   '
      write(deflun,345)'file of parameters ',infile
345   format(a,a)

	ierr=0
      
      open(92,file=INFILE,err=999,status='old')
      read(92,*,err=999,end=999) chardum,A
      read(92,*,err=999,end=999) chardum,B
      
c	do i=1,3
      read(92,*,err=999,end=999) chardum,f1
      read(92,*,err=999,end=999) chardum,a1
      read(92,*,err=999,end=999) chardum,d1
      read(92,*,err=999,end=999) chardum,v1
      read(92,*,err=999,end=999) chardum,f2
      read(92,*,err=999,end=999) chardum,a2
      read(92,*,err=999,end=999) chardum,d2
      read(92,*,err=999,end=999) chardum,v2
      read(92,*,err=999,end=999) chardum,f3
      read(92,*,err=999,end=999) chardum,a3
      read(92,*,err=999,end=999) chardum,d3
      read(92,*,err=999,end=999) chardum,v3
c	end do
           
      read(92,*,err=999,end=999) chardum,Tmin
      read(92,*,err=999,end=999) chardum,Tmax
      read(92,*,err=999,end=999) chardum,Smin
      read(92,*,err=999,end=999) chardum,Smax
      
      read(92,*,err=999,end=999) chardum,Np

      read(92,*,err=999,end=999) chardum,Q1
      read(92,*,err=999,end=999) chardum,rk1
      read(92,*,err=999,end=999) chardum,rkk1

      read(92,*,err=999,end=999) chardum,iwhat_syst
      read(92,*,err=999,end=999) chardum,tau_syst
      read(92,*,err=999,end=999) chardum,tgap_syst
	tau_syst=tau_syst*3600.*24.*365.
    	tgap_syst=tgap_syst*3600.*24.*365.
  
      close(92) 
      goto 900

    
900   return

920   format(a,1pe12.5)
921   format(a,i6)

999	ierr=1
       write(deflun,*) 'error!  unexpected end of pwk file!'
	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PRINT_PWK_PARAM(IERR)
	USE GM_PARAMETERS
	USE CONTROL_MOD
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
	character*50 chardum


	write(deflun,*)' '
	write(deflun,*)' Parameters of the ground motion model '
	write(deflun,*)' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '
      ierr=0
      chardum='''Parameter A of the ATL law,     A [m**2/m/s]   '''
      write(deflun,920) chardum,A
      chardum='''Parameter B of the PWK,         B [m**2/s**3]  '''
      write(deflun,920) chardum,B
      
c	do i=1,3
      chardum='''Frequency of 1-st peak in PWK,  f1 [Hz]        '''
      write(deflun,920) chardum,f1
      chardum='''Amplitude of 1-st peak in PWK,  a1 [m**2/Hz]   '''
      write(deflun,920) chardum,a1
      chardum='''Width of 1-st peak in PWK,      d1 [1]         '''
      write(deflun,920) chardum,d1
      chardum='''Velocity of 1-st peak in PWK,   v1 [m/s]       '''
      write(deflun,920) chardum,v1
      chardum='''Frequency of 2-nd peak in PWK,  f2 [Hz]        '''
      write(deflun,920) chardum,f2
      chardum='''Amplitude of 2-nd peak in PWK,  a2 [m**2/Hz]   '''
      write(deflun,920) chardum,a2
      chardum='''Width of 2-nd peak in PWK,      d2 [1]         '''
      write(deflun,920) chardum,d2
      chardum='''Velocity of 2-nd peak in PWK,   v2 [m/s]       '''
      write(deflun,920) chardum,v2
      chardum='''Frequency of 3-rd peak in PWK,  f3 [Hz]        '''
      write(deflun,920) chardum,f3
      chardum='''Amplitude of 3-rd peak in PWK,  a3 [m**2/Hz]   '''
      write(deflun,920) chardum,a3
      chardum='''Width of 3-rd peak in PWK,      d3 [1]         '''
      write(deflun,920) chardum,d3
      chardum='''Velocity of 3-rd peak in PWK,   v3 [m/s]       '''
      write(deflun,920) chardum,v3


c	end do
            
     
      chardum='''Minimum time,                   Tmin [s]       '''
      write(deflun,920) chardum,Tmin
      chardum='''Maximum time,                   Tmax [s]       '''
      write(deflun,920) chardum,Tmax

      chardum='''Minimum distance,               Smin [m]       '''
      write(deflun,920) chardum,Smin
      chardum='''Maximum distance,               Smax [m]       '''
      write(deflun,920) chardum,Smax

      chardum='''Number of w or k harmonics,     Np   [1]       '''
      write(deflun,921) chardum,Np

      chardum='''Ampl. of peak in systmat.P,    Q1 [m**3]       '''
      write(deflun,920) chardum,Q1
      chardum='''Wavenumber of peak in syst.P,  rk1 [1/m]       '''
      write(deflun,920) chardum,rk1
      chardum='''Width of peak in system.P,      rkk1 [1]       '''
      write(deflun,920) chardum,rkk1

      chardum='''linear or sqrt(t)->exp syst. iwhat_syst (0,1)  '''
      write(deflun,921) chardum,iwhat_syst
      chardum='''                           tau_syst [years]    '''
      write(deflun,920) chardum,tau_syst/(3600.*24.*365.)
      chardum='''(used if ist=1) time gap  tgap_syst [years]    '''
      write(deflun,920) chardum,tgap_syst/(3600.*24.*365.)

920   format(a,1pe12.5)
921   format(a,i6)

999   return
      end


*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_HARMONICS(inewparam,ierr)
	USE GM_PARAMETERS
	USE GM_HARMONICS
	USE GM_HARM_PREPARE
	USE GM_RANDOM_GEN
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
 
* calculate frequencies, wave numbers, amplitudes and phases of harmonics
*
* Maximum number of harmonics is limited by NH*NH
c      parameter (NH=250)
      real*8 k,ka,kb
c	,kmin,kmax,kmins,kmaxs
c      real*8 am(NH,NH),wh(NH),kh(NH)
c      real*8 sw(NH),cw(NH)
c      real*8 dskx(NH,NH),dckx(NH,NH),dsky(NH,NH),dcky(NH,NH)
      real*8 Stest(2),Ttest(2),wmin1(2,2),wmax1(2,2),
     >                            kmin1(2,2),kmax1(2,2)

c	real*8 wmin,wmax

c      common/harmonics/am,wh,kh,sw,cw,dskx,dckx,dsky,dcky

c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1


c      real*8 G,
c	real*8 told,difftmax
c      INTEGER*4 Nk,Nw

    
c      common/integ/G,kmin,kmax,wmin,wmax,kmins,kmaxs,Nk,Nw

c      common/randum/idum
c      common/timeold/told
c      common/maxtimediff/difftmax
  
* inewparam =0 will say to PREPARE_HARMONICS that the file
* with P(w,k) parameters has been just read thus complete
* harmonics generation should be performed
* if inewparam =1 it means that the harmonics has been already
* once calculated, this is just new seed and it is enough to 
* generate only new phases, w and k
*
c      common/filejustread/inewparam,inewparams
      
      data Ndiv/30/,pi/3.14159265358979/
      
c	write(deflun,*)'about to allocate harmonics'

      call ALLOCATE_HARMONCS (NP,IERR)

c	write(deflun,*)'after allocate'


* will just regenerate phases, w and k if inewparam =1
      if(inewparam.eq.1) goto 1000
      
c* test that requested number of harmonics is smaller than array size
c      if(Np.gt.NH) Np=NH

      write(deflun,*)' '
      write(deflun,*)' Searching the range of important w and k'
c	write(deflun,*)'Np=',Np
      
* we will estimate range of important w and k by rough 
* preliminary integration
* of P(w,k) over enough big (hopefully) range of w and k
      
      Nw=Np
      Nk=Np
* we define this wide initial range of w and k and beleive that 
* the important range is inside
*
      wmin=1./Tmax/1000.
      wmax=1./Tmin*1000.
      kmin=0.00001
      kmax=100000.
       

* ratio k_{i+1}/k_{i} is constant
*
      dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
 
* to estimate range of important w and k we will calculate four
* values, namely the mean value < [x(t,s1)-x(t,s2)]^2 > for different
* combination of variables, namely:
* t=Tmin, s1-s2=Smin
* t=Tmin, s1-s2=Smax
* t=Tmax, s1-s2=Smin
* t=Tmax, s1-s2=Smax

      Ttest(1)=Tmin
      Ttest(2)=Tmax
      Stest(1)=Smin
      Stest(2)=Smax
      
* double loop to check all these four cases

         do is=1,2
         do it=1,2 
               
* the value < [x(t,s1)-x(t,s2)]^2 > is the double integral on
* P(w,k) 2 (1-cos(wt)) 2 (1-cos(k (s1-s2))) dw/(2pi) dk/(2pi) = F(w,k)
*
* to find the important range of w we calculate first the value
* sfin = Int^{wmax}_{wmin} Int^{kmax}_{kmin} F(w,k) dw dk
* and then we calculate first the function 
* s1(w)= Int^{w}_{wmin} Int^{kmax}_{kmin} F(w,k) dw dk
* then the ratio s1(w)/sfin will be equal 0 at wmin and 1 at wmax
* the region where this function changes rapidly from 0 to 1
* gives main contribution to the integral. 
* we define the range of important w as the points where 
* s1(w)/sfin cross the level 0.01 and 0.99 for wmin and wmax
* correspondingly
* 
* to find the range of k we do the same but with s2(k)/sfin where
* s2(k)= Int^{k}_{kmin} Int^{wmax}_{wmin} F(w,k) dw dk
*

c     call PPWK(Stest(is),Ttest(it),kmin,wmin,RRR)
c	write(deflun,*)'pwk=',rrr
                
      sfin=0.
       do i=1,Nw
       w=wmin*dw**(i-1)
        do j=1,Nk
        k=kmin*dk**(j-1)
        ds=(w*dw-w)*(k*dk-k)      
        call PPWK(Stest(is),Ttest(it),k,w,RRR)
        sfin=sfin+ds*RRR
        end do
       end do
       
c	write(deflun,*)'sfin=',sfin

      if(sfin.eq.0.0) goto 500

      wmin1(is,it)=0.
      wmax1(is,it)=0.
      s1=0.
       do i=1,Nw
       w=wmin*dw**(i-1)
        do j=1,Nk
        k=kmin*dk**(j-1)
        ds=(w*dw-w)*(k*dk-k)      
        call PPWK(Stest(is),Ttest(it),k,w,RRR)
        s1=s1+ds*RRR
        end do
        s1w=s1/sfin
        if(wmin1(is,it).eq.0.0.and.s1w.gt.0.01) wmin1(is,it)=w/dw
        if(wmax1(is,it).eq.0.0.and.s1w.gt.0.99) wmax1(is,it)=w*dw
       end do   

      kmin1(is,it)=0.
      kmax1(is,it)=0.
      s1=0.
      do i=1,Nk
      k=kmin*dk**(i-1)     
       do j=1,Nw
       w=wmin*dw**(j-1)		
        ds=(w*dw-w)*(k*dk-k)      
        call PPWK(Stest(is),Ttest(it),k,w,RRR)
        s1=s1+ds*RRR
        end do
        s1k=s1/sfin
        if(kmin1(is,it).eq.0.0.and.s1k.gt.0.01) kmin1(is,it)=k/dk
        if(kmax1(is,it).eq.0.0.and.s1k.gt.0.99) kmax1(is,it)=k*dk
       end do      	
       
         end do
         end do
      

* we have found the important ranges for all of four cases,
* now we find the range that cover these four
*
         kmin=kmin1(1,1)
         kmax=kmax1(1,1)
         wmin=wmin1(1,1)
         wmax=wmax1(1,1)
         do is=1,2
         do it=1,2         
       kmin=dmin1(kmin1(is,it),kmin)
       wmin=dmin1(wmin1(is,it),wmin)
       kmax=dmax1(kmax1(is,it),kmax)
       wmax=dmax1(wmax1(is,it),wmax)
         end do
         end do
         
500   dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      
      wmax=wmax/dw
      kmax=kmax/dk

      write(deflun,*)' For the range of s from ',Smin,' to ',smax
      write(deflun,*)' and the range of t from ',tmin,' to ',tmax
      write(deflun,*)' the range of important k and w:'      
      write(deflun,*)' k from ',kmin,' to ',kmax
      write(deflun,*)' w from ',wmin,' to ',wmax
      write(deflun,*)' '      	

* the range of important k and w has been found
* now we start to find amplitude of each harmonic by 
* integration of P(w,k)

      dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      
      
* estimate maximum value of t-told for which the subroutine
* XY_PWK will still give correct values and PLOSS error will
* not happen
*

c	difftmax=1.0e+8/wmax/dw
	
c      write(deflun,*)' '
c      write(deflun,*)' The maximum allowable time difference t-told '      
c      write(deflun,*)' for calls to XY_PWK subroutine is about ',difftmax
c      write(deflun,*)' otherwise presision loss errors will happen'
c      write(deflun,*)' '

* integrate P(w,k) to find amplitude
* each cell will be split additionnaly by Ndiv*Ndiv parts
           

* start integration
       do i=1,Nw
       wa=wmin*dw**(i-1)
        do j=1,Nk
        ka=kmin*dk**(j-1)
          wb=wa*dw 
          kb=ka*dk
* the integral of P(w,k) will be stored to s                      
          s=0.

C A.S. 01/28/02  Optimize Ndiv - to decrease computation time for integration
C (inaccuracy increase is small for typical parameters)
	Ndiv=1
	corner1=PWK(wa,ka)
	corner2=PWK(wa,kb)
	corner3=PWK(wb,ka)
	corner4=PWK(wb,kb)
	corner5=PWK(sqrt(wa*wb),sqrt(ka*kb))
	corner=(corner1*corner2*corner3*corner4*corner5)**0.2
	cornerra=100.
	if(corner.ne.0.0) then
	corner1=corner1/corner
	corner2=corner2/corner
	corner3=corner3/corner
	corner4=corner4/corner
	corner5=corner5/corner
	cornerma=dmax1(corner1,corner2,corner3,corner4,corner5)
	cornermi=dmin1(corner1,corner2,corner3,corner4,corner5)
	cornerra=cornerma/cornermi
	end if
	corndiv=(3.0+28.0*cornerra)/(8.0+cornerra)
	Ndiv=corndiv
	if(Ndiv.lt. 2) Ndiv= 2
	if(Ndiv.gt.28) Ndiv=28
C A.S. 01/28/02  -- end 

      dww=(wb-wa)/Ndiv
      dkk=(kb-ka)/Ndiv
      ds=dww*dkk               
      
           do ii=1,Ndiv
           w=wa+dww*ii
                 do jj=1,Ndiv
                 k=ka+dkk*jj
                   s=s+ds*PWK(w,k)
                 end do
           end do        

* the amplitude of ij harmonic is ready
      a0=2./pi*sqrt(s)                 
* it can be negative (in fact it is not needed because anyway we will choose
* random phase, but let it be)       

c	write(deflun,*)'before ran1'
c A.S. 12/28/01 : commented out the next line to ensure that refresh_... will give 
c the same result as prepare_.. for the same random seed 
c      if(ran1_gm(idum).gt.0.5) a0=-a0
c	write(deflun,*)'after ran1, put to a(i,j),i,j=',i,j
      
* the amplitude
      am(i,j)=a0                  
          end do
       end do
                                              
      write(deflun,*)' '
      write(deflun,*)' Harmonics generation finished'
      write(deflun,*)' ' 

* here the phases, w and k will be generated or just refreshed
*
1000	continue

      dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      dww=dw**(1.0/Ndiv)
      dkk=dk**(1.0/Ndiv)

c	write(deflun,*)'about to refresh w and k'
* store frequency 
       do i=1,Nw
       wa=wmin*dw**(i-1)
       wb=wa*dw 
* we take w between wa and wb (which was the interval of PWK integration)
* with uniform distribution like here, so in principle, after many 
* seeds all frequencies will be checked. 
* this choice of w, in fact, result in about 50% inaccuracy 
* for <dx^2> for small t and big l, better results can be obtained
* if we will put w=averaged mean weighted values, but it seems
* it is not acceptable to have fixed w (and especially k) because
* lattice (or supports) may have resonance properties and
* all w and k should be present in the spectrum of signal (at least
* after big number of seeds).
*
       wh(i)=wa+ran1_gm(idum)*(wb-wa)
       end do 

* and store wave number
       do j=1,Nk
       ka=kmin*dk**(j-1)
       kb=ka*dk
* we do for k the same as for w
*       
       kh(j)=ka+ran1_gm(idum)*(kb-ka)            
       end do 

       do i=1,Nw
        do j=1,Nk
* generate random phase ij for horizontal motion
      phase=pi*2.*ran1_gm(idum)
* and store sin and cos of this phase
      dskx(i,j)=sin(phase) 
      dckx(i,j)=cos(phase)      
* generate random phase ij for vertical motion
      phase=pi*2.*ran1_gm(idum)
      dsky(i,j)=sin(phase) 
      dcky(i,j)=cos(phase)      
          end do
       end do 

c      write(deflun,*)' Harmonics phases, w and k made or refreshed'

2000	continue

* initial values of told , sinus and cosin. Remember that t.ge.0
      told=0.0
	dtold=0.0  
           do i=1,Np
           sw(i)=0.0		! it is sin(0*wh(i))
           cw(i)=1.0		!       cos(0*wh(i))
           swdt(i)=0.0		! it is sin(0*wh(i))
           cwdt(i)=1.0		!       cos(0*wh(i))
           end do

* this is to remember that harmonics have been generated
c      inewparam=1

100   format(2i4,5e9.2)
      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_SYSTHAR(inewparams,IERR)
	USE GM_PARAMETERS
	USE GM_HARMONICS_SYST
	USE GM_HARM_PREPARE
	USE GM_RANDOM_GEN
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
*   calculate wave numbers, amplitudes and phases of harmonics
*   responsible for systematic in time motion   
*   Maximum number of harmonics is limited by NH
c      parameter (NH=250)
      real*8 k,ka,kb
c	,kmin,kmax,kmins,kmaxs
c      real*8 ams(NH),khs(NH)
c      real*8 dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
      real*8 Stest(2),kmin1(2),kmax1(2)

c     common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys

c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1
      
c      common/integ/G,kmin,kmax,wmin,wmax,kmins,kmaxs,Nk,Nw
c      common/randum/idum
c      common/timeold/told
c      common/maxtimediff/difftmax
  
* inewparam =0 will say to PREPARE_SYST_HARMONICS that the file
* with P(w,k) parameters has been just read thus complete
* harmonics generation should be performed
* if inewparam =1 it means that the harmonics has been already
* once calculated, this is just new seed and it is enough to
* generate only new phases, w and k
*
c      common/filejustread/inewparam,inewparams
      
      data Ndiv/30/,pi/3.14159265358979/
      
      call ALLOCATE_HARMONCS_SYST (NP,IERR)

* will just regenerate phases and k if inewparams =1
      if(inewparams.ne.0) goto 1000
      
c* test that requested number of harmonics is smaller than array size
c      if(Np.gt.NH) Np=NH

      write(deflun,*)' '
      write(deflun,*)
     >   'Searching for the range of important k for systematic motion'
      
* we will estimate range of important k by rough 
* preliminary integration
* of Q(k) over enough big (hopefully) range of k
      
      Nk=Np
* we define this wide initial range k and beleive that 
* the important range is inside
*
      kmins=0.00001
      kmaxs=100000.
       
* ratio k_{i+1}/k_{i} is constant
*
      dk=(kmaxs/kmins)**(1./(Nk-1))	
 
* to estimate range of important k we will calculate two
* values, namely the mean value < [x(t,s1)-x(t,s2)]^2 > for different
* combination of variables, namely
* s1-s2=Smin
* s1-s2=Smax

      Stest(1)=Smin
      Stest(2)=Smax
      
* double loop to check all these four cases

         do is=1,2
               
* the value < [x(s1)-x(s2)]^2 > is the double integral on
* Q(k) 2 (1-cos(k (s1-s2))) dk/(2pi) = F(k)
*
* to find the important range of k we calculate first the value
* sfin = Int^{kmax}_{kmin} F(k) dk
* and then we calculate first the function 
* s1(k)= Int^{k}_{kmin} F(k) dk
* then the ratio s1(k)/sfin will be equal 0 at kmin and 1 at kmax
* the region where this function changes rapidly from 0 to 1
* gives main contribution to the integral. 
* we define the range of important k as the points where 
* s1(k)/sfin cross the level 0.01 and 0.99 for kmin and kmax
* correspondingly
* 
*
                 
      sfin=0.
        do j=1,Nk
        k=kmins*dk**(j-1)
        ds=(k*dk-k)      
        call QPSK(Stest(is),k,RRR)
        sfin=sfin+ds*RRR
        end do
	
      if(sfin.eq.0.0) goto 500

      kmin1(is)=0.
      kmax1(is)=0.
      s1=0.
       do i=1,Nk
       k=kmins*dk**(i-1)
        ds=(k*dk-k)      
        call QPSK(Stest(is),k,RRR)
        s1=s1+ds*RRR
        s1k=s1/sfin
        if(kmin1(is).eq.0.0.and.s1k.gt.0.01) kmin1(is)=k/dk
        if(kmax1(is).eq.0.0.and.s1k.gt.0.99) kmax1(is)=k*dk
       end do   
       
      end do
      

* we have found the important ranges for all of two cases,
* now we find the range that cover these two
*
       kmins=dmin1(kmin1(2),kmin1(1))
       kmaxs=dmax1(kmax1(2),kmax1(1))
         
500   dk=(kmaxs/kmins)**(1./(Nk-1))	
      
      kmaxs=kmaxs/dk

      write(deflun,*)' '
      write(deflun,*)' For the range of s from ',Smin,' to ',smax
      write(deflun,*)' Range of important k for systematic motion:'      
      write(deflun,*)' k from ',kmins,' to ',kmaxs
      write(deflun,*)' '      	

* the range of important k has been found
* now we start to find amplitude of each harmonic by 
* integration of Q(k)

      dk=(kmaxs/kmins)**(1./(Nk-1))	      
      

* integrate Q(k) to find amplitude
* each cell will be split additionnaly by Ndiv*Ndiv parts           

* start integration
      do j=1,Nk
        ka=kmins*dk**(j-1)
          kb=ka*dk
* the integral of Q(k) will be stored to s                      
          s=0.

      dkk=(kb-ka)/Ndiv
      ds=dkk               
      
                 do jj=1,Ndiv
                 k=ka+dkk*jj
                   s=s+ds*QPK(k)
                 end do

* the amplitude of ij harmonic is ready
C A.S. Apr-28-2002 corrected definition of amplitudes
C      a0=2./pi*sqrt(s)    <- this is wrong               
      a0=sqrt(2./pi) * sqrt(s)               
* it can be negative (in fact it is not needed because anyway we will choose
* random phase, but let it be) 
c A.S. 12/28/01 commented out, to preserve seed      
c       if(ran1_gm(idum).gt.0.5) a0=-a0
* the amplitude
       ams(j)=a0                 
      end do
                                              
      write(deflun,*)' '
      write(deflun,*)' Harmonics generation finished for systematic'
      write(deflun,*)' ' 

* here the phases, k will be generated or just refreshed
*
1000	continue

      dk=(kmaxs/kmins)**(1./(Nk-1))	
      dkk=dk**(1.0/Ndiv)

* store wave number
       do j=1,Nk
       ka=kmins*dk**(j-1)
       kb=ka*dk
       khs(j)=ka+ran1_gm(idum)*(kb-ka)            
       end do 

* will not regenerate phases if inewparams =2
      if(inewparams.eq.2) goto 2000
      
      
* we take k between ka and kb (which was the interval of QPK integration)
* with uniform distribution like here, so in principle, after many 
* seeds all frequencies will be checked. 
* this choice of k, in fact, result in about some inaccuracy 
* for <dx^2> but it seems
* it is not acceptable to have fixed k because
* lattice (or supports) may have resonance properties and
* all k should be present in the spectrum of signal (at least
* after big number of seeds).
*

        do j=1,Nk
* generate random phase j for horizontal motion
      phase=pi*2.*ran1_gm(idum)
* and store sin and cos of this phase
      dskxs(j)=sin(phase) 
      dckxs(j)=cos(phase)      
* generate random phase j for vertical motion
      phase=pi*2.*ran1_gm(idum)
      dskys(j)=sin(phase) 
      dckys(j)=cos(phase)      
          end do
 

c      write(deflun,*)' Harmonics phases and k made or refreshed for systematic'

2000	continue
      
* initial values of told , sinus and cosin. Remember that t.ge.0
      told=0.0
	dtold=0.0
      
* this is to remember that also systematic harmonics have been generated          
c      inewparams=1

100   format(2i4,5e9.2)
      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      subroutine PPWK(S,T,k,w,RRR)
	USE GM_PARAMETERS
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* it put to RRR the function 
* P(w,k) 2 (1-cos(wt)) 2(1-cos(ks)) 2/(2pi) 2/(2pi)
* the coefficient "2" in "2/(2pi)" is due to the fact that 
* we define P(w,k) so that w and k can be positive or negative,
* but we will integrate only on positive values
*
      real*8 k
c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1

      RRR=PWK(w,k)   
     >  *2.*fu(w*T)         ! 2*(1.-cos(w*T)) 
     >  *2.*fu(k*S)         ! 2*(1.-cos(k*L))
     >  *0.1013211          ! 2*2/6.28/6.28

      return
      end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      real*8 function PWK(w,k)
	USE GM_PARAMETERS
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives P(w,k) using parameters of the model
* for the model explanation see somewhere else
*
      real*8 k,Lo
c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1

C		 PWK of the "corrected ATL law":

	if(A.ne.0.0.and.B.ne.0.0) then
      Lo=B/A/w**2

      PWK=A/(w*k)**2 
     >  *fu(Lo*k) 		! that is (1.-cos(Lo*k))	
	else
	 PWK=0.0
	end if     

C               And wave contribution, three peaks
       vv1=v1
       vv2=v2
       vv3=v3
      if(v1.lt.0..or.v2.lt.0..or.v3.lt.0.) vvs=450.+1900.*exp(-w/12.5)
      
* if v < 0 then the SLAC formula is used 

       if(v1.lt.0.0) vv1=vvs
      PN1=Fmicro(w,k,vv1,f1,d1,a1)
       if(v2.lt.0.0) vv2=vvs
      PN2=Fmicro(w,k,vv2,f2,d2,a2)
       if(v3.lt.0.0) vv3=vvs
      PN3=Fmicro(w,k,vv3,f3,d3,a3)
          
      PWK=  PWK  + PN1 + PN2 + PN3 
      return
      end
      
   
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

 	real*8 function fu(x)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives (1-cos(x))
*
* if x is big we replace (1-cos(x)) by its mean value 1
*
		if(x.gt.1.e+8) then
		fu=1.
		return
		end if
	fu=2.*(sin(x/2.))**2
c			it equals to (1-cos(x))
	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	
	real*8 function Fmicro(w,k,velm,fmic,dmic,amic)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives distribution of amplitudes on k and w due to waves
* with phase velosity velm 
* 
	real*8 k,km
	km=w/velm
	Fmicro=0.
	if(k.lt.km) then
* this shape of distribution on k assumes that the waves travell
* in the plane (i.e. on our surface) and distribution of
* directions is homogenious
*
	Fmicro=2./km/sqrt(1.-k**2/km**2)
* distribution on w 
	Fmicro=Fmicro*cmic(w,fmic,dmic,amic)
	end if	
	return
	end
	
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	
	real*8 function cmic(w,fmic,dmic,amic)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives distribution of amplitudes on w due to waves
	f=w/2./3.1415926
	p0=amic
	p1=1./( ((f-fmic)/fmic*dmic)**4 +1.)
	cmic=p0*p1
900	return
	end

                   
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine QPSK(S,k,RRR)
	USE GM_PARAMETERS
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* it put to RRR the function 
* Q(k) 2(1-cos(ks)) 2/(2pi) 
* the coefficient "2" in "2/(2pi)" is due to the fact that 
* we define Q(k) so that k can be positive or negative,
* but we will integrate only on positive values
*
      real*8 k
c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1

      RRR=QPK(k)   
     >  *2.*fu(k*S)         ! 2*(1.-cos(k*L))
     >  *0.318        		! 2/6.28

      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      real*8 function QPK(k)
	USE GM_PARAMETERS
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives Q(k) using parameters of the model
* for the model explanation see somewhere else
*
      real*8 k
c      common/earth/
c     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
c     >Q1,rk1,rkk1
          
!	function cmic(w,fmic,dmic,amic)
* gives distribution of amplitudes on w due to waves
!	f=w/2./3.1415926
!!	QPK=Q1/( ((k-rk1)/rk1*rkk1)**4 +1.)
!!	qqq=(k/rk1)**4
!!	QPK=QPK*qqq /( qqq +1.)	

	QPK=Q1/( ((k/6.2832-rk1)/rk1*rkk1)**2 +1.)
	
      return
      end
      
           
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      real*8 function RAN1_GM(idum)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
*-----------------------------------------------------------------------
*            Copied from "Numerical Recipes", page 271           
*-----------------------------------------------------------------------
      INTEGER*4 idum,ia,im,iq,ir,ntab,ndiv
      real*8 am,eps,rnmx
      parameter (ia=16807,im=2147483647,am=1./im,iq=127773,ir=2836,
     >           ntab=32,ndiv=1+(im-1)/ntab,eps=1.2e-7,rnmx=1.-eps)
C  "Minimal" random number generator of Park and Miller with Bays-Durham
C   shuffle and added safeguards. Returns a uniform random deviate
C   between 0.0 and 1.0 (exclusive of the endpoint values).
C   Call with "idum" a negative number to initialize; thereafter,
C   do not alter "idum" between successive deviates in a sequence. 
C   "rnmx" should approximate the largest floating value that is
C   less than 1.
      INTEGER*4 j,k,iv(ntab),iy
      save iv,iy
      data iv/ntab*0/, iy/0/
      if (idum.le.0.or.iy.eq.0) then
          idum=max(-idum,1)
          do j=ntab+8,1,-1
              k=idum/iq
              idum=ia*(idum-k*iq)-ir*k
              if (idum.lt.0) idum=idum+im
              if (j.le.ntab) iv(j)=idum
          enddo
          iy=iv(1)
      endif
      k=idum/iq
      idum=ia*(idum-k*iq)-ir*k
      if(idum.lt.0) idum=idum+im
      j=1+iy/ndiv
      iy=iv(j)
      iv(j)=idum
      RAN1_GM=min(am*iy,rnmx)
      return
      end    
           
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      real*8 function x2pwka(A,B,t,rl)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
* gives approximation of <[x(t,s+rl)-x(t,s)]^2> for pure ATL P(w,k)
* needed only for the TESTS subroutine
*
      x2pwka=0.
      if(t.eq.0.0.or.rl.eq.0.0.or.A.eq.0.0.or.B.eq.0.0) goto 900
      pi=3.14159265358979      
      t0=sqrt(A*rl/B)
      tt0=t/t0
      q=1.+1./(1.+ ( tt0 + 1./tt0 ))
      x2pwka=A*t0*rl*tt0**2 *2./pi/(1.+2./pi*tt0) *q
900   return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine 
     > GM_MISALIGN
     > (TABS,DT,SS,DX,DY,Nelem,is0,ITFx,ITFy,INx,INy,NxTFx,NyTFy)
c MISALIGN(t)
c	USE TEST_GM_LINE
	USE GM_HARM_PREPARE
       USE CONTROL_MOD
	USE GM_S_POSITION
c	USE GM_ABSOLUTE_TIME
c	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
	IMPLICIT NONE
* calculates x and y positions of all Nelem elements at 
* a given time t
c      parameter (NELMX=300)
c      real*8 ss(NELMX),xx(NELMX),yy(NELMX)
c      common/line/ss,xx,yy,Nelem
c      common/timeold/told 
      INTEGER*4  n, Nelem, is0
      REAL*8 ss(Nelem), Dx(Nelem), Dy(Nelem), S, dxx,dyy,tabs,dt
	REAL*8 dx1,dy1
	REAL*8 dxN,dyN,s1,sN
	REAL*8 dxref,dyref,dxbeg,dybeg
	INTEGER*4 ITFx(Nelem), idtfx
	INTEGER*4 ITFy(Nelem), idtfy
	INTEGER*4 INx(Nelem), idnx
	INTEGER*4 INy(Nelem), idny
	INTEGER*4 imx,imy,NxTFx(Nelem),NyTFy(Nelem)

c	T=GM_ABS_TIME
       do n=1,Nelem
        s=ss(n)
c        call XY_PWK(t,s,x,y,dxx,dyy)
	  idtfx=ITFx(n)
	  idtfy=ITFy(n)
	  idnx=INx(n)
	  idny=INy(n)
C A.S. Apr-28-2002
	  imx=NxTFx(n)
	  imy=NyTFy(n)
c	 write(deflun,*)'n=',n,' idtfx=',idtfx,
c     > 	 ' idtfy=',idtfy,' idnx=',idnx,' idny=',idny 

      call DXDY_PWK(tabs,dt,s,dxx,dyy,idtfx,idtfy,idnx,idny,imx,imy)
        dx(n)=dxx
        dy(n)=dyy
       end do
c    added by A.S. 12/28/01
        s=-SBEG
        call DXDY_PWK(tabs,dt,s,dxref,dyref,0,0,0,0,0,0)
c                                this is beginning of reflected beamline
        s=SBEG
        call DXDY_PWK(tabs,dt,s,dxbeg,dybeg,0,0,0,0,0,0)

	if(is0.eq.1) then
c	dx1=dx(1)
c	dy1=dy(1)
c      dxN=dx(Nelem)
c	dyN=dy(Nelem)
c	s1=ss(1)
c	sN=ss(Nelem)
c	12/28/01 A.S. Changed so that point with s=SBEG and s=-SBEG do not move
	if(.not.FLIPS) then
	  dx1=dxbeg
	  dy1=dybeg
        dxN=dxref
	  dyN=dyref
	  s1=SBEG
	  sN=-SBEG
	else
	  dxN=dxbeg
	  dyN=dybeg
        dx1=dxref
	  dy1=dyref
	  sN=SBEG
	  s1=-SBEG
	end if

c	write(outlun,*) 'flips=',FLIPS
c	write(outlun,*) 'dx1=',dx1
c	write(outlun,*) 'dxN=',dxN
c	write(outlun,*) 's1=',s1
c	write(outlun,*) 'sN=',sN
c	write(outlun,*) 'ss(1)=',ss(1)
c	write(outlun,*) 'ss(Nelem)=',ss(Nelem)

       do n=1,Nelem
c        dx(n)=dx(n)-dx1
c        dy(n)=dy(n)-dy1
c
c    Fixed so that the beginning and end elements do not move 
c    Needed to use reflection of beamline
c        Note that this procedure introduces 
c        small beam angle at the entry
c    A.S. 12/20/01,  12/28/01
c
        dx(n)=dx(n)-dx1-(dxN-dx1)*(ss(n)-s1)/(sN-s1)
        dy(n)=dy(n)-dy1-(dyN-dy1)*(ss(n)-s1)/(sN-s1)
      end do
	end if

      return
      end
 

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	real*8 function a_settlement(t,tau,dtgap,isyst)
	IMPLICIT NONE
	real*8 t,tau,dtgap
	INTEGER*4 isyst
        real*8 appr_settlement
	if(isyst.eq.0)	then
	  a_settlement=t/tau
	else if (isyst.eq.1) then
	  a_settlement=appr_settlement((t+dtgap)/tau)
	end if
	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
	real*8 function appr_settlement(t)
	IMPLICIT NONE
c approximation for the ground settlement 
c  t is normalized time
	real*8 :: A1 = 2.
	real*8 :: c1 = 2.36
	real*8 sq,t
	sq=sqrt(t)
	appr_settlement=(1.-(1.-sq/(1.+A1*sq))*exp(-c1*t))
	return
	end
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      subroutine SAVE_PWK_AMPLITUDES(OUTFILE)
* save amplitudes obtained from P(w,k) into file 
	USE GM_HARMONICS
	USE GM_HARMONICS_SYST
	USE GM_HARM_PREPARE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
c
      character*(*) OUTFILE
c      
	write(deflun,*)'  '
	write(deflun,*)
     &    '   Saving a(i,j) and as(j) amplitudes into '
	write(deflun,*)
     &    '   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   '
      write(deflun,345)' the file ',outfile
345   format(a,a)
c      
      open (92,file=OUTFILE)
      write(92,101) Nw,Nk,wmin,wmax,kmin,kmax
	do i=1,Nw
	 do j=1,Nk
        write(92,102) i,j,am(i,j)
	 end do
	end do
      write(92,103) Nk,kmins,kmaxs
	do j=1,Nk
       write(92,104) j,ams(j)
	end do
      close(92)
101   format(2i6,4e14.6)
102   format(2i6, e14.6)
103   format( i6,2e14.6)
104   format( i6, e14.6)
	return
	end
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_GM2ELT_TF(ID,GMTFFILE,IERR)
* read girder transfer function from file 
	USE GM_PARAMETERS
	USE CONTROL_MOD
	USE GM_TRANSFUNCTION
	USE GM_HARM_PREPARE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 

	real*8, allocatable :: f_f(:)
	real*8, allocatable :: tfr_f(:)
	real*8, allocatable :: tfi_f(:)

	character*50 chardum
      character*(*) GMTFFILE
	logical*4 def_file
	data pi/3.14159265358979/
      
c      inewparam=0

	if(wmin.eq.0.0.or.wmax.eq.0.0) then
           WRITE(outlun,*) 'ERROR> Error in PREPARE_GM2ELT_TF: ', 
     1  'call PREPARE_HARMONICS before defigning Transfer Functions!'
	return
	end if

      def_file = GMTF(ID)%tf_defined_by_file 
      w0 =  2*pi* GMTF(ID)%F0_TF  
      q  =  GMTF(ID)%Q_TF    


	if(def_file) then
      open(92,file=GMTFFILE,err=999,status='old')
	nlines=0
10	read(92,*,end=20) f, tfr, tfi
	nlines=nlines+1
	goto 10 
20	close(92)
	if(allocated(f_f))   deallocate(f_f)
	if(allocated(tfr_f)) deallocate(tfr_f)
	if(allocated(tfi_f)) deallocate(tfi_f)
	allocate(f_f(nlines),stat=ierr)
	if(ierr.ne.0) 
     >  write(outlun,*)'ERROR>Cannot allocate f_f in PREPARE_GM2ELT_TF'
	allocate(tfr_f(nlines),stat=ierr)
	if(ierr.ne.0) 
     > write(outlun,*)'ERROR>Cannot allocate tfr_f in PREPARE_GM2ELT_TF'
	allocate(tfi_f(nlines),stat=ierr)
	if(ierr.ne.0) 
     > write(outlun,*)'ERROR>Cannot allocate tfi_f in PREPARE_GM2ELT_TF'

      open(92,file=GMTFFILE,err=999,status='old')
	do i=1,nlines
	  read(92,*) f_f(i), tfr_f(i), tfi_f(i)
	end do
	close(92)

	if(f_f(1)*2.*pi.gt.wmin.or.f_f(nlines)*2.*pi.lt.wmax) then
       write(outlun,*)
     > 'WARNING> TF file does not cover cover all [wmin,wmax] span'
       write(outlun,*)
     > 'WARNING> check (save TF to file) the extrapolated TF ! '
	end if

	end if

      dw=(wmax/wmin)**(1./(Nw-1)) 
      
	Ndiv=20

       do i=1,Nw
       wa=wmin*dw**(i-1)
          wb=wa*dw 

      dww=(wb-wa)/Ndiv
      ds=dww
      
	sr=0.0
	si=0.0
           do ii=1,Ndiv
           w=wa+dww*ii
			if(def_file) then
	         call Approximate_TF(w,tfr,tfi,f_f,tfr_f,tfi_f,nlines)
			else
	         call Standard_TF(w,w0,q,tfr,tfi)
			end if
                   sr=sr+ds*tfr
				 si=si+ds*tfi
           end do        

      GMTF(ID)%TF_RE(i)  = sr/(wb-wa)
      GMTF(ID)%TF_IM(i)  = si/(wb-wa)
	
	end do

	if(allocated(f_f))   deallocate(f_f)
	if(allocated(tfr_f)) deallocate(tfr_f)
	if(allocated(tfi_f)) deallocate(tfi_f)

900   return

999	ierr=1
       write(deflun,*) 'error!  unexpected end of tf file!'
	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine Standard_TF(w,w0,q,tfr,tfi)
* standard girder transfer function  
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
	c1=1.0 - (w/w0)**2
	c2=2.0*q*w/w0
	c3=c1**2+c2**2
	tfr=c1/c3
	tfi=c2/c3
	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine Approximate_TF(w,tfr,tfi,f_f,tfr_f,tfi_f,N)
* girder transfer function approximated from given array 
c	USE CONTROL_MOD

	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
	real*8 f_f(N), tfr_f(N),tfi_f(N)
	data iw/0/
	data pi/3.14159265358979/

	f=w/2./pi

	if(f.lt.f_f(1)) then
	 tfr=1.
	 tfi=0.
	 return
	end if

	if(f.gt.f_f(N)) then
	 tfr=0.
	 tfi=0.
	 return
	end if

c	if(f.eq.f_f(1)) then
c	 tfr=tfr_f(1)
c	 tfi=tfi_f(1)
c	 return
c	end if

	do i=1,N-1
	 if (f.ge.f_f(i).and.f.le.f_f(i+1)) goto 1
	end do
	i=N
1	tfr=tfr_f(i)+(tfr_f(i+1)-tfr_f(i))*(f-f_f(i))/(f_f(i+1)-f_f(i))
	tfi=tfi_f(i)+(tfi_f(i+1)-tfi_f(i))*(f-f_f(i))/(f_f(i+1)-f_f(i))

	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      subroutine SAVE_GM2ELT_TF(ID,OUTFILE,IERR)
* save transfer function into file 
	USE GM_TRANSFUNCTION
	USE GM_HARM_PREPARE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
c
      character*(*) OUTFILE
	data pi/3.14159265358979/
c      
	write(deflun,*)'  '
	write(deflun,*)
     &    '   Saving Transfer Function with ID = ', ID
      write(deflun,345)' into File = ',outfile
345   format(a,a)
c      
      open (92,file=OUTFILE)

      dw=(wmax/wmin)**(1./(Nw-1)) 

      do i=1,Nw
       wa=wmin*dw**(i-1)
       wb=wa*dw 
       write(92,101) sqrt(wa*wb)/2./pi,
     >	      GMTF(ID)%TF_RE(i),GMTF(ID)%TF_IM(i)
	end do

      close(92)
101   format(3e17.8)

c
c
c        do i=1, Num%elem
c          write(deflun,*) i,element(i)%name
c	 end do 
c

	return
	end
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_GM_TECH_NOISE(inewparam,ID,IERR)
* prepare tech noise 
	USE GM_PARAMETERS
	USE CONTROL_MOD
	USE GM_TECHNICAL_NOISE
	USE GM_HARM_PREPARE
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 

	data pi/3.14159265358979/
    
	if(wmin.eq.0.0.or.wmax.eq.0.0) then
           WRITE(outlun,*) 'ERROR> Error in PREPARE_GM_TECH_NOISE: ', 
     1  'call PREPARE_HARMONICS before defigning Noises!'
	return
	end if

      if(inewparam.eq.1) goto 1000


      f1 = TCHN(ID)%F1  
      a1 = TCHN(ID)%A1  
	d1 = TCHN(ID)%D1 
      f2 = TCHN(ID)%F2  
      a2 = TCHN(ID)%A2  
	d2 = TCHN(ID)%D2 
      f3 = TCHN(ID)%F3  
      a3 = TCHN(ID)%A3  
	d3 = TCHN(ID)%D3 

      dw=(wmax/wmin)**(1./(Nw-1)) 
      
	Ndiv=10

* integrate noise spectrum to find amplitude         

       do i=1,Nw
       wa=wmin*dw**(i-1)
          wb=wa*dw 
          s=0.
           dww=(wb-wa)/Ndiv
            do ii=1,Ndiv
              w=wa+dww*ii
              s=s+dww*Pof3peaks(w,f1,d1,a1,f2,d2,a2,f3,d3,a3)
            end do        
C A.S. Apr-28-2002 : corrected definition of amplitudes
C       a0=2./pi*sqrt(s) <- this is wrong                
       a0=2./sqrt(pi) * sqrt(s)             
       TCHN(ID)%am_n(i)=a0                  
      end do
                                              
* here the phases and w will be generated or just refreshed
*
1000	continue

      dw=(wmax/wmin)**(1./(Nw-1)) 

* store frequency 
       do i=1,Nw
       wa=wmin*dw**(i-1)
       wb=wa*dw 
       TCHN(ID)%wh_n(i)=wa+ran1_gm(idum)*(wb-wa)
       end do 

       do i=1,Nw
* generate random phase 
      phase=pi*2.*ran1_gm(idum)
* and store sin and cos of this phase
      TCHN(ID)%ds_n(i)=sin(phase) 
      TCHN(ID)%dc_n(i)=cos(phase)      
       end do 

c      write(deflun,*)' Harmonics phases and w are made or refreshed'

2000	continue

* initial values of told , sinus and cosin

ccc   told =0.0
ccc	dtold=0.0  

           do i=1,Np
           TCHN(ID)%sw_n(i)=0.0		! it is sin(0*wh(i))
           TCHN(ID)%cw_n(i)=1.0		!       cos(0*wh(i))
           TCHN(ID)%swdt_n(i)=0.0	! it is sin(0*wh(i))
           TCHN(ID)%cwdt_n(i)=1.0	!       cos(0*wh(i))
c
c	write(deflun,*)'ID=',ID,'am_n ',TCHN(ID)%am_n(i)
c	write(deflun,*)'w_n=',TCHN(ID)%wh_n(i)
c
           end do

	return
	end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

	real*8 function Pof3peaks(w,f1,d1,a1,f2,d2,a2,f3,d3,a3)
	IMPLICIT INTEGER*4(I-N), REAL*8(A-G,O-Z) 
	Pof3peaks=0.
	Pof3peaks=Pof3peaks + cmic(w,f1,d1,a1)
	Pof3peaks=Pof3peaks + cmic(w,f2,d2,a2)
	Pof3peaks=Pof3peaks + cmic(w,f3,d3,a3)
	return
	end
	
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
