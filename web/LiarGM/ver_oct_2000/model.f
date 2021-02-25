*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      program MODEL

*-----------------------------------------------------------------------
*                                  MODEL                           
*                         version October 1999                      
*                       Seryi@SLAC.Stanford.EDU                   
*-----------------------------------------------------------------------
*
*  conversion to C:   f2c -r8 model.f    (-r8 promote real to double)
*
*-----------------------------------------------------------------------
*        Computes horizontal x(t,s) and vertical y(t,s)       
*        position of the ground at a given time t and in a given 
*        longitudinal position s, assuming that at time t=0 we had
*        x(0,s)=0 and y(0,s)=0.  The values x(t,s) and y(t,s) will 
*        be computed using the same power spectrum P(w,k), however
*        they are independent. Parameters of approximation of 
*        the P(w,k) (PWK) can be chosen to model quiet or noisy
*        place.
*        Units are seconds and meters.
*       
*-----------------------------------------------------------------------
*                   The program needs the next files              
*       "model.data"  (if it does not exist, it will create it)    
*        for explanations of parameters see Report DAPNIA/SEA 95-04 
*        (should also appear in Phys.Rew.E, in 1 April 1997 issue)  
*                                                                  
*        Also needs one column file "positions.data" that contains  
*        longitudinal positions s of points where we will want to 
*        find x(t,s) and y(t,s). Number of lines in this file is 
*        therefore the numbers of elements. In FORTRAN version
*        this number is limited.
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
*      x(t,s) = 0.5 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * sin(w_{i} t) 
*                                                * sin(k_{j} s + phi_{ij})
*             + 0.5 sum_{i}^{Np} sum_{j}^{Np} am_{ij} * (cos(w_{i} t)-1)
*                                                * sin(k_{j} s + psi_{ij})
*      This choise of formula ensure x(t,s) = 0 at t=0.
*      The last sinus is presented in the program in other form
*      sin(k_{j} s + phi_{ij}) = sin(k_{j} s) cos(phi_{ij}) 
*                                     + cos(k_{j} s) sin(phi_{ij})
*      So, we store not phases phi_{ij}, but cos(phi_{ij}) and 
*      sin(phi_{ij})
*      The same for y(t,s) but with different phases phi_{ij}.
*      
*-----------------------------------------------------------------------
*                   Changes since the previous versions
* 12.07.96 
*  1. A bug fixed in XY_PWK. We should use both sin(wt) and cos(wt)-1
*     and not only sin(wt) as it was before
*
* 26.06.96
*  1. A bug is fixed in the PWK subroutine. The variable Lo was
*     forgotten to declare real.
*
* 12.03.96             
*  1. Calculation of sinus or cosine function from big   
*          arguments has been changed, presicion loss errors       
*          should not happen any more                               
*  2. Many comments are added (maybe still not enough) 
*  3. Harmonics generation has been changed to increase speed
*     If parameters of P(w,k) are the same, then at second and others
*     calls of PREPARE_HARMONICS subroutine only phases will
*     be refreshed (amplitudes will be the same, no need to integrate
*     P(w,k) again)
*
* Oct 99
*  1. Would like to implement systematic ground motion as
*     "x(t,s)" = "x given by P(w,k)" + time* "given by Q(k)"
*     where Q(k) is the power spectrum of the systematic 
*     comnponent
*-----------------------------------------------------------------------
* What is not done yet:
* Number of steps of integration in PREPARE_HARMONICS
* is not optimized 
*-----------------------------------------------------------------------


      parameter (NELMX=300)
* Maximum number of elements is limited by NELMX. 
* This defect can be avoided in C version.      
*
      real ss(NELMX),xx(NELMX),yy(NELMX)
* arrays for longitudinal, horizontal and vertical position, and
* number of elements 
*      
      common/line/ss,xx,yy,Nelem
      common/randum/idum
      
* if random generator called with negative idum, the generator
* will be initialized (idum then should not be touched)
* this initialization is not necessary, in fact, but we put
* it to have possibility to get different seeds from the start
* It is maybe better to read idum from input file?
*

	write(6,*)'Input positive integer to make seed='
	read (5,*)idum

c      idum= -7987413
      idum=-idum
      dummy=ran1(idum)
      
* read parameters of PWK      
      call READ_PWK_PARAM
      
      write(6,*)'   '    
      write(6,*)' We still print some information to standard output'
      write(6,*)' it can be suppressed when not needed '

* read longitudinal position of elements and count their number Nelem
      call READ_POSITIONS

* calculate frequencies, wave numbers, amplitudes and phases of harmonics
      call PREPARE_HARMONICS

      write(*,*)'before syst prep harm'
      call PREPARE_SYSTHAR
      write(*,*)'after syst prep'
       
* the main call to calculate position of elements at a given time
*  
      call MISALIGN(time)       

*     start of tests, should be commented when not needed
*     (consume processor time and also changes arrays in the 
*     common/line/, namely it put Nelem=4 and array of position
*     as ss/0. , Smin , sqrt(Smax*Smin) , Smax/
*

      call TESTS                
c      call TESTS2                
c      call TESTS3                
      
      stop
      end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine MISALIGN(t)
* calculates x and y positions of all Nelem elements at 
* a given time t
      parameter (NELMX=300)
      real ss(NELMX),xx(NELMX),yy(NELMX)
      common/line/ss,xx,yy,Nelem
      common/timeold/told 
       do n=1,Nelem
        s=ss(n)
        call XY_PWK(t,s,x,y)
        xx(n)=x
        yy(n)=y
       end do
      return
      end
 

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine READ_POSITIONS
* reads positions of elements from the file positions.data,
* counts total number of elements
      parameter (NELMX=300)
      real ss(NELMX),xx(NELMX),yy(NELMX)
      common/line/ss,xx,yy,Nelem
      
      open(92,file='positions.data',err=999,status='old')
      i=1
1     read(92,*,end=900) ss(i)
      i=i+1
      if(i.gt.NELMX) goto 999
      goto 1
900   Nelem=i-1
      goto 100 
999   write(6,*)' Error open "positions.data" or too many elements'
100   close(92)
      return
      end      

      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine XY_PWK(t,s,x,y)
* it computes x and y positions of a single element that has
* longitudinal coordinate s, at a given time t (positive)
*
      parameter (NH=250)
* arrays of: amplitude  am, frequency (omega) wh, wave number kh
      real am(NH,NH),wh(NH),kh(NH)
* arrays to store values sin(w_{i} t) and cos(w_{i} t)
* we will use only sinus, but cosine we need to calculate sinus
* for the new time t using values saved at time told
      real sw(NH),cw(NH)
      real dskx(NH,NH),dckx(NH,NH),dsky(NH,NH),dcky(NH,NH)
      real sk(NH),ck(NH)
*
      real ams(NH),khs(NH)
      real dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
*
      common/harmonics/am,wh,kh,sw,cw,dskx,dckx,dsky,dcky
      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys

      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
* told is the time t of previous use of this subroutine
      common/timeold/told 
	
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

* we calculate sin(k_{j} s) at each step. This is stupid and can 
* be avoided for the cost of two arrays (NH*NELMX). But this array can be
* very big, in our cases (50*300) = (15000) 
* What is better?
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
c           x=x + am(i,j) * sw(i) * sinkx  ! this was a bug, fixed 12.07.96
c           y=y + am(i,j) * sw(i) * sinky
           x=x + am(i,j) * ( sw(i)*sinkx + (cw(i)-1.0)*sinky )*0.5
           y=y + am(i,j) * ( sw(i)*sinky + (cw(i)-1.0)*sinkx )*0.5
         end do
        end do

* add systematic components

* we calculate sin(k_{j} s) at each step. This is stupid and can 
* be avoided for the cost of two arrays (NH*NELMX). But this array can be
* very big, in our cases (50*300) = (15000) 
* What is better?
*
           do j=1,Np
            sk(j)=sin(s*khs(j))
            ck(j)=cos(s*khs(j))
           end do

         do j=1,Np         
           sinkx=sk(j)*dckxs(j)+ck(j)*dskxs(j)
           sinky=sk(j)*dckys(j)+ck(j)*dskys(j)
c           x=x + am(i,j) * sw(i) * sinkx  ! this was a bug, fixed 12.07.96
c           y=y + am(i,j) * sw(i) * sinky
           x=x + ams(j) * t * sinkx
           y=y + ams(j) * t * sinky
         end do

      return
      end
      
            
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_HARMONICS
* calculate frequencies, wave numbers, amplitudes and phases of harmonics
*
* Maximum number of harmonics is limited by NH*NH
      parameter (NH=250)
      real k,ka,kb,kmin,kmax,kmins,kmaxs
      real am(NH,NH),wh(NH),kh(NH)
      real sw(NH),cw(NH)
      real dskx(NH,NH),dckx(NH,NH),dsky(NH,NH),dcky(NH,NH)
      real Stest(2),Ttest(2),wmin1(2,2),wmax1(2,2),kmin1(2,2),kmax1(2,2)
      common/harmonics/am,wh,kh,sw,cw,dskx,dckx,dsky,dcky
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
     
      common/integ/G,kmin,kmax,wmin,wmax,Nk,Nw,kmins,kmaxs
      common/randum/idum
      common/timeold/told
      common/maxtimediff/difftmax
  
* inewparam =0 will say to PREPARE_HARMONICS that the file
* with P(w,k) parameters has been just read thus complete
* harmonics generation should be performed
* if inewparam =1 it means that the harmonics has been already
* once calculated, this is just new seed and it is enough to 
* generate only new phases, w and k
*
      common/filejustread/inewparam,inewparams
      
      data Ndiv/30/,pi/3.14159265358979/
      
* will just regenerate phases, w and k if inewparam =1
      if(inewparam.eq.1) goto 1000
      
* test that requested number of harmonics is smaller than array size
      if(Np.gt.NH) Np=NH

      write(6,*)' '
      write(6,*)' Finding range of important w and k'
      
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
       kmin=amin1(kmin1(is,it),kmin)
       wmin=amin1(wmin1(is,it),wmin)
       kmax=amax1(kmax1(is,it),kmax)
       wmax=amax1(wmax1(is,it),wmax)
         end do
         end do
         
500   dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      
      wmax=wmax/dw
      kmax=kmax/dk

      write(6,*)' '
      write(6,*)' Range of important k and w:'      
      write(6,*)' k from ',kmin,' to ',kmax
      write(6,*)' w from ',wmin,' to ',wmax
      write(6,*)' '      	

* the range of important k and w has been found
* now we start to find amplitude of each harmonic by 
* integration of P(w,k)

      dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      
      
* estimate maximum value of t-told for which the subroutine
* XY_PWK will still give correct values and PLOSS error will
* not happen
*
	difftmax=1.0e+8/wmax/dw
	
      write(6,*)' '
      write(6,*)' The maximum allowable time difference t-told '      
      write(6,*)' for calls to XY_PWK subroutine is about ',difftmax
      write(6,*)' otherwise presision loss errors will happen'
      write(6,*)' '

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
      if(ran1(idum).gt.0.5) a0=-a0
* the amplitude
      am(i,j)=a0                  
          end do
       end do
                                              
      write(6,*)' '
      write(6,*)' Harmonics generation finished'
      write(6,*)' ' 

* here the phases, w and k will be generated or just refreshed
*
1000	continue

      dk=(kmax/kmin)**(1./(Nk-1))	
      dw=(wmax/wmin)**(1./(Nw-1)) 
      dww=dw**(1.0/Ndiv)
      dkk=dk**(1.0/Ndiv)

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
       wh(i)=wa+ran1(idum)*(wb-wa)
       end do 

* and store wave number
       do j=1,Nk
       ka=kmin*dk**(j-1)
       kb=ka*dk
* we do for k the same as for w
*       
       kh(j)=ka+ran1(idum)*(kb-ka)            
       end do 

       do i=1,Nw
        do j=1,Nk
* generate random phase ij for horizontal motion
      phase=pi*2.*ran1(idum)
* and store sin and cos of this phase
      dskx(i,j)=sin(phase) 
      dckx(i,j)=cos(phase)      
* generate random phase ij for vertical motion
      phase=pi*2.*ran1(idum)
      dsky(i,j)=sin(phase) 
      dcky(i,j)=cos(phase)      
          end do
       end do 

c      write(6,*)' Harmonics phases, w and k made or refreshed'

2000	continue

* initial values of told , sinus and cosin. Remember that t.ge.0
      told=0.0  
           do i=1,Np
           sw(i)=0.0		! it is sin(0*wh(i))
           cw(i)=1.0		!       cos(0*wh(i))
           end do

* this is to remember that harmonics have been generated
      inewparam=1

100   format(2i4,5e9.2)
      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine PREPARE_SYSTHAR
*   calculate wave numbers, amplitudes and phases of harmonics
*   responsible for systematic in time motion   
*   Maximum number of harmonics is limited by NH
      parameter (NH=250)
      real k,ka,kb,kmin,kmax,kmins,kmaxs
      real ams(NH),khs(NH)
      real dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
      real Stest(2),kmin1(2),kmax1(2)

      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
     
      common/integ/G,kmin,kmax,wmin,wmax,Nk,Nw,kmins,kmaxs
      common/randum/idum
      common/timeold/told
      common/maxtimediff/difftmax
  
* inewparam =0 will say to PREPARE_SYST_HARMONICS that the file
* with P(w,k) parameters has been just read thus complete
* harmonics generation should be performed
* if inewparam =1 it means that the harmonics has been already
* once calculated, this is just new seed and it is enough to
* generate only new phases, w and k
*
      common/filejustread/inewparam,inewparams
      
      data Ndiv/30/,pi/3.14159265358979/
      
* will just regenerate phases and k if inewparams =1
      if(inewparams.ne.0) goto 1000
      
* test that requested number of harmonics is smaller than array size
      if(Np.gt.NH) Np=NH

      write(6,*)' '
      write(6,*)' Finding range of important k for systematic motion'
      
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
       kmins=amin1(kmin1(2),kmin1(1))
       kmaxs=amax1(kmax1(2),kmax1(1))
         
500   dk=(kmaxs/kmins)**(1./(Nk-1))	
      
      kmaxs=kmaxs/dk

      write(6,*)' '
      write(6,*)' Range of important k for systematic motion:'      
      write(6,*)' k from ',kmins,' to ',kmaxs
      write(6,*)' '      	

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
      a0=2./pi*sqrt(s)                 
* it can be negative (in fact it is not needed because anyway we will choose
* random phase, but let it be)       
       if(ran1(idum).gt.0.5) a0=-a0
* the amplitude
       ams(j)=a0                 
      end do
                                              
      write(6,*)' '
      write(6,*)' Harmonics generation finished for systematic'
      write(6,*)' ' 

* here the phases, k will be generated or just refreshed
*
1000	continue

      dk=(kmaxs/kmins)**(1./(Nk-1))	
      dkk=dk**(1.0/Ndiv)

* store wave number
       do j=1,Nk
       ka=kmins*dk**(j-1)
       kb=ka*dk
       khs(j)=ka+ran1(idum)*(kb-ka)            
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
      phase=pi*2.*ran1(idum)
* and store sin and cos of this phase
      dskxs(j)=sin(phase) 
      dckxs(j)=cos(phase)      
* generate random phase j for vertical motion
      phase=pi*2.*ran1(idum)
      dskys(j)=sin(phase) 
      dckys(j)=cos(phase)      
          end do
 

c      write(6,*)' Harmonics phases and k made or refreshed for systematic'

2000	continue
      
* initial values of told , sinus and cosin. Remember that t.ge.0
      told=0.0  
      
* this is to remember that also systematic harmonics have been generated          
      inewparams=1

100   format(2i4,5e9.2)
      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      subroutine PPWK(S,T,k,w,RRR)
* it put to RRR the function 
* P(w,k) 2 (1-cos(wt)) 2(1-cos(ks)) 2/(2pi) 2/(2pi)
* the coefficient "2" in "2/(2pi)" is due to the fact that 
* we define P(w,k) so that w and k can be positive or negative,
* but we will integrate only on positive values
*
      real k
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1

      RRR=PWK(w,k)   
     >  *2.*fu(w*T)         ! 2*(1.-cos(w*T)) 
     >  *2.*fu(k*S)         ! 2*(1.-cos(k*L))
     >  *0.1013211          ! 2*2/6.28/6.28

      return
      end

*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      function PWK(w,k)
* gives P(w,k) using parameters of the model
* for the model explanation see somewhere else
*
      real k,Lo
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1

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

 	function fu(x)
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
	
	function Fmicro(w,k,velm,fmic,dmic,amic)
* gives distribution of amplitudes on k and w due to waves
* with phase velosity velm 
* 
	real k,km
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
	
	function cmic(w,fmic,dmic,amic)
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
* it put to RRR the function 
* Q(k) 2(1-cos(ks)) 2/(2pi) 
* the coefficient "2" in "2/(2pi)" is due to the fact that 
* we define Q(k) so that k can be positive or negative,
* but we will integrate only on positive values
*
      real k
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1

      RRR=QPK(k)   
     >  *2.*fu(k*S)         ! 2*(1.-cos(k*L))
     >  *0.318        		! 2/6.28

      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      function QPK(k)
* gives Q(k) using parameters of the model
* for the model explanation see somewhere else
*
      real k
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
          
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
      function RAN1(idum)
*-----------------------------------------------------------------------
*            Copied from "Numerical Recipes", page 271           
*-----------------------------------------------------------------------
      integer idum,ia,im,iq,ir,ntab,ndiv
      real RAN1,am,eps,rnmx
      parameter (ia=16807,im=2147483647,am=1./im,iq=127773,ir=2836,
     >           ntab=32,ndiv=1+(im-1)/ntab,eps=1.2e-7,rnmx=1.-eps)
C  "Minimal" random number generator of Park and Miller with Bays-Durham
C   shuffle and added safeguards. Returns a uniform random deviate
C   between 0.0 and 1.0 (exclusive of the endpoint values).
C   Call with "idum" a negative number to initialize; thereafter,
C   do not alter "idum" between successive deviates in a sequence. 
C   "rnmx" should approximate the largest floating value that is
C   less than 1.
      integer j,k,iv(ntab),iy
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
      RAN1=min(am*iy,rnmx)
      return
      end    
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      subroutine READ_PWK_PARAM
* read parameters of P(w,k) from the file and put to the common block
* if there is no input file, a version that correspond to very noisy
* conditions such as in the HERA tunnel will be created
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
      common/filejustread/inewparam,inewparams
      character*50 chardum
      
      inewparam=0
      
      open(92,file='model.data',err=999,status='old')
      read(92,*,err=999,end=999) chardum,A
      read(92,*,err=999,end=999) chardum,B
      
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
      
      read(92,*,err=999,end=999) chardum,Tmin
      read(92,*,err=999,end=999) chardum,Tmax
      read(92,*,err=999,end=999) chardum,Smin
      read(92,*,err=999,end=999) chardum,Smax
      
      read(92,*,err=999,end=999) chardum,Np

      read(92,*,err=999,end=999) chardum,Q1
      read(92,*,err=999,end=999) chardum,rk1
      read(92,*,err=999,end=999) chardum,rkk1
      
      close(92) 
      goto 900

999   close(92) 

      write(6,*)' '
      write(6,*)' Error reading file "model.data", a new'        
      write(6,*)' version of the file has been created and used'
      write(6,*)' (conditions such as HERA tunnel+SLAC systematics)'
      write(6,*)' '        
      A=1.e-16
      B=1.e-15
      
      f1=.14
      a1=10.*1.e-12
      d1=5.          
      v1=1000.
      
      f2=2.5
      a2=.001*1.e-12
      d2=1.5
      v2=400.
      
      f3=50.
      a3=1.e-7*1.e-12
      d3=1.5
      v3=400.

      Tmin=0.001
      Tmax=10000.
      Smin=1.0
      Smax=1000.
      Np=50
      dt=0.02
      
      Q1=1.0E-21
      rk1=1.0E-01
      rkk1=1.0
      
      open(92,file='model.data')
      chardum='''Parameter A of the ATL law,     A [m**2/m/s]   '''
      write(92,920) chardum,A
      chardum='''Parameter B of the PWK,         B [m**2/s**3]  '''
      write(92,920) chardum,B
      
      chardum='''Frequency of 1-st peak in PWK,  f1 [Hz]        '''
      write(92,920) chardum,f1
      chardum='''Amplitude of 1-st peak in PWK,  a1 [m**2/Hz]   '''
      write(92,920) chardum,a1
      chardum='''Width of 1-st peak in PWK,      d1 [1]         '''
      write(92,920) chardum,d1
      chardum='''Velocity of 1-st peak in PWK,   v1 [m/s]       '''
      write(92,920) chardum,v1
            
      chardum='''Frequency of 2-nd peak in PWK,  f2 [Hz]        '''
      write(92,920) chardum,f2
      chardum='''Amplitude of 2-nd peak in PWK,  a2 [m**2/Hz]   '''
      write(92,920) chardum,a2
      chardum='''Width of 2-nd peak in PWK,      d2 [1]         '''
      write(92,920) chardum,d2
      chardum='''Velocity of 2-nd peak in PWK,   v2 [m/s]       '''
      write(92,920) chardum,v2
            
      chardum='''Frequency of 3-rd peak in PWK,  f3 [Hz]        '''
      write(92,920) chardum,f3
      chardum='''Amplitude of 3-rd peak in PWK,  a3 [m**2/Hz]   '''
      write(92,920) chardum,a3
      chardum='''Width of 3-rd peak in PWK,      d3 [1]         '''
      write(92,920) chardum,d3
      chardum='''Velocity of 3-rd peak in PWK,   v3 [m/s]       '''
      write(92,920) chardum,v3
      
      chardum='''Minimum time,                   Tmin [s]       '''
      write(92,920) chardum,Tmin
      chardum='''Maximum time,                   Tmax [s]       '''
      write(92,920) chardum,Tmax

      chardum='''Minimum distance,               Smin [m]       '''
      write(92,920) chardum,Smin
      chardum='''Maximum distance,               Smax [m]       '''
      write(92,920) chardum,Smax

      chardum='''Number of w or k harmonics,     Np   [1]       '''
      write(92,921) chardum,Np

      chardum='''Ampl. of peak in systematic.P,Q1  [m**3*Hz**2] '''
      write(92,920) chardum,Q1
      chardum='''Wavenumber of peak in syst.P,  rk1 [1/m]       '''
      write(92,920) chardum,rk1
      chardum='''Width of peak in system.P,      rkk1 [1]       '''
      write(92,920) chardum,rkk1
      
      close(92)

920   format(a,1pe12.5)
921   format(a,i6)
     
900   continue

      chardum='''Parameter A of the ATL law,     A [m**2/m/s]   '''
      write(6,920) chardum,A
      chardum='''Parameter B of the PWK,         B [m**2/s**3]  '''
      write(6,920) chardum,B
      
      chardum='''Frequency of 1-st peak in PWK,  f1 [Hz]        '''
      write(6,920) chardum,f1
      chardum='''Amplitude of 1-st peak in PWK,  a1 [m**2/Hz]   '''
      write(6,920) chardum,a1
      chardum='''Width of 1-st peak in PWK,      d1 [1]         '''
      write(6,920) chardum,d1
      chardum='''Velocity of 1-st peak in PWK,   v1 [m/s]       '''
      write(6,920) chardum,v1
            
      chardum='''Frequency of 2-nd peak in PWK,  f2 [Hz]        '''
      write(6,920) chardum,f2
      chardum='''Amplitude of 2-nd peak in PWK,  a2 [m**2/Hz]   '''
      write(6,920) chardum,a2
      chardum='''Width of 2-nd peak in PWK,      d2 [1]         '''
      write(6,920) chardum,d2
      chardum='''Velocity of 2-nd peak in PWK,   v2 [m/s]       '''
      write(6,920) chardum,v2
            
      chardum='''Frequency of 3-rd peak in PWK,  f3 [Hz]        '''
      write(6,920) chardum,f3
      chardum='''Amplitude of 3-rd peak in PWK,  a3 [m**2/Hz]   '''
      write(6,920) chardum,a3
      chardum='''Width of 3-rd peak in PWK,      d3 [1]         '''
      write(6,920) chardum,d3
      chardum='''Velocity of 3-rd peak in PWK,   v3 [m/s]       '''
      write(6,920) chardum,v3
      
      chardum='''Minimum time,                   Tmin [s]       '''
      write(6,920) chardum,Tmin
      chardum='''Maximum time,                   Tmax [s]       '''
      write(6,920) chardum,Tmax

      chardum='''Minimum distance,               Smin [m]       '''
      write(6,920) chardum,Smin
      chardum='''Maximum distance,               Smax [m]       '''
      write(6,920) chardum,Smax

      chardum='''Number of w or k harmonics,     Np   [1]       '''
      write(6,921) chardum,Np

      chardum='''Ampl. of peak in systematic.P,Q1  [m**3*Hz**2] '''
      write(6,920) chardum,Q1
      chardum='''Wavenumber of peak in syst.P,  rk1 [1/m]       '''
      write(6,920) chardum,rk1
      chardum='''Width of peak in system.P,      rkk1 [1]       '''
      write(6,920) chardum,rkk1

      return
      end
      
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################

      function x2pwka(A,B,t,rl)
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
      
      subroutine TESTS
* do some tests using generated harmonics
* 
      parameter (NELMX=300)
      parameter (NH=250)
      real ss(NELMX),xx(NELMX),yy(NELMX)
      real dx(3,20),dy(3,20)
      real xma(7),xmi(7)
      common/line/ss,xx,yy,Nelem
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1
      common/randum/idum
      common/timeold/told            
      common/maxtimediff/difftmax
            
      write(6,*)' '
      write(6,*)' Start of TESTS'
      
      Nelem=4
      ss(1)=0.
      ss(2)=Smin
      ss(3)=sqrt(Smax*Smin)
      ss(4)=Smax

      write(6,*)'  ' 
      write(6,*)' write to ftn08 t, x(0), x(s1), x(s2), x(s3)'
      write(6,*)' write to ftn09 t, y(0), y(s1), y(s2), y(s3)'
      write(6,*)' where s1, s2, s3 =',Smin,sqrt(Smax*Smin),Smax
      write(6,*)' Nt is number of points on time'

	Nt=1000    
	Tmaxhere=Tmax
	
       write(6,*)'Input Nt='
       read (5,*)Nt

	dtmaxhere=Tmaxhere/Nt
	
      write(6,*)'  ' 
	write(6,*)' In tests dtmax=',dtmaxhere
	
	  if(dtmaxhere.gt.difftmax) then
	    Tmaxhere=difftmax*Nt
      write(6,*)'  ' 
	write(6,*)'TESTS: too big Tmax, localy changed to ',Tmaxhere
	write(6,*)' (or one can change number of steps in TESTS)'
	  end if	

      do time=0.0,Tmaxhere,Tmaxhere/Nt
      call MISALIGN(time)
      write(8,100) time,xx(1),xx(2),xx(3),xx(4)
      write(9,100) time,yy(1),yy(2),yy(3),yy(4)
      end do
      write(6,*)' finished ' 


*****
      write(6,*)'  ' 
      write(6,*)' write to ftn13 s, x(t1), x(t2) ... x(t5)'
      write(6,*)' write to ftn14 s, y(t1), y(t2) ... y(t5)'
      write(6,*)' where s in meters'
      write(6,*)' max time ', Tmaxhere

	dt=2.e+8/7	!Tmaxhere/7		!5
      write(6,*)'  ' 
	write(6,*)' In files 13,14 dt=',dt
*

	do it=1,7	!5
	t=dt*it
        call XY_PWK(t,27000./1277,x,y)
        xmi(it)=x
        call XY_PWK(t,27000.,x,y)
        xma(it)=x
	end do

       do n=1,1277	!300
c       s=3000./300*n	! length of the SLC linac is 2 miles ...
       s=27000./1277*n	! length of the LEP is 27 km ...
	do it=1,7	!5
	t=dt*it
        call XY_PWK(t,s,x,y)
        xx(it)=x		! (it) used not as was supposed to, but OK
        yy(it)=y
	end do
          write(13,150) 
     > s,(xx(it)-xmi(it)-(xma(it)-xmi(it))/1276.*(n-1) ,it=1,7)	!5)
          write(14,150) s,(yy(it),it=1,7)	!5)
       end do

*
            
      Nt=20
      Nseed=50
      Tmaxhere=Tmax
      
5000  dt=(Tmaxhere/Tmin)**(1./(Nt-1))      
      dtmaxhere=Tmin*dt**(Nt-1)-Tmin*dt**(Nt-2)
      write(6,*)'  ' 
	write(6,*)' In tests dtmax=',dtmaxhere
            
	  if(dtmaxhere.gt.difftmax) then
	    Tmaxhere=difftmax/(1.-1./dt)
      write(6,*)'  ' 
      write(6,*)'TESTS: too big Tmax, localy changed to ',Tmaxhere	    
	write(6,*)' (or one can change array size in TESTS)'
	    goto 5000
	  end if	
	         
* here we will calculate <[x(t,s+rl)-x(t,s)]^2> for different
* t and sl, with Nseed number of averaging. Each time we should 
* generate new harmonics thus it is time consuming
* the approximate value of <[x(t,s+rl)-x(t,s)]^2> for the 
* "corrected ATL" will be also calculated and if our model
* has no wave contribution (amplitudes of all three peaks are zero)
* then these calculated and analytical values should be close
* it allows to see that program works well or not
* for example it can allow to estimate number of harmonics
* that we really need to describe ground motion with  
* desired accuracy 
* 

      write(6,*)'  ' 
      write(6,*)' write t,rms_x, rms_y, rms_quiet_pwk '
      write(6,*)' to ftn10 for ds=',Smin
      write(6,*)' to ftn11 for ds=',sqrt(Smax*Smin)
      write(6,*)' to ftn12 for ds=',Smax            
      write(6,*)' number of seeds=',Nseed
      write(6,*)'  ' 

        
      do iseed=1,Nseed
c      write(6,*)' iseed=',iseed      
      call PREPARE_HARMONICS
      call PREPARE_SYSTHAR
              do n=1,Nt
              t=Tmin*dt**(n-1)      
              call MISALIGN(t)      
                      do k=1,3
                      dx(k,n)=dx(k,n)+(xx(1)-xx(k+1))**2
                      dy(k,n)=dy(k,n)+(yy(1)-yy(k+1))**2
                      end do      
              end do      
      end do
      
      do n=1,Nt
      t=Tmin*dt**(n-1)
         do k=1,3
         dx(k,n)=sqrt(dx(k,n)/Nseed)
         dy(k,n)=sqrt(dy(k,n)/Nseed)
         end do
      write(10,100)t,dx(1,n),dy(1,n),sqrt(x2pwka(A,B,t,Smin))
      write(11,100)t,dx(2,n),dy(2,n),sqrt(x2pwka(A,B,t,sqrt(Smin*Smax)))
      write(12,100)t,dx(3,n),dy(3,n),sqrt(x2pwka(A,B,t,Smax))
      end do
      
      write(6,*)' TESTS finished.  ' 
      write(6,*)'  Check files ftn08, ftn09, ftn10, ftn11, ftn12 '
      write(6,*)' In C version the names are '
      write(6,*)' fort.8, fort.9, fort.10, fort.11, fort.12 '
             
100   format(5e13.5)
150   format(8e13.5)
      return
      end    
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################


*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      subroutine TESTS2
* do some tests using generated harmonics
* 
      parameter (NELMX=300)
      parameter (NH=250)
      real ss(NELMX),xx(NELMX),yy(NELMX)
      real dx(3,20),dy(3,20)
      character*20 aaa
      real zd(300),yd(300),ym(300),yq(300)
      real optsx(NH),optcx(NH),optsy(NH),optcy(NH),optam(NH),optkh(NH)
      
      real ams(NH),khs(NH)
      real dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
      
      common/line/ss,xx,yy,Nelem
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1     
      common/randum/idum
      common/timeold/told            
      common/maxtimediff/difftmax

      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys
      
      write(6,*)' '
      write(6,*)' Start of TESTS2'
      
      open(unit=1,file='slc_83v.dat')
      read(1,101) aaa
      read(1,101) aaa
101   format(20a)   

      npoi=276
      do i=1,npoi
       read(1,*) zd(i),yd(i)
       yd(i)=yd(i)/1000.
      end do
      close(1)
      
      write(6,*)' File read'
          
	sumd_min=0.0
	
	Nseedmax=5000
	write(*,*)'input Nseedmax='
	read(*,*)Nseedmax

      do iseed=1,Nseedmax

      do i=1,npoi
        call XY_PWK(Tmax,zd(i),x,y)
       ym(i)=y
      end do

      write(6,*)' model ym generated'
      
      sym2=0.0
      sydym=0.0
      do i=1,npoi
        sym2=sym2+ym(i)**2
	sydym=sydym+ym(i)*yd(i)
      end do
      scale=sydym/sym2
      
      do i=1,npoi
       ym(i)=ym(i)*scale
      end do
            
c      write(6,*)' model scaled by *',scale
      
      
      sumd=0.0
      do i=1,npoi
        sumd=sumd+(ym(i)-yd(i))**2
      end do
      sumd=sumd/npoi
      
      if(sumd_min.ne.0.0) then
       if(sumd.lt.sumd_min) then
       sumd_min=sumd
       opt_scl=scale
       do j=1,Np
       optsx(j)=dskxs(j)
       optcx(j)=dckxs(j)
       optsy(j)=dskys(j)
       optcy(j)=dckys(j)
       optam(j)=ams(j)
       optkh(j)=khs(j)
       end do
      write(6,*)'Best  <dy**2>=',sumd,' min<dy**2>=',sumd_min
      write(6,*)'   at iseed=',iseed,' scale=',scale
        open(unit=1,file='testme.dat')
        do i=1,npoi
         write(1,100) zd(i),yd(i),ym(i)
        end do
        close(1)

      open(unit=1,file='opt.dat')
      write(1,*)'# scale=',opt_scl,' sum_min=',sumd_min,' seed=',iseed
      do j=1,Np
      write(1,150)optsx(j),optcx(j),optsy(j),optcy(j),optam(j),optkh(j)
      end do
      close(1)

       end if
	
	
      else
       sumd_min=sumd
      end if
      
	if(100*(iseed/100).eq.iseed) then
      write(6,*)' <dy**2>=',sumd,' min<dy**2>=',sumd_min              
      write(6,*)' iseed=',iseed      
	end if
      
      
c      call PREPARE_HARMONICS
      call PREPARE_SYSTHAR
c      write(6,*)' back to beginning, iseed=',iseed

      end do	! loop of iseed
            
      write(6,*)' TESTS2 finished.  ' 
      
      open(unit=1,file='opt.dat')
      write(1,*)'# scale=',opt_scl,' sum_min=',sumd_min
      do j=1,Np
      write(1,150)optsx(j),optcx(j),optsy(j),optcy(j),optam(j),optkh(j)
      end do
      close(1)
      write(6,*)' TESTS2 finished, see file opt.dat for phases ' 

             
100   format(5e13.5)
150   format(6e13.5)
      return
      end    
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
      
      subroutine TESTS3
* do some tests using generated harmonics
* 
      parameter (NELMX=300)
      parameter (NH=250)
      real ss(NELMX),xx(NELMX),yy(NELMX)
      real dx(3,20),dy(3,20)
      character*20 aaa
      real zd(300),yd(300),ym(300),yq(300)
      real optsx(NH),optcx(NH),optsy(NH),optcy(NH),optam(NH),optkh(NH)
      
      real ams(NH),khs(NH)
      real dskxs(NH),dckxs(NH),dskys(NH),dckys(NH)
      
      common/line/ss,xx,yy,Nelem
      common/earth/
     >A,B,f1,a1,d1,v1,f2,a2,d2,v2,f3,a3,d3,v3,Tmax,Tmin,Smax,Smin,Np,
     >Q1,rk1,rkk1     
      common/randum/idum
      common/timeold/told            
      common/maxtimediff/difftmax

      common/harmonics_syst/ams,khs,dskxs,dckxs,dskys,dckys
      common/filejustread/inewparam,inewparams
      
      write(6,*)' '
      write(6,*)' Start of TESTS3'
      
      open(unit=1,file='slc_83v.dat')
      read(1,101) aaa
      read(1,101) aaa
101   format(20a)   

      npoi=276
      do i=1,npoi
       read(1,*) zd(i),yd(i)
       yd(i)=yd(i)/1000.
      end do
      close(1)
      
      write(6,*)' File read'
          
*=======================================================
* add systematic components
* we calculate sin(k_{j} s) at each step. This is stupid and can 
* be avoided for the cost of two arrays (NH*NELMX). But this array can be
* very big, in our cases (50*300) = (15000) 
* What is better?
*
c           do j=1,Np
c            sk(j)=sin(s*khs(j))
c            ck(j)=cos(s*khs(j))
c           end do
c
c         do j=1,Np         
c           sinkx=sk(j)*dckxs(j)+ck(j)*dskxs(j)
c           sinky=sk(j)*dckys(j)+ck(j)*dskys(j)
c           x=x + ams(j) * t * sinkx
c           y=y + ams(j) * t * sinky
c         end do
c
*========================================================



	Nseedmax=5000
	write(*,*)'input Nseedmax='
	read(*,*)Nseedmax

	sumd_min=0.0
	
      do iseed=1,Nseedmax

* find phase and amplitude a la fouriert

	do j0=1,Np
	 s_ys=0.0
	 s_yc=0.0
	 s_ss=0.0
	 s_cc=0.0
	   do i=1,npoi
	     sinus=sin(zd(i)*khs(j0))
	     cosin=cos(zd(i)*khs(j0))	
	     s_ys=s_ys+yd(i)*sinus
	     s_yc=s_yc+yd(i)*cosin
	     s_ss=s_ss+sinus**2
	     s_cc=s_cc+cosin**2
	   end do
	   phase_j0=atan2(s_yc*s_ss,s_ys*s_cc)
	   dckxs(j0)=cos(phase_j0)
	   dskxs(j0)=sin(phase_j0)
	   if(dckxs(j0).ne.0.0)then
	     ams(j0)=s_ys/(s_ss*dckxs(j0))/Tmax
	   else
	     ams(j0)=s_yc/(s_cc*dskxs(j0))/Tmax
	   end if  
	end do

	write(*,*)'phases and amplitudes are redefined'

	
      do i=1,npoi
        call XY_PWK(Tmax,zd(i),x,y)
       ym(i)=y
      end do

      write(6,*)' model ym generated with redefined phases and ampl.'
      
      sym2=0.0
      sydym=0.0
      do i=1,npoi
        sym2=sym2+ym(i)**2
	sydym=sydym+ym(i)*yd(i)
      end do
      scale=sydym/sym2
      
c      do i=1,npoi
c       ym(i)=ym(i)*scale
c      end do
            
c      write(6,*)' model scaled by *',scale
      
      
      sumd=0.0
      do i=1,npoi
        sumd=sumd+(ym(i)-yd(i))**2
      end do
      sumd=sumd/npoi
      
      if(sumd_min.ne.0.0) then
       if(sumd.lt.sumd_min) then
       sumd_min=sumd
       opt_scl=scale
       do j=1,Np
       optsx(j)=dskxs(j)
       optcx(j)=dckxs(j)
       optsy(j)=dskys(j)
       optcy(j)=dckys(j)
       optam(j)=ams(j)
       optkh(j)=khs(j)
       end do
      write(6,*)'Best  <dy**2>=',sumd,' min<dy**2>=',sumd_min
      write(6,*)'   at iseed=',iseed,' scale=',scale
        open(unit=1,file='testme.dat')
        do i=1,npoi
         write(1,100) zd(i),yd(i),ym(i)
        end do
        close(1)

      open(unit=1,file='opt.dat')
      write(1,*)'# scale=',opt_scl,' sum_min=',sumd_min,' seed=',iseed
      do j=1,Np
      write(1,150)optsx(j),optcx(j),optsy(j),optcy(j),optam(j),optkh(j)
      end do
      close(1)

       end if
	
	
      else
       sumd_min=sumd
      end if
      
	if(100*(iseed/100).eq.iseed) then
      write(6,*)' <dy**2>=',sumd,' min<dy**2>=',sumd_min              
      write(6,*)' iseed=',iseed      
	end if
      
      
c      call PREPARE_HARMONICS

c will only regenerate k and not phases
	inewparams=2
	
      call PREPARE_SYSTHAR
      
c      write(6,*)' back to beginning, iseed=',iseed

      end do	! loop of iseed
            
      write(6,*)' TESTS3 finished.  ' 
      
      open(unit=1,file='opt.dat')
      write(1,*)'# scale=',opt_scl,' sum_min=',sumd_min
      do j=1,Np
      write(1,150)optsx(j),optcx(j),optsy(j),optcy(j),optam(j),optkh(j)
      end do
      close(1)
      write(6,*)' TESTS3 finished, see file opt.dat for phases ' 

             
100   format(5e13.5)
150   format(6e13.5)
      return
      end    
*#######################################################################
*23456789*123456789*123456789*123456789*123456789*123456789*123456789*12
*#######################################################################
