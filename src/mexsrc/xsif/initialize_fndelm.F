      subroutine initialize_fndelm( ival, kval)!, klist )

      use xsif_elements

      implicit none
                                                             
      integer*4 ival
      character*8 kval!, klist(maxelm)
      character*8 k1

      CALL RDLOOK(kval,8,Kelem(1),1,ielem1,Ival)                          
c      IEDAT(1,Ival) = 0                                                 
c      IEDAT(2,Ival) = 0                                                 
c      IEDAT(3,Ival) = 0                                                 
c      IELIN(Ival) = lval                                               
c      KELEM(Ival) = Kval                                               
c      KETYP(ival) = '    '                                              

      return
      end      