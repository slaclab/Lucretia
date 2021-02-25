      INTEGER*4 FUNCTION LENGTH(STRING)                                    
C
C     member of MAD INPUT PARSER
C
C---- FIND LAST NON-BLANK CHARACTER IN "STRING"                          
C----------------------------------------------------------------------- 
      SAVE
      CHARACTER*(*)     STRING                                           
C----------------------------------------------------------------------- 
      DO 10 L = LEN(STRING), 1, -1                                       
        IF (STRING(L:L) .NE. ' ') GO TO 20                               
   10 CONTINUE                                                           
      L = 1                                                              
   20 LENGTH = L                                                         
      RETURN                                                             
      END                                                                
