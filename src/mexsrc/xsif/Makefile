# Makefile for xsif
# created by k.furukawa, nov.17.2000.
# Makefile for Lucretia mex files for passing xsif files
# Requirements:
# 	* requires intel fortran compiler
#	* before use, make sure intel compile environment is set up
#	  correctly and "ifort" points to the fortran compiler binary
# Useage:
#	
# 	* Type "make realclean" to start fresh
#	* Edit FORCOMP variable to point at you system Matlab mex binary if mex not on search path
#       * "make install" to move built files to respective Lucretia directories
# MOD:
#     GW, 05Aug2011:
#		Setup to build mex files
#     PT, 04apr2005:
#               Add PutElemParsToMatlab.f to library.
#     PT, 12Jan2004:
#		support for longer element names, EALIGN and
#		SELECT statements.  New files:
#			ealign.f
#			select.f
#			read_range.f.
#		Add KICKER element, expand HKICK and VKICK 
#		element types to HKICKER and VKICKER, SROT/YROT
#		to SROTATIO/YROTATIO.
#     PT, 26Jun2003:
#               added support for dynamic allocation.  New files:
#                       XSIF_ALLOCATE_INITIAL
#                       MORE_PARS
#                       MORE_ELTS
#                       MORE_LINKS
#                       MORE_CALLS
#                       MORE_WAKES
#                       XSIF_MEM_MANAGE
#                       XSIF_RELEASE_MEM
#                       XOPEN_STACK_MANAGE
#                       XSIF_STACK_SEARCH
#                       XCLOSE_STACK_MANAGE.
#               Changed Solaris compiler for bug workaround.
#     PT, 18Nov2002:
#		added support for new xsif subroutines
#			DECODE_PART_NAME
#			RD_LOGICAL
#			SET_BEAM_DEFAULTS.
#       Linda Hendrickson, 22Jan2002:
#               Added mat-liar option for solaris.
#               If setenv MATLIAR T then it uses options needed for
#               matlab-liar interface building.
#	PT, 22Aug2001:
#		added options relevant for linux compilation on
#		lahey F95 compiler.
#	PT, 01Feb2001:
#		Changed MODSUF for Solaris from .M to .mod, per
#		sudden version change at SLAC...
#	PT, 31Jan2001:
#		added dependencies from previous makefile_sun in 
#		public AFS space, replaced DIMAD_ modules with
#		XSIF_ modules.  Added PARCHK utility function to list.
#		Removed obsolete XSIF_PARSE function.
#

FORCOMP = mex FFLAGS='$$FFLAGS -fdollar-ok'

#WHICHMEX="$(shell which mex)"
#$(info MEX is [${WHICHMEX}])

# Support Linux or OSX (64-bit systems only)
ifeq ($(shell uname),Linux)
  MEXEXT = mexa64
  OSSUF = _linux64
else
  MEXEXT = mexmaci64
  OSSUF = _maci64
endif  
  MODSUF = .mod
  CFLG = -c

 LCLPATH = .
 XSIFLIBS = libxsif$(OSSUF).a
#
XSIFParse.$(MEXEXT): libxsif$(OSSUF).a XSIFParse.F GetXSIFDictionary.$(MEXEXT)
	$(FORCOMP) XSIFParse.F -I. -L. -lxsif$(OSSUF)

GetXSIFDictionary.$(MEXEXT): libxsif$(OSSUF).a GetXSIFDictionary.F
	$(FORCOMP) GetXSIFDictionary.F -I. -L. -lxsif$(OSSUF)

install:: GetXSIFDictionary.$(MEXEXT) XSIFParse.$(MEXEXT)
	mv GetXSIFDictionary.$(MEXEXT) ../../LatticeGeneration/
	mv XSIFParse.$(MEXEXT) ../../LatticeGeneration/

libxsif$(OSSUF).a(%.o): %.F
	$(FORCOMP) $(CFLG) $<
	ar ru $@ $%
	ranlib $@
	rm -f $%     

# dimad_* is used in others

libxsif$(OSSUF).a:	\
	libxsif$(OSSUF).a(xsif_size_pars.o) \
	libxsif$(OSSUF).a(xsif_inout.o) \
	libxsif$(OSSUF).a(xsif_constants.o) \
	libxsif$(OSSUF).a(xsif_elem_pars.o) \
	libxsif$(OSSUF).a(xsif_elements.o) \
	libxsif$(OSSUF).a(xsif_interfaces.o) \
	libxsif$(OSSUF).a(arr_to_str.o) \
	libxsif$(OSSUF).a(arrcmp.o) \
	libxsif$(OSSUF).a(clear.o) \
	libxsif$(OSSUF).a(decexp.o) \
	libxsif$(OSSUF).a(decfrm.o) \
	libxsif$(OSSUF).a(declst.o) \
	libxsif$(OSSUF).a(decobs.o) \
	libxsif$(OSSUF).a(decode_part_name.o) \
	libxsif$(OSSUF).a(decpar.o) \
	libxsif$(OSSUF).a(decpnt.o) \
	libxsif$(OSSUF).a(decuse.o) \
	libxsif$(OSSUF).a(defpar.o) \
	libxsif$(OSSUF).a(dump.o) \
	libxsif$(OSSUF).a(ealign.o) \
	libxsif$(OSSUF).a(elmdef.o) \
	libxsif$(OSSUF).a(expand.o) \
	libxsif$(OSSUF).a(fndelm.o) \
	libxsif$(OSSUF).a(fndpar.o) \
	libxsif$(OSSUF).a(frmset.o) \
	libxsif$(OSSUF).a(int_to_char.o) \
	libxsif$(OSSUF).a(intrac.o) \
	libxsif$(OSSUF).a(jobnam.o) \
	libxsif$(OSSUF).a(length.o) \
	libxsif$(OSSUF).a(line.o) \
	libxsif$(OSSUF).a(newlft.o) \
	libxsif$(OSSUF).a(newlst.o) \
	libxsif$(OSSUF).a(newrgt.o) \
	libxsif$(OSSUF).a(opdef.o) \
	libxsif$(OSSUF).a(ovflow.o) \
	libxsif$(OSSUF).a(param.o) \
	libxsif$(OSSUF).a(parchk.o) \
	libxsif$(OSSUF).a(parcon.o) \
	libxsif$(OSSUF).a(parevl.o) \
	libxsif$(OSSUF).a(parord.o) \
	libxsif$(OSSUF).a(parprt.o) \
	libxsif$(OSSUF).a(prline.o) \
	libxsif$(OSSUF).a(rd_filename.o) \
	libxsif$(OSSUF).a(rd_logical.o) \
	libxsif$(OSSUF).a(rd_wakefilename.o) \
	libxsif$(OSSUF).a(rdend.o) \
	libxsif$(OSSUF).a(rdfail.o) \
	libxsif$(OSSUF).a(rdfile.o) \
	libxsif$(OSSUF).a(rdfind.o) \
	libxsif$(OSSUF).a(rdinit.o) \
	libxsif$(OSSUF).a(rdint.o) \
	libxsif$(OSSUF).a(rdlabl.o) \
	libxsif$(OSSUF).a(rdline.o) \
	libxsif$(OSSUF).a(rdlook.o) \
	libxsif$(OSSUF).a(rdnext.o) \
	libxsif$(OSSUF).a(rdnumb.o) \
	libxsif$(OSSUF).a(rdpara.o) \
	libxsif$(OSSUF).a(rdptab.o) \
	libxsif$(OSSUF).a(rdskip.o) \
	libxsif$(OSSUF).a(rdtest.o) \
	libxsif$(OSSUF).a(rdwarn.o) \
	libxsif$(OSSUF).a(rdword.o) \
	libxsif$(OSSUF).a(read_range.o) \
	libxsif$(OSSUF).a(select.o) \
	libxsif$(OSSUF).a(set_beam_defaults.o) \
	libxsif$(OSSUF).a(title.o) \
	libxsif$(OSSUF).a(value.o) \
	libxsif$(OSSUF).a(vms_timedate.o) \
	libxsif$(OSSUF).a(xcall.o) \
	libxsif$(OSSUF).a(xclose.o) \
	libxsif$(OSSUF).a(xopen.o) \
	libxsif$(OSSUF).a(xpath.o) \
	libxsif$(OSSUF).a(xpath_expand.o) \
	libxsif$(OSSUF).a(xretrn.o) \
	libxsif$(OSSUF).a(xsif_cmd_loop.o) \
	libxsif$(OSSUF).a(xsif_header.o) \
	libxsif$(OSSUF).a(xsif_io_close.o) \
	libxsif$(OSSUF).a(xsif_io_setup.o) \
	libxsif$(OSSUF).a(xuse.o) \
	libxsif$(OSSUF).a(xuse2.o) \
	libxsif$(OSSUF).a(xsif_allocate_initial.o) \
	libxsif$(OSSUF).a(more_pars.o) \
	libxsif$(OSSUF).a(more_elts.o) \
	libxsif$(OSSUF).a(more_links.o) \
	libxsif$(OSSUF).a(more_wakes.o) \
        libxsif$(OSSUF).a(more_calls.o) \
	libxsif$(OSSUF).a(xsif_mem_manage.o) \
	libxsif$(OSSUF).a(xsif_release_mem.o) \
	libxsif$(OSSUF).a(xopen_stack_manage.o) \
	libxsif$(OSSUF).a(xsif_stack_search.o) \
	libxsif$(OSSUF).a(xclose_stack_manage.o) \
	libxsif$(OSSUF).a(PutElemParsToMatlab.o) \

#

libxsif$(OSSUF).a(clear.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)   xsif_constants$(MODSUF)

libxsif$(OSSUF).a(decexp.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(decfrm.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(declst.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(decobs.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(decode_part_name.o) : xsif_elem_pars$(MODSUF) xsif_inout$(MODSUF)   \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(decpar.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF) xsif_elem_pars$(MODSUF)

libxsif$(OSSUF).a(decpnt.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(decuse.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)           \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(defpar.o) : xsif_size_pars$(MODSUF)   xsif_elements$(MODSUF)        

libxsif$(OSSUF).a(dump.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)             \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(ealign.o) : xsif_size_pars$(MODSUF)   xsif_inout$(MODSUF)             \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(elmdef.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)  xsif_elem_pars$(MODSUF)  xsif_constants$(MODSUF)

libxsif$(OSSUF).a(expand.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(fndelm.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(fndpar.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(frmset.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(line.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)              \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(newlft.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(newlst.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(newrgt.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(opdef.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)             \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(ovflow.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(param.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)             \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(parchk.o) : xsif_inout$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(parcon.o) : xsif_size_pars$(MODSUF)  xsif_elements$(MODSUF)

libxsif$(OSSUF).a(parevl.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(parord.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(parprt.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)            \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(rd_filename.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rd_logical.o) : xsif_inout$(MODSUF) 

libxsif$(OSSUF).a(rd_wakefilename.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)   \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(rdend.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdfail.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdfile.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdfind.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdinit.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdint.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdlabl.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdline.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdnext.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdnumb.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdpara.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdptab.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdskip.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdtest.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdwarn.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(rdword.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(read_range.o) :  xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF) \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(select.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF) \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(set_beam_defaults.o) : xsif_constants$(MODSUF) \
   xsif_elements$(MODSUF) xsif_elem_pars$(MODSUF)

libxsif$(OSSUF).a(title.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(value.o) : xsif_size_pars$(MODSUF)  xsif_inout$(MODSUF)             \
   xsif_elements$(MODSUF)

libxsif$(OSSUF).a(xcall.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xclose.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xopen.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xpath.o) : xsif_inout$(MODSUF)   xsif_interfaces$(MODSUF)

libxsif$(OSSUF).a(xpath_expand.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xretrn.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xsif_cmd_loop.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xsif_header.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xsif_io_close.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xsif_io_setup.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xuse.o) : xsif_inout$(MODSUF)  xsif_size_pars$(MODSUF) \
                            xsif_elements$(MODSUF)

libxsif$(OSSUF).a(xuse2.o) : xsif_inout$(MODSUF)

libxsif$(OSSUF).a(xsif_allocate_initial.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(more_pars.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(more_elts.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(more_links.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(more_calls.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF)

libxsif$(OSSUF).a(more_wakes.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(xsif_mem_manage.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) \
                                             xsif_elements$(MODSUF)

libxsif$(OSSUF).a(xsif_release_mem.o) : xsif_size_pars$(MODSUF) \
                                             xsif_inout$(MODSUF) 

libxsif$(OSSUF).a(xopen_stack_manage.o) : xsif_inout$(MODSUF) \
                                             xsif_interfaces$(MODSUF)

libxsif$(OSSUF).a(xsif_stack_search.o) : xsif_inout$(MODSUF) 

libxsif$(OSSUF).a(xclose_stack_manage.o) : xsif_inout$(MODSUF) 

xsif_constants$(MODSUF) : libxsif$(OSSUF).a(xsif_constants.o)

xsif_elem_pars$(MODSUF) : libxsif$(OSSUF).a(xsif_elem_pars.o)

xsif_elements$(MODSUF) : xsif_size_pars$(MODSUF) libxsif$(OSSUF).a(xsif_elements.o)

xsif_inout$(MODSUF) : xsif_size_pars$(MODSUF) libxsif$(OSSUF).a(xsif_inout.o)

xsif_interfaces$(MODSUF) : libxsif$(OSSUF).a(xsif_interfaces.o)
  
xsif_size_pars$(MODSUF) : libxsif$(OSSUF).a(xsif_size_pars.o)



clean:	
	-rm -f *.o

realclean:	clean
	-rm -f libxsif$(OSSUF).a
	-rm -f *$(MODSUF)
	-rm -f *.$(MEXEXT)

release:
	-mv *.$(MEXEXT) ../../LatticeGeneration
