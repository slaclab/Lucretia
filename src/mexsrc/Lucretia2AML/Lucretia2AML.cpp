#include <iostream>
#include <fstream>
#include <cstdlib>
#include "XSIFParser.hpp"
#include "UAPUtilities.hpp"
#include "AMLReader.hpp"
#include "AMLLatticeExpander.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
 /* This function will take the Lucretia lattice in the global workspace and convert it
  * to an AML lattice.
  * The first thing needed is to check for the existence of BEAMLINE, and error out if it 
  * can't find it.*/
  string fileout;
  bool ok, dogirders = true, doFL = false, verbose = true, xsif_conv = false;

  if (nrhs==1) {
    mxArray *BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
    if (!BEAMLINEmx) {
      mexErrMsgTxt("No BEAMLINE in the workspace.");
    }
    fileout = "testfile.aml";

    if (!mxIsLogical(prhs[0])) mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    if (!(mxGetN(prhs[0])==1)) mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    bool *log_in = mxGetLogicals(prhs[0]), *xsifin = log_in;
    xsif_conv = *xsifin;
  }

  else if (nrhs==2) {
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1])) mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    string in_1 = mxArrayToString(prhs[0]);
    if (!strcmp(in_1.c_str(),"input")) {
      mxArray *filein = mxCreateString(mxArrayToString(prhs[1]));
      if (verbose) cout << "Loading " << mxArrayToString(prhs[1]) << endl;
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");
      fileout = "testfile.aml";
    }
    else if (!strcmp(in_1.c_str(),"output")) {
      mxArray *BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
      if (!BEAMLINEmx) {
        mexErrMsgTxt("No BEAMLINE in the workspace.");
      }
      fileout = mxArrayToString(prhs[1]);
    }
    else mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
  }

  else if (nrhs==3) {
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1]) || !mxIsLogical(prhs[2]))
      mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    if (!(mxGetN(prhs[2])==1)) mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    bool *log_in = mxGetLogicals(prhs[2]), *xsifin = log_in;
    xsif_conv = *xsifin;

    string in_1 = mxArrayToString(prhs[0]);                                                             
    if (!strcmp(in_1.c_str(),"input")) {                                                                
      mxArray *filein = mxCreateString(mxArrayToString(prhs[1]));                                       
      if (verbose) cout << "Loading " << mxArrayToString(prhs[1]) << endl;
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");                                                 
      fileout = "testfile.aml";                                                                         
    }                                                                                                   
    else if (!strcmp(in_1.c_str(),"output")) {                                                          
      mxArray *BEAMLINEmx = mexGetVariable("global", "BEAMLINE");                                       
      if (!BEAMLINEmx) {                                                                                
        mexErrMsgTxt("No BEAMLINE in the workspace.");                                                  
      }                                                                                                 
      fileout = mxArrayToString(prhs[1]);                                                               
    }                                                                                                   
    else mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
  }

  else if (nrhs==4) {
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1]) || !mxIsChar(prhs[2]) || !mxIsChar(prhs[3])) {
      mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    }
    string in_1 = mxArrayToString(prhs[0]);
    string in_2 = mxArrayToString(prhs[2]);
    if (!strcmp(in_1.c_str(),"input") && !strcmp(in_2.c_str(),"output")) {
      mxArray *filein = mxCreateString(mxArrayToString(prhs[1]));
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");
      fileout = mxArrayToString(prhs[3]);
    }
    else if (!strcmp(in_2.c_str(),"input") && !strcmp(in_1.c_str(),"output")) {
      mxArray *filein = mxCreateString(mxArrayToString(prhs[3]));
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");
      fileout = mxArrayToString(prhs[1]);
    }
    else mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
  }

  else if (nrhs==5) {
    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1]) || !mxIsChar(prhs[2]) || !mxIsChar(prhs[3]) || !mxIsLogical(prhs[4]))
      mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    if (!(mxGetN(prhs[4])==1)) mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
    bool *log_in = mxGetLogicals(prhs[4]), *xsifin = log_in;
    xsif_conv = *xsifin;

    string in_1 = mxArrayToString(prhs[0]);                                                             
    string in_2 = mxArrayToString(prhs[2]);                                                             
    if (!strcmp(in_1.c_str(),"input") && !strcmp(in_2.c_str(),"output")) {                              
      mxArray *filein = mxCreateString(mxArrayToString(prhs[1]));
      if (verbose) cout << "Loading " << mxArrayToString(prhs[1]) << endl;
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");
      fileout = mxArrayToString(prhs[3]);
    }
    else if (!strcmp(in_2.c_str(),"input") && !strcmp(in_1.c_str(),"output")) {
      mxArray *filein = mxCreateString(mxArrayToString(prhs[3]));
      if (verbose) cout << "Loading " << mxArrayToString(prhs[1]) << endl;
      mexCallMATLAB(0,NULL,1,&filein,"load_n_testlat");
      fileout = mxArrayToString(prhs[1]);
    }
    else mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
  }

  else if (nrhs==0) {
    mxArray *BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
    if (!BEAMLINEmx) {
      mexErrMsgTxt("No BEAMLINE in the workspace.");
    }
    fileout = "testfile.aml";
  }

  else {
    mexErrMsgTxt("Incorrect inputs to Lucretia2AML");
  }

  mxArray *BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
  if (!BEAMLINEmx) { mexErrMsgTxt("No BEAMLINE array in the global workspace. Terminating."); }

  mxArray *FLmx = mexGetVariable("global", "FL");
  mxArray *INSTRmx=NULL, *HwInfomx=NULL, *FLGIRDSmx=NULL, *FLPSmx=NULL, *FLINSTRmx=NULL;
  if (FLmx && mxGetN(FLmx)>0) {
    doFL = true;
    INSTRmx = mexGetVariable("global", "INSTR");
    HwInfomx = mxGetField(FLmx, 0, "HwInfo");
    if (HwInfomx) {
      FLGIRDSmx = mxGetField(HwInfomx, 0, "GIRDER");
      FLPSmx = mxGetField(HwInfomx, 0, "PS");
      FLINSTRmx = mxGetField(HwInfomx, 0, "INSTR");
    }
  }

  mxArray *GIRDERmx = mexGetVariable("global", "GIRDER");
  if (!GIRDERmx) dogirders = false;
  else {
    for (int i=0 ; i<mxGetN(GIRDERmx) ; i++) {
      mxArray *GCell = mxGetCell(GIRDERmx, i);
      mxArray *GEles = mxGetField(GCell, 0, "Element");
      double *Ele1 = mxGetPr(GEles);
      double *Ele2 = Ele1 + (mxGetN(GEles)-1);
      for (double n=*Ele1 ; n<(*Ele2+0.5) ; n++) {
        mxArray *BLEle = mxGetCell(BEAMLINEmx, (int)n-1);
        mxArray *GNum = mxGetField(BLEle, 0, "Girder");
        if (!GNum) {
          mxAddField(BLEle, "Girder");
          mxSetField(BLEle, 0, "Girder", mxCreateDoubleScalar(i+1));
          mxAddField(BLEle, "L2AMLadded");
          mxSetField(BLEle, 0, "L2AMLadded", mxCreateLogicalScalar(true));
        }
      }
    }
  }

 /* Create the master node -- <AML_representation>. This will hold all the element definitions.*/
  string masternodename = "AML_representation";
  UAPNode *AMLRepNode;
  AMLRepNode = new UAPNode(ELEMENT_NODE, masternodename.c_str());
  UAPNode *LabNode = AMLRepNode->addChild(ELEMENT_NODE, "laboratory");

 /* Create another node to hold the sector (girder) and mover definitions.
  * Add this to the previous node at the end.*/
  string sectornodename = "sector_defs";
  UAPNode *SectorNode;
  SectorNode = new UAPNode(ELEMENT_NODE, sectornodename.c_str());
  UAPNode *TopMachineNode = SectorNode->addChild(ELEMENT_NODE, "machine");
  UAPNode *MachineNode = TopMachineNode->addChild(ELEMENT_NODE, "sector");
  MachineNode->addAttribute("name", "ConvertedLattice");

 /* Loop around all the elements of BEAMLINE and send the cell array to LEle2UAPNode. */
  int numele = mxGetM(BEAMLINEmx);
  mxArray *Elemx, *Blockmx;
  UAPNode *EleNode;
  int i=0;
  while (i<numele) {
   /*  Get the cell array corresponding to this data point.*/
    Elemx = mxGetCell(BEAMLINEmx, i);

   /* Find the Block field if it exists.*/
    Blockmx = mxGetField(Elemx, 0, "Block");
   /* If Block exists, then this element is one part of a larger physical object,
    * and we need to pass a list of elements to LEle2UAPNode.*/
    if (Blockmx) {
     /* Find the limits of the block.*/
      mxArrayList LucEleList;
      double *blockstartptr = mxGetPr(Blockmx), *blockendptr = blockstartptr+1;
      double blockstart = *blockstartptr, blockend = *blockendptr;

     /* Push each element to the end of a list.*/
      for (int EleInt=(int)blockstart-1; EleInt!=blockend; EleInt++) {
        LucEleList.push_back( mxGetCell(BEAMLINEmx, (int)EleInt) );
      }
     /* Pass this list to LEle2UAPNode.*/
      EleNode = LEle2UAPNode(i, LabNode, LucEleList, FLPSmx, INSTRmx, FLINSTRmx);

     /* Advance to the end of the block so that the i index is correct.*/
      i = (int)(blockend-1);
    }
   /* If there's no Block, then life is easy.*/
    else {
      EleNode = LEle2UAPNode(i, LabNode, Elemx, FLPSmx, INSTRmx, FLINSTRmx);
    }

   /* If there's a GIRDER here, add it as a sector to MachineNode, otherwise, just add the element.*/
    if (dogirders) {
      mxArray *BLGFieldmx = mxGetField(Elemx, 0, "Girder");
      double GirderNum = 0;
     /* If it found a girder, then figure out its number.*/
      if (BLGFieldmx) GirderNum = mxGetScalar(BLGFieldmx);
     /* Most elements have a girder field even if there's not girder, 
      * so only do the work if the girder!=0.*/
      if (GirderNum!=0) {
        UAPNode *GirderSector, *ThisSector, *CopyNode, *SecNode, *ThisSector2;
       /* Get a list of the sectors we have already added.*/
        NodeList SectorList = LabNode->getChildrenByName("sector");
  
       /* If this list is zero length, then we haven't added any yet, so we don't have to search.*/
        if (SectorList.size()==0){
         /* Add a "sector" node, and give it a name attribute, with value "Girder" + GirderNum.*/
          ThisSector = LabNode->addChild("sector");
          GirderSector = ThisSector->addChild("girder");
          ThisSector2 = MachineNode->addChild("sector");
          //GirderSector->addAttribute("name","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
          ThisSector->addAttribute("name","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
          ThisSector2->addAttribute("ref","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
         /* Then add a copy of the element node to this sector*/
          CopyNode = ThisSector->addChildCopy(EleNode);
         /* We don't need the subtree of EleNode, so loop around it's children and delete them.*/
          NodeList CopyNodeList = CopyNode->getChildren();
          NodeListIter CopyIter=CopyNodeList.begin(); 
          for (CopyIter ; CopyIter!=CopyNodeList.end() ; CopyIter++){
            UAPNode* TempNode = *CopyIter;
            TempNode->deleteTree();
          }
          string nameattrib = CopyNode->getAttributeString("name");
          CopyNode->removeAttribute("name");
          CopyNode->addAttribute("ref",nameattrib);
        } /*if (SectorList.size()==0)*/
       /* If the list isn't zero length, then we have to search to make sure there are no 
        * girders already there*/
        else {
          bool assigned=false;
         /* Loop around an iterator pointing to each sector node.*/
          NodeListIter SecIter=SectorList.begin(); 
          for (SecIter ; SecIter!=SectorList.end() ; SecIter++) {
           /* Assign the iterator to a ptr and get its name*/
            SecNode = *SecIter;
            string CurrNameStr = SecNode->getAttributeString("name");
           /* Assign the name of this girder to a string for comparison with the name of this sector.*/
            string NameStr = "Girder"+BasicUtilities::double_to_string(GirderNum, ok);
           /* If they're the same, then add the element to this sector, and delete its 
            * children as before*/
            if (strcmp(CurrNameStr.c_str(),NameStr.c_str()) == 0){
              UAPNode* OrientNode = SecNode->getChildByName("orientation");
              if (OrientNode) CopyNode = SecNode->addChildCopy(EleNode, OrientNode);
              else CopyNode = SecNode->addChildCopy(EleNode);
              NodeList CopyNodeList = CopyNode->getChildren();
              NodeListIter CopyIter=CopyNodeList.begin(); 
              for (CopyIter; CopyIter!=CopyNodeList.end(); CopyIter++){
                UAPNode* TempNode = *CopyIter;
                TempNode->deleteTree();
              }
              string nameattrib = CopyNode->getAttributeString("name");
              CopyNode->removeAttribute("name");
              CopyNode->addAttribute("ref",nameattrib);
             /* Set a bool to indicate that this element has been assigned to a sector*/
              assigned=true;
             /* And stop looping around the machinenode looking for the sector.*/
              break;
            } /*if (strcmp(CurrNameStr.c_str(),NameStr.c_str()) == 0)*/
          } /*for (SecIter ; SecIter!=SectorList.end() ; SecIter++)*/
         /* If we've finished the loop and still not assigned this element to a sector, then do it now*/
          if(!assigned) {
           /* Create a new sector child with the appropriate name*/
            ThisSector = LabNode->addChild("sector");
            GirderSector = ThisSector->addChild("girder");
            ThisSector2 = MachineNode->addChild("sector");
            //GirderSector->addAttribute("name","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
            ThisSector->addAttribute("name","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
            ThisSector2->addAttribute("ref","Girder"+BasicUtilities::double_to_string(GirderNum, ok));
           /* Add a copy of EleNode, then loop around its children, deleting them.*/
            CopyNode = ThisSector->addChildCopy(EleNode);
            NodeList CopyNodeList = CopyNode->getChildren();
            NodeListIter CopyIter=CopyNodeList.begin(); 
            for (CopyIter; CopyIter!=CopyNodeList.end(); CopyIter++){
              UAPNode* TempNode = *CopyIter;
              TempNode->deleteTree();
            }
            string nameattrib = CopyNode->getAttributeString("name");
            CopyNode->removeAttribute("name");
            CopyNode->addAttribute("ref",nameattrib);
          } /*if(!assigned)*/
        } /*else*/
  
        if (!GirderSector->getChildByName("orientation")) {
          addmover(mxGetCell(GIRDERmx, (int)GirderNum-1), GirderSector);
        }

        if (doFL && !GirderSector->getChildByName("control_sys") && FLGIRDSmx) {
          UAPNode *CtrlSysNode = GirderSector->addChild("control_sys");
          addFloodLand(CtrlSysNode, FLGIRDSmx, GirderNum);
          NodeList ChildList = CtrlSysNode->getChildren();
          if (ChildList.size()==0) CtrlSysNode->deleteTree();
        } /* doFL and control_sys*/

      } /*if (GirderNum!=0)*/
      else { /* We haven't found a girder here, so we just add the element to the MachineNode. */
        UAPNode* CopyNode = MachineNode->addChildCopy(EleNode);
        NodeList CopyNodeList = CopyNode->getChildren();
        NodeListIter CopyIter=CopyNodeList.begin();
        for (CopyIter; CopyIter!=CopyNodeList.end(); CopyIter++){
          UAPNode* TempNode = *CopyIter;
          TempNode->deleteTree();
        }
        string nameattrib = CopyNode->getAttributeString("name");
        CopyNode->removeAttribute("name");
        CopyNode->addAttribute("ref",nameattrib);
      }
    } /* if (dogirders) */

   /* Advance to the next element.*/
    i++;
  }

 /* The list of elements is now complete.
  * Before we go on to build the machine, we should remove any duplicates from this list. 
  * First, remove any empty girders from sector definitions.*/
  NodeList SectEles = LabNode->getChildrenByName("sector");
  NodeListIter SectElesIter = SectEles.begin();
  for (SectElesIter; SectElesIter!=SectEles.end(); SectElesIter++){
    UAPNode* SectNode = *SectElesIter;
    UAPNode* GirderNode = SectNode->getChildByName("girder");
    if (GirderNode) {
      NodeList GNodeList = GirderNode->getChildren();
      if (GNodeList.size()==0) GirderNode->deleteTree();
    }
  }

  NodeList AllEles = LabNode->getChildren();
  double ListLength = AllEles.size();
  NodeListIter AllElesIter = AllEles.begin();
  list<string> dupeList;
  int count = 0, numdupes=0;
  for (AllElesIter; AllElesIter!=--AllEles.end(); AllElesIter++) {
    count++;
    NodeListIter Iter2 = AllEles.begin();
    for (int i=0; i<count; i++) Iter2++;
    for (Iter2; Iter2!=AllEles.end(); Iter2++) {
      UAPNode *Node1 = *AllElesIter;
      UAPNode *Node2 = *Iter2;
      if (equalNodes(Node1, Node2)) {
        dupeList.push_back(Node2->getAttributeString("name"));
        LabNode->removeChild(Node2);
        numdupes++;
        break;
      }
    }
  }
  if (verbose && numdupes>0) {
    dupeList.sort();
    string prev = "";
    for (list<string>::iterator iter = dupeList.begin(); iter!=dupeList.end(); iter++) {
      if (!(*iter==prev)) {
        prev=*iter;
        cout << "Element " << *iter << " is duplicated" << endl;
      }
    }
    cout << "\n" << numdupes << " duplicate entries found.  Retaining original entries, and";
    cout << " deleting the remainder.\nIt would help to have unique names for each entry";
    cout << " in the Lucretia lattice.\nIf you are confident that any Lucretia elements";
    cout << " sharing the same name are IDENTICAL\n(e.g. the lattice was originally parsed";
    cout << " from XSIF), you may safely ignore this message." << endl;
  }

 /* We now have AMLRepNode containing the element and controller definitions, and
  * SectorNode containing the sector and girder defs.  Add this to the end of AMLRepNode
  * before writing it out to a file.*/
  UAPNode *CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","This finishes the definition of the elements");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","The following defines the machine");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","++++++++++++++++++++++++++++++++++++");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","In the following, a sector represents a Lucretia girder, which has been defined above.");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","The orientation node of the sector, represents the orientation of the girder");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","as a whole, while the orientation of the girder node represents the state");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","of the mover present at that object.  Only the degrees of freedom available");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","to the mover are shown.");
  CommentNode = LabNode->addChild(ELEMENT_NODE, "comment");
  CommentNode->addAttribute("text","Elements shown without a sector are girderless.");

  LabNode->addChildCopy(TopMachineNode);

 /* Create an AMLReader object, and use it to convert the AML representation to a file.*/
  AMLReader Converter;
  if (Converter.AMLRepToAMLFile(AMLRepNode, fileout));
  else {
    AMLRepNode->deleteTree();
    mexErrMsgTxt("AML file creation failed.");
    return ;
  }

  if (xsif_conv) {
    string out_file("testfile.xsif");
    AMLReader reader;
    MADCore* converter;
    converter = new XSIFParser();
  
    converter->AMLRepToXFile (AMLRepNode, out_file);
  }

 /* Delete the AMLRepNode dynamic memory and return to the command line.*/
  AMLRepNode->deleteTree();
  SectorNode->deleteTree();

  return ;
}

