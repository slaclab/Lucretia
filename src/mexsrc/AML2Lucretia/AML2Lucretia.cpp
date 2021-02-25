#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "AML/AMLReader.hpp"
#include "AML/AMLLatticeExpander.hpp"
#include "mex.h"
#include "matrix.h"

#include "AML2Lucretia.hpp"

using namespace std;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  UAPNode* UAPModel;
  string filein;

  if (nrhs==1) {
    filein = mxArrayToString(prhs[0]);
  }
  else if (nrhs==0) {
    filein = "expand.aml";
  }

  // Read AML lattice file and create an AML representation subtree.
  cout << "Expanding file: " << filein << endl;
  AMLReader reader;
  UAPModel = reader.AMLFileToAMLRep (filein);
  if (!UAPModel) { cout << "Stopping here." << endl; return ; }

  // Expand the AML representation subtree.
  try {
    AMLLatticeExpander LE;
    LE.AMLExpandLattice(UAPModel);
  } catch (UAPException err) {
    cerr << err << endl;
    return ;
  }

  // Check the tree for problems
  UAPModel->checkTree(); 

 /* UAPModel now contains the expanded tree for the entire lattice.
  * This contains two child nodes -- one is a representation of the input AML
  * and the other is the fully expanded (instantiated?) lattice.  It is this
  * second node we want to examine.
  * Therefore we get a list of the children of the UAPModel node, and assign the
  * last child in this node to another UAPNode* -- "ExLatticeNode".
  * We then expand this to find its children. */

  NodeList nlist = UAPModel->getChildren();
  UAPNode* ExLatticeNode = UAPModel->getChildByName(nlist.back()->getName());
  NodeList exlist = ExLatticeNode->getChildren();
  NodeListIter exlistiter = exlist.begin();

 /* ExLatticeNode is the parent node of the expanded lattice, ie the entire machine
  * exlist is a list of children of this node
  *       -- control_list, global, and machine.
  * First assign control_list to its own UAPNode*, then remove it from ExLatticeNode,
  * and examine the new node for the girder and controller subnodes. */

  UAPNode* BigContNode = ExLatticeNode->getChildByName("control_list");
  ExLatticeNode->removeChild(BigContNode);
  NodeList GIRDERList = BigContNode->getChildrenByName("girder");
  NodeList PSList = BigContNode->getChildrenByName("controller");

  // Now assign the global node to its own UAPNode* and remove it from ExLatticeNode.
  UAPNode* GlobalNode = ExLatticeNode->getChildByName("global");
  ExLatticeNode->removeChild(GlobalNode);

  // The machine nodes should be the only ones remaining in ExLatticeNode, so assign them to
  // a NodeList.
  NodeList MachineList=ExLatticeNode->getChildren();
  cout << "Number of machines in this lattice file: " << MachineList.size() << endl;
  if (MachineList.size() > 1) cout << "Will concatenate them in the order they appear." << endl;
  cout << "\n";

 /* At this point we have two useful nodes:	
  *        BigContNode -- the root of the PS, GIRDER, KLYSTRON tree
  *        GlobalNode  -- the root of any global variables
  * We also have a list of pointers to each of the "machine" nodes in the container: MachineList
  * BigContNode has two children:
  *        GirderNode  -- A node containing all the GIRDER info.
  *        ControlNode -- A node containing all the PS and KLYSTRON info. */

 /* Now we'll loop around the machines in MachineList, and around each of the elements
  * of those machines, extracting the information as we go.
  * Set an iterator to point at the first machine in MachineList */
  NodeListIter MachineListIter = MachineList.begin();

 /* Count the number of elements in this machine.*/
  int ele_num=0;
  UAPNode* MachineNode;
  UAPNode* TLattNode;
  for (MachineListIter; MachineListIter != MachineList.end(); ++MachineListIter) {
    MachineNode = *MachineListIter;
    TLattNode = MachineNode->getChildByName("tracking_lattice");
    NodeList TLattList = TLattNode->getChildren();
    NodeListIter TLattIter = TLattList.begin();
    for (TLattIter; TLattIter!=TLattList.end(); ++TLattIter) ele_num++;
  }

 /* Make an empty BEAMLINE.*/
  mxArray* BEAMLINEmx = mxCreateCellMatrix(ele_num, 1);
 /* Send it to the global workspace.*/
  mexPutVariable("global","BEAMLINE",BEAMLINEmx);

  MachineListIter = MachineList.begin();
  int machinecounter=0;
  ele_num=0;
  UAPNode* MListNode;
  UAPNode* BeamNode;
  UAPNode* LattNode;
  UAPNode* NPartNode;
  UAPAttribute* NPartDesAttrib;
  UAPNode* ENode;
  UAPAttribute* EDesAttrib;	
  UAPNode* EleNode;
  mxArray* EleStruc;
  beamdef beamparams;
  for (MachineListIter; MachineListIter != MachineList.end(); ++MachineListIter) {
    // This loop is around the machines present in the lattice

    MachineNode = *MachineListIter;
    UAPAttribute* MachineName = MachineNode->getAttribute("name");
    cout << "Machine Name: " << MachineName->getValue() << endl;

    TLattNode = MachineNode->getChildByName("tracking_lattice");
    MListNode = MachineNode->getChildByName("master_list");
    BeamNode = MachineNode->getChildByName("beam");
    LattNode = MachineNode->getChildByName("lattice");

   /* MachineNode is (surprise surprise) the machine node.
    * TLattNode, MListNode, BeamNode, and LattNode, are children of MachineNode,
    * and represent the tracking_lattice, master_list, beam, and lattice nodes.*/

    /* Let's determine some of the beam parameters and assign them to global variables.
     * We only need to do this on the first iteration of this loop. */
    if (machinecounter == 0) {
      bool ok;
      NPartNode = BeamNode->getChildByName("n_particles");
      NPartDesAttrib = NPartNode->getAttribute("design");
      string NPartDesstr = NPartDesAttrib->getValue();
      beamparams.BunchPopulation = BasicUtilities::string_to_double(NPartDesstr,ok);

      ENode = BeamNode->getChildByName("total_energy");
      EDesAttrib = ENode->getAttribute("design");
      string EDesstr = EDesAttrib->getValue();
      beamparams.DesignBeamP = BasicUtilities::string_to_double(EDesstr,ok) / 1e9;
    }

   /* Next we loop around the children of tracking_lattice.*/

    NodeList TLattList = TLattNode->getChildren();
    NodeListIter TLattIter = TLattList.begin();
    for (TLattIter; TLattIter!=TLattList.end(); ++TLattIter) {
      // This loop is around the children of tracking_lattice.

     /* Each child of tracking_lattice is a lattice element.
      * Assign this to EleNode, and pass to UAPNode2Lucretia. */
      EleNode = *TLattIter;
      EleStruc = UAPNode2Lucretia (EleNode, beamparams);

     /* UAPNode2Lucretia returned a pointer to a cell array.
      * Now add this to BEAMLINE.*/
      int rowLen = mxGetN(BEAMLINEmx);
      int colLen = mxGetM(BEAMLINEmx);
      mxArray* clptr = mxGetCell(BEAMLINEmx, ele_num);
      mxFree(clptr);
      mxSetCell(BEAMLINEmx, ele_num, EleStruc);
      mexPutVariable("global","BEAMLINE",BEAMLINEmx);

      ele_num++;
    } // TLattIter loop

    cout << "\n";

    machinecounter++;
  } // MachineListIter loop

  BEAMLINEmx = mexGetVariable("global","BEAMLINE");

 /* Count the number of GIRDERs and PSs. */
  NodeListIter PSListIter = PSList.begin();
  NodeListIter GIRDERListIter = GIRDERList.begin();

  int PScounter = 0;
  for (PSListIter; PSListIter!=PSList.end(); ++PSListIter) {
    PScounter++;
    UAPNode* PSNode = *PSListIter;
    UAPNode* PSSlaveNode = PSNode->getChildByName("slave");

    UAPAttribute* PSTargetAttrib = PSSlaveNode->getAttribute("target");
    string TargetName = PSTargetAttrib->getValue();

    mxArray* inds;
    mxArray* inargs[3];
    inargs[0] = BEAMLINEmx;
    inargs[1] = mxCreateString("Name");
    inargs[2] = mxCreateString(TargetName.c_str());
    mexCallMATLAB(1,&inds,3,inargs,"findcells");

    mxArray* inargs1[2];
    inargs1[0] = inds;
    inargs1[1] = mxCreateDoubleScalar(PScounter);
    mexCallMATLAB(0,NULL,2,inargs1,"AssignToPS");
  }


  int GIRDERCounter = 0;
  for (GIRDERListIter; GIRDERListIter!=GIRDERList.end(); GIRDERListIter++) {
    GIRDERCounter++;
    UAPNode* GIRDERNode = *GIRDERListIter;

    NodeList SlaveList = GIRDERNode->getList(SLAVE);
    NodeListIter SlaveListIter = SlaveList.begin();

    for (SlaveListIter; SlaveListIter!=SlaveList.end(); ++SlaveListIter) {
      UAPNode* GirderSlaveNode = *SlaveListIter;
      UAPAttribute* GirderSlaveAttrib = GirderSlaveNode->getAttribute("name");
      string TargetName = GirderSlaveAttrib->getValue();

      mxArray* inds;
      mxArray* inargs[3];
      inargs[0] = BEAMLINEmx;
      inargs[1] = mxCreateString("Name");
      inargs[2] = mxCreateString(TargetName.c_str());
      mexCallMATLAB(1,&inds,3,inargs,"findcells");

      mxArray* inargs1[3];
      inargs1[0] = inds;
      inargs1[1] = mxCreateDoubleScalar(GIRDERCounter);
      inargs1[2] = mxCreateDoubleScalar(0);
      mexCallMATLAB(0,NULL,3,inargs1,"AssignToGirder");
    }
  }

  mxArray* rhs[3];
  rhs[0] = mxCreateDoubleScalar(1);
  rhs[1] = mxCreateDoubleScalar(ele_num);
  rhs[2] = mxCreateDoubleScalar(0);
  if (mexCallMATLAB(0,NULL,3,rhs,"SetSPositions")) cout << "Didn't work :(" << endl;
  mxDestroyArray(*rhs);

  mxArray* rhs1[5];
  rhs1[0] = mxCreateDoubleScalar(1);
  rhs1[1] = mxCreateDoubleScalar(ele_num);
  rhs1[2] = mxCreateDoubleScalar(beamparams.BunchPopulation);
  rhs1[3] = mxCreateDoubleScalar(beamparams.DesignBeamP);
  rhs1[4] = mxCreateDoubleScalar(0);
  if (mexCallMATLAB(0,NULL,5,rhs1,"UpdateMomentumProfile")) cout << "Didn't work :(" << endl;
  mxDestroyArray(*rhs1);

  cout << "Num Elements: " << ele_num << endl;

  return ;
}

