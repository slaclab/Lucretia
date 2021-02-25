  mxArray *PSglobalmx = mexGetVariable("global","PS");
  if (PSglobalmx) {
    mxArray *PSmx = mxGetField(Elemx, 0, "PS");
    double PSnum = mxGetScalar(PSmx);
    mxArray *PSamplmx = mxGetField(PSglobalmx, (int)PSnum-1, "Ampl");
    mxArray *PSdamplmx = mxGetField(PSglobalmx, (int)PSnum-1, "dAmpl");
    mxArray *PSStep = mxGetField(PSglobalmx, (int)PSnum-1, "Step");

    UAPNode *PSNode = AMLRepNode->addChild(ELEMENT_NODE, "controller");
    char PSName[10];
    strcpy(PSName, "PS");
    strcat(PSName, BasicUtilities::double_to_string(PSnum, ok).c_str());
    PSNode->addAttribute("name", PSName, false);
    PSNode->addAttribute("variation", "ABSOLUTE", false);
    PSNode->addAttribute("default_attribute", "bend:g", false);
    PSNode->addAttribute("design", BasicUtilities::double_to_string(mxGetScalar(PSamplmx), ok), false);
    PSNode->addAttribute("err", BasicUtilities::double_to_string(mxGetScalar(PSdamplmx), ok), false);
    PSNode->addAttribute("step", BasicUtilities::double_to_string(mxGetScalar(PSStep), ok), false);

    UAPNode *SlaveNode = PSNode->addChild(ELEMENT_NODE, "slave");
    SlaveNode->addAttribute("target", Namestr);
    char express[100];
    strcpy(express, BasicUtilities::double_to_string(Bdoub/Ldoub, ok).c_str());
    strcat(express, " * ");
    strcat(express, PSName);
    strcat(express, "[@actual]");
    SlaveNode->addAttribute("expression", express);
  }

