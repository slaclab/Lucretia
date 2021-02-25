#!/bin/bash

printenv | grep G4 | grep DATA | sed 's/G4\w*=//' | xargs -I@ cp -r @ ~/Lucretia/src/ExtProcess/G4Data/