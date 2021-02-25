#!/bin/sh
#Run from Lucretia/ : $ source package/package_linux64.sh
tar --bzip2 -X package/tarExcludeList_mexa64.txt -cpf package/Lucretia_linux64.tar-bz2 *

