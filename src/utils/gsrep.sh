#!/bin/sh
find *.m | xargs grep -l $1 | xargs sed -i '' -e s/$1/$2/g
