#!/bin/bash

# cd /Users/bernier2/Documents/GitHub/HEDM_geometry

sketchbin=$1
echo "Using sketch binary: $sketchbin";

inputname=$2

$sketchbin -o ${PWD}/$inputname.sk.out ${PWD}/$inputname.sk
latex ${PWD}/$inputname.tex
dvips -o ${PWD}/$inputname.ps ${PWD}/$inputname.dvi
dvipdf ${PWD}/$inputname.dvi ${PWD}/$inputname.pdf
