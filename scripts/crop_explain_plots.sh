#!/bin/bash

shopt -s globstar

# files=`find ../logs -name "ShapleyValueSampling.pdf" -print`
explain_name=ShapleyValueSampling
paths_ar=../logs/**/autoregressive/${explain_name}_cropped.pdf
paths_pi=../logs/**/physics_informed/${explain_name}.pdf

N=6

for file in $paths_pi; do
	((i=i%N)); ((i++==0)) && wait
	filename=${file%${explain_name}.pdf}${explain_name}_cropped.pdf
	python crop_pdf.py $file $filename --preset explain
done
