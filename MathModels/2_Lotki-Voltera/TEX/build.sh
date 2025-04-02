#!/bin/zsh

pdftk rpz.pdf cat 2-end output cutted.pdf

pdftk titlePage2.pdf cutted.pdf cat output final.pdf

rm cutted.pdf
