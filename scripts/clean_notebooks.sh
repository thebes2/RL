#!/bin/bash

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$1"
git add "$1"
