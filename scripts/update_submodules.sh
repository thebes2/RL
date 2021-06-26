#!/bin/bash

git submodule foreach git pull origin ${1:-master}
