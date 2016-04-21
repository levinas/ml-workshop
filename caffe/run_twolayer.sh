#! /bin/bash

caffe train --solver twolayer.solver.pt 2>&1 |tee twolayer.log 
