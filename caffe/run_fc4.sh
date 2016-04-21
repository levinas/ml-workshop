#! /bin/bash

caffe train --solver fc4.solver.pt 2>&1 |tee fc4.log 
