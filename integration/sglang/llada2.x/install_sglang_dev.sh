#!/bin/bash

pip3 install -e .[srt_npu]
python3 -c "import sglang, os; print(sglang.__file__)"