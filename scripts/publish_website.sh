#!/bin/bash                                                                                                                                                                                                      
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build and push updated ClassyVision site."
  echo ""
  exit 1
}

# Current directory (needed for cleanup later)                                                                                                                                                                   
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory                                                                                                                                                                                       
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit

# Clone both master & gh-pages branches                                                                                                                                                                          
git clone git@github.com:facebookresearch/ClassyVision.git ClassyVision-master
git clone --branch gh-pages git@github.com:facebookresearch/ClassyVision.git ClassyVision-gh-pages

cd ClassyVision-master/website || exit

# Build site, tagged with "latest" version; baseUrl set to /versions/latest/                                                                                                                                   
yarn
yarn run build

cd .. || exit
./scripts/build_docs.sh -b

cd "${WORK_DIR}" || exit
rm -rf ClassyVision-gh-pages/*
touch ClassyVision-gh-pages/CNAME
echo "classyvision.ai" > ClassyVision-gh-pages/CNAME
mv ClassyVision-master/website/build/ClassyVision/* ClassyVision-gh-pages/

cd ClassyVision-gh-pages || exit
git add .
git commit -m 'Update latest version of site'
git push

# Clean up                                                                                                                                                                                                       
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
