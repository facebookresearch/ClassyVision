#!/bin/bash                                                                                                                                                                                                      
usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build and push updated ClassyVision site. Will either update latest or bump stable version."
  echo ""
  exit 1
}

# Command to strip out Algolia (search functionality) form siteConfig.js                                                                                                                                         
# Algolia only indexes stable build, so we'll remove from older versions                                                                                                                                         
REMOVE_ALGOLIA_CMD="import os, re; "
REMOVE_ALGOLIA_CMD+="c = open('siteConfig.js', 'r').read(); "
REMOVE_ALGOLIA_CMD+="out = re.sub('algolia: \{.+\},', '', c, flags=re.DOTALL); "
REMOVE_ALGOLIA_CMD+="f = open('siteConfig.js', 'w'); "
REMOVE_ALGOLIA_CMD+="f.write(out); "
REMOVE_ALGOLIA_CMD+="f.close(); "

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

# disable search for non-stable version (can't use sed b/c of newline)                                                                                                                                         
python3 -c "$REMOVE_ALGOLIA_CMD"

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
