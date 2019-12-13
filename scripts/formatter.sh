#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: isort should be run in a Python enironment where all the packages
# required by the package are installed and the project isn't in the
# directory where isort is being called from. We fix the latter in the following
# lines

# cd to the scripts directory
# this is needed for isort to work as expected
cd "$(dirname "$0")/" || exit 1

GIT_URL_1="https://github.com/facebookresearch/ClassyVision.git"
GIT_URL_2="git@github.com:facebookresearch/ClassyVision.git"

UPSTREAM_URL="$(git config remote.upstream.url)"

if [ -z "$UPSTREAM_URL" ]
then
    echo "Setting upstream remote to $GIT_URL_1"
    git remote add upstream "$GIT_URL_1"
elif [ "$UPSTREAM_URL" != "$GIT_URL_1" ] && \
     [ "$UPSTREAM_URL" != "$GIT_URL_2" ]
then
    echo "upstream remote set to $UPSTREAM_URL."
    echo "Please delete the upstream remote or set it to $GIT_URL_1 to use this script."
    exit 1
fi

# fetch upstream
git fetch upstream

CHANGED_FILES="$(git diff --name-only upstream/master | grep '\.py$')"
# add ../ to all the files and remove newlines
CHANGED_FILES="$(echo "$CHANGED_FILES" | sed 's/^/..\//' | tr '\n' ' ')"

if [ "$CHANGED_FILES" != "" ]
then
    if [ ! "$(black --version)" ]
    then
        echo "Please install black."
        exit 1
    fi
    if [ ! "$(isort --version)" ]
    then
        echo "Please install isort."
        exit 1
    fi

    # run isort
    cmd="isort $CHANGED_FILES"
    echo "Running command \"$cmd\""
    ($cmd)

    # run black
    cmd="black $CHANGED_FILES"
    echo "Running command \"$cmd\""
    ($cmd)
else
    echo "No changes made to any Python files. Nothing to do."
    exit 0
fi

CHANGED_FILES="$(git diff --name-only | grep '\.py$' | tr '\n' ' ')"

if [ "$CHANGED_FILES" != "" ]
then
    # need this so that CircleCI fails
    exit 1
fi
