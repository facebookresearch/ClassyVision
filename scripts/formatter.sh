#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# cd to the project directory
cd "$(dirname "$0")/.." || exit 1

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

CHANGED_FILES="$(git diff --name-only upstream/master | grep '\.py$' | tr '\n' ' ')"

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
    cmd="isort -o classy_vision -o torchelastic -o visdom -o tensorboardX \
    -o progressbar $CHANGED_FILES"
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
    echo "Running git diff:"
    git --no-pager diff
    # need this so that CircleCI fails
    exit 1
fi
