#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

CMD="black"
CHANGED_FILES="$(git diff --name-only master... | grep '\.py$' | tr '\n' ' ')"

while getopts bs opt; do
  case $opt in
    s)
      CMD="isort"
      ;;

    b)
      CMD="black"
      ;;

    *)
      CMD="black"
  esac

  done

if [ "$CHANGED_FILES" != "" ]
then
    if [ "$CMD" = "black" ]
    then
        command -v black >/dev/null || \
            ( echo "Please install black." && false )
        # only output if something needs to change
        black --check "$CHANGED_FILES"
    else
        isort -v | grep 'VERSION' >/dev/null || \
            ( echo "Please install isort." && false )

        # output number of files with incorrectly sorted imports
        isort "$CHANGED_FILES" -c | grep -c 'ERROR'
    fi
fi
