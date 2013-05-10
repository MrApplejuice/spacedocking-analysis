#!/bin/bash

version=$1
echo "GAME_VERSION_STRING=\"\\\"$version\\\"\""

version_index=1
while [ ! -z "$version" ] ; do
  echo GAME_VERSION_VALUE_$version_index=$(sed -E 's/^([0-9]+)(\..+$|$)/\1/' <<< $version)
  version_index=$[ $version_index + 1 ]
  version=$(sed -E 's/^[0-9]+(\.(.*)$|)$/\2/' <<< $version)
done
