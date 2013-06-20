#/!bin/bash

START_JOBS=$(jobs | wc -l)

for f in $@; do
  jobs > /dev/null
  CURRENT_JOBS=$(jobs | wc -l)
  while [ $[ $CURRENT_JOBS - $START_JOBS ] -ge 8 ] ; do
    echo waiting
    sleep 1
    jobs > /dev/null
    CURRENT_JOBS=$(jobs | wc -l)
  done

  yes | avconv -i "$f" -c:v mpeg4 -qscale 8 -an "${f%.*}.avi"
done


