#!/bin/sh
#curl model data
#unzip model data
curl http://nlp.stanford.edu/data/glove.twitter.27B.zip /data/glove.twitter.27B.zip --create-dirs #curl embedding data
unzip /data/glove.twitter.27B.zip