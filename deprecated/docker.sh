#!/bin/sh
docker build pc-img .
docker run --name pc-container pc-image