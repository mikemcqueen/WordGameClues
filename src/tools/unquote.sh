#!/bin/sh
sed -E s/^[[:space:]]*[\"]\(http.*\)[\"][,]?[[:space:]]*$/\\1/g $1

