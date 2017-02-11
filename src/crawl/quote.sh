#!/bin/sh
sed -E s/^[[:space:]]*\(http[^,[:space:]]*\)[[:space:]]*\([,]?\)[[:space:]]*$/\"\\1\"\\2/g $1
