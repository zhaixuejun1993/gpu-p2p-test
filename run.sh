#!/bin/bash
#./build/reproduce-event-hang/reproduce-event-hang 200 0
for ((i=1; i<=200; i++))
do
    echo "------ $i ------"
    ./build/reproduce-event-hang/reproduce-event-hang 200 0
done
