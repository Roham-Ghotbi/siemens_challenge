#!/bin/bash

pyro4-ns &
python shared_info.py &
python server.py &
