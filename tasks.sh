#!/bin/bash
source credentials.conf
SYNAPSE_EMAIL=${SYNAPSE_EMAIL} SYNAPSE_AUTH=${SYNAPSE_AUTH} invoke download unzip-all preprocess