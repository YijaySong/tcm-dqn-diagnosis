#!/bin/bash

#relation=$base@schemastaging@organization_extra@phone_number.@base@schemastaging@phone_sandbox@service_location
relation="concept_agentbelongstoorganization"
python sl_policy.py $relation
python policy_agent.py $relation retrain
python policy_agent.py $relation test

