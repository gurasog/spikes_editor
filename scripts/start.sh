#!/bin/bash
cd "$(dirname "$0")/.."

# Create a virtual environment if it does not exist
if [ ! -d ".venv" ]
then
	python3.9 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

# Launch scripts
python3.9 build_dash.py Rat_spikes_series_dataset.json Rat_spikes_series.mat                           10125 > /dev/null 2>&1 &
python3.9 build_dash.py Rat_spikes_series_concervative_dataset.json Rat_spikes_series_concervative.mat 10126 > /dev/null 2>&1 &
python3.9 build_dash.py Rat_spikes_28_long_alternative_dataset.json Rat_spikes_28_long_alternative.mat 10127 > /dev/null 2>&1 &
python3.9 build_dash.py Rat_spikes_28_dataset.json Rat_spikes_28.mat                                   10128 > /dev/null 2>&1 &
python3.9 build_dash.py Rat_spikes_28_long_dataset.json Rat_spikes_28_long.mat                         10129 > /dev/null 2>&1 &

disown -a  # Detach all jobs
deactivate  # Deactivate python environment
