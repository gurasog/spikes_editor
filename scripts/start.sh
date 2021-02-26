#!/bin/bash
cd "$(dirname "$0")/.."

# Create a virtual environment if it does not exist
if [ ! -d ".venv" ] then
	python3.9 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

# Launch scripts
python3.9 dash_for_editor_with_spike_data_on_classical_rats.py  > /dev/null 2>&1 &
python3.9 dash_for_editor_with_spike_data.py                    > /dev/null 2>&1 &

disown -a  # Detach all jobs
deactivate  # Deactivate python environment