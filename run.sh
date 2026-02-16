#!/bin/bash
# Helper script to run the tumor classification application

# Activate virtual environment if it exists
if [ -d "env" ]; then
    source env/bin/activate
fi

# Run the application with provided arguments
python -m src.main "$@"

