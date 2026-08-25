# Stage 0 - Create from Python3.12 image
FROM python:3.12-slim as stage0

# Stage 1 - Debian dependencies
FROM stage0 as stage1
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y curl zip python3-dev build-essential libhdf5-serial-dev netcdf-bin libnetcdf-dev

# Stage 2 - Create virtual environment and install dependencies
FROM stage1 as stage2
COPY requirements.txt /app/requirements.txt
RUN /usr/local/bin/python3 -m venv /app/env
RUN /app/env/bin/pip install -r /app/requirements.txt

# Stage 3 - Copy MetroMan code
FROM stage2 as stage3
COPY ./moi /app/moi/
COPY ./sos_read /app/sos_read/
COPY ./CalValSeparation_basin_stratified_v2.csv /app/CalValSeparation_basin_stratified_v2.csv

# Stage 4 - Execute algorithm
FROM stage3 as stage4
COPY run_MOI.py /app/run_MOI.py
LABEL version="1.0" \
	description="Containerized MOI algorithm." \
	"confluence.contact"="ntebaldi@umass.edu" \
	"algorithm.contact"="durand.8@osu.edu"
ENTRYPOINT ["/app/env/bin/python3", "/app/run_MOI.py"]
