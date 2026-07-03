#!/bin/bash
# ============================================================================
#  render_in_container.sh  —  entrypoint run INSIDE the Apptainer container
# ============================================================================
#  Activates the existing $HPCVAULT conda env (isaaclab + skrl + tensegrity_pick)
#  without needing the `conda` command (the cluster conda / module system is not
#  present inside the container), then runs render_play.py. All arguments are
#  forwarded verbatim to render_play.py.
#
#  Expects these env vars to be passed in from the host job (apptainer --env):
#    ISAAC_ENV_PREFIX  absolute path of the conda env prefix on $HPCVAULT
#    PROJECT_PATH      absolute path of the repo (…/studentische-arbeiten)
# ============================================================================
set -uo pipefail

: "${ISAAC_ENV_PREFIX:?ISAAC_ENV_PREFIX not set}"
: "${PROJECT_PATH:?PROJECT_PATH not set}"

# Isaac Sim / Kit want a writable XDG_RUNTIME_DIR; the login/compute env doesn't
# set one. Point it at node-local scratch.
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/xdg-${SLURM_JOB_ID:-$$}"
mkdir -p "$XDG_RUNTIME_DIR"

# Compute nodes have no direct internet — route Isaac Sim's startup network
# calls through the NHR@FAU proxy so they don't hang (mirrors train_alex.sh).
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export no_proxy=localhost,127.0.0.1,.nhr.fau.de,.fau.de
export NO_PROXY=localhost,127.0.0.1,.nhr.fau.de,.fau.de

# --- activate the conda env manually (no `conda` binary needed) -------------
export CONDA_PREFIX="${ISAAC_ENV_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# The activate.d scripts reference unbound vars and return non-zero; relax -u.
set +u
# setenv.sh: sets ISAACLAB_PATH and sources _isaac_sim/setup_conda_env.sh
# (which puts Isaac Sim's runtime libs on LD_LIBRARY_PATH).
source "${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh"
# torch_gomp.sh: prepend torch's libgomp to LD_PRELOAD.
source "${CONDA_PREFIX}/etc/conda/activate.d/torch_gomp.sh"
set -u

cd "${PROJECT_PATH}/src/tensegrity_pick"

echo "[container] python : $(command -v python)"
echo "[container] glibc  : $(ldd --version | head -1)"
echo "[container] render : render_play.py $*"

exec python scripts/rendering/render_play.py "$@"
