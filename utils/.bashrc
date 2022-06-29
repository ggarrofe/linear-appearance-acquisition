# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# Use NSS shared database for Thunderbird and Firefox
export NSS_DEFAULT_DB_TYPE="sql"

# set your default printer
export PRINTER="ICTMono"

# set CUDA path
export PATH=/vol/cuda/11.3.1-cudnn8.2.1/bin/${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/vol/cuda/11.3.1-cudnn8.2.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export QT_SELECTION=/usr/lib/x86_64-linux-gnu

# put your aliases here,
alias pd="pushd"
alias rm="rm -i"
alias colmap=/data/gg921/bin/colmap