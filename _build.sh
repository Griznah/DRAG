#!/bin/bash
cd /home/bombom/repos/drag
export PATH="$HOME/.local/bin:$PATH"
exec podman compose build drag-app
