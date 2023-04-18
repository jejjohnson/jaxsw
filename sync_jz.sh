#!/bin/bash

# Configure these variables to your reality
REMOTE_HOST=jean_zay
REMOTE_FOLDER=/gpfswork/rech/cli/uvo53rl/projects_rsync/jaxsw
LOCAL_FOLDER=/Users/eman/code_projects/jaxsw/

# initial sync step
rsync -a --progress . "$REMOTE_HOST:$REMOTE_FOLDER"
# rsync -av -P jean_zay:/gpfswork/rech/cli/uvo53rl/projects_rsync .

# initial check may take some time, later changes may be instant
fswatch \
--one-per-batch \ 
--recursive \
--latency 0 \
--verbose \
--access \
--monitor fsevents_monitor \
"$LOCAL_FOLDER" | xargs -I{} rsync -a --progress "$LOCAL_FOLDER" "$REMOTE_HOST:$REMOTE_FOLDER"
