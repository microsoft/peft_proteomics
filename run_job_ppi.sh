# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
RUN_NAME="$1"
CONFIG="$2"
DEVICES="${@:3}"
echo "$CONFIG", "$RUN_NAME", "$DEVICES"
python ppi/main.py --run_name "$RUN_NAME" --config "$CONFIG" --devices $DEVICES