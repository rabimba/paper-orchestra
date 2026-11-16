# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# By default, all jobs offload compute to puffingbird (2x H100 NVL GPUs, 1TB RAM on 8TB SSD)
# Pass --local to execute locally instead of offloading to puffingbird.
if [[ "${1:-}" == "--local" ]]; then
    shift
    echo "Running locally..."
    PYTHONUNBUFFERED=1 exec "${SCRIPT_DIR}/.venv/bin/python" paper_writing_cli.py "$@"
else
    echo "Offloading job to puffingbird..."
    exec "${SCRIPT_DIR}/scripts/offload_to_puffingbird.sh" "$@"
fi



