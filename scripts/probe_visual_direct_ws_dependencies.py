from __future__ import annotations

from pathlib import Path

import jax
import msgpack
import numpy as np
import websockets

from openpi_client import websocket_client_policy
import train_expert


KIN_ROOT = Path(
    "/cluster/nvme6/pijiaqi/real-time-chunking-kinetix"
).resolve()
OPENPI_CLIENT_ROOT = Path(
    "/cluster/nvme6/pijiaqi/openpi_kinetix/packages/openpi-client/src"
).resolve()

train_expert_path = Path(train_expert.__file__).resolve()
client_path = Path(websocket_client_policy.__file__).resolve()

print(f"TRAIN_EXPERT_FILE={train_expert_path}", flush=True)
print(f"OPENPI_CLIENT_FILE={client_path}", flush=True)
print(f"JAX_DEVICES={jax.devices()}", flush=True)
print(f"NUMPY_VERSION={np.__version__}", flush=True)
print(f"MSGPACK_VERSION={msgpack.__version__}", flush=True)
print(f"WEBSOCKETS_VERSION={websockets.__version__}", flush=True)

if KIN_ROOT not in train_expert_path.parents:
    raise RuntimeError(
        f"train_expert imported from wrong location: {train_expert_path}"
    )
if OPENPI_CLIENT_ROOT not in client_path.parents:
    raise RuntimeError(
        f"openpi_client imported from wrong location: {client_path}"
    )
if any(device.platform != "cpu" for device in jax.devices()):
    raise RuntimeError(f"Kinetix client is not CPU-only: {jax.devices()}")

print("VISUAL_DIRECT_WS_DEPENDENCIES_OK", flush=True)
