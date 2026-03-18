
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

EVENT_CONSUMER_LOGGER_PATH=BASE_DIR / 'storages' / 'logs' / 'event_consumer.log'
EVENT_CONSUMER_LOGGER_LEVEL="INFO"

TRACKER_MANAGER_LOGGER_PATH=BASE_DIR / 'storages' / 'logs' / 'tracker_manager.log'
TRACKER_MANAGER_LOGGER_LEVEL="INFO"

WS_WORKER_LOGGER_PATH=BASE_DIR / 'storages' / 'logs' / 'ws_manager.log'
WS_WORKER_LOGGER_LEVEL="INFO"

# CAM_F="AXIS-FE-01"
CAM_F="f"
CAM_B="B"
CAM_D='d'