#!/usr/bin/env python3
from __future__ import annotations

import atexit
import itertools
import json
import logging
import os
import select
import signal
import subprocess
import sys
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum, unique
from threading import Thread
from typing import IO, Iterator, NamedTuple, Type

try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False

from logger_extras import RelativeTimeFilter, TieredFormatter

try:
    from logger_extras import MQTTHandler  # type: ignore[attr-defined]
except ImportError:
    MQTTHandler = None

REL_TIME_FILTER: RelativeTimeFilter | None = None


logging.raiseExceptions = False  # Don't print stack traces when the USB is removed
LOGGER = logging.getLogger('runusb')

PROC_FILE = '/proc/mounts'
ROBOT_FILE = 'robot.py'
METADATA_FILENAME = 'metadata.json'
LOG_NAME = 'log.txt'
USERCODE_LEVEL = 35  # Between INFO and WARNING
logging.addLevelName(USERCODE_LEVEL, "USERCODE")
USERCODE_LOGGER = logging.getLogger('usercode')

# the directory under which all USBs will be mounted
MOUNTPOINT_DIR = os.environ.get('RUNUSB_MOUNTPOINT_DIR', '/media')
# This will be populated if we have the config file
# url format: mqtt[s]://[<username>[:<password>]@]<host>[:<port>]/<topic_root>
MQTT_URL = None
MQTT_TOPIC_ROOT = ''
MQTT_CLIENT = None
MQTT_CONFIG_FILE = '/etc/sbot/mqtt.conf'


class MQTTVariables(NamedTuple):
    host: str
    port: int | None
    topic_prefix: str
    use_tls: bool
    username: str | None
    password: str | None


class Mountpoint(NamedTuple):
    mountpoint: str
    filesystem: str


VERBOTEN_FILESYSTEMS = (
    'autofs',
    'bpf',
    'cgroup',
    'cgroup2',
    'configfs',
    'debugfs',
    'devpts',
    'devtmpfs',
    'fusectl',
    'hugetlbfs',
    'mqueue',
    'proc',
    'pstore',
    'rpc_pipefs',
    'securityfs',
    'sysfs',
    'tracefs',
)


class LedStatus(Enum):
    NoUSB = (False, False, False)  # Off
    Running = (False, False, True)  # Blue
    Killed = (True, False, True)  # Magenta
    Finished = (False, True, False)  # Green
    Crashed = (True, False, False)  # Red


class LEDController():
    @unique
    class LEDs(IntEnum):
        BOOT_100 = 13
        CODE = 11
        COMP = 16
        WIFI = 8
        STATUS_RED = 26
        STATUS_GREEN = 20
        STATUS_BLUE = 21

    def __init__(self) -> None:
        if IS_PI:
            LOGGER.debug("Configuring LED controller")
            self._register_exit()
            atexit.register(GPIO.cleanup)  # type: ignore[attr-defined]
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([led.value for led in self.LEDs], GPIO.OUT, initial=GPIO.LOW)

    def _register_exit(self) -> None:
        """
        Ensure `atexit` triggers on `SIGTERM`.

        > The functions registered via [`atexit`] are not called when the program is
        killed by a signal not handled by Python
        """

        if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
            # If a signal handler is already present for SIGTERM,
            # this is sufficient for `atexit` to trigger, so do nothing.
            return

        def handle_signal(handled_signum: int, frame) -> None:
            """Semi-default signal handler for SIGTERM, enough for atexit."""
            USERCODE_LOGGER.error(signal.strsignal(handled_signum))
            exit(128 + handled_signum)  # 143 for SIGTERM

        # Add the null-ish signal handler
        signal.signal(signal.SIGTERM, handle_signal)

    def mark_start(self) -> None:
        if IS_PI:
            GPIO.output(self.LEDs.BOOT_100, GPIO.HIGH)

    def set_comp(self, value: bool) -> None:
        if IS_PI:
            GPIO.output(self.LEDs.COMP, GPIO.HIGH if value else GPIO.LOW)

    def set_code(self, value: bool) -> None:
        if IS_PI:
            GPIO.output(self.LEDs.CODE, GPIO.HIGH if value else GPIO.LOW)

    def set_wifi(self, value: bool) -> None:
        if IS_PI:
            GPIO.output(self.LEDs.WIFI, GPIO.HIGH if value else GPIO.LOW)

    def set_status(self, value: LedStatus) -> None:
        if IS_PI:
            GPIO.output(self.LEDs.STATUS_RED, GPIO.HIGH if value.value[0] else GPIO.LOW)
            GPIO.output(self.LEDs.STATUS_GREEN, GPIO.HIGH if value.value[1] else GPIO.LOW)
            GPIO.output(self.LEDs.STATUS_BLUE, GPIO.HIGH if value.value[2] else GPIO.LOW)

        # Also send the status over MQTT
        if MQTT_CLIENT is not None:
            MQTT_CLIENT.publish(
                f'{MQTT_TOPIC_ROOT}/state',
                json.dumps({"state": value.name}),
                qos=1,
                retain=True,
            )


LED_CONTROLLER = LEDController()


@unique
class USBType(Enum):
    ROBOT = 'ROBOT'
    METADATA = 'METADATA'
    INVALID = 'INVALID'  # We dont care about this drive


def detect_usb_type(mountpoint: str) -> USBType:
    # only mountpoints under MOUNTPOINT_DIR are considered
    if not mountpoint.startswith(MOUNTPOINT_DIR):
        LOGGER.debug(
            f"Disregarding filesystem {mountpoint!r}. Is not under {MOUNTPOINT_DIR}.",
        )
        return USBType.INVALID

    # Check for the existence of the robot code
    if os.path.exists(os.path.join(mountpoint, ROBOT_FILE)):
        return USBType.ROBOT

    # Check for existence of a metadata file
    if os.path.exists(os.path.join(mountpoint, METADATA_FILENAME)):
        return USBType.METADATA

    LOGGER.info(
        f"Disregarding filesystem {mountpoint!r}. Is not forbidden but lacks either "
        f"{ROBOT_FILE} or {METADATA_FILENAME} file.",
    )

    return USBType.INVALID


class FSTabReader(object):
    def __init__(self) -> None:
        self.handle = open(PROC_FILE)

    def close(self) -> None:
        self.handle.close()

    def read(self) -> Iterator[Mountpoint]:
        self.handle.seek(0)

        for line in self.handle:
            (_, mountpoint_raw, filesystem, _, _, _) = line.split(' ')

            yield Mountpoint(
                mountpoint=self.mountpoint_decode(mountpoint_raw),
                filesystem=filesystem,
            )

    def watch(self, timeout=None) -> bool:
        _, _, changed = select.select([], [], [self.handle], timeout)

        if changed:
            LOGGER.debug("Detected change in procfile")

        return bool(changed)

    @staticmethod
    def mountpoint_decode(mountpoint_raw: str) -> str:
        """Decode whitespace characters in mountpoint path."""
        return mountpoint_raw.replace('\\040', ' ').replace('\\011', '\t')


class USBHandler(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, mountpoint_path: str) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class RobotUSBHandler(USBHandler):
    def __init__(self, mountpoint_path: str) -> None:
        self._setup_logging(mountpoint_path)
        LED_CONTROLLER.set_code(True)
        LED_CONTROLLER.set_status(LedStatus.Running)

        env = dict(os.environ)
        env["SBOT_METADATA_PATH"] = MOUNTPOINT_DIR
        if MQTT_URL is not None:
            # pass the mqtt url to the robot for camera images
            env["SBOT_MQTT_URL"] = MQTT_URL
        self.process = subprocess.Popen(
            [sys.executable, '-u', ROBOT_FILE],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            bufsize=1,  # line buffered
            cwd=mountpoint_path,
            env=env,
            text=True,
            start_new_session=True,  # Put the process in a new process group
        )

        self.thread = Thread(target=self._watch_process)
        self.thread.start()
        self.log_thread = Thread(
            target=self._log_output, args=(self.process.stdout,))
        self.log_thread.start()

    def cleanup(self) -> None:
        self._send_signal(signal.SIGTERM)
        try:
            # Wait for the process to exit
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # The process did not exit after 5 seconds, so kill it.
            self._send_signal(signal.SIGKILL)

    def close(self) -> None:
        self.cleanup()
        LED_CONTROLLER.set_status(LedStatus.NoUSB)
        LED_CONTROLLER.set_code(False)
        USERCODE_LOGGER.removeHandler(self.handler)

    def _send_signal(self, sig: int) -> None:
        if self.process.poll() is not None:
            # Process has already exited, so the kill() call would fail.
            return
        os.killpg(self.process.pid, sig)

    def _watch_process(self) -> None:
        # Wait for the process to complete
        self.process.wait()
        if self.process.returncode != 0:
            USERCODE_LOGGER.warning(f"Process exited with code {self.process.returncode}")
            LED_CONTROLLER.set_status(LedStatus.Crashed)
        else:
            USERCODE_LOGGER.info("Your code finished successfully.")
            LED_CONTROLLER.set_status(LedStatus.Finished)

        # Start clean-up
        self.cleanup()

    def _setup_logging(self, log_dir: str) -> None:
        self._rotate_old_logs(log_dir)
        self.handler = logging.FileHandler(
            os.path.join(log_dir, LOG_NAME),
            mode='w',  # Overwrite the log file
        )
        REL_TIME_FILTER.reset_time_reference()  # type: ignore[union-attr]
        self.handler.setFormatter(TieredFormatter(
            fmt='[%(reltime)08.3f - %(levelname)s] %(message)s',
            level_fmts={
                USERCODE_LEVEL: '[%(reltime)08.3f] %(message)s',
            },
        ))
        USERCODE_LOGGER.addHandler(self.handler)
        LOGGER.info('Starting user code')

    def _log_output(self, pipe: IO[str]) -> None:
        """
        Log the output of the process to the logger.

        This is done in a separate thread to avoid blocking the main thread.
        """
        for line in iter(pipe.readline, ''):
            USERCODE_LOGGER.log(USERCODE_LEVEL, line.rstrip('\n'))
        LOGGER.info('Process output finished')

    def _rotate_old_logs(self, log_dir: str) -> None:
        """
        Add a suffix to the old log file, if it exists.

        Suffixes are of the form log-<n>.txt, where <n> is the smallest
        integer that didn't already exist.
        """
        if not os.path.exists(os.path.join(log_dir, LOG_NAME)):
            return
        for i in itertools.count(1):
            new_name = os.path.join(log_dir, f'log-{i}.txt')
            if not os.path.exists(new_name):
                break

        os.rename(os.path.join(log_dir, LOG_NAME), new_name)


class MetadataUSBHandler(USBHandler):
    def __init__(self, mountpoint_path: str) -> None:
        # NOTE the comp LED just represents the presence of a comp mode USB
        # not whether comp mode is enabled
        LED_CONTROLLER.set_comp(True)

    def close(self) -> None:
        LED_CONTROLLER.set_comp(False)


class AutorunProcessRegistry(object):
    TYPE_HANDLERS: dict[USBType, Type[USBHandler]] = {
        USBType.ROBOT: RobotUSBHandler,
        USBType.METADATA: MetadataUSBHandler,
    }

    def __init__(self) -> None:
        self.mountpoint_handlers: dict[str, USBHandler] = {}

    def update_filesystems(self, mountpoints: Iterator[Mountpoint]) -> None:
        actual_mountpoints = {
            x.mountpoint
            for x in mountpoints
            if self._is_viable_mountpoint(x)
        }

        expected_mountpoints = set(self.mountpoint_handlers.keys())

        # Handle newly detected filesystems
        for new_mountpoint in (actual_mountpoints - expected_mountpoints):
            self._detect_new_mountpoint_path(new_mountpoint)

        # Handle now-dead filesystems
        for old_mountpoint in (expected_mountpoints - actual_mountpoints):
            self._detect_dead_mountpoint_path(old_mountpoint)

    def _detect_new_mountpoint_path(self, path: str) -> None:
        usb_type = detect_usb_type(path)
        LOGGER.info(f"Found new mountpoint: {path} ({usb_type})")
        handler_class = self.TYPE_HANDLERS[usb_type]
        handler = handler_class(path)
        LOGGER.info("  -> launched handler")
        self.mountpoint_handlers[path] = handler

    def _detect_dead_mountpoint_path(self, path: str) -> None:
        LOGGER.info(f"Lost mountpoint: {path}")
        handler = self.mountpoint_handlers[path]
        handler.close()
        LOGGER.info("  -> closed handler")
        del self.mountpoint_handlers[path]

    def _is_viable_mountpoint(self, mountpoint: Mountpoint) -> bool:
        # Drop restricted types
        if mountpoint.filesystem in VERBOTEN_FILESYSTEMS:
            return False

        # Sanity: never consider the root filesystem
        if mountpoint.mountpoint == '/':
            return False

        # Defer to the declared filter by path
        return detect_usb_type(mountpoint.mountpoint) is not USBType.INVALID


def set_mqtt_url(config: MQTTVariables) -> None:
    global MQTT_URL
    if config.username is not None and config.password is not None:
        auth = f"{config.username}:{config.password}@"
    elif config.username is not None:
        auth = f"{config.username}@"
    else:
        auth = ""

    port_str = (f":{config.port}" if config.port is not None else "")
    scheme = 'mqtts' if config.use_tls else 'mqtt'

    MQTT_URL = (
        f"{scheme}://{auth}{config.host}{port_str}/{config.topic_prefix}"
    )


def read_mqtt_config_file() -> MQTTVariables | None:
    """
    Read the MQTT config file and return the config.

    Returns None if the file does not exist or is invalid.
    """
    if not os.path.exists(MQTT_CONFIG_FILE):
        return None

    try:
        with open(MQTT_CONFIG_FILE) as f:
            config_dict = json.load(f)
            config = MQTTVariables(
                host=config_dict['host'],
                port=config_dict.get('port', None),
                topic_prefix=config_dict['topic_prefix'],
                use_tls=config_dict.get('use_tls', False),
                username=config_dict.get('username', None),
                password=config_dict.get('password', None),
            )
            set_mqtt_url(config)
            return config
    except Exception as e:
        LOGGER.error(f"Failed to read MQTT config file: {e}")
        return None


def setup_usercode_logging() -> None:
    global REL_TIME_FILTER, MQTT_CLIENT, MQTT_TOPIC_ROOT
    REL_TIME_FILTER = RelativeTimeFilter()
    USERCODE_LOGGER.addFilter(REL_TIME_FILTER)
    USERCODE_LOGGER.setLevel(logging.DEBUG)

    if MQTTHandler is not None:
        # If we have relative logging, we should also have the MQTT handler
        mqtt_config = read_mqtt_config_file()

        if mqtt_config is not None:
            handler = MQTTHandler(
                host=mqtt_config.host,
                topic=f"{mqtt_config.topic_prefix}/logs",
                port=mqtt_config.port,
                use_tls=mqtt_config.use_tls,
                username=mqtt_config.username,
                password=mqtt_config.password,
                connected_topic=f"{mqtt_config.topic_prefix}/connected",
                connected_callback=lambda: LED_CONTROLLER.set_wifi(True),
                disconnected_callback=lambda: LED_CONTROLLER.set_wifi(False),
            )
            MQTT_CLIENT = handler.mqtt
            MQTT_TOPIC_ROOT = mqtt_config.topic_prefix

            handler.setLevel(logging.INFO)
            handler.setFormatter(TieredFormatter(
                fmt='[%(reltime)08.3f - %(levelname)s] %(message)s',
                level_fmts={
                    USERCODE_LEVEL: '[%(reltime)08.3f] %(message)s',
                },
            ))
            USERCODE_LOGGER.addHandler(handler)


def main():
    logging.basicConfig(level=logging.DEBUG)
    setup_usercode_logging()

    fstab_reader = FSTabReader()

    registry = AutorunProcessRegistry()

    LED_CONTROLLER.mark_start()
    LED_CONTROLLER.set_status(LedStatus.NoUSB)

    # Initial pass (in case an autorun FS is already mounted)
    registry.update_filesystems(fstab_reader.read())

    try:
        while True:
            if fstab_reader.watch():
                registry.update_filesystems(fstab_reader.read())
    except KeyboardInterrupt:
        # Tell the registry that all filesystems were unmounted, which has the
        # effect of making it do cleanup.
        registry.update_filesystems([])  # type: ignore


if __name__ == '__main__':
    main()
