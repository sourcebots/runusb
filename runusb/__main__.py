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
import uuid
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
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
    from paho.mqtt.client import Client as MQTTClient  # type: ignore[import-untyped,unused-ignore]
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
MQTT_CONFIG_FILE = '/etc/sbot/mqtt.conf'


@dataclass
class MqttSettings:
    # url format: mqtt[s]://[<username>[:<password>]@]<host>[:<port>]/<topic_root>
    url: str | None = None
    active_config: MQTTVariables | None = None
    client: MQTTClient | None = None
    active_usercode: RobotUSBHandler | None = None
    extra_data: dict[str, str] = field(default_factory=lambda: {"run_uuid": ""})


# This will be populated if we have the config file
MQTT_SETTINGS = MqttSettings()


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
        mqtt_client = MQTT_SETTINGS.client
        if mqtt_client is not None and MQTT_SETTINGS.active_config is not None:
            topic_prefix = MQTT_SETTINGS.active_config.topic_prefix
            mqtt_client.publish(
                f'{topic_prefix}/state',
                json.dumps(dict(state=value.name, **MQTT_SETTINGS.extra_data)),
                qos=1,
                retain=True,
            )


LED_CONTROLLER = LEDController()


def mqtt_on_stop_action(client, userdata, message):
    LOGGER.info("Received stop action")
    try:
        payload = json.loads(message.payload)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to decode stop action message.")
        return

    if payload.get('pressed') is not True:
        LOGGER.info("Stop action had incorrect payload, ignoring.")
        return

    if MQTT_SETTINGS.active_usercode is not None:
        # Run the cleanup function to stop the usercode but allow it to be
        # restarted without reinserting the USB
        MQTT_SETTINGS.active_usercode.killed = True
        MQTT_SETTINGS.active_usercode.cleanup()


def mqtt_on_reset_action(client, userdata, message):
    LOGGER.info("Received reset action")
    try:
        payload = json.loads(message.payload)
    except json.JSONDecodeError:
        LOGGER.warning("Failed to decode reset action message.")
        return

    if payload.get('pressed') is not True:
        LOGGER.info("Reset action had incorrect payload, ignoring.")
        return

    if MQTT_SETTINGS.active_usercode is not None:
        # The reset function will stop the usercode and wait for it to finish,
        # if it was running, before restarting it
        MQTT_SETTINGS.active_usercode.reset()


def mqtt_connected_actions():
    """Actions to perform when the MQTT client connects."""
    LED_CONTROLLER.set_wifi(True)
    if MQTT_SETTINGS.client is not None:
        mqtt_client = MQTT_SETTINGS.client
        assert MQTT_SETTINGS.active_config is not None
        topic_prefix = MQTT_SETTINGS.active_config.topic_prefix
        mqtt_client.message_callback_add(f"{topic_prefix}/stop", mqtt_on_stop_action)
        mqtt_client.message_callback_add(f"{topic_prefix}/reset", mqtt_on_reset_action)
        mqtt_client.subscribe(f"{topic_prefix}/stop", qos=1)
        mqtt_client.subscribe(f"{topic_prefix}/reset", qos=1)


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
        self.mountpoint_path = mountpoint_path

        if MQTT_SETTINGS.active_usercode is not None:
            raise RuntimeError("There is already a usercode running")
        else:
            MQTT_SETTINGS.active_usercode = self

        self._setup_logging(mountpoint_path)
        LED_CONTROLLER.set_code(True)

        self.env = dict(os.environ)
        self.env["SBOT_METADATA_PATH"] = MOUNTPOINT_DIR
        if MQTT_SETTINGS.url is not None:
            # pass the mqtt url to the robot for camera images
            self.env["SBOT_MQTT_URL"] = MQTT_SETTINGS.url

        self.start()

    def start(self) -> None:
        run_uuid = uuid.uuid4().hex
        MQTT_SETTINGS.extra_data["run_uuid"] = run_uuid
        self.env["run_uuid"] = run_uuid
        self.killed = False
        REL_TIME_FILTER.reset_time_reference()  # type: ignore[union-attr]
        LED_CONTROLLER.set_status(LedStatus.Running)
        self.process = subprocess.Popen(
            [sys.executable, '-u', ROBOT_FILE],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            bufsize=1,  # line buffered
            cwd=self.mountpoint_path,
            env=self.env,
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

        # Ensure logs have finished writing
        self.log_thread.join()

        # Sync filesystems after run finishes
        os.sync()

    def close(self) -> None:
        self.cleanup()
        MQTT_SETTINGS.extra_data["run_uuid"] = ""  # Reset the run UUID
        LED_CONTROLLER.set_status(LedStatus.NoUSB)
        LED_CONTROLLER.set_code(False)

        # Explicitly close handler before removing it
        self.handler.close()
        USERCODE_LOGGER.removeHandler(self.handler)
        MQTT_SETTINGS.active_usercode = None

    def reset(self) -> None:
        self.cleanup()
        # Wait for the process to finish
        self.process.wait()
        self.log_thread.join()
        self.start()

    def _send_signal(self, sig: int) -> None:
        if self.process.poll() is not None:
            # Process has already exited, so the kill() call would fail.
            return
        os.killpg(self.process.pid, sig)

    def _watch_process(self) -> None:
        # Wait for the process to complete
        self.process.wait()
        if self.killed:
            USERCODE_LOGGER.warning("Your code was stopped.")
            LED_CONTROLLER.set_status(LedStatus.Killed)
        elif self.process.returncode != 0:
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
        # Write through to avoid buffering the log file since the USB might be
        # removed at any time
        self.handler.stream.reconfigure(write_through=True)
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
        try:
            handler = handler_class(path)
        except RuntimeError as e:
            LOGGER.error(f"Failed to launch handler: {e}")
        else:
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
    if config.username is not None and config.password is not None:
        auth = f"{config.username}:{config.password}@"
    elif config.username is not None:
        auth = f"{config.username}@"
    else:
        auth = ""

    port_str = (f":{config.port}" if config.port is not None else "")
    scheme = 'mqtts' if config.use_tls else 'mqtt'

    MQTT_SETTINGS.url = (
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
    global REL_TIME_FILTER
    REL_TIME_FILTER = RelativeTimeFilter()
    USERCODE_LOGGER.addFilter(REL_TIME_FILTER)
    USERCODE_LOGGER.setLevel(logging.DEBUG)

    if MQTTHandler is not None:
        # If we have relative logging, we should also have the MQTT handler
        mqtt_config = read_mqtt_config_file()
        MQTT_SETTINGS.active_config = mqtt_config

        if mqtt_config is not None:
            handler = MQTTHandler(
                host=mqtt_config.host,
                topic=f"{mqtt_config.topic_prefix}/logs",
                port=mqtt_config.port,
                use_tls=mqtt_config.use_tls,
                username=mqtt_config.username,
                password=mqtt_config.password,
                connected_topic=f"{mqtt_config.topic_prefix}/connected",
                connected_callback=mqtt_connected_actions,
                disconnected_callback=lambda: LED_CONTROLLER.set_wifi(False),
                extra_data=MQTT_SETTINGS.extra_data,
            )
            MQTT_SETTINGS.client = handler.mqtt

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
