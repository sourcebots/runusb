#!/usr/bin/env python3

import os.path
import re
import sys
import select
import logging
import argparse
import subprocess
import collections


LOGGER = logging.getLogger('runusb.legacy.py')


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Automatically run from mountpoints with .autorun",
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help="enable verbose output",
        action='store_const',
        const=logging.INFO,
        dest='log_level',
        default=logging.WARNING,
    )

    parser.add_argument(
        '--debug',
        help="enable very verbose output",
        action='store_const',
        const=logging.DEBUG,
        dest='log_level',
    )

    parser.add_argument(
        '--runfile',
        help="name of the autorun file to detect",
        action='store',
        default='.autorun',
    )

    parser.add_argument(
        '--spawn-image',
        help="root filesystem for the container",
        action='store',
        default='/mnt/rootmirror',
    )

    parser.add_argument(
        '--bind',
        help="mountpoint(s) to bind inside the container",
        action='append',
        default=[],
    )

    parser.add_argument(
        '--proc-mounts',
        help="alternative path to watch from /proc/mounts",
        action='store',
        default='/proc/mounts',
    )

    parser.add_argument(
        '--enable-network',
        help="enable networking inside the container",
        action='store_true',
        dest='networking',
    )

    parser.add_argument(
        '--disable-network',
        help="disable networking inside the container (the default)",
        action='store_false',
        dest='networking',
        default=False,
    )

    return parser


VERBOTEN_FILESYSTEMS = (
    'cgroup',
    'configfs',
    'debugfs',
    'devpts',
    'devtmpfs',
    'hugetlbfs',
    'mqueue',
    'proc',
    'sysfs',
)


Mountpoint = collections.namedtuple('Mountpoint', (
    'mountpoint',
    'device',
    'filesystem',
    'options',
))


class FSTabReader(object):
    def __init__(self, path):
        self.path = path
        self.handle = open(path)

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.handle.__exit__(*args, **kwargs)

    def close(self):
        self.handle.close()

    def read(self):
        self.handle.seek(0)

        for line in self.handle:
            stripped_line = line.strip()

            try:
                # Split the line into the standard 6 parts; the ValueError
                # will protect us from ill-formatted lines
                (device, mountpoint, filesystem, options, _, _) = \
                    line.split(' ')
            except ValueError:
                pass

            if options:
                options_tuple = tuple(options.split(','))
            else:
                options = ()

            yield Mountpoint(
                mountpoint=mountpoint,
                device=device,
                filesystem=filesystem,
                options=options,
            )

    def watch(self, timeout=None):
        _, _, changed = select.select([], [], [self.handle], timeout)

        if changed:
            LOGGER.debug("Detected change in %s", self.path)

        return bool(changed)



class AutorunProcessRegistry(object):
    def __init__(self, *, open_process, close_process, mountpoint_filter):
        self.open_process = open_process
        self.close_process = close_process
        self.mountpoint_filter = mountpoint_filter

        self.mountpoint_processes = {}

    def update_filesystems(self, mountpoints):
        actual_mountpoint_paths = {
            x.mountpoint
            for x in mountpoints
            if self._is_viable_mountpoint(x)
        }

        expected_mountpoint_paths = {
            x for x in self.mountpoint_processes.keys()
        }

        # Handle newly detected filesystems
        for new_mountpoint_path in (
            actual_mountpoint_paths -
            expected_mountpoint_paths
        ):
            self._detect_new_mountpoint_path(new_mountpoint_path)

        # Handle now-dead filesystems
        for old_mountpoint_path in (
            expected_mountpoint_paths -
            actual_mountpoint_paths
        ):
            self._detect_dead_mountpoint_path(old_mountpoint_path)

    def _detect_new_mountpoint_path(self, path):
        LOGGER.info("Found new mountpoint: %s", path)
        process = self.open_process(path)
        LOGGER.info("  -> launched process")
        self.mountpoint_processes[path] = process

    def _detect_dead_mountpoint_path(self, path):
        LOGGER.info("Lost mountpoint: %s", path)
        process = self.mountpoint_processes[path]
        self.close_process(path, process)
        LOGGER.info("  -> closed process")
        del self.mountpoint_processes[path]

    def _is_viable_mountpoint(self, mountpoint):
        # Drop restricted types
        if mountpoint.filesystem in VERBOTEN_FILESYSTEMS:
            LOGGER.debug(
                "Disregarding filesystem %s due to forbidden filesystem "
                "type %s",
                mountpoint.mountpoint,
                mountpoint.filesystem,
            )

            return False

        # Sanity: never consider the root filesystem
        if mountpoint.mountpoint == '/':
            return False

        # Defer to the declared filter by path
        return self.mountpoint_filter(mountpoint.mountpoint)


def main(args=sys.argv[1:]):
    options = argument_parser().parse_args(args)

    logging.basicConfig(level=options.log_level)

    def get_machinename(path):
        return re.sub(r'[^\w]', r'_', path).strip('_')

    # Actual process drivers
    def open_process(path):
        command = ['systemd-nspawn']

        # Specify the root directory for the nspawn container
        command.extend(('--directory', options.spawn_image))

        # Mount the root directory read-only
        command.append('--read-only')

        # Bring in any requested bind mounts
        for bind_mount in options.bind:
            command.extend(('--bind', bind_mount))

        # Disable networking if requested
        if not options.networking:
            command.append('--private-network')

        # Mount a tmpfs as /tmp
        command.extend(('--tmpfs', '/tmp'))

        # Bind-mount the actual path
        command.extend(('--bind', path))

        # Attach a machine ID, use the path as the ID
        print(get_machinename(path))
        command.extend(('--machine', get_machinename(path)))

        # Run the autorun file with bash -c
        command.extend(('/bin/bash', '-c'))

        # cd to the mountpoint and run the autorun file
        command.append(
            'cd "{mountpoint}" ; script -ec ./"{autorun}" -f log.txt'.format(
                mountpoint=path,
                autorun=options.runfile,
            ),
        )

        return subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def close_process(path, process):
        # With the mountpoint now missing the only sensible thing to do is
        # to die horribly.
        print(get_machinename(path))
        command = ('machinectl', 'terminate', get_machinename(path))
        subprocess.call(command)
        process.wait(timeout=5)
        try:
            process.kill()
        except ProcessLookupError:
            pass

    def is_autorun_mountpoint(path):
        # subpaths of the root image are excluded
        if path.startswith(options.spawn_image):
            return False

        # Check for the existence of the autorun file
        return os.path.exists(os.path.join(path, options.runfile))

    try:
        fstab_reader = FSTabReader(options.proc_mounts)
    except FileNotFoundError:
        # No /proc/mounts, check whether the user is actually on Linux at all
        if not sys.platform.startswith('linux'):
            print(
                "Could not find {mounts} - this is a Linux-specific API and "
                "you seem to be on {platform}.".format(
                    mounts=options.proc_mounts,
                    platform=sys.platform,
                ),
            )
        else:
            print(
                "Could not find {mounts} - is procfs mounted?".format(
                    mounts=options.proc_mounts,
                ),
            )
        exit(1)

    registry = AutorunProcessRegistry(
        open_process=open_process,
        close_process=close_process,
        mountpoint_filter=is_autorun_mountpoint,
    )

    # Initial pass (in case an autorun FS is already mounted)
    registry.update_filesystems(fstab_reader.read())

    try:
        while True:
            if fstab_reader.watch():
                registry.update_filesystems(fstab_reader.read())
    except KeyboardInterrupt:
        # Tell the registry that all filesystems were unmounted, which has the
        # effect of making it do cleanup.
        registry.update_filesystems([])


if __name__ == '__main__':
    main()