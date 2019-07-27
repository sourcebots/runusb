Automatic USB running
=====================

[![CircleCI](https://circleci.com/gh/sourcebots/runusb.svg?style=shield)](https://circleci.com/gh/sourcebots/runusb)

Watches for mounted filesystems with `.autorun` files, and runs those files,
well, automatically.

Building the Debian package
---------------------------

Install through apt:

* `build-essential`
* `devscripts`
* `debhelper`
* `dh-systemd`

And then run, from the root of the project:

* `debuild -uc -us`
