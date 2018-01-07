Automatic USB running
=====================

[![CircleCI](https://circleci.com/gh/sourcebots/runusb.svg?style=shield&circle-token=37b85813fead1813c6daf82f8d95ccb9931408df)](https://circleci.com/gh/sourcebots/runusb)

Watches for mounted filesystems with `.autorun` files, and runs those files,
well, automatically.

The files are run in a systemd nspawn container, which limits the havoc one
can accidentally wreak, and causes consistent behaviour when shutting down.

Building the Debian package
---------------------------

Install through apt:

* `build-essential`
* `devscripts`
* `debhelper`
* `dh-systemd`

And then run, from the root of the project:

* `debuild -uc -us`
