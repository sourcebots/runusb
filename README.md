Automatic USB running
=====================

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
