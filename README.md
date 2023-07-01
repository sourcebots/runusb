Automatic USB running
=====================

![Linting workflow](https://github.com/sourcebots/runusb/actions/workflows/pylint.yml/badge.svg)

Watches for mounted filesystems with `.autorun` files, and runs those files,
well, automatically.

Building the Debian package
---------------------------

Install through apt:

* `build-essential`
* `devscripts`
* `debhelper`

And then run, from the root of the project:

* `debuild -uc -us`

Alternatively, you can download the prebuilt package from the GitHub releases.
