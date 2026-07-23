# Python runtime dependencies, vendored so the project installs entirely offline.
Built via: pip download --only-binary=:all: --platform manylinux2014_x86_64 --python-version 311 --abi cp311.
Dockerfile installs with: pip install --no-index --find-links vendor/pywheels .
