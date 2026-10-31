# LCM-X local embeddings dependency

This companion declares `fastembed>=0.8.0,<1.0` for the Hermes host to install
and retain across host updates. It registers no tools or hooks and downloads no
model. Its version stays stable across LCM-X releases: only dependency changes
should rebuild the host's shared environment.

Use `scripts/install.sh` from the LCM-X checkout, or follow the native install
command in [embeddings setup](../../docs/embeddings-setup.md). The host owns
installation consent. Declining leaves LCM-X available with full-text recall.
Manage this companion with the host's install, enable, disable and uninstall
commands. Removing it may remove fastembed when the host next syncs dependencies.
