# Vendored dependencies

This directory holds third-party code copied verbatim into the repo so it
travels with our package. **Do not modify** these files. If upstream ships
a fix, refresh with `scripts/sync_vendor.sh /path/to/upstream`.

## `millionaire_client/`

Provided by the NLP course staff (PoliMi 2025/26) for talking to the
PoliMillionaire game server. Original layout assumes the package sits next
to the notebook in Google Drive; vendoring it here means our package works
identically locally and in Colab without `sys.path` hacks.

Provenance: distributed alongside the assignment brief on WeBeep, version
tag in `millionaire_client/__init__.py`.
