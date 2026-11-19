- Fix: switching `LCM_HOST_MESSAGE_UID` from `shadow` to `off` now stops the session-end fallback from using
  saved host UID bindings to reject a matching prefix and leave its tail unstored (#2970).
