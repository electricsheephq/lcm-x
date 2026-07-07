"""LCM bypass-lineage session tracking.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``BypassLineageMixin``
(WS5 seam). The methods record and hand off which sessions descend from an
LCM-bypassed session so later turns can recognise the lineage. State stays on
the engine (accessed via ``self``, including the shared auxiliary-session
lock); mixing this in leaves every call site and ``self._*`` reference
unchanged.
"""

from typing import Optional


class BypassLineageMixin:
    def _has_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> bool:
        with self._auxiliary_session_lock:
            if session_id not in self._lcm_bypass_lineage_session_ids:
                return False
            if platform is None:
                return True
            platforms = self._lcm_bypass_lineage_platforms.get(session_id) or set()
            return not platforms or platform in platforms

    def _mark_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> None:
        if not session_id:
            return
        platform = self._session_platform if platform is None else str(platform or "")
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.add(session_id)
            self._lcm_bypass_lineage_platforms.setdefault(session_id, set()).add(platform)
            self._lcm_session_last_platform[session_id] = platform
            self._lcm_session_last_bypassed[session_id] = True

    def _unmark_lcm_bypass_lineage_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.discard(session_id)
            self._lcm_bypass_lineage_platforms.pop(session_id, None)

    def _handoff_lcm_bypass_lineage(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        new_platform: str = "",
    ) -> None:
        with self._auxiliary_session_lock:
            if old_session_id:
                self._lcm_bypass_lineage_session_ids.add(old_session_id)
            if new_session_id:
                new_platform = str(new_platform or "")
                self._lcm_bypass_lineage_session_ids.add(new_session_id)
                self._lcm_bypass_lineage_platforms.setdefault(new_session_id, set()).add(new_platform)
                self._lcm_session_last_platform[new_session_id] = new_platform
                self._lcm_session_last_bypassed[new_session_id] = True

    def _compression_boundary_from_lcm_bypassed_session(self, old_session_id: str) -> bool:
        if not old_session_id:
            return False
        if old_session_id in self._lcm_session_last_bypassed:
            return bool(self._lcm_session_last_bypassed.get(old_session_id))
        if old_session_id == self._session_id:
            return bool(
                self._bypasses_lcm_context_management()
                or self._session_id_matches_lcm_bypass_filters(
                    old_session_id,
                    platform=self._session_platform,
                )
            )
        return bool(
            self._has_lcm_bypass_lineage_session(old_session_id)
            or self._session_id_matches_lcm_bypass_filters(old_session_id)
        )
