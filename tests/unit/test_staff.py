"""
Unit tests for staff scheduling helpers.

`_smooth_schedule` is a pure function on lists of headcounts — a greedy
"min-shift" smoother that prevents 1-hour stubs in the schedule. Tests here
pin the specific behaviours the kitchen actually relies on:

  * No 1-hour or 2-hour stubs survive the smoother (extended to min_shift).
  * Existing long shifts are unchanged.
  * All-zero input passes through untouched (no synthesised hours).
  * The total number of staffed hours never decreases.

If `_smooth_schedule` is ever rewritten, these tests force a code review.
"""

from __future__ import annotations

import pytest

from rrp.forecasting.staff import _smooth_schedule


class TestSmoothSchedule:
    def test_all_zero_passes_through(self) -> None:
        assert _smooth_schedule([0.0] * 24) == [0.0] * 24

    def test_long_shift_unchanged(self) -> None:
        # 6-hour shift, already above the 4-hour minimum — must not be padded.
        raw = [0.0] * 12 + [3.0] * 6 + [0.0] * 6
        assert _smooth_schedule(raw, min_shift_hours=4) == raw

    def test_one_hour_stub_extended(self) -> None:
        raw = [0.0] * 10 + [2.0] + [0.0] * 13
        smoothed = _smooth_schedule(raw, min_shift_hours=4)
        # The original 2.0 is preserved at its position and the immediately
        # following hours are padded to at least 1.0 (no 1-hour stubs survive).
        assert smoothed[10] == 2.0
        assert smoothed[11] >= 1.0
        assert smoothed[12] >= 1.0
        assert smoothed[13] >= 1.0
        assert smoothed[:10] == raw[:10]

    def test_two_hour_stub_first_two_unchanged(self) -> None:
        raw = [0.0] * 8 + [3.0, 2.0] + [0.0] * 14
        smoothed = _smooth_schedule(raw, min_shift_hours=4)
        # The original headcount values at the start of the shift are preserved.
        assert smoothed[8] == 3.0
        assert smoothed[9] == 2.0
        # And subsequent hours are padded so the shift is at least 4 hours.
        assert smoothed[10] >= 1.0
        assert smoothed[11] >= 1.0

    def test_padding_does_not_reduce_existing(self) -> None:
        """Padding must use max() — never overwrite a higher existing count."""
        raw = [0.0] * 6 + [5.0, 0.0, 4.0] + [0.0] * 15
        smoothed = _smooth_schedule(raw, min_shift_hours=4)
        # Indices 6..9 are 5,0,4,0 -> after smoothing the run starting at 6
        # is padded to 4 hours, but we don't drop the 4.0 at index 8.
        assert smoothed[6] == 5.0
        assert smoothed[8] == 4.0  # existing higher value preserved

    def test_total_staffed_hours_never_decrease(self) -> None:
        """Property: smoothing only adds hours, never removes them."""
        cases = [
            [0.0] * 24,
            [1.0] * 24,
            [0.0] * 10 + [2.0] + [0.0] * 13,
            [0.0] * 8 + [1.0, 0.0, 2.0, 0.0, 3.0] + [0.0] * 11,
        ]
        for raw in cases:
            smoothed = _smooth_schedule(raw, min_shift_hours=4)
            assert sum(smoothed) >= sum(raw)
            assert len(smoothed) == len(raw)

    def test_min_shift_one_is_no_op(self) -> None:
        """min_shift=1 means every hour is already a valid shift -> identity."""
        raw = [0.0, 1.0, 0.0, 2.0, 0.0]
        assert _smooth_schedule(raw, min_shift_hours=1) == raw

    @pytest.mark.parametrize("min_shift", [3, 4, 6])
    def test_no_stub_shorter_than_min_shift_remains(self, min_shift: int) -> None:
        """After smoothing, no run of nonzero headcounts is shorter than min_shift."""
        raw = [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0]
        smoothed = _smooth_schedule(raw, min_shift_hours=min_shift)

        i = 0
        n = len(smoothed)
        while i < n:
            if smoothed[i] > 0:
                end = i
                while end < n and smoothed[end] > 0:
                    end += 1
                run_length = end - i
                # Either the run is at least min_shift long, OR it was extended
                # to the end of the array (can't pad past day-end).
                assert run_length >= min_shift or end == n, (
                    f"Found {run_length}-hour run at index {i} with "
                    f"min_shift={min_shift}: {smoothed}"
                )
                i = end
            else:
                i += 1
