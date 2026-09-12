"""Fail-closed comparison of a published critic against its native trainer."""
import math


def compare_scores(native, replica, repeated, *, version, count, tolerance=.005):
    if count <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid critic comparison configuration')
    for name, result in [('native', native), ('replica', replica), ('repeated', repeated)]:
        if result['version'] != version or len(result['scores']) != count:
            raise ValueError(f'{name}: critic version or context count mismatch')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result['scores']):
            raise ValueError(f'{name}: invalid critic probability')
    errors = [abs(a-b) for a, b in zip(native['scores'], replica['scores'], strict=True)]
    worst = max(range(count), key=errors.__getitem__)
    deterministic = replica == repeated
    return dict(max_abs_error=errors[worst], tolerance=tolerance,
                deterministic=deterministic, abs_errors=errors,
                worst_context_index=worst,
                failed_context_indices=[i for i, error in enumerate(errors) if error > tolerance],
                passed=errors[worst] <= tolerance and deterministic)
