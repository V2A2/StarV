"""Pytest configuration and hooks for StarV test suite."""

import inspect
import os
import sys
import pytest

# Ensure repository root is at the front of sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Use non-interactive matplotlib backend in automated test runs
os.environ.setdefault('MPLBACKEND', 'Agg')


def pytest_collection_modifyitems(session, config, items):
    """Skip legacy Test methods that require non-fixture positional arguments."""
    for item in items:
        if isinstance(item, pytest.Function):
            sig = inspect.signature(item.obj)
            required_params = [
                p.name for p in sig.parameters.values()
                if p.name != 'self'
                and p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            # If the method requires parameters that are not registered fixtures, skip it
            fixture_defs = getattr(session, '_fixturemanager', None)
            available_fixtures = set(fixture_defs._arg2fixturedefs.keys()) if fixture_defs else set()
            missing = [p for p in required_params if p not in available_fixtures]
            if missing:
                item.add_marker(
                    pytest.mark.skip(reason=f"Requires explicit benchmark arguments: {missing}")
                )


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Fail pytest item if a legacy StarV Test instance increments self.n_fails."""
    instance = getattr(pyfuncitem, 'instance', None)
    before_fails = getattr(instance, 'n_fails', 0) if instance is not None else 0
    yield
    if instance is not None:
        after_fails = getattr(instance, 'n_fails', 0)
        if after_fails > before_fails:
            pytest.fail(
                f"{pyfuncitem.name} recorded {after_fails - before_fails} failure(s) (n_fails={after_fails})"
            )
