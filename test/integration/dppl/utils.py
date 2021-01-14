
import os
import pytest

def on_travis():
    return "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true"

skip_on_travis = pytest.mark.skipif(on_travis(), reason="This test takes too long to run on Travis CI.")
