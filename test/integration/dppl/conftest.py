
## from pytest docs: https://docs.pytest.org/en/latest/example/simple.html#detect-if-running-from-within-a-pytest-run

def pytest_configure(config):
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    import sys

    del sys._called_from_test