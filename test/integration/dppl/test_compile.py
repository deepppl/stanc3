# /* Copyright 2018
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# */

from os.path import splitext, basename, dirname
import subprocess
from contextlib import contextmanager
import ast
import pytest

def compile(stanfile):
    name = basename(stanfile)
    subprocess.check_call(["dune","exec","stanc","--","--pyro",stanfile])
    pyfile = splitext(stanfile)[0] + ".py"
    with open(pyfile) as f:
        compiled_code = f.read()
    return compiled_code

def code_to_normalized(code):
    return ast.dump(ast.parse(code), annotate_fields=False)

@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail("DID RAISE {0}".format(exception))

# Note about setting verbose=False

# if we use verbose mode, then the code generates annotated types.
# it sets simple=1 in the AnnAssign constructor
# to avoid parenthesis for simple identifiers
# This is good, as it makes the generated python code nicer.
# unfortunately, when parsing back in the code, python sometimes
# sets simple=0.  When we then compare, it fails.
# The simplest solution, taken here, is just not to generate type annotations
# for this and similar examples

def normalize_and_compare(src_file, target_file, verbose=True):
    with open(target_file) as f:
        target_code = f.read()
    target = code_to_normalized(target_code)
    compiled = compile(src_file)
    assert code_to_normalized(compiled) == target

compile_tests = [
    ('coin', None),
    ('coin_guide', None),
    ('coin_vect', None),
    ('coin_vectorized', None),
    ('coin_guide_init', None),
    ('coin_reverted', None),
    ('coin_transformed_data', None),
    ('gaussian', None),
    ('gaussian_log_density', None),
    ('double_gaussian', None),
    ('multimodal', None),
    ('multimodal_guide', None),
    ('aspirin', None),
    ('log_normal', None),
    ('operators', None),
    ('operators-expr', None),
    ('simple_init', None),
    ('missing_data', None),
    ('neal_funnel', None),
    ('linear_regression', None),
    ('kmeans', None),
    ('schools', None),
    ('logistic', None),
    ('lda', 'XXX TODO?: simplex XXX'),
    ('cockroaches', None),
    ('posterior_twice', None),
    ('seeds', None),
    ('gaussian_process', None),
    ('regression_matrix', None),
    ('squared_error', None),
    ('vectorized_probability', None),
    ('mlp', 'XXX TODO: deep XXX'),
    ('mlp_default_init', 'XXX TODO: deep XXX'),
    ('vae_inferred_shape', 'XXX TODO: deep XXX'),
    ('vae', 'XXX TODO: deep XXX'),
    ('bayes_nn', 'XXX TODO: deep XXX'),
    ('lstm', 'Type inference + XXX TODO: deep XXX'),
]
@pytest.mark.parametrize('test_name, fail', compile_tests)
def test_normalize_and_compare(test_name, fail):
    if fail:
        pytest.xfail(fail)
    filename = f'good/{test_name}.stan'
    target_file = f'target_py/{test_name}.py'
    normalize_and_compare(filename, target_file)
