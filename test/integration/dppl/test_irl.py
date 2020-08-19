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

from deepppl import dpplc
from deepppl.translation.exceptions import *
from contextlib import contextmanager

import ast
import pytest


def code_to_normalized(code):
    return ast.dump(ast.parse(code), annotate_fields=False)


from contextlib import contextmanager


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
    config = dpplc.Config()
    
    compiled = dpplc.stan2astpyFile(src_file, config, verbose)
    assert code_to_normalized(compiled) == target

def compile(filename):
    config = dpplc.Config()
    dpplc.stan2astpyFile(filename, config, verbose=True)

 
compile_tests = [
    ('coin_vectorized', None), 
    ('coin_guide_init', None), 
    ('gaussian', None), 
    ('gaussian_log_density', None), 
    ('double_gaussian', None), 
    ('log_normal', None), 
    ('operators', None), 
    ('operators-expr', None),
    ('simple_init', None), 
    ('missing_data', None),
    ('neal_funnel', None), 
    ('coin_reverted', 'In stan lines are not ordered'),
    ('coin_transformed_data', 'Type Inference'), 
    ('lstm', 'Type inference'),
    ('linear_regression', 'Type inference'),
    ('gaussian_process', 'Type inference'),
    ('regression_matrix', 'Type inference'),
    ('squared_error', 'Type inference'),
    ('vectorized_probability', 'Type inference'),
]
@pytest.mark.parametrize('test_name, fail', compile_tests)   
def test_normalize_and_compare(test_name, fail):
    if fail:
        pytest.xfail(fail)
    filename = f'deepppl/tests/good/{test_name}.stan'
    target_file = f'deepppl/tests/target_py/{test_name}.py'
    normalize_and_compare(filename, target_file)
    
compile_tests_notype = [
    ('aspirin', None),
    ('mlp', None), 
    ('mlp_default_init', None),
    ('kmeans', None),
    ('schools', None),
    ('logistic', 'Type infernce'), 
    ('lda', None),
    ('bayes_nn', None),  
    ('coin', None),
    ('coin_guide', None),
    ('coin_vect', None),
    ('cockroaches', None), 
    ('multimodal', None),
    ('multimodal_guide', None),
    ('posterior_twice', None),
    ('seeds', None),
    ('vae_inferred_shape', None),
    ('vae', None)  
]
@pytest.mark.parametrize('test_name, fail', compile_tests_notype)   
def test_normalize_and_compare_notype(test_name, fail):
    if fail:
        pytest.xfail(fail)
    filename = f'deepppl/tests/good/{test_name}.stan'
    target_file = f'deepppl/tests/target_py/{test_name}.py'
    normalize_and_compare(filename, target_file, verbose=False)

raise_tests = [
    ('coin_guide_missing_var', MissingGuideException, None),
    ('coin_guide_sample_obs', ObserveOnGuideException, None),
    ('mlp_undeclared_parameters', UndeclaredParametersException, None),
    ('mlp_undeclared_net1', UndeclaredNetworkException, None),
    ('mlp_undeclared_net2', UndeclaredNetworkException, None),
    ('mlp_missing_guide', MissingGuideNetException, None),
    ('coin_unknown_identifier', UndeclaredVariableException, None),
    ('coin_unknown_distribution', UnknownDistributionException, None),
    ('coin_already_declared_var', AlreadyDeclaredException, None),
    ('mlp_unsupported_prop', UnsupportedProperty, None),
    ('mlp_unsupported_prop1', UnsupportedProperty, None),
    ('mlp_incorrect_shape1', IncompatibleShapes, 'Type inference'),
    ('mlp_incorrect_shape2', IncompatibleShapes, 'Type inference'),
    ('mlp_incorrect_shape3', IncompatibleShapes, 'Type inference'),
    ('mlp_incorrect_shape4', IncompatibleShapes, 'Type inference')
]

@pytest.mark.parametrize('test_name, error, fail', raise_tests)
def test_raise_error(test_name, error, fail):
    if fail:
        pytest.xfail(fail)
    with pytest.raises(error):
        filename = f'deepppl/tests/good/{test_name}.stan'
        compile(filename)
        
        
not_raise_tests = [
    ('coin_guide_missing_model', MissingModelException, None),
    ('coin_invalid_sampling', InvalidSamplingException, None),
    ('coin_invalid_sampling2', NonRandomSamplingException, None),
    ('mlp_missing_model', MissingPriorNetException, 'Type inference'),
    
]

@pytest.mark.parametrize('test_name, error, fail', not_raise_tests)
def test_not_raise_error(test_name, error, fail):
    if fail:
        pytest.xfail(fail)
    with not_raises(error):
        filename = f'deepppl/tests/good/{test_name}.stan'
        compile(filename)