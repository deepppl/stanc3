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
from deepppl.translation.sdim import KnownDimension, Dimension, \
        Dnew, Dnamed, Dshape, Druntime, Dconstant, Groups
from deepppl.translation.stype import Type_, \
        Treal, Tint, Tindexed, Tvector, Trow_vector, Tmatrix, Tarray, \
        Tnamed, Tnew

from contextlib import contextmanager
import pytest


def code_to_normalized(code):
    return ast.dump(ast.parse(code))

from contextlib import contextmanager

@contextmanager
def not_raises(exception):
  try:
    yield
  except exception:
    raise pytest.fail("DID RAISE {0}".format(str(exception)))

### This section tests unification of dimensions
def test_unify_dim_variable_variable():
        Dnew().unify(Dnew())

def test_unify_dim_variable_named_variable():
        Dnew().unify(Dnamed({}, "x"))

def test_unify_dim_named_variable_named_variable_same_wrong_envs():
        Dnamed({}, "x").unify(Dnamed({}, "x"))

def test_unify_dim_named_variable_named_variable_same():
        denv={}
        Dnamed(denv, "x").unify(Dnamed(denv, "x"))

def test_unify_dim_named_variable_named_variable_different():
        denv={}
        Dnamed(denv, "x").unify(Dnamed(denv, "y"))

def test_unify_dim_variable_path():
        equalities = Groups()
        Dnew().unify(Dshape("mlp.size"))
        assert(len(equalities.groups())==0)

def test_unify_dim_named_variable_path():
        denv = {}
        equalities = Groups()
        Dnamed(denv, "x").unify(Dshape("mlp.size"))
        assert(len(equalities.groups())==0)

def test_unify_dim_path_path_same():
        equalities = Groups()
        Dshape("mlp.size").unify(Dshape("mlp.size"))
        assert(len(equalities.groups())==0)

def test_unify_dim_path_path_different():
        equalities = Groups()
        Dshape("mlp.size").unify(Dshape("mlp.other_size"), equalities=equalities)
        assert(len(equalities.groups())==1)


def test_unify_dim_path_path_different_via_named():
        denv = {}
        equalities = Groups()
        Dnamed(denv, "x").unify(Dshape("mlp.size"),equalities=equalities)
        Dshape("mlp.size_other").unify(Dnamed(denv, "y"),equalities=equalities)
        Dnamed(denv, "x").unify(Dnamed(denv, "y"), equalities=equalities)
        assert(len(equalities.groups())==1)


def test_unify_dim_constant_same():
        Dconstant(3).unify(Dconstant(3))

def test_unify_dim_constant_different():
        with pytest.raises(IncompatibleDimensions):
                Dconstant(3).unify(Dconstant(4))

def test_unify_dim_constant_path():
        equalities = Groups()
        Dconstant(3).unify(Dshape(4), equalities=equalities)
        assert(len(equalities.groups())==1)

### This section tests unification of types (with anonymous dimensions)
def test_unify_int_int():
        Tint().unify(Tint())

def test_unify_real_real():
        Treal().unify(Treal())

def test_unify_int_real():
        with pytest.raises(IncompatibleTypes):
                Tint().unify(Treal())

def test_unify_real_int():
        with pytest.raises(IncompatibleTypes):
                Treal().unify(Tint())

def test_unify_array_real():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Treal())

def test_unify_array_vector():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Tvector())

def test_unify_array_indexed():
        Tarray(Treal()).unify(Tindexed())

def test_unify_array_indexed_real():
        Tarray(Treal()).unify(Tindexed(Treal()))

def test_unify_array_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Tarray(Treal()).unify(Tindexed(Tint()))

def test_unify_vector_indexed_real():
        Tvector().unify(Tindexed(Treal()))

def test_unify_vector_indexed_int():
        with pytest.raises(IncompatibleTypes):
                Tvector().unify(Tindexed(Tint()))

def test_unify_matrix_indexed1():
        Tmatrix().unify(Tindexed())

def test_unify_matrix_indexed_real1():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Treal()))

def test_unify_matrix_indexed2():
        Tmatrix().unify(Tindexed(Tindexed()))

def test_unify_matrix_indexed_real2():
        Tmatrix().unify(Tindexed(Tindexed(Treal())))

def test_unify_matrix_indexed3():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Tindexed(Tindexed())))

def test_unify_matrix_indexed_real3():
        with pytest.raises(IncompatibleTypes):
                Tmatrix().unify(Tindexed(Tindexed(Tindexed(Treal()))))

def test_unify_variable_int():
        Tnew().unify(Tindexed(Tint()))

def test_unify_variable_indexed():
        Tnew().unify(Tindexed(Tint()))

def test_unify_named_variable_int():
        tenv = {}
        Tnamed(tenv, "x").unify(Tindexed(Tint()), tenv=tenv)

def test_unify_named_variable_indexed():
        tenv = {}
        Tnamed(tenv, "x").unify(Tindexed(Tint()), tenv=tenv)

def test_unify_named_variable_variable():
        tenv = {}
        Tnamed(tenv, "x").unify(Tnew(), tenv=tenv)

def test_unify_named_variable_named_variable_same():
        tenv = {}
        Tnamed(tenv, "x").unify(Tnamed(tenv, "x"), tenv=tenv)

def test_unify_named_variable_named_variable_different():
        tenv = {}
        Tnamed(tenv, "x").unify(Tnamed(tenv, "y"), tenv=tenv)

def test_unify_named_variable_named_variable_different_no_conflict():
        tenv = {}
        Tnamed(tenv, "x").unify(Tint(), tenv=tenv)
        Tnamed(tenv, "y").unify(Treal(), tenv=tenv)

def test_unify_named_variable_named_variable_same_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                Tnamed(tenv, "x").unify(Tint(), tenv=tenv)
                Tnamed(tenv, "x").unify(Treal(), tenv=tenv)

def test_unify_named_variable_named_variable_different_conflict():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                Tnamed(tenv, "x").unify(Tint(), tenv=tenv)
                Tnamed(tenv, "y").unify(Treal(), tenv=tenv)
                Tnamed(tenv, "y").unify(Tnamed(tenv, "x"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_real():
        tenv = {}
        Tmatrix().unify(Tindexed(Tnamed(tenv, "x")), tenv=tenv)
        Tindexed(Tnamed(tenv, "y")).unify(Tnamed(tenv, "x"), tenv=tenv)
        Treal().unify(Tnamed(tenv, "y"), tenv=tenv)

def test_unify_matrix_indexed_named_variable_int():
        with pytest.raises(IncompatibleTypes):
                tenv = {}
                Tmatrix().unify(Tindexed(Tnamed(tenv, "x")), tenv=tenv)
                Tindexed(Tnamed(tenv, "y")).unify(Tnamed(tenv, "x"), tenv=tenv)
                Tint().unify(Tnamed(tenv, "y"), tenv=tenv)
