import jax
import jax.random as jrandom

from jaxsw._src.operators.functional import advection as F_adv

KEY = jrandom.PRNGKey(123)

U_1D = jrandom.normal(key=KEY, shape=(20,))


def test_plusminus():
    u_plus, u_minus = F_adv.plusminus(U_1D, way=1)

    assert u_plus.sum() > 0.0
    assert u_minus.sum() < 0.0

    u_plus, u_minus = F_adv.plusminus(U_1D, way=-1)

    assert u_plus.sum() < 0.0
    assert u_minus.sum() > 0.0


def test_plusminus_custom():
    fn = jax.nn.relu
    u_plus, u_minus = F_adv.plusminus(U_1D, way=1, fn=fn)

    assert u_plus.sum() > 0.0
    assert u_minus.sum() < 0.0

    u_plus, u_minus = F_adv.plusminus(U_1D, way=-1, fn=fn)

    assert u_plus.sum() < 0.0
    assert u_minus.sum() > 0.0
