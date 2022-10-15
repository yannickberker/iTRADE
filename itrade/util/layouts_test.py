"""Test implementation of the Plate class."""
import pytest

from ..util.layouts import Plate


def test_range() -> None:
    """Test :py:meth:`~layouts.Plate.range`."""
    assert all(isinstance(p, Plate) for p in Plate.range(5))
    assert list(Plate.range(5)) == list(range(5))


def test_init() -> None:
    """Test :py:meth:`~layouts.Plate.__init__`."""
    assert Plate("P1")
    assert Plate("1")
    assert Plate(0)
    with pytest.raises(NotImplementedError):
        Plate([0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unexpected Plate input string"):
        Plate("N1")
    with pytest.raises(ValueError, match="unexpected Plate input string"):
        Plate("1P")
    with pytest.raises(ValueError, match="negative Plate index"):
        Plate(-1)


def test_str() -> None:
    """Test :py:meth:`~layouts.Plate.__str__`."""
    assert str(Plate(0)) == "P1"


def test_eq() -> None:
    """Test :py:meth:`~layouts.Plate.__eq__`."""
    assert Plate("P1") == Plate("1") == Plate(0)
    assert Plate("P1") == "P1"
    assert Plate("1") == "1"
    assert Plate(0) == 0
    assert Plate(Plate(0)) == Plate(0)
    assert Plate(0) > -1


def test_gt() -> None:
    """Test :py:meth:`~layouts.Plate.__gt__`."""
    assert Plate("P3") > Plate("2") > Plate(0)
    assert Plate("P1") <= "P1"


def test_add() -> None:
    """Test :py:meth:`~layouts.Plate.__add__`."""
    assert Plate(0) + 1 == Plate(1)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = Plate("1") + Plate("1")  # type: ignore[operator]
    with pytest.raises(ValueError, match="negative Plate index"):
        _ = Plate("1") + (-1)


def test_sub() -> None:
    """Test :py:meth:`~layouts.Plate.__sub__`."""
    assert Plate("2") - 1 == Plate("1")
    assert Plate("3") - Plate("2") == 1
    with pytest.raises(ValueError, match="negative Plate index"):
        _ = Plate("1") - 1


def test_int() -> None:
    """Test :py:meth:`~layouts.Plate.__int__`."""
    assert int(Plate("1")) == 0


def test_index() -> None:
    """Test :py:meth:`~layouts.Plate.__index__`."""
    assert [1][Plate(0)]
