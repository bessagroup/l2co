import pytest

from l2co.example import add_one


def test_add_one():
    i = 3
    j = add_one(i)
    assert j == (i+1)


@pytest.mark.requires_dependency("abaqus")
def test_add_two():
    i = 3
    j = add_one(i)
    k = add_one(j)
    assert k == (i+2)

@pytest.mark.requires_dependency("pandas")
def test_add_three():
    i = 3
    j = add_one(i)
    k = add_one(j)
    m = add_one(k)
    assert m == (i+3)

if __name__ == "__main__":  # pragma: no cover
    pytest.main()