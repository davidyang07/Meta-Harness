from geometry import Line, Point


def test_point_origin():
    p = Point(0, 0)
    assert p.x == 0 and p.y == 0


def test_point_distance_3_4_5():
    assert Point(0, 0).distance_to(Point(3, 4)) == 5.0


def test_point_equality():
    assert Point(1, 2) == Point(1, 2)
    assert Point(1, 2) != Point(2, 1)


def test_line_length():
    assert Line(Point(0, 0), Point(3, 4)).length() == 5.0


def test_line_contains_endpoint_start():
    line = Line(Point(0, 0), Point(2, 2))
    assert line.contains(Point(0, 0))


def test_line_contains_endpoint_end():
    line = Line(Point(0, 0), Point(2, 2))
    assert line.contains(Point(2, 2))


def test_line_contains_midpoint():
    line = Line(Point(0, 0), Point(4, 4))
    assert line.contains(Point(2, 2))


def test_line_does_not_contain_external():
    line = Line(Point(0, 0), Point(2, 0))
    assert not line.contains(Point(1, 1))


def test_line_does_not_contain_extension():
    """Past the endpoint, on the same infinite line — must be False."""
    line = Line(Point(0, 0), Point(2, 0))
    assert not line.contains(Point(3, 0))


def test_line_contains_uses_floating_point_tolerance():
    """A point within the README's 1e-6 tolerance of the segment is on it."""
    line = Line(Point(0, 0), Point(2, 0))
    near_point = Point(1.0, 1e-7)
    assert line.contains(near_point), (
        f"Point({near_point.x}, {near_point.y}) should be on the segment."
    )


def test_line_contains_zero_length_segment():
    """A segment whose start and end coincide contains that point only."""
    degenerate = Line(Point(1, 1), Point(1, 1))
    assert degenerate.contains(Point(1, 1)), (
        "A zero-length segment contains its (single) endpoint."
    )
    assert not degenerate.contains(Point(2, 2)), (
        "A zero-length segment at (1,1) does NOT contain other points."
    )
