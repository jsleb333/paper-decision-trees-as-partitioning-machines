from partitioning_machines.utils.special import wedderburn_etherington


def test_wedderburn_etherington():
    assert wedderburn_etherington(0) == 0
    assert wedderburn_etherington(1) == 1
    assert wedderburn_etherington(2) == 1
    answers = [0, 1, 1, 1, 2, 3, 6, 11, 23, 46, 98, 207, 451, 983, 2179, 4850, 10905, 24631, 56011]
    assert [wedderburn_etherington(i) for i in range(len(answers))] == answers
    assert wedderburn_etherington(40)
