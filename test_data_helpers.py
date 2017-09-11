from data_helpers import RWBatchGenerator


def test_dw_batch_generator():
    walks = [[1, 2, 3, 4], [4, 5, 6]]

    g = RWBatchGenerator(walks, 2, 2, 1)  # 2 batches

    # iterate 3 rounds
    expected_batches = [{(2, 1), (2, 3)}, {(3, 2), (3, 4)}, {(5, 4), (5, 6)}] * 100
    for exp in expected_batches:
        batches, labels = g.next_batch()
        assert set(zip(batches, labels)) == exp

    g = RWBatchGenerator(walks, 8, 2, 1)  # 8 batches
    batches, labels = g.next_batch()
    assert set(zip(batches, labels)) == {(2, 1), (2, 3), (3, 2), (3, 4), (5, 4), (5, 6)}
    assert set(list(zip(batches, labels))[-2:]) == {(2, 1), (2, 3)}  # the last two loops back
