from cosmo import detect, profile


def test_box_inside_same_label_keeps_bigger():
    dets = [[100, 100, 150, 400, "trousers", 0.50],   # one leg
            [100, 100, 220, 400, "trousers", 0.45]]   # whole pair
    out = detect.clean_boxes(dets, 1000, 1000)
    assert len(out) == 1 and out[0]["box"] == (100, 100, 220, 400)


def test_outfit_box_does_not_swallow_pieces():
    dets = [[100, 100, 300, 300, "hoodie", 0.60],
            [100, 100, 300, 800, "jeans", 0.40],       # big box over the whole outfit
            [100, 400, 300, 800, "jeans", 0.55]]
    labels = sorted(d["label"] for d in detect.clean_boxes(dets, 1000, 1000))
    assert labels == ["hoodie", "jeans"]


def test_pair_of_shoes_merges_side_by_side_and_stacked():
    side = [[100, 800, 200, 900, "sneakers", 0.4], [220, 800, 320, 900, "sneakers", 0.4]]
    stacked = [[100, 500, 300, 600, "sneakers", 0.4], [100, 620, 300, 720, "sneakers", 0.4]]
    assert len(detect.clean_boxes(side, 1000, 1000)) == 1
    assert len(detect.clean_boxes(stacked, 1000, 1000)) == 1


def test_weak_and_tiny_boxes_dropped():
    dets = [[0, 0, 500, 500, "sweater", 0.12], [0, 0, 20, 20, "jewelry", 0.9]]
    assert detect.clean_boxes(dets, 1000, 1000) == []


def test_pairings_count_images_and_skip_same_category():
    image_of = [0, 0, 1, 1, 1]
    cats = ["sweater", "jeans", "sweater", "jeans", "sweater"]
    group_of = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2}   # pieces 0,2 share sweater group 0
    pairs = dict(profile.pairings(image_of, group_of, cats))
    assert pairs[(0, 1)] == 2      # sweater group 0 + jeans group 1 in both images
    assert (0, 2) not in pairs     # two sweater groups: not a pairing
