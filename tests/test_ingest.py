import io
import stat
import zipfile
from pathlib import Path

from PIL import Image

from cosmo import ingest


def _png_bytes(color, size=(64, 64), pattern=False):
    im = Image.new("RGB", size, color)
    if pattern:  # give phash some structure to work with
        for x in range(size[0] // 2):
            for y in range(size[1]):
                im.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            if isinstance(data, zipfile.ZipInfo):
                zf.writestr(data, b"/etc/passwd")
            else:
                zf.writestr(name, data)
    return buf.getvalue()


def test_rejects_path_traversal(tmp_path):
    z = _zip([("../evil.png", _png_bytes((1, 2, 3))), ("ok.png", _png_bytes((9, 9, 9)))])
    dest = tmp_path / "out"
    dest.mkdir()
    res = ingest.safe_extract(z, dest)
    assert res.rejected_unsafe == 1
    assert not (tmp_path / "evil.png").exists()
    assert (dest / "ok.png").exists()


def test_skips_symlinks_junk_and_non_images(tmp_path):
    link = zipfile.ZipInfo("link.png")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    z = _zip([
        ("link.png", link),
        ("__MACOSX/._a.png", b"junk"),
        (".hidden.png", b"junk"),
        ("notes.txt", b"hello"),
        ("a.png", _png_bytes((10, 20, 30))),
    ])
    res = ingest.safe_extract(z, tmp_path)
    assert res.skipped_other == 4
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["a.png"]


def test_corrupt_image_does_not_crash(tmp_path):
    z = _zip([("bad.jpg", b"not really a jpeg"), ("good.png", _png_bytes((5, 5, 5)))])
    res = ingest.safe_extract(z, tmp_path)
    ingest.load_images(tmp_path, res)
    assert res.unreadable == 1
    assert len(res.items) == 1


def test_bad_zip_raises_cleanly(tmp_path):
    import pytest
    with pytest.raises(zipfile.BadZipFile):
        ingest.safe_extract(b"definitely not a zip", tmp_path)


def test_removes_near_duplicates(tmp_path):
    a = Image.open(io.BytesIO(_png_bytes((200, 30, 30), (200, 200), pattern=True)))
    a.save(tmp_path / "a.png")
    a.resize((150, 150)).save(tmp_path / "a_small.jpg", quality=85)  # same picture, re-encoded
    b = Image.new("RGB", (200, 200), (30, 30, 200))
    for x in range(200):
        for y in range(0, 200, 40):
            for dy in range(20):
                b.putpixel((x, y + dy), (240, 240, 0))  # horizontal stripes
    b.save(tmp_path / "b.png")
    res = ingest.load_images(tmp_path)
    assert res.duplicates == 1
    assert len(res.items) == 2


def test_downscales_and_converts(tmp_path):
    Image.new("RGBA", (3000, 1500), (0, 0, 0, 0)).save(tmp_path / "big.png")
    res = ingest.load_images(tmp_path)
    im = res.items[0].image
    assert im.mode == "RGB"
    assert max(im.size) == 1024
