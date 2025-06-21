import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scraper_wiki as sw


def test_extract_images_basic():
    html = (
        "<div class='thumb'>"
        "<img src='//upload.wikimedia.org/img.jpg'/>"
        "<div class='thumbcaption'>Cap</div>"
        "</div>"
    )
    res = sw.extract_images(html)
    assert res == [{"image_url": "https://upload.wikimedia.org/img.jpg", "caption": "Cap"}]


def test_extract_images_no_caption():
    html = "<figure><img src='pic.png'/></figure>"
    res = sw.extract_images(html)
    assert res == [{"image_url": "pic.png", "caption": ""}]
