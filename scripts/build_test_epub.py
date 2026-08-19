"""
Build a canonical test EPUB ("The Translator's Sampler") exercising the
features the translation pipeline must handle on real EPUBs.

The book is deliberately tiny — about ten 450-token chunks total — and is
built only with the Python standard library (no Pillow/ebooklib). All
literary content is public-domain English text (Carroll, Poe, Aesop, Bacon,
Shakespeare) plus a short editorial framing written for this fixture.

Features exercised
------------------
- EPUB 3 package + legacy NCX
- Cover image (real PNG, generated programmatically)
- Inline figure with <figure>/<img>/<figcaption>
- Multiple XHTML spine items: cover, title, foreword, 6 chapters, glossary
- Rich inline formatting: <em>, <strong>, <i>, <q>, <span class="dropcap">
- Block formatting: <blockquote>, ordered + unordered lists
- Poetry: stanza/line break markup (<p class="verse">, <br/>)
- Dialogue with curly quotes and em-dashes
- Footnote with intra-document link (epub:type="noteref" / "footnote")
- Foreign-language span (xml:lang) — a Latin tag
- Definition list glossary (<dl>/<dt>/<dd>) of recurring proper nouns
- Cross-references between chapters (anchor links)

Run
---
    python scripts/build_test_epub.py
    python scripts/build_test_epub.py --out path/to/sampler.epub
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# PNG generation (stdlib only — no Pillow)
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _make_png(width: int, height: int, pixel_fn) -> bytes:
    """Build a minimal 8-bit RGB PNG from a (x, y) -> (r, g, b) function."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: None
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            raw.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    iend = _png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def make_cover_png() -> bytes:
    """A soft vertical gradient with a darker band — readable on any reader."""
    w, h = 600, 900

    def pixel(x, y):
        t = y / (h - 1)
        # Dusk gradient: warm cream at top -> deep plum at bottom
        r = int(245 * (1 - t) + 60 * t)
        g = int(231 * (1 - t) + 30 * t)
        b = int(200 * (1 - t) + 80 * t)
        # Darker spine band on the left for character
        if x < 40:
            r, g, b = int(r * 0.55), int(g * 0.55), int(b * 0.6)
        return r, g, b

    return _make_png(w, h, pixel)


def make_illustration_png() -> bytes:
    """A small abstract motif used to test inline <figure> handling."""
    w, h = 320, 200

    def pixel(x, y):
        # Concentric soft rings on cream background
        cx, cy = w / 2, h / 2
        dx, dy = x - cx, y - cy
        d = (dx * dx + dy * dy) ** 0.5
        ring = (int(d) // 18) % 2
        if ring:
            return 90, 70, 110
        return 240, 232, 215

    return _make_png(w, h, pixel)


# ---------------------------------------------------------------------------
# Static EPUB content
# ---------------------------------------------------------------------------

MIMETYPE = b"application/epub+zip"

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

CSS = """@charset "utf-8";
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.5; margin: 1em 1.2em; }
h1, h2, h3 { font-family: "Palatino Linotype", Palatino, Georgia, serif; }
h1 { font-size: 1.6em; margin-top: 1.2em; }
h2 { font-size: 1.25em; }
p  { text-indent: 1.2em; margin: 0 0 0.4em 0; }
p.noindent, p.verse, p.attribution { text-indent: 0; }
p.verse { margin: 0; }
.stanza { margin: 0 0 0.9em 1.5em; }
.dropcap { font-size: 2.6em; line-height: 0.9; float: left; padding: 0.05em 0.08em 0 0; font-weight: bold; }
blockquote { margin: 0.6em 2em; font-style: italic; }
figure { text-align: center; margin: 1em 0; }
figcaption { font-size: 0.9em; font-style: italic; }
dl dt { font-weight: bold; margin-top: 0.6em; }
dl dd { margin: 0 0 0.4em 1.2em; }
.cover { text-align: center; margin: 0; padding: 0; }
.cover img { max-width: 100%; height: auto; }
a.noteref { vertical-align: super; font-size: 0.75em; text-decoration: none; }
hr.scene { border: 0; text-align: center; margin: 1em 0; }
hr.scene:after { content: "\\002042"; font-size: 1.2em; }
"""


def xhtml(title: str, body: str, *, lang: str = "en") -> str:
    """Wrap a body fragment in a well-formed XHTML5 (EPUB 3) document."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops" '
        f'xml:lang="{lang}" lang="{lang}">\n'
        '<head>\n'
        '  <meta charset="utf-8"/>\n'
        f'  <title>{title}</title>\n'
        '  <link rel="stylesheet" type="text/css" href="styles/main.css"/>\n'
        '</head>\n'
        f'<body>\n{body}\n</body>\n</html>\n'
    )


# ---------------------------------------------------------------------------
# Spine documents — public-domain literary content
# ---------------------------------------------------------------------------

COVER_XHTML = xhtml(
    "Cover",
    '<section epub:type="cover" class="cover">\n'
    '  <img src="images/cover.png" alt="Cover of The Translator’s Sampler"/>\n'
    '</section>',
)

TITLE_XHTML = xhtml(
    "Title Page",
    """<section epub:type="titlepage">
  <h1>The Translator&#8217;s Sampler</h1>
  <p class="attribution"><em>A miniature anthology assembled for testing</em></p>
  <p class="attribution">Edited by the Test Fixture Department</p>
  <hr class="scene"/>
  <p class="noindent">This volume gathers six short pieces in the public domain,
  arranged so that a translator&#8212;human or machine&#8212;may rehearse
  the small troubles of prose, verse, dialogue, and citation in the span
  of a single afternoon.</p>
</section>""",
)

FOREWORD_XHTML = xhtml(
    "Foreword",
    """<section epub:type="foreword">
  <h1>Foreword</h1>
  <p><span class="dropcap">A</span> good translation, it has been said,
  is like a pane of glass: one notices it only when it is flawed. The
  pieces collected here were chosen for the variety of glasswork they
  demand. The reader will find a child&#8217;s reverie, a dirge, a
  fable, an essay of counsel, a sonnet, and an ekphrastic glance at an
  ornament&#8212;each making its own small claim on the translator.</p>
  <p>Where the originals show their age, we have left their spellings
  and punctuation as we found them. A short <a href="glossary.xhtml">glossary</a>
  closes the book.</p>
  <p class="attribution"><em>&#8212; The Editors</em></p>
</section>""",
)

# Lewis Carroll, Alice's Adventures in Wonderland (1865), opening of Ch. I.
CH01_XHTML = xhtml(
    "I. Down the Rabbit-Hole",
    """<section epub:type="chapter">
  <h1>I. Down the Rabbit-Hole</h1>
  <p class="noindent"><span class="dropcap">A</span>lice was beginning
  to get very tired of sitting by her sister on the bank, and of having
  nothing to do: once or twice she had peeped into the book her sister
  was reading, but it had no pictures or conversations in it,
  <q>and what is the use of a book,</q> thought Alice,
  <q>without pictures or conversation?</q></p>

  <p>So she was considering in her own mind (as well as she could,
  for the hot day made her feel very sleepy and stupid) whether the
  pleasure of making a daisy-chain would be worth the trouble of getting
  up and picking the daisies, when suddenly a <strong>White Rabbit</strong>
  with pink eyes ran close by her.</p>

  <p>There was nothing so very remarkable in that; nor did Alice
  think it so very much out of the way to hear the Rabbit say to itself,
  <q>Oh dear! Oh dear! I shall be late!</q> (when she thought it over
  afterwards, it occurred to her that she ought to have wondered at
  this, but at the time it all seemed quite natural); but when the
  Rabbit actually <em>took a watch out of its waistcoat-pocket</em>,
  and looked at it, and then hurried on, Alice started to her feet, for
  it flashed across her mind that she had never before seen a rabbit
  with either a waistcoat-pocket, or a watch to take out of it, and,
  burning with curiosity, she ran across the field after it, and was
  just in time to see it pop down a large rabbit-hole under the hedge.</p>

  <p>In another moment down went Alice after it, never once considering
  how in the world she was to get out again.</p>

  <hr class="scene"/>

  <p>The rabbit-hole went straight on like a tunnel for some way, and
  then dipped suddenly down, so suddenly that Alice had not a moment to
  think about stopping herself before she found herself falling down what
  seemed to be a very deep well.</p>

  <p>Either the well was very deep, or she fell very slowly, for she
  had plenty of time as she went down to look about her, and to wonder
  what was going to happen next. First, she tried to look down and make
  out what she was coming to, but it was too dark to see anything; then
  she looked at the sides of the well, and noticed that they were filled
  with cupboards and book-shelves: here and there she saw maps and
  pictures hung upon pegs. She took down a jar from one of the shelves
  as she passed; it was labelled <em>&#8220;ORANGE MARMALADE&#8221;</em>,
  but to her great disappointment it was empty: she did not like to drop
  the jar for fear of killing somebody underneath, so managed to put it
  into one of the cupboards as she fell past it.</p>

  <p><q>Well!</q> thought Alice to herself. <q>After such a fall as this,
  I shall think nothing of tumbling down stairs! How brave they&#8217;ll
  all think me at home! Why, I wouldn&#8217;t say anything about it, even
  if I fell off the top of the house!</q> (Which was very likely true.)</p>
</section>""",
)

# Edgar Allan Poe, "A Dream Within a Dream" (1849), in full.
CH02_XHTML = xhtml(
    "II. A Dream Within a Dream",
    """<section epub:type="chapter">
  <h1>II. A Dream Within a Dream</h1>
  <p class="attribution">by Edgar Allan Poe</p>

  <div class="stanza">
    <p class="verse">Take this kiss upon the brow!</p>
    <p class="verse">And, in parting from you now,</p>
    <p class="verse">Thus much let me avow&#8212;</p>
    <p class="verse">You are not wrong, who deem</p>
    <p class="verse">That my days have been a dream;</p>
    <p class="verse">Yet if hope has flown away</p>
    <p class="verse">In a night, or in a day,</p>
    <p class="verse">In a vision, or in none,</p>
    <p class="verse">Is it therefore the less <em>gone</em>?</p>
    <p class="verse"><strong>All</strong> that we see or seem</p>
    <p class="verse">Is but a dream within a dream.</p>
  </div>

  <div class="stanza">
    <p class="verse">I stand amid the roar</p>
    <p class="verse">Of a surf-tormented shore,</p>
    <p class="verse">And I hold within my hand</p>
    <p class="verse">Grains of the golden sand&#8212;</p>
    <p class="verse">How few! yet how they creep</p>
    <p class="verse">Through my fingers to the deep,</p>
    <p class="verse">While I weep&#8212;while I weep!</p>
    <p class="verse">O God! can I not grasp</p>
    <p class="verse">Them with a tighter clasp?</p>
    <p class="verse">O God! can I not save</p>
    <p class="verse">One from the pitiless wave?</p>
    <p class="verse">Is <em>all</em> that we see or seem</p>
    <p class="verse">But a dream within a dream?</p>
  </div>
</section>""",
)

# Aesop, "The Fox and the Grapes" (V. S. Vernon Jones translation, 1912 — PD).
CH03_XHTML = xhtml(
    "III. The Fox and the Grapes",
    """<section epub:type="chapter">
  <h1>III. The Fox and the Grapes</h1>
  <p class="attribution">From Aesop, after V. S. Vernon Jones</p>

  <p>A hungry Fox saw some fine bunches of grapes hanging from a vine
  that was trained along a high trellis, and did his best to reach them
  by jumping as high as he could into the air. But it was all in vain,
  for they were just out of reach: so he gave up trying, and walked away
  with an air of dignity and unconcern, remarking,
  <q>I thought those grapes were ripe, but I see now they are quite sour.</q></p>

  <blockquote>
    <p><strong>Moral.</strong> It is easy to despise what you cannot get.</p>
  </blockquote>

  <h2>A reader&#8217;s aside</h2>
  <p>The fable rewards close reading on three counts:</p>
  <ol>
    <li>The Fox <em>narrates his own failure</em> in the present tense
    of pretence&#8212;a rhetorical move worth preserving in any
    translation.</li>
    <li>The trellis is not incidental: it places the grapes within sight
    and out of reach, the very geometry of envy.</li>
    <li>The closing maxim was, in Latin, often given as
    <span xml:lang="la"><em>acerba uva</em></span>&#8212;literally,
    &#8220;the bitter grape.&#8221;</li>
  </ol>
  <p>Other animals make similar arguments throughout the corpus:</p>
  <ul>
    <li>The Wolf and the Crane</li>
    <li>The Dog in the Manger</li>
    <li>The Lion&#8217;s Share</li>
  </ul>
</section>""",
)

# Francis Bacon, "Of Studies" (1597) — slightly abridged. PD.
CH04_XHTML = xhtml(
    "IV. Of Studies",
    """<section epub:type="chapter">
  <h1>IV. Of Studies</h1>
  <p class="attribution">by Sir Francis Bacon, 1597</p>

  <p><span class="dropcap">S</span>tudies serve for delight, for
  ornament, and for ability. Their chief use for delight, is in
  privateness and retiring; for ornament, is in discourse; and for
  ability, is in the judgment and disposition of business. For
  expert men can execute, and perhaps judge of particulars, one by
  one; but the general counsels, and the plots and marshalling of
  affairs, come best from those that are learned.<a class="noteref"
  epub:type="noteref" href="#fn1" id="fn1ref">1</a></p>

  <p>To spend too much time in studies is sloth; to use them too
  much for ornament, is affectation; to make judgment wholly by
  their rules, is the humour of a scholar. They perfect nature, and
  are perfected by experience: for natural abilities are like
  natural plants, that need pruning by study; and studies themselves
  do give forth directions too much at large, except they be bounded
  in by experience.</p>

  <p>Crafty men <em>contemn</em> studies, simple men <em>admire</em>
  them, and wise men <em>use</em> them; for they teach not their own
  use; but that is a wisdom without them, and above them, won by
  observation. Read not to contradict and confute; nor to believe
  and take for granted; nor to find talk and discourse; but to
  weigh and consider.</p>

  <blockquote>
    <p>Some books are to be tasted, others to be swallowed, and some
    few to be chewed and digested; that is, some books are to be read
    only in parts; others to be read, but not curiously; and some few
    to be read wholly, and with diligence and attention.</p>
  </blockquote>

  <p>Reading maketh a full man; conference a ready man; and writing
  an exact man. And therefore, if a man write little, he had need
  have a great memory; if he confer little, he had need have a
  present wit; and if he read little, he had need have much cunning,
  to seem to know that he doth not.</p>

  <p>Histories make men wise; poets witty; the mathematics subtile;
  natural philosophy deep; moral grave; logic and rhetoric able to
  contend. <em>Abeunt studia in mores</em><a class="noteref"
  epub:type="noteref" href="#fn2" id="fn2ref">2</a>&#8212;studies pass
  into character. Nay, there is no stond or impediment in the wit, but
  may be wrought out by fit studies: like as diseases of the body
  may have appropriate exercises.</p>

  <aside epub:type="footnote" id="fn1">
    <p><a href="#fn1ref">1.</a> Bacon glosses three uses of study:
    private pleasure, public speech, and the conduct of affairs.</p>
  </aside>
  <aside epub:type="footnote" id="fn2">
    <p><a href="#fn2ref">2.</a> <span xml:lang="la"><em>Abeunt studia
    in mores</em></span>&#8212;Ovid, <em>Heroides</em> XV.</p>
  </aside>
</section>""",
)

# Shakespeare, Sonnet XVIII (1609). PD.
CH05_XHTML = xhtml(
    "V. Sonnet XVIII",
    """<section epub:type="chapter">
  <h1>V. Sonnet XVIII</h1>
  <p class="attribution">by William Shakespeare</p>

  <div class="stanza">
    <p class="verse">Shall I compare thee to a summer&#8217;s day?</p>
    <p class="verse">Thou art more lovely and more temperate:</p>
    <p class="verse">Rough winds do shake the darling buds of May,</p>
    <p class="verse">And summer&#8217;s lease hath all too short a date:</p>
    <p class="verse">Sometime too hot the eye of heaven shines,</p>
    <p class="verse">And often is his gold complexion dimm&#8217;d;</p>
    <p class="verse">And every fair from fair sometime declines,</p>
    <p class="verse">By chance, or nature&#8217;s changing course, untrimm&#8217;d;</p>
    <p class="verse">But thy eternal summer shall not fade,</p>
    <p class="verse">Nor lose possession of that fair thou ow&#8217;st;</p>
    <p class="verse">Nor shall Death brag thou wander&#8217;st in his shade,</p>
    <p class="verse">When in eternal lines to time thou grow&#8217;st:</p>
    <p class="verse" style="margin-left:1.5em;">So long as men can breathe, or eyes can see,</p>
    <p class="verse" style="margin-left:1.5em;">So long lives this, and this gives life to thee.</p>
  </div>
</section>""",
)

# Original short chapter built around the illustration. Written for this fixture.
CH06_XHTML = xhtml(
    "VI. A Small Ornament",
    """<section epub:type="chapter">
  <h1>VI. A Small Ornament</h1>
  <p><span class="dropcap">T</span>he editors close the volume with a
  device borrowed from the printer&#8217;s tray: a pattern of rings,
  set on cream, of the kind that once adorned the colophons of
  provincial presses. It bears no meaning of its own. It is offered
  here as a final test for the translator&#8217;s eye&#8212;to see
  whether the caption travels alongside the figure, and whether the
  alternative text survives the crossing.</p>

  <figure>
    <img src="images/illustration.png"
         alt="Concentric rings on a cream ground &#8212; a printer&#8217;s ornament."/>
    <figcaption><em>Ornament.</em> Pattern of concentric rings, after a
    nineteenth-century colophon.</figcaption>
  </figure>

  <p>For the recurring proper nouns of this anthology, the reader is
  referred to the <a href="glossary.xhtml">glossary</a>. For the
  opening fall, return to <a href="ch01.xhtml">Chapter I</a>.</p>
</section>""",
)

GLOSSARY_XHTML = xhtml(
    "Glossary",
    """<section epub:type="glossary">
  <h1>Glossary</h1>
  <p class="noindent">A short index of recurring proper nouns and
  technical terms, given here so the translator may keep them
  consistent across the book.</p>

  <dl>
    <dt>Alice</dt>
    <dd>The child protagonist of <em>Alice&#8217;s Adventures in
    Wonderland</em>. The name should be left unchanged in most
    target languages.</dd>

    <dt>White Rabbit</dt>
    <dd>The waistcoated rabbit in Chapter <a href="ch01.xhtml">I</a>.
    Capitalize as a character name.</dd>

    <dt>Trellis</dt>
    <dd>The latticework that holds up the grape-vine in Chapter
    <a href="ch03.xhtml">III</a>.</dd>

    <dt>Marmalade</dt>
    <dd>British orange preserve. Preserve the brand-like
    typography (<em>&#8220;ORANGE MARMALADE&#8221;</em>) where it
    appears on a jar.</dd>

    <dt>Sonnet</dt>
    <dd>A fourteen-line poem in iambic pentameter; here, in the
    English (Shakespearean) form, with a final couplet.</dd>

    <dt>Colophon</dt>
    <dd>The printer&#8217;s emblem at the end of a book; see
    Chapter <a href="ch06.xhtml">VI</a>.</dd>

    <dt><span xml:lang="la">Abeunt studia in mores</span></dt>
    <dd>Latin: <em>studies pass into character</em>. From Ovid,
    quoted by Bacon. Keep the Latin in italics and provide a gloss
    on first occurrence.</dd>

    <dt>Ekphrasis</dt>
    <dd>A verbal description of a visual artwork. Chapter
    <a href="ch06.xhtml">VI</a> is a miniature instance.</dd>
  </dl>
</section>""",
)


# ---------------------------------------------------------------------------
# Manifest / spine / nav
# ---------------------------------------------------------------------------

SPINE = [
    # (id,        href,             title,                    nav?)
    ("cover",     "cover.xhtml",    "Cover",                  False),
    ("title",     "title.xhtml",    "Title Page",             False),
    ("foreword",  "foreword.xhtml", "Foreword",               True),
    ("ch01",      "ch01.xhtml",     "I. Down the Rabbit-Hole", True),
    ("ch02",      "ch02.xhtml",     "II. A Dream Within a Dream", True),
    ("ch03",      "ch03.xhtml",     "III. The Fox and the Grapes", True),
    ("ch04",      "ch04.xhtml",     "IV. Of Studies",         True),
    ("ch05",      "ch05.xhtml",     "V. Sonnet XVIII",        True),
    ("ch06",      "ch06.xhtml",     "VI. A Small Ornament",   True),
    ("glossary",  "glossary.xhtml", "Glossary",               True),
]

BOOK_ID = "urn:uuid:6f3b1e9a-translator-sampler-fixture-v1"
TITLE = "The Translator's Sampler"
AUTHOR = "Various (public domain)"
LANG = "en"


def build_opf() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest_items = [
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '    <item id="css" href="styles/main.css" media-type="text/css"/>',
        '    <item id="cover-img" href="images/cover.png" media-type="image/png" properties="cover-image"/>',
        '    <item id="illus" href="images/illustration.png" media-type="image/png"/>',
    ]
    for sid, href, _title, _in_nav in SPINE:
        manifest_items.append(
            f'    <item id="{sid}" href="{href}" media-type="application/xhtml+xml"/>'
        )

    spine_items = "\n".join(
        f'    <itemref idref="{sid}"/>' for sid, _, _, _ in SPINE
    )

    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0"
         unique-identifier="bookid" xml:lang="{LANG}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{BOOK_ID}</dc:identifier>
    <dc:title>{TITLE}</dc:title>
    <dc:creator>{AUTHOR}</dc:creator>
    <dc:language>{LANG}</dc:language>
    <dc:date>{now}</dc:date>
    <dc:description>A miniature, public-domain anthology assembled to exercise
    every feature an EPUB translator must handle: chapters, formatting,
    poetry, dialogue, footnotes, images, and a glossary.</dc:description>
    <meta property="dcterms:modified">{now}</meta>
    <meta name="cover" content="cover-img"/>
  </metadata>
  <manifest>
{chr(10).join(manifest_items)}
  </manifest>
  <spine toc="ncx">
{spine_items}
  </spine>
</package>
"""


def build_nav() -> str:
    items = "\n".join(
        f'      <li><a href="{href}">{title}</a></li>'
        for _sid, href, title, in_nav in SPINE
        if in_nav
    )
    body = f"""<nav epub:type="toc" id="toc">
  <h1>Table of Contents</h1>
  <ol>
{items}
  </ol>
</nav>"""
    return xhtml("Table of Contents", body)


def build_ncx() -> str:
    nav_points = []
    play_order = 1
    for sid, href, title, in_nav in SPINE:
        if not in_nav:
            continue
        nav_points.append(
            f"""    <navPoint id="nav-{sid}" playOrder="{play_order}">
      <navLabel><text>{title}</text></navLabel>
      <content src="{href}"/>
    </navPoint>"""
        )
        play_order += 1
    return f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="{BOOK_ID}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{TITLE}</text></docTitle>
  <navMap>
{chr(10).join(nav_points)}
  </navMap>
</ncx>
"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

SPINE_DOCS = {
    "cover.xhtml":    COVER_XHTML,
    "title.xhtml":    TITLE_XHTML,
    "foreword.xhtml": FOREWORD_XHTML,
    "ch01.xhtml":     CH01_XHTML,
    "ch02.xhtml":     CH02_XHTML,
    "ch03.xhtml":     CH03_XHTML,
    "ch04.xhtml":     CH04_XHTML,
    "ch05.xhtml":     CH05_XHTML,
    "ch06.xhtml":     CH06_XHTML,
    "glossary.xhtml": GLOSSARY_XHTML,
}


def build_epub(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w") as z:
        # mimetype must be the first entry, stored (uncompressed), no extra fields.
        zi = zipfile.ZipInfo("mimetype")
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, MIMETYPE)

        def write(name: str, data: bytes | str) -> None:
            if isinstance(data, str):
                data = data.encode("utf-8")
            z.writestr(
                zipfile.ZipInfo(name),
                data,
                compress_type=zipfile.ZIP_DEFLATED,
            )

        write("META-INF/container.xml", CONTAINER_XML)
        write("OEBPS/content.opf", build_opf())
        write("OEBPS/nav.xhtml", build_nav())
        write("OEBPS/toc.ncx", build_ncx())
        write("OEBPS/styles/main.css", CSS)
        write("OEBPS/images/cover.png", make_cover_png())
        write("OEBPS/images/illustration.png", make_illustration_png())
        for name, doc in SPINE_DOCS.items():
            write(f"OEBPS/{name}", doc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/translation_sampler.epub"),
        help="Output EPUB path (default: tests/fixtures/translation_sampler.epub)",
    )
    args = parser.parse_args()

    build_epub(args.out)

    size_kb = args.out.stat().st_size / 1024
    words = sum(len(d.split()) for d in SPINE_DOCS.values())
    print(f"Wrote {args.out} ({size_kb:.1f} KB, ~{words} XHTML words across {len(SPINE_DOCS)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
