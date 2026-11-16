# Recovering paragraph boundaries without rewriting translation

An EPUB completed all 160 translation units but remained pending publication.
The final DOM gate found three unusually short paragraphs in one chapter.
Comparison with the source and saved XHTML established that the affected
passages were present: a dialogue had been split, subsequent paragraph contents
shifted, and the final two passages were joined. The saved checkpoint already
contained that distribution; final packaging did not introduce it.

The existing alignment repair found the correct eight-paragraph region but
could not accept even source-grounded, unique anchors. One paragraph began
immediately after terminal punctuation without an intervening space. The
anchor parser required whitespace before every start. Separating that passage
also added a paragraph separator to the flattened comparison, which rejected
the otherwise lossless repair.

Both limitations were reproduced offline. Production logs recorded two rejected
alignment attempts, but did not retain their responses; the exact responses
must not be inferred from this reproduction. Retained records also do not prove
whether translation, refinement, or a prior semantic repair first shifted the
paragraph boundaries.

The correction belongs in the existing alignment repair. Paragraph-boundary
evidence must remain distinct from inline emphasis boundaries. Unique ordered
anchors, exact content conservation, source structure, cache fingerprints, and
bounded requests remain mandatory. No proportional redistribution, new
translation stage, threshold relaxation, or unconditional publication is used.

Recovery reassembles the existing translated checkpoints and reruns publication
and whole-book quality validation. Accepted alignment anchors remain durable
and tied to the original source, region, and exact repaired result so a later
rebuild can reuse them without another model request.
