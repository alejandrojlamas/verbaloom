# EPUB fidelity recovery fixes

The Great Gatsby reached the end of its translation but retained two failed
XHTML units. Saved audit journals confirmed two different causes, rather than
a provider outage or lost translation checkpoint:

* A deterministic language check mistook comma-separated spelling examples
  for untranslated narrative. Two saved semantic assessments approved the
  exact text, but the local hard rejection still blocked publication.
* A repair corrected an unsupported kinship detail. A subsequent auditor
  rejected a different, unchanged passage that the previous adjudication had
  explicitly considered faithful. Conflict recovery required distinct changed
  renderings on both sides, so this evidence could not reach adjudication.

The existing bounded recovery exhausted its repair allowance and retained a
terminal journal. Replaying that same journal did not make progress. This was
not a reason to discard checkpoints, replenish request limits, or waive the
publication gate.

## Changes

`fidelity_supervisor.py` treats a narrow source-backed word-form enumeration
as contextual evidence only when the same individually comma-separated forms
occur in both texts, the source has a local linguistic list introduction, the
surrounding sentence is translated, and contextual auditing is enabled.
Explicit glossary translation requirements remain binding. Ordinary prose,
quoted sentences, architecture lists, source echoes, and unrelated neighboring
residue retain their existing checks. The existing exact-candidate recheck can
reuse a favorable saved judgment without another model request.

`epub/audit_repair.py` also recognizes conflicts concerning unchanged cited
evidence after a prior cited defect was edited. Both assessments must supply
grounded source/candidate excerpts and different objective findings. This only
permits the existing bounded adjudication: it does not approve a candidate.
The same decisive-pass, checksum, cumulative-attempt, and budget checks apply.

`epub/body_serializer.py` keeps paragraph-leading decorative initials and their
following smallcaps text in one editable text node, including meaningful
spaces. This prevents the isolated initial from being protected as markup and
left in the source language. Visible source text, anchors, smallcaps styling,
emphasis, and semantic attributes remain protected.

## Recovery and deployment

Source changes do not reload an already-running Python service. The observed
service predated the preceding installation, so deployment must include an
idle-state check, checkpoint backup, managed restart, and health verification.

Valid legacy checkpoint slots remain compatible. For this affected job, three
XHTML artifacts have independently verified malformed chapter openings; a
backup followed by targeted invalidation uses the existing missing-artifact
recovery path. Other completed chapters and the spelling-example checkpoint
remain reusable. The original job ID, source book, configured roles, and
cumulative cost accounting must be retained.

## Regression coverage

Tests cover old terminal journals, grounded and ungrounded conflicts, audit
rejections, low confidence, exhausted allowances, exact saved-language
rechecks, glossary obligations, genuine untranslated prose, and decorative
initial round-trips. Network-disabled replay of the real spelling-example
checkpoint approved both units with unchanged text, zero model requests, and
unchanged audit/repair counters. Synthetic tests contain no book excerpts or
credentials.

The real recovery subsequently exposed the same lexical-example false
positive in the independently configured publication gate. All 1,785 source
units were present, but one mixed-language flag on the nine-block editorial
note excluded that whole group from audited coverage. This was not evidence
of nine missing passages. Publication must classify only the exact proven
literal spans and continue checking every surrounding passage; a whole-file
exemption or a fabricated successful semantic review would be incorrect.

The subsequent whole-book entity check identified a separate finalization
defect: sentence-spacing cleanup inserted a space inside a mixed-case visible
URL after `www.`. The source and saved translated XHTML both contained the
correct URL; the link destination was unchanged. Final inline hygiene now
protects exact URL, email, DOI and explicit path spans during spacing repair,
while processing surrounding prose and removing source artifacts as before.
Tests reproduce corruption of each literal type, mixed-case EPUB anchor labels,
and ordinary sentence repairs. Reassembly from intact translated checkpoints
can therefore correct the final artifact without retranslating those chapters.
The publication and universal typography validators use the same exact-span
proof from `technical_identifiers.py`; intact URL punctuation is not a spacing
error, while joined prose immediately before or after it remains a failure.
Entity comparison still rejects a modified URL. No paragraph or quality gate
is exempted, and literal spans are recalculated after every formatter pass.

The optional download diagnostic also misread a hyphenated, elongated word
ending in `-os` as the standalone regional pronoun. Its shared locale matcher
now treats hyphen-linked syllables as part of a word while retaining ordinary
pronouns after dialogue dashes, spaces, and punctuation. This only corrects
diagnostic token boundaries; it does not alter the published book.
