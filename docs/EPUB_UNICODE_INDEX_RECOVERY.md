# Recovering Unicode index identities

A recoverable EPUB stopped in its name index after saving 486 of 510 chunks
and all 307 other XHTML resources. The provider responded successfully, but
the language gate repeatedly rejected an unchanged name entry. Automatic
recovery exhausted its existing bounded attempts without losing the checkpoint.

The structural classifier recognized only ASCII letters and Spanish accented
letters. It split `François` into `Fran` and `ois`, treating the latter as
ordinary untranslated prose. This prevented the existing identity-only index
path from preserving a legitimate name and page locator. The same issue
affected other non-Spanish diacritics; an individual-name exception would not
solve it.

The index word classifier now recognizes Unicode letters and normalizes a
temporary classification value to NFC. It never normalizes or rewrites the
exported source. Existing requirements for index context, capitalization,
locators, qualifiers, and descriptive text remain unchanged. Uncased scripts
do not receive a new unconditional identity exemption. Normal translation,
structural recovery, and publication metadata classification reuse these index
rules. Bibliography, title, formula, glossary, and watermark classification keep
their existing tokenization; broadening those policies is outside this fix.

Tests cover composed and decomposed names, exact output preservation, recovery
of a previously failed unit without a provider call, and rejection of ordinary
descriptive prose. No model, dependency, processing stage, or publication
threshold is added. Recovery resumes the original checkpoint and still runs the
normal final publication and quality checks.

## Mixed-language qualifier validation

The first live recovery completed the remaining translation chunks and exposed
a second, independent conflict in the final audit. The auditor correctly asked
for translated honorifics and descriptive qualifiers, but the local language
gate could reject their translations because most words in an index are names.
Several generic honorifics were absent from the gate's existing qualifier set.
Country descriptions introduced by ranks and determiner-led epithets also
needed structural treatment rather than an unbounded dictionary of exonyms.

The language check now recognizes these bounded qualifier edits in index
entries while retaining the leading identity, other name tokens, entry order,
and numeric locators. The existing semantic auditor remains responsible for
meaning and completeness. Actual source prose, altered identity anchors,
changed numbers, and omitted entries do not receive this language exemption.

A saved favorable semantic assessment can be rechecked against the corrected
local rule only for the same source and candidate hashes, with no factual or
structural objections. A negative semantic assessment cannot become an
approval through this path.

The legacy compatible repair path did not retain rejected repair responses.
Those responses cannot be replayed or reconstructed. For the affected job only,
the existing `epub_audit_repair_rounds` resume option can be raised from two to
four, retaining lifetime attempt counts, cached decisions, completed work, and
the monetary budget. This is an explicit bounded recovery operation, not a new
global retry policy. Regression coverage verifies that previously approved
units stay cached and every newly generated repair still requires audit.
