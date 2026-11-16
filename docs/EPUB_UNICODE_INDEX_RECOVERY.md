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
descriptive prose. No model, dependency, processing stage, retry allowance, or
publication threshold is added. Recovery resumes the original checkpoint and
still runs the normal final publication and quality checks.
