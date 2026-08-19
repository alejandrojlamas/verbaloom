/**
 * Single source of truth for the Sample & Quick-test sampling defaults.
 *
 * Both the Sample tab (its #sampleNSamples / #sampleMaxChars inputs are seeded
 * from here) and the Translate-tab quick test read these, so there is exactly
 * one place to change the default. The backend mirrors the same values in
 * sample_routes.py (DEFAULT_N_SAMPLES / DEFAULT_MAX_CHARS) as a safety net for
 * requests that omit the fields — keep the two in sync.
 */
export const SAMPLE_DEFAULT_N_SAMPLES = 5;
export const SAMPLE_DEFAULT_MAX_CHARS = 400;
