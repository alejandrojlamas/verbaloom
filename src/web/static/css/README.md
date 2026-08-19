# Frontend CSS Structure

The frontend keeps legacy classes and IDs because the JavaScript modules bind to
them. New UI work should use the internal design-system contracts first and add
feature-specific CSS only when a screen needs it.

- `design-system.css`: generic tokens and reusable components. It must not
  contain feature IDs such as `#glossary...` or `#profilePrep...`.
- `app-components.css`: page-specific rules extracted from
  `translation_interface.html`. This file can reference concrete IDs when the
  template needs a layout fix.
- `../style.css`: legacy compatibility and the current visual theme. Avoid
  growing it for new feature work.

Preferred component classes:

- Buttons: `.ds-button`, `.ds-button--primary`, `.ds-button--secondary`,
  `.ds-button--danger`, `.ds-icon-button`
- Tabs: `.ds-tabs`, `.ds-tab`
- Cards: `.ds-card`, `.ds-card__header`, `.ds-card__body`
- Progress: `.ds-progress`, `.ds-progress__bar`
- Tables: `.ds-table`, `.ds-table--fixed`
- Modals/sheets: `.ds-modal`, `.ds-modal__panel`, `.ds-mobile-sheet`

Existing classes like `.btn`, `.tab-nav`, `.main-card`, `.progress-bar-container`,
`.file-table`, and `.modal-overlay` are bridged in `design-system.css` so older
markup keeps working while new markup can use the component vocabulary directly.
