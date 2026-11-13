/** Per-tab request ownership; never stores credentials, prompts or book text. */
const KEY = 'verbaloom.pending-submission.v1';
const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export const SubmissionReceipt = {
    read() {
        const raw = sessionStorage.getItem(KEY);
        if (!raw) return null;
        const receipt = JSON.parse(raw);
        if (!ID.test(receipt.submissionId) || !receipt.file || typeof receipt.file.name !== 'string') {
            throw new Error('Invalid saved submission receipt');
        }
        return receipt;
    },
    begin(file) {
        const bytes = crypto.getRandomValues(new Uint8Array(16));
        bytes[6] = (bytes[6] & 15) | 64;
        bytes[8] = (bytes[8] & 63) | 128;
        const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
        const submissionId = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
        const receipt = { submissionId, translationId: `trans_${hex}`, cancelRequested: false,
            file: Object.fromEntries(['name', 'fileType', 'operation', 'transformMode', 'transformLabel',
                'sourceLanguage', 'targetLanguage'].map(key => [key, file[key] || ''])) };
        sessionStorage.setItem(KEY, JSON.stringify(receipt));
        return receipt;
    },
    cancel() {
        const receipt = this.read();
        if (receipt) {
            receipt.cancelRequested = true;
            sessionStorage.setItem(KEY, JSON.stringify(receipt));
        }
        return receipt;
    },
    clear(submissionId) {
        if (this.read()?.submissionId === submissionId) sessionStorage.removeItem(KEY);
    },
};
