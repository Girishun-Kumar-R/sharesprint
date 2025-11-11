(() => {
    const blockedCombo = (event) => {
        const key = event.key.toLowerCase();
        if (event.ctrlKey || event.metaKey) {
            if (["s", "p", "u", "c"].includes(key)) return true;
            if (event.shiftKey && ["i", "j", "c"].includes(key)) return true;
        }
        if (["f12"].includes(key)) return true;
        return false;
    };

    document.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('dragstart', (e) => e.preventDefault());
    document.addEventListener('copy', (e) => e.preventDefault());
    document.addEventListener('keydown', (e) => {
        if (blockedCombo(e)) {
            e.preventDefault();
            e.stopPropagation();
        }
    });

    const deterrent = document.createElement('div');
    deterrent.className = 'screen-deterrent';
    deterrent.textContent = 'Previews are protected';
    document.body.appendChild(deterrent);

    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox hidden';
    lightbox.innerHTML = '<canvas></canvas>';
    document.body.appendChild(lightbox);
    const canvas = lightbox.querySelector('canvas');
    const ctx = canvas.getContext('2d');

    let currentImg = null;

    const openLightbox = (src) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const maxW = window.innerWidth * 0.8;
            const maxH = window.innerHeight * 0.8;
            let { width, height } = img;
            const ratio = Math.min(maxW / width, maxH / height, 1);
            width = Math.floor(width * ratio);
            height = Math.floor(height * ratio);
            canvas.width = width;
            canvas.height = height;
            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(img, 0, 0, width, height);
            lightbox.classList.remove('hidden');
            currentImg = src;
        };
        img.src = src;
    };

    const closeLightbox = () => {
        lightbox.classList.add('hidden');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        currentImg = null;
    };

    lightbox.addEventListener('click', closeLightbox);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeLightbox();
    });

    const bindLightbox = (nodeList) => {
        Array.from(nodeList).forEach((el) => {
            el.addEventListener('click', (evt) => {
                evt.preventDefault();
                const source = el.getAttribute('src') || el.dataset.src;
                if (source) openLightbox(source);
            });
        });
    };

    bindLightbox(document.querySelectorAll('.thumb img'));
    bindLightbox(document.querySelectorAll('.preview img'));

    document.querySelectorAll('.veil').forEach((veil) => {
        veil.addEventListener('click', (evt) => {
            const src = veil.dataset.src;
            if (src) {
                evt.preventDefault();
                openLightbox(src);
                return;
            }
            const link = veil.closest('a');
            if (link) {
                evt.preventDefault();
                link.click();
            }
        });
    });

    const pluralize = (count) => (count === 1 ? 'photo' : 'photos');

    const initSelectionForms = () => {
        const forms = document.querySelectorAll('[data-selection-form]');
        forms.forEach((form) => {
            const checkboxes = Array.from(form.querySelectorAll('.photo-select'));
            if (!checkboxes.length) return;
            const submit = form.querySelector('[data-selection-submit]');
            const counter = form.querySelector('[data-selection-count]');

            const updateState = () => {
                const count = checkboxes.filter((cb) => cb.checked).length;
                if (counter) counter.textContent = `${count} ${pluralize(count)} selected`;
                if (submit) submit.disabled = count === 0;
            };

            checkboxes.forEach((checkbox) => {
                checkbox.addEventListener('change', updateState);
            });
            updateState();
        });
    };

    initSelectionForms();

    const initBackLinks = () => {
        const links = document.querySelectorAll('[data-back-link]');
        links.forEach((link) => {
            link.addEventListener('click', (evt) => {
                const fallback = link.getAttribute('href');
                const ref = document.referrer ? new URL(document.referrer, window.location.origin) : null;
                const sameOrigin = ref && ref.origin === window.location.origin;
                if (sameOrigin) {
                    evt.preventDefault();
                    window.history.back();
                } else if (fallback) {
                    // allow default navigation
                } else {
                    evt.preventDefault();
                }
            });
        });
    };

    initBackLinks();
})();
