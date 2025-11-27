document.addEventListener('DOMContentLoaded', () => {
    // Array of common image extensions for validation
    const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'];

    const fileInput = document.getElementById('file');
    const urlInput = document.getElementById('image_url');
    const nameDisplay = document.getElementById('file-name');
    const urlError = document.getElementById('url-error');
    const form = document.getElementById('prediction-form');
    let dropArea = document.querySelector('.card');

    // --- Core UI Logic ---
    window.setFileName = function(input) {
        urlError.style.display = 'none';

        if (input.id === 'file') {
            // File chosen: display name, clear URL field
            const fn = fileInput.files[0] ? fileInput.files[0].name : 'No file chosen';
            nameDisplay.innerText = fn;
            if (fileInput.files[0]) {
                urlInput.value = "";
            }
        } else if (input.id === 'image_url') {
            // URL entered: clear file input, update name display
            if (urlInput.value.trim() !== "") {
                fileInput.value = ""; // Clear file input
                nameDisplay.innerText = "Image URL ready for prediction.";
            } else {
                nameDisplay.innerText = fileInput.files[0] ? fileInput.files[0].name : 'No file chosen';
            }
        }
    };

    // --- Validation Logic ---
    function validateAndSubmit(event) {
        urlError.style.display = 'none';

        // 1. Check if both are empty
        if (urlInput.value.trim() === "" && fileInput.files.length === 0) {
            urlError.innerText = "⚠️ Error: Please upload a file or provide a URL.";
            urlError.style.display = 'block';
            event.preventDefault();
            return false;
        }

        // 2. Check URL for file extension if file is NOT provided
        if (urlInput.value.trim() !== "" && fileInput.files.length === 0) {
            const url = urlInput.value.trim().toLowerCase();

            // Simple check to see if the URL contains a common image extension
            const isImageLink = IMAGE_EXTENSIONS.some(ext => url.includes(ext));

            if (!isImageLink) {
                urlError.innerText = "⚠️ Error: URL must be a direct link to an image file (.jpg, .png, etc.).";
                urlError.style.display = 'block';
                event.preventDefault();
                return false;
            }
        }

        // 3. Prevent form submission if both have content (only one allowed)
        if (urlInput.value.trim() !== "" && fileInput.files.length > 0) {
             urlError.innerText = "⚠️ Error: Please use EITHER the file upload OR the URL input, not both.";
             urlError.style.display = 'block';
             event.preventDefault();
             return false;
        }

        return true; // Proceed with submission
    }

    // Attach validation to the form's submit event
    if (form) {
        form.addEventListener('submit', validateAndSubmit);
    }

    // --- Drag and Drop Logic ---
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        if (dropArea) {
            dropArea.addEventListener(eventName, preventDefaults, false);
        }
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    if (dropArea) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
        });

        // Handle dropped files/URLs
        dropArea.addEventListener('drop', (e) => {
            const dataTransfer = e.dataTransfer;
            const url = dataTransfer.getData('text/uri-list');

            if (url) {
                // Dropped a URL (e.g., dragging an image from a browser)
                urlInput.value = url;
                setFileName(urlInput);
            } else if (dataTransfer.files.length > 0) {
                // Dropped an image file
                fileInput.files = dataTransfer.files;
                setFileName(fileInput);
            } else {
                // Attempt to grab text that might be a URL
                const textData = dataTransfer.getData('text/plain');
                if (textData && (textData.startsWith('http://') || textData.startsWith('https://'))) {
                    urlInput.value = textData;
                    setFileName(urlInput);
                } else {
                    nameDisplay.innerText = "Drop failed. Please drop an image file or an image from a website.";
                }
            }
        }, false);
    }
});